# -*- coding: utf-8 -*-
"""Globalized Newton solver for the quasi-static equilibrium, shared by 2D and 3D.

Why this exists
---------------
The equilibrium residual is ``f(q) = -dE/dq - F_contact - W`` and the solver
looks for ``f(q_free) = 0``.  A plain full-step Newton iteration

    q_free <- q_free - J_ff^{-1} f_free

is only reliable while ``J_ff`` is positive definite.  A compressed rod is
*post-buckled*: the straight configuration is still an exact equilibrium, but
``J_ff`` picks up negative eigenvalues (one pair per buckling mode), and in 3D
there are twice as many as in 2D because each mode exists in both the y and
the z direction.  On such an indefinite system an undamped Newton step can
increase the residual without bound — in practice it blows up from ~1e0 to
~1e21 within a handful of iterations, burns all ``max_newton_iter``
iterations, and prints "Maximum number of iterations reached."

Two standard safeguards fix that, and neither changes the answer when plain
Newton was already working:

1. **Non-monotone backtracking line search** — Newton's residual on this
   problem is legitimately non-monotone: a converging 3D sequence looks like
   ``1.5e6 -> 2.7e6 -> 7.8e5 -> ... -> 6.8e2 -> 1.8e4 -> 4.4e2``, with
   temporary jumps of 40x on the way down.  Demanding a decrease at *every*
   iteration rejects all of those and cripples the solver (measured: 27
   iterations and 94 backtracks per step instead of 10 clean full steps).  So
   the step is accepted whenever it stays at or below the largest residual of
   the last ``nonmonotone_memory`` iterations (Grippo's criterion), and only
   backtracked when it exceeds even that — which a genuine runaway
   (``1e15 -> 1e21``) always does.
2. **Adaptive Tikhonov regularization** — if no step length helps, ``J_ff`` is
   shifted by ``lambda * scale * I`` (lambda escalating) which bends the step
   back toward gradient descent on ``||f||``.  On success lambda relaxes again.

Both are on by default and configured by the ``newton:`` block of the YAML
config.  Set ``line_search: false`` to recover the original undamped iteration
exactly.
"""
from __future__ import annotations

import numpy as np

from contact import get_contact_energy, get_contact_forces


def solve_equilibrium(q_start, *,
                      energy,
                      W,
                      free_index,
                      dim,
                      tol,
                      obstacles=(),
                      n_pos=None,
                      max_iter=1000,
                      q_old=None,
                      u_old=None,
                      dt=None,
                      m=None,
                      mMat=None,
                      use_inertia=False,
                      line_search=True,
                      max_backtrack=12,
                      nonmonotone_memory=8,
                      reg=0.0,
                      max_reg_escalations=6,
                      max_stalled_steps=10,
                      verbose=False):
    """Solve ``f(q_free) = 0`` starting from ``q_start`` (boundary DOFs already set).

    Returns ``(q, flag, J, info)`` where ``flag`` is 1 on convergence and -1
    otherwise, ``J`` is the equilibrium Jacobian at the returned ``q`` (the
    adjoint gradient consumes it), and ``info`` carries solver diagnostics.
    """
    q = np.asarray(q_start, dtype=float).copy()
    free = free_index

    def residual(qq, want_jac):
        """f = F_inertia - F_elastic - F_contact - W, and optionally J = df/dq."""
        if use_inertia:
            F_in = m / dt * ((qq - q_old) / dt - u_old)
        else:
            F_in = 0.0

        if want_jac:
            F_el, J_el = energy.force_and_jacobian(qq)
            Fc, Jc = get_contact_forces(qq, obstacles, dim=dim, n_pos=n_pos)
            J = -J_el - Jc
            if use_inertia:
                J = J + mMat / dt ** 2
        else:
            # Force-only evaluation: the line search never needs the Hessian,
            # which is the expensive part of the assembly.
            F_el = energy.force(qq)
            Fc = get_contact_forces(qq, obstacles, dim=dim, forces_only=True, n_pos=n_pos)
            J = None

        return F_in - F_el - Fc - W, J

    lam = float(reg)
    n_backtracks = 0
    n_regularized = 0
    stalled = 0

    f, J = residual(q, True)
    r = float(np.linalg.norm(f[free]))

    # Sliding window of recent residuals; a step is acceptable if it does not
    # exceed the worst of them (non-monotone / Grippo criterion).
    memory = max(1, int(nonmonotone_memory))
    recent = [r]

    for it in range(max_iter):
        if not np.isfinite(r):
            return q, -1, J, _info(it, r, n_backtracks, n_regularized, "residual overflow")
        if r < tol:
            return q, 1, J, _info(it, r, n_backtracks, n_regularized, "")

        J_ff = J[np.ix_(free, free)]
        f_free = f[free]
        scale = max(float(np.abs(np.diag(J_ff)).max()), 1.0)

        accepted = False
        for _ in range(max_reg_escalations):
            A = J_ff + lam * scale * np.eye(J_ff.shape[0]) if lam > 0.0 else J_ff
            try:
                dq = np.linalg.solve(A, f_free)
            except np.linalg.LinAlgError:
                lam = max(lam * 10.0, 1e-10)
                n_regularized += 1
                continue
            if not np.all(np.isfinite(dq)):
                lam = max(lam * 10.0, 1e-10)
                n_regularized += 1
                continue

            r_ref = max(recent)
            alpha = 1.0
            for b in range(max_backtrack if line_search else 1):
                q_try = q.copy()
                q_try[free] = q[free] - alpha * dq
                f_try, _ = residual(q_try, False)
                r_try = float(np.linalg.norm(f_try[free]))
                if np.isfinite(r_try) and (r_try <= r_ref or not line_search):
                    accepted = True
                    n_backtracks += b
                    break
                alpha *= 0.5
            if accepted:
                break
            lam = max(lam * 10.0, 1e-8)
            n_regularized += 1

        if not accepted:
            # Nothing helped. Fall back to the plain undamped step (the
            # original behaviour) so this is never worse than before, but give
            # up after a few consecutive stalls instead of spinning to
            # max_iter.
            stalled += 1
            if stalled > max_stalled_steps:
                return q, -1, J, _info(it, r, n_backtracks, n_regularized,
                                       "line search and regularization both stalled")
            q_try = q.copy()
            q_try[free] = q[free] - dq
        else:
            stalled = 0
            lam = max(lam * 0.5, float(reg))

        q = q_try
        f, J = residual(q, True)
        r = float(np.linalg.norm(f[free]))
        recent.append(r)
        if len(recent) > memory:
            recent.pop(0)

    return q, -1, J, _info(max_iter, r, n_backtracks, n_regularized,
                           "max iterations reached")


def _info(iters, residual, n_backtracks, n_regularized, reason):
    return {
        "iterations": iters,
        "residual": residual,
        "n_backtracks": n_backtracks,
        "n_regularized": n_regularized,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Stability, the potential, and the energy descent of the snap rule
# ---------------------------------------------------------------------------
#
# The residual above is f = dPi/dq for the total potential
#
#     Pi(q) = E(q) + E_contact(q) - W . q
#
# (quasi-static, no inertia), and J = df/dq is its Hessian.  An equilibrium is
# a stationary point of Pi; the physically realized ones are its strict local
# minima, where J_ff is positive definite (paper, Sec. III-A, Assumption 2).
# Newton finds any stationary point near its warm start, saddles included --
# a straight rod pushed past its Euler load is the classic case -- so the
# environments check every converged solve with :func:`is_stable` and, when
# the boundary path crosses a fold, let the rod fall into the next valley by
# descending Pi (:func:`minimize_potential`).  That descent is the overdamped
# limit of the snap-through: each iterate lowers the potential, so it cannot
# stop on the saddle it left and ends in a minimum, which is the branch the
# rollout then continues on (src/continuation.py).


def is_stable(J, free_index):
    """Whether the equilibrium with residual Jacobian ``J`` is a strict local
    minimum of the potential: ``J_ff`` positive definite, decided by a Cholesky
    factorization of its symmetric part (the adjoint reuses the factor)."""
    H = J[np.ix_(free_index, free_index)]
    H = 0.5 * (H + H.T)
    if not np.all(np.isfinite(H)):
        return False
    try:
        np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return False
    return True


def total_potential(q, *, energy, W, obstacles=(), dim=2, n_pos=None):
    """``Pi(q) = E(q) + E_contact(q) - W . q`` -- the potential whose gradient
    is the quasi-static residual and whose Hessian is ``J``."""
    q = np.asarray(q, dtype=float)
    return float(energy.energy(q)
                 + get_contact_energy(q, obstacles, dim=dim, n_pos=n_pos)
                 - W @ q)


def _grad_and_hessian(q, energy, W, obstacles, dim, n_pos):
    """``(dPi/dq, d2Pi/dq2)`` over all DOFs, assembled as in solve_equilibrium."""
    F_el, J_el = energy.force_and_jacobian(q)
    Fc, Jc = get_contact_forces(q, obstacles, dim=dim, n_pos=n_pos)
    return -F_el - Fc - W, -J_el - Jc


def minimize_potential(q_start, *,
                       energy,
                       W,
                       free_index,
                       dim,
                       tol,
                       obstacles=(),
                       n_pos=None,
                       max_iter=500,
                       max_backtrack=30,
                       armijo_c=1.0e-4,
                       kick=1.0e-3,
                       max_kicks=4,
                       max_shift_escalations=12,
                       verbose=False):
    """Descend the total potential over the free DOFs from ``q_start``.

    The boundary DOFs of ``q_start`` are held.  The direction is the Newton
    step on ``Pi`` with the Hessian made positive definite by a shift,
    ``(J_ff + mu * scale * I) d = -g``, ``mu`` raised until the Cholesky
    factorization succeeds, and the step length is chosen by backtracking
    under the Armijo condition *on the potential*,

        Pi(q + alpha d) <= Pi(q) + c * alpha * g . d,

    not on the residual norm: every iterate lowers ``Pi``, which is what
    keeps the descent off the saddle it starts next to.  It stops when the
    gradient norm is below ``tol`` (the Newton tolerance), and a final
    Cholesky confirms the end point is a minimum.

    A start that is itself a stationary point -- the straight rod compressed
    past its Euler load with nothing to break the symmetry -- has zero
    gradient but negative curvature.  There the step is ``kick`` metres (at
    the largest node) along the most negative eigenvector of ``J_ff``, with a
    deterministic sign, after which the descent continues; the kick is the
    second-order descent direction at a saddle, not a random perturbation.

    Returns ``(q, flag, J, info)`` like :func:`solve_equilibrium`: ``flag`` is
    1 when the descent converged to a stable minimum and -1 otherwise, ``J``
    is the Hessian at the returned ``q``, and ``info`` records the iterations,
    the energy drop ``Pi(q_start) - Pi(q)``, the largest node displacement,
    and the counts of shifts, backtracks and kicks.
    """
    q = np.asarray(q_start, dtype=float).copy()
    free = free_index
    npos = q.size if n_pos is None else int(n_pos)

    def potential(qq):
        return total_potential(qq, energy=energy, W=W, obstacles=obstacles, dim=dim, n_pos=n_pos)

    Pi = potential(q)
    Pi_start = Pi
    f, J = _grad_and_hessian(q, energy, W, obstacles, dim, n_pos)
    mu = 0.0
    n_shifts = n_backtracks = n_kicks = 0
    stable = False
    reason = "max iterations reached"

    def finish(it, flag, reason):
        disp = (q[:npos].reshape(-1, dim) - np.asarray(q_start, dtype=float)[:npos].reshape(-1, dim))
        info = {
            "iterations": it,
            "residual": float(np.linalg.norm(f[free])),
            "n_shifts": n_shifts,
            "n_backtracks": n_backtracks,
            "n_kicks": n_kicks,
            "energy_start": Pi_start,
            "energy_end": Pi,
            "energy_drop": Pi_start - Pi,
            "max_displacement": float(np.linalg.norm(disp, axis=1).max()) if disp.size else 0.0,
            "stable": stable,
            "reason": reason,
        }
        if verbose:
            print(f"[descent] {'ok' if flag > 0 else 'FAIL'}: {info}")
        return q, flag, J, info

    for it in range(max_iter):
        g = f[free]
        r = float(np.linalg.norm(g))
        if not np.isfinite(r) or not np.isfinite(Pi):
            return finish(it, -1, "potential or gradient overflow")
        H = J[np.ix_(free, free)]
        H = 0.5 * (H + H.T)
        scale = max(float(np.abs(np.diag(H)).max()), 1.0)
        try:
            np.linalg.cholesky(H)
            pd = True
        except np.linalg.LinAlgError:
            pd = False

        if r < tol:
            if pd:
                stable = True
                return finish(it, 1, "")
            # A stationary point that is not a minimum: leave it along the
            # most negative curvature direction.
            if n_kicks >= max_kicks:
                return finish(it, -1, "stationary point with negative curvature, kicks exhausted")
            w, V = np.linalg.eigh(H)
            v = V[:, 0]
            v = v * np.sign(v[np.argmax(np.abs(v))])       # deterministic sign
            d = kick * v / np.abs(v).max()
            n_kicks += 1
            kicked = True
        else:
            kicked = False
            d = None
            for _ in range(max_shift_escalations):
                A = H + mu * scale * np.eye(H.shape[0]) if mu > 0.0 else H
                try:
                    L = np.linalg.cholesky(A)
                except np.linalg.LinAlgError:
                    mu = max(mu * 10.0, 1.0e-8)
                    n_shifts += 1
                    continue
                y = np.linalg.solve(L, -g)
                cand = np.linalg.solve(L.T, y)
                if np.all(np.isfinite(cand)) and float(g @ cand) < 0.0:
                    d = cand
                    break
                mu = max(mu * 10.0, 1.0e-8)
                n_shifts += 1
            if d is None:
                return finish(it, -1, "no descent direction")

        gd = float(g @ d)
        alpha = 1.0
        accepted = False
        for b in range(max_backtrack):
            q_try = q.copy()
            q_try[free] = q[free] + alpha * d
            Pi_try = potential(q_try)
            if np.isfinite(Pi_try) and Pi_try <= Pi + armijo_c * alpha * gd:
                accepted = True
                n_backtracks += b
                break
            alpha *= 0.5
        if not accepted:
            if kicked:
                return finish(it, -1, "kick along the negative curvature did not lower the potential")
            # Bend the step further toward steepest descent and try again.
            mu = max(mu * 10.0, 1.0e-8)
            n_shifts += 1
            if mu > 1.0e8:
                return finish(it, -1, "line search on the potential stalled")
            continue

        q, Pi = q_try, Pi_try
        f, J = _grad_and_hessian(q, energy, W, obstacles, dim, n_pos)
        mu = mu * 0.1 if mu > 1.0e-12 else 0.0

    return finish(max_iter, -1, reason)
