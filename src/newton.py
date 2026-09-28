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

from contact import get_contact_forces


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
