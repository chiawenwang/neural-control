# -*- coding: utf-8 -*-
"""Quasi-static stepping that follows the boundary path (paper, Sec. III-A).

The equilibrium simulators (SimulatorEnv_2D / SimulatorEnv_3D) call
:func:`quasistatic_step` for every command.  What it adds to a plain Newton
solve is the three assumptions the state-based gradient rests on:

1. **Continuation.**  The realized branch should depend on the *continuous*
   boundary path, not on how many control steps it is cut into.  A step from
   the previous pose ``u_prev`` to the commanded ``u`` is solved from the
   previous equilibrium; if Newton does not converge, or converges to an
   unstable point, the step is cut into 2, 4, 8, ... (up to
   ``newton.max_substeps``) substeps, each warm-started from the last.  The
   substeps interpolate the *pose* ``u`` linearly and map every intermediate
   pose through ``Gamma = env._control_node_from_pose`` -- interpolating the
   clamped nodes directly would change the length of the clamped edge.

2. **Stability.**  Every converged solve, main step and substep alike, is
   checked with a Cholesky factorization of ``J_ff`` (:func:`newton.is_stable`),
   so each state the rollout keeps is a strict local minimum of the energy
   under its boundary (Assumption 2).  A failed check triggers the
   subdivision above.

3. **Snap-through.**  When even the finest substep has no stable equilibrium
   near the last one, the boundary path has crossed a fold.  The rod then
   falls into the neighbouring valley: the potential ``E + E_c - W.q`` under
   the failed substep's boundary is descended from the last stable state
   (:func:`newton.minimize_potential`, the overdamped limit), and the
   remaining substeps continue from the minimum it reaches.  The step is
   marked ``snap`` and records the energy drop and the largest node
   displacement.  Because the descent is deterministic and the landing valley
   is stable, a small change of the trajectory parameters lands in the same
   valley (Assumption 3), which is what makes the gradient across a snap
   well defined.

   A fold crossed with a small step is often resolved by Newton itself: from
   the last state before the fold it converges straight to the stable
   equilibrium in the next valley, and the stability check passes.  That is
   the same snap, so it is flagged by the paper's displacement indicator
   (eq. 23): a step whose free nodes move by more than ``snap_disp_ratio``
   times the boundary nodes plus ``snap_disp_eps`` rod lengths is a jump.
   On the paths checked, a steep but continuous response (the onset of
   buckling) moves the free nodes at most ~13 times the boundary and under
   0.03 m per step, a flip of the arch 90-470 times and 0.4-0.6 m, with the
   same landing valley for 25 to 200 control steps.

The three are switched by ``newton.continuation``, ``newton.stability_check``
and ``newton.snap_rule``; with all of them off a step is the single Newton
solve it used to be, bit for bit.  The returned ``J`` is always the Jacobian
at the state the step ends in, after substeps and snaps, since that is what
the adjoint consumes.
"""
from __future__ import annotations

import numpy as np

from newton import is_stable, minimize_potential, total_potential


def node_displacements(env, q_old, q_new):
    """Largest node displacement of the free nodes and of the clamped nodes."""
    dim = env.dim
    n_pos = int(getattr(env, "n_pos", None) or q_old.size)
    P0 = np.asarray(q_old, dtype=float)[:n_pos].reshape(-1, dim)
    P1 = np.asarray(q_new, dtype=float)[:n_pos].reshape(-1, dim)
    d = np.linalg.norm(P1 - P0, axis=1)
    clamped = np.zeros(P0.shape[0], dtype=bool)
    clamped[np.unique(env.fixed_index[env.fixed_index < n_pos] // dim)] = True
    d_free = float(d[~clamped].max()) if (~clamped).any() else 0.0
    d_bound = float(d[clamped].max()) if clamped.any() else 0.0
    return d_free, d_bound


def _stability(flag, J, free_index, check):
    """True / False from the Cholesky test; None when the check is off."""
    if flag < 0:
        return False
    return is_stable(J, free_index) if check else None


def _descend(env, q_from, control_node):
    """Energy descent under ``control_node`` from the free DOFs of ``q_from``."""
    cfg = env.newton_cfg
    q_start = env._apply_boundary(q_from, control_node)
    q, flag, J, info = minimize_potential(
        q_start,
        energy=env.energy,
        W=env.W,
        free_index=env.free_index,
        dim=env.dim,
        tol=env.tol,
        obstacles=env.obstacles,
        n_pos=getattr(env, "n_pos", None),
        max_iter=int(cfg.get("snap_max_iter", 500)),
        kick=float(cfg.get("snap_kick", 1.0e-3)),
        verbose=bool(cfg.get("verbose", False)),
    )
    return q, flag, J, info


def _follow_path(env, u_from, u_to, n, check, allow_snap):
    """Solve the ``n`` substeps of the straight pose path ``u_from -> u_to``.

    Each substep is warm-started from the previous one and, with ``check``,
    accepted only if stable.  A substep without a stable equilibrium ends the
    path (``ok=False``) unless ``allow_snap``: then the potential is descended
    from the last stable state under that substep's boundary and the path
    continues from the minimum reached.
    """
    q = env.q
    iterations = 0
    snaps = []
    last = None
    for j in range(1, n + 1):
        u_j = u_from + (j / n) * (u_to - u_from)
        node_j = env._control_node_from_pose(u_j)
        q_new, flag, J, ninfo = env._solve(q, node_j, False)
        iterations += int(ninfo["iterations"])
        stable = _stability(flag, J, env.free_index, check)
        if flag < 0 or stable is False:
            if not allow_snap:
                return {"ok": False, "iterations": iterations, "fail_at": j, "snaps": snaps,
                        "reason": "newton failed" if flag < 0 else "unstable"}
            q_new, flag, J, dinfo = _descend(env, q, node_j)
            iterations += int(dinfo["iterations"])
            snaps.append({"kind": "descent", "substep": j, "of": n,
                          "energy_drop": dinfo["energy_drop"],
                          "max_displacement": dinfo["max_displacement"],
                          "iterations": dinfo["iterations"],
                          "kicks": dinfo["n_kicks"]})
            if flag < 0:
                return {"ok": False, "iterations": iterations, "fail_at": j, "snaps": snaps,
                        "reason": "energy descent: " + dinfo["reason"]}
            ninfo = dinfo
        q = q_new
        last = (q, J, ninfo)
    q, J, ninfo = last
    return {"ok": True, "q": q, "J": J, "newton": ninfo, "iterations": iterations,
            "snaps": snaps, "reason": ""}


def tangent_prediction(env, control_node):
    """First-order predictor of the next equilibrium (``newton.predictor: tangent``).

    ``x_f ~ x_f,prev + S dx_b`` with ``S = -J_ff^{-1} J_fb`` at the committed
    equilibrium (its Hessian ``env._last_J``, kept by the environment), i.e.
    the interior follows the boundary along the tangent of the branch instead
    of staying where it was.  Returns the plain warm start ``env.q`` when no
    Hessian is available for the current state.  Newton then converges to the
    same equilibrium of the branch from a closer start.
    """
    J = getattr(env, "_last_J", None)
    if J is None:
        return env.q
    free, fixed = env.free_index, env.fixed_index
    xb_new = control_node if env.boundary_rows is None else control_node[env.boundary_rows]
    dxb = np.asarray(xb_new, dtype=float) - env.q[fixed]
    J_ff = J[np.ix_(free, free)]
    J_fb = J[np.ix_(free, fixed)]
    try:
        from scipy.linalg import cho_factor, cho_solve
        dxf = -cho_solve(cho_factor(J_ff, lower=True, check_finite=False), J_fb @ dxb,
                         check_finite=False)
    except Exception:                                   # noqa: BLE001 -- not SPD: no prediction
        return env.q
    q = env.q.copy()
    q[free] += dxf
    q[fixed] = xb_new
    return q


def quasistatic_step(env, control, use_inertia=False):
    """Equilibrium at ``Gamma(control)``, reached along the pose path from ``env.u_prev``.

    Returns a dict with ``q``, ``flag`` (1 converged, -1 not), ``J`` (the
    Jacobian at ``q``), ``newton`` (the final solve's diagnostics),
    ``substeps``, ``newton_iterations`` (summed over every solve the step
    made), ``stable`` (True / False, or None when the check is off),
    ``snap`` with ``snap_by`` (``descent``, ``displacement`` or ``both``)
    and the list ``snaps`` of descents and jumps, and the step's largest
    free-node and clamped-node displacements.  The environment's state is
    not touched; the caller commits the result.
    """
    cfg = env.newton_cfg
    quasi = not use_inertia
    check = quasi and bool(cfg.get("stability_check", False))
    cont = quasi and bool(cfg.get("continuation", False))
    snap = quasi and bool(cfg.get("snap_rule", False))
    max_sub = int(cfg.get("max_substeps", 0))
    control = np.asarray(control, dtype=float)

    def finish(out):
        """Add the displacement indicator and settle the snap flag."""
        q_old = env.q
        d_free, d_bound = node_displacements(env, q_old, out["q"]) if out["flag"] > 0 else (0.0, 0.0)
        out["max_displacement"] = d_free
        out["boundary_displacement"] = d_bound
        by = "descent" if out["snaps"] else None
        if snap and out["flag"] > 0 and out["stable"] is not False:
            kappa = float(cfg.get("snap_disp_ratio", 10.0))
            eps = float(cfg.get("snap_disp_eps", 0.05)) * float(getattr(env, "rod_length", 1.0))
            if d_free > kappa * d_bound + eps:
                by = "both" if by else "displacement"
                if not out["snaps"]:
                    kw = dict(energy=env.energy, W=env.W, obstacles=env.obstacles, dim=env.dim,
                              n_pos=getattr(env, "n_pos", None))
                    out["snaps"] = [{"kind": "jump", "substep": None, "of": None,
                                     "energy_drop": (total_potential(q_old, **kw)
                                                     - total_potential(out["q"], **kw)),
                                     "max_displacement": d_free,
                                     "boundary_displacement": d_bound,
                                     "iterations": out["newton_iterations"], "kicks": 0}]
        out["snap"] = by is not None
        out["snap_by"] = by
        return out

    node = env._control_node_from_pose(control)
    q_start = env.q
    pred = quasi and cfg.get("predictor", "none") == "tangent"
    if pred:
        q_start = tangent_prediction(env, node)
    q1, flag1, J1, ninfo1 = env._solve(q_start, node, use_inertia)
    stable1 = _stability(flag1, J1, env.free_index, check)
    if pred and q_start is not env.q and not (flag1 > 0 and stable1 is not False):
        # The predicted start did not give a stable equilibrium: fall back to
        # the plain warm start before subdividing, so the predictor can only
        # save iterations, never change what the stepping does otherwise.
        it0 = int(ninfo1["iterations"])
        q1, flag1, J1, ninfo1 = env._solve(env.q, node, use_inertia)
        ninfo1 = dict(ninfo1, iterations=int(ninfo1["iterations"]) + it0)
        stable1 = _stability(flag1, J1, env.free_index, check)
    out = {"q": q1, "flag": flag1, "J": J1, "newton": ninfo1, "substeps": 1,
           "newton_iterations": int(ninfo1["iterations"]), "stable": stable1,
           "snap": False, "snaps": [], "reason": ninfo1.get("reason", "")}
    if flag1 > 0 and stable1 is not False:
        return finish(out)

    iterations = out["newton_iterations"]
    u_prev = getattr(env, "u_prev", None)
    if cont and u_prev is not None and max_sub >= 2:
        n = 2
        while n <= max_sub:
            finest = 2 * n > max_sub
            r = _follow_path(env, np.asarray(u_prev, dtype=float), control, n, check,
                             snap and finest)
            iterations += r["iterations"]
            if r["ok"]:
                return finish({"q": r["q"], "flag": 1, "J": r["J"], "newton": r["newton"],
                               "substeps": n, "newton_iterations": iterations,
                               "stable": True if check else None,
                               "snap": bool(r["snaps"]), "snaps": r["snaps"], "reason": ""})
            out["reason"] = f"substeps {n}: {r['reason']} at substep {r['fail_at']}"
            n *= 2
    elif snap:
        q, flag, J, dinfo = _descend(env, env.q, node)
        iterations += int(dinfo["iterations"])
        if flag > 0:
            s = {"kind": "descent", "substep": 1, "of": 1, "energy_drop": dinfo["energy_drop"],
                 "max_displacement": dinfo["max_displacement"],
                 "iterations": dinfo["iterations"], "kicks": dinfo["n_kicks"]}
            return finish({"q": q, "flag": 1, "J": J, "newton": dinfo, "substeps": 1,
                           "newton_iterations": iterations, "stable": True, "snap": True,
                           "snaps": [s], "reason": ""})
        out["reason"] = "energy descent: " + dinfo["reason"]

    out["newton_iterations"] = iterations
    return finish(out)
