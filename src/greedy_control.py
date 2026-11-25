"""Greedy shape control: no policy, no horizon, one quasi-static step at a time.

Every step moves the controlled gripper by the increment that most lowers the
shape error of the equilibrium it lands in -- a one-step lookahead on the
real physics.  The increment is found in two parts:

1. **Direction.**  At the current equilibrium the implicit function theorem
   gives the shape's sensitivity to the gripper pose (the same single-step
   sensitivity the adjoint uses, :func:`implicit_grad_tools.local_dxf_dxb`)::

       x(u + du) ~ x(u) + G du,      G = dx/du_c   (all node positions)

   and the direction is the Levenberg-Marquardt step of the linearized
   objective, scaled so that ``angle_scale`` metres count as one radian and
   clipped to the trust radius ``step_cap``.

2. **Step length.**  Each trial length ``alpha * du`` is taken as a real
   ``env.step`` from the current state (continuation, stability check, snap
   rule; src/continuation.py), and the trial with the lowest objective is
   kept if it lowers the objective.  Otherwise the trust radius halves; it
   grows back by 1.5x after every accepted step.  A trial that fails to
   converge, ends unstable, hits the singular vertical gripper tangent or
   wraps the twist model's holonomy (|dH| > pi in one step, where the twist
   energy is discontinuous, src/rodframes.py) is rejected.

The objective is the mean squared node error against the current target plus
an optional quadratic penalty ``reg`` on the controlled DOFs' distance from a
reference: their start values, or -- for the DOFs given in ``ref_final`` --
``u0 + lam (ref_final - u0)``, a feedforward that moves with the schedule
(e.g. the wrist yaw a multi-turn helix needs to be a *stable* equilibrium,
which a shape-only objective does not see until the rod snaps).  The target follows a schedule ``lam -> shape``: ``direct``
uses ``lam = 1`` throughout, ``homotopy`` starts at ``lam = 0`` and advances
by ``dlam`` whenever the rms error to the current target is below
``advance_tol``, so the rod follows a family of shapes into the final one.

Being greedy, the method stops at the first local minimum of the objective
along the path it took: when the trust radius falls below ``step_min``, or
(once ``lam = 1``) when ``patience`` accepted steps lowered the objective by
less than ``rel_tol`` of its value.  ``time_limit`` (seconds) caps the wall
time, so a run grinding through snapping trials ends as stuck.  With
``max_retreats > 0`` a collapse of the trust radius before ``lam = 1`` is not
the end: ``lam`` steps back by ``retreat``, ``dlam`` halves and the trust
radius resets (the usual step-size control of numerical continuation).
"""
from __future__ import annotations

import time

import numpy as np

from implicit_grad_tools import local_dxf_dxb, _boundary_jacobian

#: The ``greedy:`` block of configs/default.yaml, for callers without a config.
DEFAULTS = {
    "schedule": "homotopy", "dlam": 0.01, "advance_tol": 0.02, "step_cap": 0.01,
    "step_min": 1.0e-4, "angle_scale": 0.1, "alphas": [1.0, 0.5, 0.25], "damping": 1.0e-4,
    "reg": None, "ref_final": None, "max_steps": 800, "patience": 50, "rel_tol": 1.0e-3,
    "time_limit": None, "max_retreats": 0, "retreat": 0.05,
}
SCHEDULES = ("homotopy", "direct")


def shape_sensitivity(env, J, u, cols):
    """``dx/du[cols]``, shape ``(n_pos, len(cols))``, at the equilibrium whose Jacobian is ``J``.

    Free nodes through ``-J_ff^{-1} J_fb``, clamped nodes through the
    boundary map directly; ``x`` is every node position (no twist DOFs).
    """
    dxf = local_dxf_dxb(J, env.free_index, env.fixed_index, jac_reg=1e-9)
    B = _boundary_jacobian(u, env.deltaL, env.dim, getattr(env, "twist", False),
                           getattr(env, "boundary_rows", None))
    G = np.zeros((env.n_dof, len(u)))
    G[env.free_index] = dxf @ B
    G[env.fixed_index] = B
    return G[:env.n_pos][:, cols]


def _holonomy(env):
    if not getattr(env, "twist", False):
        return None
    from rodframes import holonomy
    return float(holonomy(env.positions()))


def greedy_shape_control(env, u0, J0, target_fn, cols, angle_cols=(), opts=None, log=print,
                         log_every=10):
    """Drive the rod from its current equilibrium toward ``target_fn(1.0)`` greedily.

    ``env`` holds the start state (settled at ``u0``, Jacobian ``J0``).
    ``target_fn(lam)`` returns the ``(nv, 3)`` target at schedule value
    ``lam``; ``cols`` are the control entries the greedy may move (the right
    gripper), ``angle_cols`` the subset that are angles.  ``opts`` overrides
    :data:`DEFAULTS`.  Returns a dict of per-step records (index 0 = start)
    and the reason it stopped; the environment is left in the final state.
    """
    o = dict(DEFAULTS, **(opts or {}))
    if o["schedule"] not in SCHEDULES:
        raise ValueError(f"greedy.schedule must be one of {SCHEDULES}, got {o['schedule']!r}")
    cols = np.asarray(cols)
    nc = len(cols)
    scale = np.ones(nc)
    scale[np.isin(cols, angle_cols)] = 1.0 / float(o["angle_scale"])   # normalized -> control units
    reg = np.zeros(nc) if o["reg"] is None else np.asarray(o["reg"], dtype=float)
    if reg.shape != (nc,):
        raise ValueError(f"greedy.reg needs {nc} entries (one per controlled DOF), got {reg.shape}")
    u_start = np.asarray(u0, dtype=float)[cols].copy()
    ref_final = u_start.copy()
    if o["ref_final"] is not None:
        rf = list(o["ref_final"])
        if len(rf) != nc:
            raise ValueError(f"greedy.ref_final needs {nc} entries (null keeps the start value)")
        ref_final = np.array([u_start[i] if v is None else float(v) for i, v in enumerate(rf)])

    def u_ref_at(lam_):
        return u_start + lam_ * (ref_final - u_start)
    final = np.asarray(target_fn(1.0), dtype=float)
    homotopy = o["schedule"] == "homotopy"

    def objective(X, u, tgt):
        e = float(np.mean(np.sum((X - tgt) ** 2, axis=1)))
        return e + float(reg @ (u[cols] - u_ref_at(lam)) ** 2), e

    def rms(X, tgt):
        return float(np.sqrt(np.mean(np.sum((X - tgt) ** 2, axis=1))))

    u = np.asarray(u0, dtype=float).copy()
    J = J0
    lam = 0.0 if homotopy else 1.0
    cap = float(o["step_cap"])
    dlam = float(o["dlam"])
    retreats = 0
    rec = {k: [] for k in ("step", "lam", "u", "q", "objective", "rms_target", "rms_final",
                           "holonomy", "twist", "snap", "substeps", "alpha", "cap", "trials")}
    rejected = {"newton": 0, "unstable": 0, "singular": 0, "wrap": 0, "no_decrease": 0}
    snap_trials = 0                                       # trials (kept or not) that snapped

    def record(k, tgt, info=None, alpha=0.0, trials=0):
        X = env.positions()
        rec["step"].append(k)
        rec["lam"].append(lam)
        rec["u"].append(u.copy())
        rec["q"].append(env.q.copy())
        rec["objective"].append(objective(X, u, tgt)[0])
        rec["rms_target"].append(rms(X, tgt))
        rec["rms_final"].append(rms(X, final))
        rec["holonomy"].append(_holonomy(env))
        rec["twist"].append(env.energy.total_twist(env.q) if getattr(env, "twist", False) else None)
        rec["snap"].append(bool(info["snap"]) if info else False)
        rec["substeps"].append(int(info["substeps"]) if info else 0)
        rec["alpha"].append(alpha)
        rec["cap"].append(cap)
        rec["trials"].append(trials)

    tgt = np.asarray(target_fn(lam), dtype=float)
    record(0, tgt)
    t0 = time.time()
    stop = "max_steps"
    k = 0
    for it in range(int(o["max_steps"])):
        if o["time_limit"] is not None and time.time() - t0 > float(o["time_limit"]):
            stop = f"time limit {float(o['time_limit']):g} s"
            break
        # Advance only once per accepted step: the last record must have been
        # scored against the current target.  (Advancing on rejected iterations
        # too let the target run away from a rod that was not moving.)
        if (homotopy and lam < 1.0 and rec["lam"][-1] == lam
                and rec["rms_target"][-1] < o["advance_tol"]):
            lam = min(1.0, lam + dlam)
            tgt = np.asarray(target_fn(lam), dtype=float)
        X = env.positions()
        obj0, _ = objective(X, u, tgt)

        # LM direction in normalized units: d (scaled) -> du = scale * d
        G = shape_sensitivity(env, J, u, cols) * scale[None, :]
        n = X.shape[0]
        r = (X - tgt).ravel()
        R = n * reg * scale ** 2                         # the penalty's curvature, normalized
        A = G.T @ G
        A = A + float(o["damping"]) * np.trace(A) / nc * np.eye(nc) + np.diag(R)
        g = G.T @ r + n * reg * scale * (u[cols] - u_ref_at(lam))
        d = -np.linalg.solve(A, g)
        big = np.max(np.abs(d))
        if big > cap:
            d *= cap / big

        state = env.get_state()
        H0 = _holonomy(env)
        best = None
        trials = 0
        for a in o["alphas"]:
            env.set_state(state)
            un = u.copy()
            un[cols] += a * d * scale
            trials += 1
            try:
                _, _, info = env.step(un)
            except ValueError:                           # vertical gripper tangent
                rejected["singular"] += 1
                continue
            snap_trials += bool(info["snap"])
            if not info["converged"]:
                rejected["newton"] += 1
                continue
            if info["stable"] is False:
                rejected["unstable"] += 1
                continue
            if H0 is not None and abs(_holonomy(env) - H0) > np.pi:
                rejected["wrap"] += 1
                continue
            obj, _ = objective(env.positions(), un, tgt)
            if best is None or obj < best[0]:
                best = (obj, un, info, env.get_state(), a)
        if best is None or best[0] >= obj0:
            rejected["no_decrease"] += best is not None
            env.set_state(state)
            cap *= 0.5
            if cap < o["step_min"]:
                if homotopy and lam < 1.0 and retreats < int(o["max_retreats"]):
                    retreats += 1
                    lam = max(rec["lam"][-1] - float(o["retreat"]), 0.0)
                    dlam *= 0.5
                    cap = float(o["step_cap"])
                    tgt = np.asarray(target_fn(lam), dtype=float)
                    if log:
                        log(f"  retreat {retreats}: lam -> {lam:.3f}, dlam -> {dlam:.4f}")
                    continue
                stop = "trust radius below step_min"
                break
            continue

        k += 1
        _, u, info, s_new, a = best
        env.set_state(s_new)
        J = info["J"]
        cap = min(float(o["step_cap"]), 1.5 * cap)
        record(k, tgt, info, a, trials)
        if log and k % log_every == 0:
            hol = rec["holonomy"][-1]
            log(f"  step {k:4d}  lam {lam:.2f}  rms target {1e3 * rec['rms_target'][-1]:6.1f} mm  "
                f"helix {1e3 * rec['rms_final'][-1]:6.1f} mm  u_c {np.round(u[cols], 3)}"
                + (f"  H {hol:+.2f}  twist {rec['twist'][-1]:+.2f}" if hol is not None else "")
                + f"  {time.time() - t0:.0f}s")
        p = int(o["patience"])
        if lam >= 1.0 and len(rec["objective"]) > p and rec["lam"][-p - 1] >= 1.0:
            old = rec["objective"][-p - 1]
            if old - rec["objective"][-1] < float(o["rel_tol"]) * old:
                stop = f"objective dropped < {o['rel_tol']:g} (relative) over {p} steps"
                break

    out = {key: np.array(v) if key not in ("holonomy", "twist") else
           np.array([np.nan if x is None else x for x in v]) for key, v in rec.items()}
    out.update(stop=stop, rejected=rejected, snap_trials=snap_trials, time=time.time() - t0,
               lam_final=lam, retreats=retreats,
               steps=k, iterations=it + 1 if o["max_steps"] else 0)
    return out
