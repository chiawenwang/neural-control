# -*- coding: utf-8 -*-
"""Task P (path tracking): reference paths and the one-step reference tracker.

Reference path.  The mid node follows ``p(lambda)``, anchored at its position
``m0`` in the initial equilibrium; step ``k`` (k = 1..M, ``lambda_k = k/M``)
targets ``p(lambda_k)``.  ``reference_path`` returns the rows ``p(lambda_1)``
.. ``p(lambda_M)`` (shape (M, 3)), the same targets as
env_factory.make_target_trajectory(``sine_xz``) and train_tracking.py.

Reference tracker.  A one-step horizon with the exact per-step sensitivity
(paper Sec. VII-A, "Greedy"): at the equilibrium of step k-1 the implicit
function theorem gives the mid node's sensitivity to the gripper pose,
``G = dx_mid/du`` (3 x n_u, greedy_control.shape_sensitivity, the same
single-step sensitivity the adjoint uses), and the increment is the damped
least-squares (Levenberg-Marquardt) step towards ``p(lambda_k)``::

    du = W G_W^T (G_W G_W^T + (mu s_max)^2 I)^{-1} e,   G_W = G W,
    W = diag(1, 1, 1, 1/a, 1/a, 1/a)        (a = angle_scale [m per rad])

bounded by the step cap (the inf-norm of ``du / W``, i.e. metres, with
angles counted as ``a`` metres per radian; the designed motion uses
``max_step``, the greedy baseline the largest step of our solutions).  Each
trial length ``alpha * du`` is a real quasi-static step from the previous
equilibrium; a trial that fails (Newton, unstable, model error), leaves the
workspace, passes through a peg, or -- the task's rule -- is flagged as a
snap-through, is rejected.  The trial with the lowest error at step k is
kept; if none is admissible, or none is better than holding the pose, the
pose is held.  Exactly M steps are taken, so the result is a command
sequence of the horizon (M, n_u), which replays to the same states.

It serves twice: ``designed=True`` (cap ``max_step``) gives the designed
motion that shows the path is reachable -- the common initialization is a
perturbation of it, as the designed motion of a shape target in S -- and,
with the cap of our solutions, the greedy baseline.
"""
from __future__ import annotations

import time

import numpy as np

from greedy_control import shape_sensitivity

PATH_KINDS = ("sine_xz", "line")


def reference_path(spec, m0, M):
    """Rows ``p(lambda_k)``, k = 1..M, of the path ``spec`` anchored at ``m0``: (M, 3).

    ``kind: sine_xz``: ``m0 + (dx t, dy t, amp sin(2 pi periods t))``;
    ``kind: line``:    ``m0 + t (end - m0)`` (``end`` absolute, or ``d`` an offset).
    """
    kind = spec.get("kind", "sine_xz")
    m0 = np.asarray(m0, dtype=float).reshape(3)
    t = (np.arange(1, M + 1) / M)[:, None]
    if kind == "sine_xz":
        d = np.array([float(spec["dx"]), float(spec["dy"]), 0.0])
        P = m0 + t * d
        P[:, 2] += float(spec["amp"]) * np.sin(2 * np.pi * float(spec.get("periods", 0.5)) * t[:, 0])
        return P
    if kind == "line":
        end = np.asarray(spec["end"], float) if "end" in spec else m0 + np.asarray(spec["d"], float)
        return m0 + t * (end - m0)
    raise ValueError(f"path kind must be one of {PATH_KINDS}, got {kind!r}")


def describe(spec):
    if "via" in spec:
        return f"mid-node path of a designed motion through {len(spec['via'])} via points"
    if spec.get("kind", "sine_xz") == "sine_xz":
        return (f"half sine: dx {spec['dx']:+.3f} m, dy {spec['dy']:+.3f} m, z amp {spec['amp']:.3f} m "
                f"({spec.get('periods', 0.5)} periods)")
    return f"straight line to {spec.get('end', spec.get('d'))}"


def mid_sensitivity(env, J, u, k_end, node):
    """``dx_node/du_gripper`` (3, n_u - k_end) at the equilibrium with Jacobian ``J``."""
    cols = np.arange(k_end, len(u))
    G = shape_sensitivity(env, J, u, cols)
    return G[3 * node:3 * node + 3]


def track_reference(env, checkpoint, u0, ref, node, *, cap, damping=0.03, angle_scale=0.2,
                    alphas=(1.0, 0.5, 0.25), ws_bounds=None, reject_snap=True, pegs=(),
                    log=None, log_every=25):
    """Follow ``ref`` (M, 3) with the mid node ``node``, one quasi-static step per row.

    ``cap``: bound on ``max |du / W|`` per step (see the module docstring).
    ``ws_bounds``: ``(lo, hi)`` of the gripper's controls, or None; a trial
    is clipped into the box (the clipped pose is what is executed).
    Returns a dict: ``U`` (M, n_u) commands, ``q_list``, per-step ``err``,
    ``alpha`` (0 = held), ``held``, ``rejected`` counts, ``ok`` (all M
    steps committed) and ``time``.  ``env`` is left in the final state.
    """
    t0 = time.time()
    u = np.asarray(u0, dtype=float).copy()
    k_end = len(u) // 2
    k = len(u) - k_end
    W = np.ones(k)
    W[3:] = 1.0 / float(angle_scale)
    env.set_state(checkpoint)
    J = env._last_J
    if J is None:
        raise RuntimeError("the checkpoint has no equilibrium Jacobian (settle the start first)")
    M = len(ref)
    U, q_list, err, alpha_used = [], [], [], []
    rej = {"newton": 0, "unstable": 0, "error": 0, "snap": 0, "tunnel": 0, "workspace_clip": 0}
    held = 0
    q_prev = np.asarray(checkpoint["q"]).copy()
    ok = True
    for j in range(M):
        st = env.get_state()
        x = env.positions(q_prev)[node]
        e = np.asarray(ref[j], float) - x
        G = mid_sensitivity(env, J, u, k_end, node) * W[None, :]
        smax = max(np.linalg.svd(G, compute_uv=False)[0], 1e-12)
        d = G.T @ np.linalg.solve(G @ G.T + (float(damping) * smax) ** 2 * np.eye(3), e)
        big = np.max(np.abs(d))
        if big > cap:
            d *= cap / big
        du = W * d
        best = None
        for a in alphas:
            env.set_state(st)
            un = u.copy()
            un[k_end:] += a * du
            if ws_bounds is not None:
                c = np.clip(un[k_end:], ws_bounds[0], ws_bounds[1])
                rej["workspace_clip"] += bool(np.any(c != un[k_end:]))
                un[k_end:] = c
            try:
                _, _, info = env.step(un)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError):
                rej["error"] += 1
                continue
            if not info["converged"]:
                rej["newton"] += 1
                continue
            if info.get("stable") is False:
                rej["unstable"] += 1
                continue
            if reject_snap and info.get("snap"):
                rej["snap"] += 1
                continue
            if any(p.crossed(q_prev, info["q"]) for p in pegs):
                rej["tunnel"] += 1
                continue
            ej = float(np.linalg.norm(env.positions(info["q"])[node] - ref[j]))
            if best is None or ej < best[0]:
                best = (ej, un, env.get_state(), info, a)
        # hold the pose when no trial is admissible or none beats holding
        env.set_state(st)
        try:
            _, _, hinfo = env.step(u.copy())
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError):
            hinfo = {"converged": False}
        h_ok = bool(hinfo.get("converged") and hinfo.get("stable") is not False
                    and not (reject_snap and hinfo.get("snap")))
        if h_ok:
            eh = float(np.linalg.norm(env.positions(hinfo["q"])[node] - ref[j]))
            if best is None or eh <= best[0]:
                best = (eh, u.copy(), env.get_state(), hinfo, 0.0)
        if best is None:
            ok = False
            env.set_state(st)
            if log:
                log(f"  tracker: no admissible step (not even holding) at step {j}")
            break
        ej, u, s_new, info, a = best
        env.set_state(s_new)
        held += a == 0.0
        J = info["J"]
        q_prev = np.asarray(info["q"]).copy()
        U.append(u.copy())
        q_list.append(q_prev)
        err.append(ej)
        alpha_used.append(a)
        if log and (j + 1) % log_every == 0:
            log(f"  step {j + 1:4d}/{M}  error {ej * 1e3:6.2f} mm  alpha {a:.2f}  "
                f"u {np.round(u[k_end:], 3)}  {time.time() - t0:.0f}s")
    err = np.asarray(err)
    return {"U": np.asarray(U), "q_list": q_list, "err": err, "alpha": np.asarray(alpha_used),
            "held": int(held), "rejected": rej, "ok": bool(ok and len(U) == M), "steps": len(U),
            "rms": float(np.sqrt(np.mean(err ** 2))) if len(err) else np.inf,
            "max": float(err.max()) if len(err) else np.inf, "time": time.time() - t0}
