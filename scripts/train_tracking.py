"""Task 2: path tracking -- the rod's mid node follows a reference path.

The controller is the adjoint neural control of :mod:`neural_control`, the
same one point-to-point uses; only the loss differs.  Here every step carries
a term, ``mean_k |mid_k - ref_k|^2`` over the N steps of the horizon, so the
adjoint pushes a gradient back from each of them rather than from the final
pose alone.  A step whose Newton solve fails is masked out of the loss and the
rollout continues; an epoch is *valid* only when all N steps converged.

Where the reference comes from:

  --path NAME   one of the named presets in ``PATHS`` below -- straight lines,
                sine curves and circular arcs, all anchored at the mid node's
                own starting position.  These are the paths the RRT comparison
                uses, and the ones ``verify`` checks for feasibility.
  (none)        the config's ``task.tracking.trajectory_type``
                (src/env_factory.py: segment, line, circle, figure8).

Methods, as in point-to-point:

  nc       from a random policy.
  rrt_nc   from a verified plan: ``verify`` plans the path segment by segment
           with the RRT of src/dlo_rrt.py on a coarse rod, the policy is
           fitted to the plan, and training fine-tunes it (at a smaller
           learning rate -- NC's lr knocks a fitted warm start off).

The rollouts start from a *ready pose*, not from the straight rod: the right
gripper first moves ``--ready-dx`` towards the fixed end (stable steps), and
gravity bows the rod downwards.  At the straight, taut rod the mid node's
response to the gripper is singular (a millimetre of compression drops the
sagging mid node by centimetres), so no path can be followed smoothly from
there.  Both methods start from the same ready state.

``verify`` is what keeps the comparison honest: each of ``--segments``
segments is planned from where the previous one ended, resampled to its
N/segments rollout steps, and replayed with stable steps; a segment is kept
only if the mid node ends within ``--wp-tol`` of the waypoint and never
strays more than ``--path-tol`` from the reference on the way.  A path is
feasible if at least one planner seed gets through every segment.  The
verified command sequence is also replayed open loop on the fine rod
("RRT open loop").  With ``workspace.enabled`` the gripper is confined to
that box throughout: the planner grows the tree inside it, and the neural
control projects every command onto it (the penalty acts on the unprojected
command).

``--grad-method`` (default ``train.grad_method``, ``state``) picks the adjoint
gradient: ``state`` is exact, ``proxy`` the frozen-tangent rate formulation
(see src/implicit_grad_tools.py).  ``--optimizer`` (default
``train.optimizer``, ``adam``) picks the update: ``adam`` is the fixed-rate
Adam with gradient clipping, ``linesearch`` the paper's Algorithm 1 -- the
Adam direction with a backtracking step length on full rollouts.  A tracking
objective has to stay on the branch of the initial equilibrium, so in that
mode a trial with a snap-through is rejected, as is one with a Newton
failure, an unstable state or a command outside the workspace.
Single-horizon training only.

Usage
-----
    python scripts/train_tracking.py                              # the config's curve
    python scripts/train_tracking.py --grad-method proxy
    python scripts/train_tracking.py --optimizer linesearch
    python scripts/train_tracking.py verify --path line_a --out DIR
    python scripts/train_tracking.py --path line_a --method nc --seed 0 --out DIR
    python scripts/train_tracking.py --path line_a --method rrt_nc --plan-seed 0 --out DIR
    python scripts/train_tracking.py jobs --out DIR      # the commands for a sweep
    python scripts/train_tracking.py report --out DIR
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from blas_threads import limit_blas_threads
limit_blas_threads()

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from config import load_config
from dlo_rrt import RRTPlanner, PlannerParams, resample_path, stable_step
from env_factory import (
    build_env, reset_env, env_dim, env_twist, control_dim, end_control_dim,
    initial_control, make_target_trajectory, material_summary, node_position,
    task_target, dloss_dq_free_node,
)
from neural_control import (
    BoundaryVelocityNet, reinit_net_, resolve_device, segment_lengths,
    rollout, execute_controls, adjoint_gradient, surrogate_loss, apply_update,
    fit_policy, workspace_bounds, workspace_projection, workspace_penalty,
    GRAD_METHODS, OPTIMIZERS, LineSearch, trial_rule, flat_gradient,
    get_parameters, set_parameters,
)
from runlog import RunLogger
from utils import maybe_save_trajectory_gif


#: Candidate reference paths, all starting at the mid node's position m0 in
#: the ready pose, (0.495, 0, -0.0626) m.  For a bowed rod the mid node sits
#: near the surface x ~ 0.5 - 1.43 (y^2 + z^2) (fitted on the coarse rod):
#: moving it sideways or up and down along that surface is easy, moving it
#: off the surface along x needs large gripper motions.  The line and sine
#: end points lie on the surface, and their chords stay within ~5 mm of it;
#: the arcs turn the bow plane about the rod axis (arc_a, arc_b) or run in a
#: plane tilted towards the fixed end (arc_c).
#:   line  m0 -> end
#:   sine  the line m0 -> end plus amp * sin(2 pi periods t) along `normal`
#:   arc   circle through m0 with centre m0 - radius * e1, turning towards e2
#:         by `angle` degrees
PATHS = {
    "line_a": ("line", {"end": (0.481, 0.10, -0.06)}),
    "line_b": ("line", {"end": (0.490, -0.08, 0.02)}),
    "line_c": ("line", {"end": (0.467, 0.06, -0.14)}),
    "sine_a": ("sine", {"end": (0.481, 0.10, -0.06), "normal": (0.0, 0.0, 1.0),
                        "amp": 0.025, "periods": 1.0}),
    "sine_b": ("sine", {"end": (0.467, 0.06, -0.14), "normal": (0.0, 0.79, 0.61),
                        "amp": 0.025, "periods": 1.0}),
    "sine_c": ("sine", {"end": (0.481, -0.10, -0.06), "normal": (0.0, 0.0, 1.0),
                        "amp": 0.025, "periods": 1.0}),
    "arc_a": ("arc", {"e1": (0.0, 0.0, -1.0), "e2": (0.0, 1.0, 0.0),
                      "radius": 0.0626, "angle": 120.0}),
    "arc_b": ("arc", {"e1": (0.0, 0.0, -1.0), "e2": (0.0, -1.0, 0.0),
                      "radius": 0.0626, "angle": 180.0}),
    "arc_c": ("arc", {"e1": (0.0, 0.0, -1.0), "e2": (-0.12, 0.99, 0.0),
                      "radius": 0.09, "angle": 90.0}),
}


def ready_pose(env, cfg, dx, steps=20):
    """Move the right gripper ``dx`` towards the fixed end with stable steps.

    Leaves ``env`` in the ready state and returns the ready control.
    """
    u0 = initial_control(cfg)
    k = end_control_dim(env.dim, env_twist(cfg))
    u_r = u0.copy()
    u_r[k] -= dx
    for j in range(1, steps + 1):
        info = stable_step(env, u0 + (u_r - u0) * j / steps)
        if not info["stable"]:
            raise RuntimeError(f"the ready motion fails at step {j}")
    return u_r


def reference_path(name, m0, N):
    """m_ref at lambda_k = k/N, k = 0..N; shape (N+1, 3), row 0 = m0."""
    kind, spec = PATHS[name]
    t = np.arange(N + 1)[:, None] / N
    m0 = np.asarray(m0, dtype=float)
    if kind in ("line", "sine"):
        ref = m0 + t * (np.asarray(spec["end"], float) - m0)
        if kind == "sine":
            n = np.asarray(spec["normal"], float)
            ref = ref + spec["amp"] * np.sin(2 * np.pi * spec["periods"] * t) * n / np.linalg.norm(n)
        return ref
    e1 = np.asarray(spec["e1"], float)
    e1 /= np.linalg.norm(e1)
    e2 = np.asarray(spec["e2"], float)
    e2 = e2 - (e2 @ e1) * e1
    e2 /= np.linalg.norm(e2)
    R, th = spec["radius"], np.deg2rad(spec["angle"]) * t
    return m0 - R * e1 + R * (np.cos(th) * e1 + np.sin(th) * e2)


def mid_of(env, q):
    return np.asarray(node_position(q, env.dim, env.mid_node), dtype=float)


def coarse_config(cfg, nv):
    cfg_c = copy.deepcopy(cfg)
    cfg_c["rod"]["nv"] = nv
    return cfg_c


# ------------------------------------------------------------- verification

class TrackingRRT(RRTPlanner):
    """RRTPlanner with a damped, backtracking task-space extension.

    The base planner's greedy step (damping 1e-4) explodes along the mid
    node's weak direction, whose singular value is ~0.016 in the ready pose:
    an 8 mm request becomes a 20 cm gripper move, the step fails, and the tree
    falls back to random detours.  Here the damping scales with the largest
    singular value (Levenberg-Marquardt), and a step that does not reduce the
    error is halved, then retried with less damping, before it is given up.
    """

    lm_damps = (0.03, 0.01)

    def _extend_task(self, m_rand, toward_goal=False):
        i = self._nearest_mid(m_rand, exclude=self.dead if toward_goal else ())
        if i is None:
            return 0
        first, n_before = i, len(self.nodes)
        S = np.ones(self.k)
        S[self.dim:] = 1.0 / self.p.w_rot
        for _ in range(self.p.greedy_steps):
            nd = self.nodes[i]
            e = m_rand - nd.mid
            if np.linalg.norm(e) < 1e-3:
                break
            G = self.mid_jacobian(nd) * S
            s_max = np.linalg.svd(G, compute_uv=False)[0]
            new = None
            for damp in self.lm_damps:
                du = S * (G.T @ np.linalg.solve(G @ G.T + (damp * s_max) ** 2 * np.eye(self.dim), e))
                for frac in (1.0, 0.5, 0.25, 0.125):
                    u_to = nd.u.copy()
                    u_to[self.k:2 * self.k] += frac * du
                    cand = self._steer(i, u_to)
                    if cand is not None and np.linalg.norm(m_rand - cand.mid) < np.linalg.norm(e):
                        new = cand
                        break
                if new is not None:
                    break
            if new is None:
                break
            i = self._add(new)
            if self._goal_reached(i):
                break
        if toward_goal and not self._goal_reached(i):
            self.dead.add(first)
            self.dead.update(range(n_before, len(self.nodes)))
        return i


def chain_plan(cfg, bounds, ref, seed, args):
    """Follow ``ref`` on the coarse rod, one RRT-planned segment at a time.

    Returns (summary, arrays).  ``arrays["U_plan"]`` (N, n_u) is the rollout
    command sequence (row j = command of step j+1) when the chain succeeds.
    """
    cfg_c = coarse_config(cfg, args.plan_nv)
    env = build_env(cfg_c, max_newton_iter=args.plan_newton_iter)
    reset_env(env, cfg_c)
    dim, k = env.dim, end_control_dim(env.dim, env_twist(cfg_c))
    N = len(ref) - 1
    steps = N // args.segments
    if steps * args.segments != N:
        raise ValueError(f"--segments must divide N = {N}")
    u_cur = ready_pose(env, cfg_c, args.ready_dx)
    state = env.get_state()
    U_rows, mids, seg_log = [], [], []
    t0 = time.time()
    for j in range(1, args.segments + 1):
        target = ref[j * steps]
        ref_seg = ref[(j - 1) * steps + 1:j * steps + 1]
        done = False
        for attempt in range(args.seg_tries):
            p = PlannerParams(seed=seed * 100003 + j * 101 + attempt, goal_tol=args.wp_tol,
                              time_limit=args.seg_time, max_iter=args.seg_iter,
                              p_ts=0.9, p_goal=0.9, shortcut_iters=args.shortcuts)
            n_ang = int(np.sum(np.isfinite(bounds[0][dim:])))
            p.box_min, p.box_max = tuple(bounds[0][:dim]), tuple(bounds[1][:dim])
            p.ang_min = tuple(bounds[0][dim:dim + n_ang])
            p.ang_max = tuple(bounds[1][dim:dim + n_ang])
            planner = TrackingRRT(env, u_cur, target, p)
            res = planner.plan(state)
            if not res["success"]:
                continue
            U_seg = resample_path(res["U"], steps, k, dim, p.w_rot)
            # Replay the resampled segment exactly as the full plan will be replayed.
            env.set_state(state)
            seg_mids, ok = [], True
            for u in U_seg:
                inf = stable_step(env, u)
                if not inf["stable"]:
                    ok = False
                    break
                seg_mids.append(mid_of(env, inf["q"]))
            if not ok:
                continue
            err = np.linalg.norm(np.array(seg_mids) - ref_seg, axis=1)
            if err[-1] > args.wp_tol + 0.003 or err.max() > args.path_tol:
                continue
            state, u_cur = env.get_state(), U_seg[-1].copy()
            U_rows.append(U_seg)
            mids.extend(seg_mids)
            seg_log.append({"segment": j, "attempt": attempt, "iterations": res["iterations"],
                            "nodes": res["nodes"], "max_err": float(err.max()),
                            "end_err": float(err[-1])})
            done = True
            break
        if not done:
            return ({"seed": seed, "success": False, "failed_segment": j,
                     "time": time.time() - t0, "segments": seg_log}, {})
    U_plan = np.concatenate(U_rows)
    mids = np.array(mids)
    err = np.linalg.norm(mids - ref[1:], axis=1)
    summ = {"seed": seed, "success": True, "time": time.time() - t0, "segments": seg_log,
            "coarse_rms": float(np.sqrt(np.mean(err ** 2))), "coarse_max": float(err.max())}
    return summ, {"U_plan": U_plan, "coarse_mid": mids}


def cmd_verify(args):
    cfg = load_config(args.config)
    dim, k = env_dim(cfg), end_control_dim(env_dim(cfg), env_twist(cfg))
    bounds = workspace_bounds(cfg, dim, k)
    if bounds is None:
        raise SystemExit("path tracking runs in the bounded workspace; enable workspace in the config")
    env = build_env(cfg)
    reset_env(env, cfg)
    ready_pose(env, cfg, args.ready_dx)
    checkpoint = env.get_state()
    m0 = mid_of(env, checkpoint["q"])
    ref = reference_path(args.path, m0, env.Nsteps)
    out = Path(args.out) / "plans"
    out.mkdir(parents=True, exist_ok=True)
    length = float(np.sum(np.linalg.norm(np.diff(ref, axis=0), axis=1)))
    print(f"[verify {args.path}] {PATHS[args.path][0]}, length {length:.3f} m, "
          f"start {np.round(ref[0], 4)}, end {np.round(ref[-1], 4)}", flush=True)
    results = []
    for seed in args.plan_seeds:
        summ, arrs = chain_plan(cfg, bounds, ref, seed, args)
        if summ["success"]:
            # the plan on the fine rod, open loop, with the controller's own stepping
            q_list, ok = execute_controls(env, checkpoint, arrs["U_plan"])
            summ["fine_converged"] = bool(ok)
            fine_mid = np.array([mid_of(env, q) for q in q_list])
            arrs["fine_mid"] = fine_mid
            if ok:
                e = np.linalg.norm(fine_mid - ref[1:], axis=1)
                summ["fine_rms"], summ["fine_max"] = float(np.sqrt(np.mean(e ** 2))), float(e.max())
            np.savez_compressed(out / f"{args.path}_p{seed}.npz", ref=ref, **arrs)
            print(f"[verify {args.path}] seed {seed}: feasible on the coarse rod in "
                  f"{summ['time']:.0f}s, rms {summ['coarse_rms'] * 1e3:.1f} mm, max "
                  f"{summ['coarse_max'] * 1e3:.1f} mm; fine rod open loop: "
                  + (f"rms {summ['fine_rms'] * 1e3:.1f} mm, max {summ['fine_max'] * 1e3:.1f} mm"
                     if ok else f"solver failed at step {len(q_list)}"), flush=True)
        else:
            print(f"[verify {args.path}] seed {seed}: no chain past segment "
                  f"{summ['failed_segment']} ({summ['time']:.0f}s)", flush=True)
        results.append(summ)
    verdict = {"path": args.path, "kind": PATHS[args.path][0], "spec": PATHS[args.path][1],
               "length": length, "m0": ref[0].tolist(), "end": ref[-1].tolist(),
               "feasible": any(r["success"] for r in results), "plans": results,
               "settings": {"segments": args.segments, "wp_tol": args.wp_tol,
                            "path_tol": args.path_tol, "plan_nv": args.plan_nv,
                            "ready_dx": args.ready_dx}}
    json.dump(verdict, open(out / f"{args.path}_verify.json", "w"), indent=1, default=float)
    print(f"[verify {args.path}] feasible: {verdict['feasible']}", flush=True)


# ----------------------------------------------------------------- training

def tracking_loss_terms(env, q_list, conv, ref, N):
    """``(per-step error, loss, dL/dq_free per step)`` for the mid node.

    The loss is the mean squared mid-node error over the horizon; a step whose
    solve failed contributes nothing and gets no adjoint source.
    """
    dim = env.dim
    mids = np.array([mid_of(env, q) for q in q_list])
    err = np.linalg.norm(mids - ref[1:len(q_list) + 1], axis=1)
    loss = float(np.sum(err[conv[:len(q_list)]] ** 2) / N)
    a_free = np.zeros((len(q_list), len(env.free_index)), dtype=float)
    for i in np.where(conv[:len(q_list)])[0]:
        a_free[i] = dloss_dq_free_node(q_list[i], env.free_index, dim, env.mid_node,
                                       ref[i + 1]) / N
    return err, loss, a_free, mids


def train_track_linesearch(env, checkpoint, net, u0, ref, cfg, epochs, device, lr, bounds,
                           run=None, log_every=10, tag=""):
    """Algorithm 1 for the tracking loss: :class:`neural_control.LineSearch`.

    Same inputs and outputs as :func:`train_track`, with the optimizer
    replaced.  A trial is accepted only under the Armijo condition and the
    trajectory rule: every step converged and stable, every command inside
    the workspace (no projection in this mode), and no snap-through -- the
    rollout has to stay on the branch of the initial equilibrium.  The
    accepted rollout is reused for the next gradient; training stops when no
    step is accepted.
    """
    N = env.Nsteps
    n_end = net.out_dim
    dlam = 1.0 / N
    gm = cfg.train.grad_method
    ls = LineSearch(net, lr, c=float(cfg.train.linesearch.c),
                    tau_min=float(cfg.train.linesearch.tau_min))
    hist = {k: [] for k in ("loss", "rms", "max", "n_conv", "valid", "ws_loss", "time",
                            "mid_paths", "u_paths", "tau", "rejected", "rejected_reasons",
                            "rollouts", "snaps", "fallback")}
    hist["stop"] = None
    t_start = time.time()

    def evaluate(xi):
        set_parameters(net, xi)
        env.set_state(checkpoint)
        infos = []
        u_list, r_list, q_list, J_list, conv = rollout(env, net, u0, N, dlam, device,
                                                       project=None, infos=infos)
        ok, reason = trial_rule(infos, conv, u_list, n_end, bounds, snap_ok=False)
        err, loss, a_free, mids = tracking_loss_terms(env, q_list, conv, ref, N)
        return {"ok": ok, "reason": reason, "loss": loss if conv.all() else np.inf,
                "err": err, "a_free": a_free, "mids": mids, "u_list": u_list, "r_list": r_list,
                "q_list": q_list, "J_list": J_list, "conv": conv, "infos": infos}

    def gradient(tr):
        g_seq = adjoint_gradient(env, tr["u_list"], tr["J_list"], tr["a_free"], cfg.train.jac_reg,
                                 grad_method=gm)
        return flat_gradient(net, surrogate_loss(tr["u_list"], tr["r_list"], g_seq, dlam, device,
                                                 grad_method=gm))

    def snaps_of(tr):
        return int(sum(1 for i in tr["infos"] if i.get("snap")))

    def record(ep, tr, rec, fallback):
        err = tr["err"]
        hist["loss"].append(tr["loss"])
        hist["rms"].append(float(np.sqrt(np.mean(err ** 2))))
        hist["max"].append(float(err.max()))
        hist["n_conv"].append(int(tr["conv"].sum()))
        hist["valid"].append(True)
        hist["ws_loss"].append(0.0)
        hist["time"].append(time.time() - t_start)
        hist["mid_paths"].append(np.vstack([ref[:1], tr["mids"]]))
        hist["u_paths"].append(np.stack([u.detach().cpu().numpy() for u in tr["u_list"]]))
        hist["tau"].append(rec["tau"])
        hist["rejected"].append(rec["n_rejected"])
        hist["rejected_reasons"].append([r["reason"] for r in rec["rejected"]])
        hist["rollouts"].append(ls.n_rollouts)
        hist["snaps"].append(snaps_of(tr))
        hist["fallback"].append(fallback)
        if run is not None:
            run.log(epoch=ep, total_loss=tr["loss"], mean_track_error=float(err.mean()),
                    max_track_error=float(err.max()), converged_ratio=1.0,
                    newton_failures=env.n_newton_failures, tau=rec["tau"],
                    rejected=rec["n_rejected"], rollouts=ls.n_rollouts, snaps=snaps_of(tr),
                    fallback=fallback)
        if ep % log_every == 0 or ep == epochs - 1 or rec["tau"] is None:
            print(f"[{tag} ep {ep:03d}] rms={hist['rms'][-1] * 1e3:7.2f} mm  "
                  f"tau={'-' if rec['tau'] is None else f'{rec['tau']:g}'}  "
                  f"rejected={rec['n_rejected']}  rollouts={ls.n_rollouts}  "
                  f"t={hist['time'][-1]:7.1f}s", flush=True)

    xi = get_parameters(net)
    ls.n_rollouts += 1
    cur = evaluate(xi)
    if not cur["ok"]:
        hist["stop"] = {"epoch": 0, "reason": f"initial rollout rejected: {cur['reason']}"}
        print(f"[{tag}] the initial rollout is rejected ({cur['reason']}): Algorithm 1 needs a "
              "feasible start, no update is made", flush=True)
        if run is not None:
            run.log(epoch=0, converged_ratio=float(cur["conv"].mean()),
                    newton_failures=env.n_newton_failures, note=hist["stop"]["reason"])
    else:
        g = gradient(cur)
        for ep in range(epochs):
            d, fallback = ls.direction(g)
            acc, rec = ls.search(xi, g, cur["loss"], d, evaluate)
            record(ep, cur, rec, fallback)
            if acc is None:
                hist["stop"] = {"epoch": ep, "reason": rec["stop"]}
                print(f"[{tag} ep {ep:03d}] line search stopped: {rec['stop']}", flush=True)
                break
            xi = get_parameters(net)
            cur = acc
            g = gradient(cur)
    err = cur["err"]
    final = {"valid": bool(cur["conv"].all()), "n_conv": int(cur["conv"].sum()),
             "rms": float(np.sqrt(np.mean(err ** 2))), "max": float(err.max()),
             "end_err": float(err[-1])}
    final_U = np.stack([u.detach().cpu().numpy() for u in cur["u_list"]])
    return (hist, final, np.vstack([ref[:1], cur["mids"]]), final_U, cur["q_list"],
            time.time() - t_start)


def train_track(env, checkpoint, net, u0, ref, cfg, epochs, device, lr, bounds, ws_weight,
                run=None, log_every=10, tag=""):
    """Mean squared tracking error of the mid node, adjoint gradients, Adam.

    ``train.optimizer: linesearch`` goes to :func:`train_track_linesearch`.
    """
    if cfg.train.optimizer == "linesearch":
        return train_track_linesearch(env, checkpoint, net, u0, ref, cfg, epochs, device, lr,
                                      bounds, run=run, log_every=log_every, tag=tag)
    N = env.Nsteps
    n_end = net.out_dim
    dlam = 1.0 / N
    opt = optim.Adam(net.parameters(), lr=lr)
    project = workspace_projection(bounds, u0, n_end)
    hist = {k: [] for k in ("loss", "rms", "max", "n_conv", "valid", "ws_loss", "time",
                            "mid_paths", "u_paths")}
    t_start = time.time()
    for ep in range(epochs):
        env.set_state(checkpoint)
        opt.zero_grad(set_to_none=True)
        u_list, r_list, q_list, J_list, conv = rollout(env, net, u0, N, dlam, device,
                                                       project=project, stop_on_failure=False)
        err, loss, a_free, mids = tracking_loss_terms(env, q_list, conv, ref, N)
        L_ws_t = workspace_penalty(net, u0, u_list, n_end, bounds, ws_weight, device, dlam)
        if conv.any():
            g_seq = adjoint_gradient(env, u_list, J_list, a_free, cfg.train.jac_reg,
                                     grad_method=cfg.train.grad_method)
            apply_update(net, opt, surrogate_loss(u_list, r_list, g_seq, dlam, device,
                                                  grad_method=cfg.train.grad_method) + L_ws_t)
        valid = bool(conv.all())
        u_seq = np.stack([u.detach().cpu().numpy() for u in u_list])
        hist["loss"].append(loss)
        hist["rms"].append(float(np.sqrt(np.mean(err ** 2))) if valid else float("nan"))
        hist["max"].append(float(err.max()) if valid else float("nan"))
        hist["n_conv"].append(int(conv.sum()))
        hist["valid"].append(valid)
        hist["ws_loss"].append(float(L_ws_t.detach()))
        hist["time"].append(time.time() - t_start)
        hist["mid_paths"].append(np.vstack([ref[:1], mids]))
        hist["u_paths"].append(u_seq)
        if run is not None:
            run.log(epoch=ep, total_loss=loss, mean_track_error=float(err.mean()),
                    max_track_error=float(err.max()), converged_ratio=float(conv.mean()),
                    newton_failures=env.n_newton_failures)
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"[{tag} ep {ep:03d}] rms={hist['rms'][-1] * 1e3:7.2f} mm  "
                  f"converged {conv.sum()}/{N}  ws_pen={hist['ws_loss'][-1]:.2e}  "
                  f"t={hist['time'][-1]:7.1f}s", flush=True)
    # the policy after the last update
    env.set_state(checkpoint)
    with torch.no_grad():
        u_list, _, q_list, _, conv = rollout(env, net, u0, N, dlam, device, project=project,
                                             stop_on_failure=False)
    err, _, _, mids = tracking_loss_terms(env, q_list, conv, ref, N)
    final = {"valid": bool(conv.all()), "n_conv": int(conv.sum()),
             "rms": float(np.sqrt(np.mean(err ** 2))), "max": float(err.max()),
             "end_err": float(err[-1])}
    final_U = np.stack([u.detach().cpu().numpy() for u in u_list])
    return (hist, final, np.vstack([ref[:1], mids]), final_U, q_list,
            time.time() - t_start)


def train_track_rhc(env, checkpoint, net, u0, ref, cfg, device, lr, bounds, ws_weight,
                    run=None):
    """Receding horizon: optimize one segment of the path, commit it, move on.

    Each segment is trained against its own slice of the reference and then
    rolled out once more to be committed, so the next segment warm-starts from
    the state the committed one ended in.
    """
    N = env.Nsteps
    seg_lens = segment_lengths(N, cfg.rhc.num_segments)
    seg_bounds = np.cumsum([0] + seg_lens)
    n_end = net.out_dim
    project = workspace_projection(bounds, u0, n_end)
    u_current = u0.clone()
    opt = optim.Adam(net.parameters(), lr=lr)
    losses, q_traj = [], []
    t_start = time.time()

    for seg_idx, T_seg in enumerate(seg_lens):
        if cfg.rhc.reinit_policy_each_segment:
            reinit_net_(net)
            opt = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
        dlam = 1.0 / T_seg
        ref_seg = ref[seg_bounds[seg_idx]:seg_bounds[seg_idx + 1] + 1]
        print(f"\n{'=' * 70}\nRHC segment {seg_idx + 1}/{len(seg_lens)}  (T={T_seg})\n{'=' * 70}")

        best = float("inf")
        for it in range(cfg.train.iterations_per_segment):
            env.set_state(checkpoint)
            opt.zero_grad(set_to_none=True)
            u_list, r_list, q_list, J_list, conv = rollout(env, net, u_current, T_seg, dlam,
                                                           device, project=project,
                                                           stop_on_failure=False)
            _, loss, a_free, _ = tracking_loss_terms(env, q_list, conv, ref_seg, T_seg)
            if not conv.any():
                continue
            g_seq = adjoint_gradient(env, u_list, J_list, a_free, cfg.train.jac_reg,
                                     grad_method=cfg.train.grad_method)
            L_ws = workspace_penalty(net, u_current, u_list, n_end, bounds, ws_weight, device,
                                     dlam)
            apply_update(net, opt, surrogate_loss(u_list, r_list, g_seq, dlam, device,
                                                  grad_method=cfg.train.grad_method) + L_ws)
            best = min(best, loss)

        env.set_state(checkpoint)
        with torch.no_grad():
            u_list, _, q_list, _, conv = rollout(env, net, u_current, T_seg, dlam, device,
                                                 project=project, stop_on_failure=False)
        checkpoint = env.get_state()
        u_current = u_list[-1].detach().clone()
        q_traj.extend(q_list)
        losses.append(best)
        print(f"[segment {seg_idx}] best_loss={best:.6e}  converged {conv.sum()}/{T_seg}")
        if run is not None:
            run.log(segment=seg_idx, horizon=int(T_seg), best_loss=float(best),
                    converged_ratio=float(conv.mean()),
                    newton_failures=env.n_newton_failures)
    return losses, q_traj, time.time() - t_start


def plot_tracking(ref, mids, dim, title, path, env=None, q_init=None, q_final=None):
    """Reference against what the mid node actually did.

    With ``env`` the scene is drawn around the two paths: the gripper
    workspace, the obstacles (the peg), the pipe, and -- from ``q_init`` /
    ``q_final`` -- the rod at the start and at the end of the rollout.
    """
    import utils
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111) if dim == 2 else fig.add_subplot(111, projection='3d')
    coords = lambda p: [p[:, c] for c in range(dim)]       # noqa: E731
    extent = [ref, mids]
    if env is not None:
        utils.draw_obstacles(ax, env)
        utils.draw_workspace(ax, env)
        if getattr(env, 'pipe', None) is not None:
            env.pipe.draw(ax)
        if getattr(env, 'workspace_box', None) is not None:
            extent.append(np.stack(env.workspace_box))
        for obs in getattr(env, 'obstacles', []) or []:
            if hasattr(obs, 'bounds'):
                extent.append(np.stack(obs.bounds()))
        for q, style, label in ((q_init, dict(color='0.6', lw=1.2), "Initial rod"),
                                (q_final, dict(color='k', lw=1.8), "Final rod")):
            if q is not None:
                P = utils._positions(q, env, dim)
                ax.plot(*coords(P), label=label, **style)
                extent.append(P)
    ax.plot(*coords(ref), 'b--', linewidth=2, label="Reference path")
    ax.plot(*coords(mids), 'r-', linewidth=1.5, label="Mid node")
    ax.plot(*[[v] for v in ref[0]], 'go', markersize=7, label="Start")
    ax.plot(*[[v] for v in ref[-1]], 'bx', markersize=10, markeredgewidth=2.5, label="Path end")
    if dim == 2:
        ax.set_aspect('equal')
    else:
        pts = np.vstack(extent)
        lo, hi = pts.min(0), pts.max(0)
        half = 0.55 * (hi - lo).max()
        ctr = 0.5 * (lo + hi)
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[1] - half, ctr[1] + half)
        ax.set_zlim(ctr[2] - half, ctr[2] + half)
        ax.set_zlabel("z")
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:      # matplotlib < 3.3
            pass
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def build_reference(cfg, env, args, m0):
    """``(ref (N+1, dim), name)``: row 0 is where the mid node starts.

    A named preset is anchored at ``m0``; otherwise the config's curve is used
    and ``m0`` is prepended, so row ``k+1`` is step ``k``'s target either way.
    """
    N = env.Nsteps
    if args.path:
        return reference_path(args.path, m0, N), args.path
    kind = cfg.task.tracking.trajectory_type
    curve = make_target_trajectory(N, env.rod_length, traj_type=kind, dim=env.dim,
                                   start=m0, end=task_target(cfg),
                                   sine=cfg.task.tracking.get("sine_xz"))
    return np.vstack([m0, curve]), kind


def cmd_train(args):
    cfg = load_config(args.config)
    if args.grad_method is not None:        # the command line wins over the YAML
        cfg["train"]["grad_method"] = args.grad_method
    if args.optimizer is not None:
        cfg["train"]["optimizer"] = args.optimizer
    grad_method = cfg.train.grad_method
    optimizer = cfg.train.optimizer
    if optimizer == "linesearch" and cfg.rhc.enabled:
        raise SystemExit("train.optimizer: linesearch is implemented for single-horizon "
                         "training; set rhc.enabled: false")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dim, twist = env_dim(cfg), env_twist(cfg)
    k = end_control_dim(dim, twist)
    bounds = workspace_bounds(cfg, dim, k)
    if args.method == "rrt_nc" and bounds is None:
        raise SystemExit("rrt_nc plans inside the gripper workspace; enable workspace in the config")
    env = build_env(cfg)
    reset_env(env, cfg)
    u0_np = ready_pose(env, cfg, args.ready_dx) if args.ready_dx else initial_control(cfg)
    checkpoint = env.get_state()
    N = env.Nsteps
    m0 = mid_of(env, checkpoint["q"])
    ref, ref_name = build_reference(cfg, env, args, m0)
    u0 = torch.tensor(u0_np, dtype=torch.float32, device=torch.device("cpu"))
    device = resolve_device(cfg.train.device if args.device is None else args.device)
    u0 = u0.to(device)
    net = BoundaryVelocityNet(hidden=cfg.train.hidden, out_dim=k).to(device)

    lr = args.lr if args.lr is not None else (cfg.train.lr if args.method == "nc" else 1e-4)
    epochs = cfg.train.epochs if args.epochs is None else args.epochs
    name = (f"{ref_name}_{args.method}_s{args.seed}"
            + (f"_p{args.plan_seed}" if args.method == "rrt_nc" else "")
            + (f"_lr{lr:g}" if args.method == "nc" and lr != cfg.train.lr else "")
            + ("_proxy" if grad_method == "proxy" else "")
            + ("_ls" if optimizer == "linesearch" else ""))
    run = RunLogger(cfg, env=env, root=_ROOT, tag=name,
                    enabled=bool((cfg.get("output") or {}).get("run_log", True)))
    out = Path(args.out) if args.out else Path(run.artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Task: path tracking  (adjoint neural control, method {args.method})")
    print(f"Adjoint gradient: {grad_method}  "
          f"({'--grad-method' if args.grad_method else 'train.grad_method'})")
    print(f"Optimizer: {optimizer}  "
          f"({'--optimizer' if args.optimizer else 'train.optimizer'})")
    print(f"Environment: {type(env).__name__}  (dim={dim}, twist={twist}, "
          f"control={control_dim(dim, twist)}D)")
    print(f"Rod material: {material_summary(cfg)}")
    budget = (f"receding horizon, {cfg.rhc.num_segments} segments x "
              f"{cfg.train.iterations_per_segment} updates" if cfg.rhc.enabled
              else f"single horizon, {epochs} epochs")
    print(f"Reference: {ref_name}, {N} steps from {np.round(ref[0], 4)} to "
          f"{np.round(ref[-1], 4)}   device: {device}   lr {lr:g}")
    print(f"Horizon: {N} steps, {budget}")
    print(f"Workspace: {'off' if bounds is None else 'on, ' + ('hard constraint (trials outside it are rejected)' if optimizer == 'linesearch' else 'penalty ' + str(args.ws_weight))}")

    summary = {"path": ref_name, "method": args.method, "grad_method": grad_method,
               "optimizer": optimizer,
               "seed": args.seed, "lr": float(lr),
               "lr_default": float(cfg.train.lr), "epochs": epochs, "N": int(N),
               "nv": int(env.nv), "ready_dx": float(args.ready_dx),
               "plan_seed": args.plan_seed if args.method == "rrt_nc" else None,
               "workspace": None if bounds is None else
               {"lo": bounds[0].tolist(), "hi": bounds[1].tolist(),
                "weight": float(args.ws_weight)}}
    arrays = {"ref": ref}

    if args.method == "rrt_nc":
        # ---- global: the verified plan for this path and planner seed
        plans_dir = out / "plans"
        verdict = json.load(open(plans_dir / f"{args.path}_verify.json"))
        if abs(verdict["settings"]["ready_dx"] - args.ready_dx) > 1e-12:
            raise SystemExit("--ready-dx differs from the one the plan was verified with")
        plan = next((p for p in verdict["plans"] if p["seed"] == args.plan_seed), None)
        if plan is None or not plan["success"]:
            raise SystemExit(f"no verified plan for {args.path} with planner seed {args.plan_seed}")
        pz = np.load(plans_dir / f"{args.path}_p{args.plan_seed}.npz")
        summary["plan"] = plan
        arrays.update({"U_plan": pz["U_plan"], "plan_coarse_mid": pz["coarse_mid"],
                       "plan_fine_mid": pz["fine_mid"]})
        # ---- warm start
        t_fit = time.time()
        summary["fit"] = fit_policy(net, u0, pz["U_plan"], k, dim=dim, iters=args.fit_iters)
        summary["fit"]["time"] = time.time() - t_fit
        print(f"[{name}] policy fit: {summary['fit']}", flush=True)

    env.n_newton_failures = 0
    if cfg.rhc.enabled:
        losses, q_traj, t_train = train_track_rhc(env, checkpoint, net, u0, ref, cfg, device,
                                                  lr, bounds, args.ws_weight, run=run)
        mids = np.vstack([ref[:1], np.array([mid_of(env, q) for q in q_traj])])
        err = np.linalg.norm(mids[1:] - ref[1:len(mids)], axis=1)
        final = {"valid": True, "n_conv": len(q_traj),
                 "rms": float(np.sqrt(np.mean(err ** 2))), "max": float(err.max()),
                 "end_err": float(err[-1])}
        summary["segment_loss"] = losses
        arrays["final_mid"] = mids
    else:
        hist, final, mids, final_U, q_traj, t_train = train_track(
            env, checkpoint, net, u0, ref, cfg, epochs, device, lr, bounds, args.ws_weight,
            run=run, tag=name)
        losses = hist["loss"]
        for key in ("loss", "rms", "max", "n_conv", "valid", "ws_loss", "time"):
            summary[f"epoch_{key}"] = hist[key]
        if optimizer == "linesearch":
            for key in ("tau", "rejected", "rejected_reasons", "rollouts", "snaps", "fallback"):
                summary[f"epoch_{key}"] = hist[key]
            summary["stop"] = hist.get("stop")
            summary["rollouts_total"] = int(hist["rollouts"][-1]) if hist["rollouts"] else 0
            summary["rejected_total"] = int(sum(hist["rejected"]))
        arrays.update({"final_mid": mids, "final_U": final_U,
                       "epoch_mid_paths": np.array(hist["mid_paths"], dtype=np.float32),
                       "epoch_u_paths": np.array(hist["u_paths"], dtype=np.float32)})

    summary.update({"train_time": t_train, "final": final, "status": "done",
                    "newton_failures_total": int(env.n_newton_failures),
                    "unstable_steps_total": int(env.n_unstable),
                    "snaps_total": int(env.n_snaps)})
    np.savez_compressed(out / f"{name}.npz", **arrays)
    json.dump(summary, open(out / f"{name}.json", "w"), indent=1, default=float)

    plot_tracking(ref, arrays["final_mid"], dim, f"Mid-node tracking ({ref_name})",
                  os.path.join(run.artifact_dir, "tracking_result.png"),
                  env=env, q_init=checkpoint["q"], q_final=q_traj[-1])
    if losses:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(len(losses)), losses, linewidth=1.5, color="steelblue")
        ax.set_xlabel("RHC segment" if cfg.rhc.enabled else "Epoch")
        ax.set_ylabel("Tracking loss")
        ax.set_title(f"Tracking loss ({ref_name})")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(run.artifact_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)
    maybe_save_trajectory_gif(cfg, q_traj, env, ref[1:], save_path=str(run.artifact_dir),
                              title=f'DLO tracking ({ref_name})')

    run.finish(task="tracking", trajectory=ref_name, method=args.method,
               grad_method=grad_method, optimizer=optimizer,
               unstable_steps_total=env.n_unstable, snaps_total=env.n_snaps,
               mode="rhc" if cfg.rhc.enabled else "single_horizon",
               final_loss=float(losses[-1]) if losses else None,
               best_loss=float(np.nanmin(losses)) if losses else None,
               losses=[float(l) for l in losses], epochs_recorded=len(losses),
               final_error=final["rms"], newton_failures_total=env.n_newton_failures,
               device=str(device))
    print(f"[{name}] done: final rms {final['rms'] * 1e3:.2f} mm (valid={final['valid']}), "
          f"train {t_train:.0f}s", flush=True)


# ------------------------------------------------------------------- report

COLORS = {"nc": "#2a78d6", "nc_lr": "#2a78d6", "rrt": "#eb6834", "rrt_nc": "#1baf7a"}
LABELS = {"nc": "NC (direct)", "nc_lr": "NC (direct), fine-tune lr",
          "rrt_nc": "RRT + NC", "rrt": "RRT open loop (fine rod)"}


def _variant(r):
    if r["method"] == "rrt_nc":
        return "rrt_nc"
    return "nc" if abs(r["lr"] - r["lr_default"]) < 1e-12 else "nc_lr"


def _first_below(x, thr):
    idx = np.where(np.asarray(x) <= thr)[0]
    return int(idx[0]) if len(idx) else None


def _fmt(v, scale=1e3, nd=1):
    return "-" if v is None or not np.isfinite(v) else f"{v * scale:.{nd}f}"


def _fisher_greater(a, n_a, b, n_b):
    """One-sided Fisher exact p for 'success rate of a > that of b'."""
    from math import comb
    total, succ = n_a + n_b, a + b

    def p_table(x):
        return comb(n_a, x) * comb(n_b, succ - x) / comb(total, succ)
    return sum(p_table(x) for x in range(a, min(n_a, succ) + 1))


def report(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    verdicts = {p: json.load(open(out / "plans" / f"{p}_verify.json"))
                for p in PATHS if (out / "plans" / f"{p}_verify.json").exists()}
    runs = [json.load(open(f)) for f in sorted(out.glob("*.json"))]
    # A job is recognised by its own fields, so a directory that also holds a
    # run's own files (summary.json, another task's job) is read regardless.
    runs = [r for r in runs if isinstance(r, dict) and r.get("status") == "done"
            and {"path", "method", "epoch_rms"} <= set(r)]
    for r in runs:
        r["_v"] = _variant(r)
        rms = np.array(r["epoch_rms"], dtype=float)
        r["_best"] = float(np.nanmin(rms)) if np.isfinite(rms).any() else np.nan
        r["_best_ep"] = int(np.nanargmin(rms)) if np.isfinite(rms).any() else None
        r["_n_valid"] = int(np.sum(r["epoch_valid"]))
    n_steps = int(runs[0]["N"]) if runs else 0
    n_nodes = int(runs[0]["nv"]) if runs else 0
    L = ["# Path tracking in the bounded workspace: RRT + NC versus NC", ""]
    L.append("Errors are mid-node distances to the reference at the same continuation step, "
             f"on the fine rod. RMS over the {n_steps} steps; *valid* = all {n_steps} Newton "
             "solves converged (invalid rollouts are excluded from every result). 'best' = "
             "best valid epoch; 'final' = the policy after the last update.")
    L.append("")
    L.append("## Paths and their feasibility check")
    L.append("")
    L.append("Coarse rod (planner model) following the path with chained RRT segments; "
             f"fine rod = the same commands executed open loop on the {n_nodes}-node rod.")
    L.append("")
    L.append("| path | shape | length [m] | end point [m] | planner seeds feasible | coarse rms / max [mm] | fine open-loop rms / max [mm] |")
    L.append("|---|---|---|---|---|---|---|")
    for p, v in verdicts.items():
        ok = [q for q in v["plans"] if q["success"]]
        co = ", ".join(f"{_fmt(q['coarse_rms'])} / {_fmt(q['coarse_max'])}" for q in ok) or "-"
        fi = ", ".join((f"{_fmt(q.get('fine_rms'))} / {_fmt(q.get('fine_max'))}"
                        if q.get("fine_converged") else "solver fails") for q in ok) or "-"
        L.append(f"| {p} | {v['kind']} | {v['length']:.3f} | {np.round(v['end'], 3).tolist()} | "
                 f"{len(ok)}/{len(v['plans'])} | {co} | {fi} |")
    L.append("")
    paths = [p for p in verdicts
             if verdicts[p]["feasible"] and any(r["path"] == p for r in runs)]
    thr = (0.005, 0.010)
    agg = {v: {"n": 0, "best": [0, 0], "final": [0, 0], "valid_final": 0}
           for v in ("nc", "nc_lr", "rrt_nc")}
    L.append("## Runs")
    L.append("")
    L.append("| path | method | run | start rms | best rms (epoch) | final rms | final valid | "
             "epochs to 10 / 5 mm | valid epochs |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for p in paths:
        for r in sorted([r for r in runs if r["path"] == p],
                        key=lambda r: (r["_v"], r.get("plan_seed") or 0, r["seed"])):
            rms = np.array(r["epoch_rms"], dtype=float)
            e10, e5 = _first_below(np.nan_to_num(rms, nan=9.9), 0.010), \
                _first_below(np.nan_to_num(rms, nan=9.9), 0.005)
            run = (f"plan {r['plan_seed']}, seed {r['seed']}" if r["_v"] == "rrt_nc"
                   else f"seed {r['seed']}")
            fin = r["final"]
            L.append(f"| {p} | {LABELS[r['_v']]} | {run} | {_fmt(rms[0])} | "
                     f"{_fmt(r['_best'])} ({r['_best_ep']}) | {_fmt(fin['rms'])} | "
                     f"{'yes' if fin['valid'] else 'no'} | {e10 if e10 is not None else '-'} / "
                     f"{e5 if e5 is not None else '-'} | {r['_n_valid']}/{len(rms)} |")
            a = agg[r["_v"]]
            a["n"] += 1
            for i, t in enumerate(thr):
                a["best"][i] += int(np.isfinite(r["_best"]) and r["_best"] <= t)
                a["final"][i] += int(fin["valid"] and fin["rms"] <= t)
            a["valid_final"] += int(fin["valid"])
    L.append("")
    L.append("## Final rms per path [mm]")
    L.append("")
    L.append("Final policy, sorted; '-' marks a final rollout with a failed Newton step.")
    L.append("")
    L.append(f"| path | {LABELS['rrt_nc']} | {LABELS['nc']} | {LABELS['nc_lr']} |")
    L.append("|---|---|---|---|")
    for p in paths:
        cells = []
        for v in ("rrt_nc", "nc", "nc_lr"):
            fin = sorted((r["final"]["rms"] if r["final"]["valid"] else np.inf)
                         for r in runs if r["path"] == p and r["_v"] == v)
            cells.append(", ".join(_fmt(x) if np.isfinite(x) else "-" for x in fin) or "")
        L.append(f"| {p} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Success rates over all runs")
    L.append("")
    L.append("One-sided Fisher exact test of RRT + NC against each NC variant.")
    L.append("")
    L.append("| criterion | RRT + NC | NC (direct) | p | NC, fine-tune lr | p |")
    L.append("|---|---|---|---|---|---|")
    g, n, nl = agg["rrt_nc"], agg["nc"], agg["nc_lr"]
    for i, t in enumerate(thr):
        for key, lab in (("best", "best valid rms"), ("final", "final policy rms")):
            pv, pl = (_fisher_greater(g[key][i], g["n"], b[key][i], b["n"])
                      if g["n"] and b["n"] else np.nan for b in (n, nl))
            L.append(f"| {lab} <= {t * 1e3:.0f} mm | {g[key][i]}/{g['n']} | {n[key][i]}/{n['n']} | "
                     f"{pv:.3f} | {nl[key][i]}/{nl['n']} | {pl:.3f} |")
    L.append(f"| final rollout valid | {g['valid_final']}/{g['n']} | {n['valid_final']}/{n['n']} | - | "
             f"{nl['valid_final']}/{nl['n']} | - |")
    L.append("")
    # ---- figures
    if paths:
        nc_ = len(paths)
        cols = min(3, nc_)
        rows = int(np.ceil(nc_ / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.2 * rows), squeeze=False)
        for ax, p in zip(axes.flat, paths):
            for r in [r for r in runs if r["path"] == p]:
                rms = np.array(r["epoch_rms"], dtype=float) * 1e3
                ax.plot(rms, color=COLORS[r["_v"]], lw=1.4,
                        ls="--" if r["_v"] == "nc_lr" else "-", alpha=0.85)
            ok = [q for q in verdicts[p]["plans"] if q["success"] and q.get("fine_converged")]
            for q in ok:
                ax.axhline(q["fine_rms"] * 1e3, color=COLORS["rrt"], ls="-.", lw=1.0, alpha=0.7)
            ax.set_yscale("log")
            ax.set_title(f"{p}  ({verdicts[p]['kind']})")
            ax.set_xlabel("epoch")
            ax.set_ylabel("rms tracking error [mm]")
            ax.grid(alpha=0.3)
        for ax in list(axes.flat)[nc_:]:
            ax.axis("off")
        handles = [plt.Line2D([], [], color=COLORS[v], ls=s, label=LABELS[v])
                   for v, s in (("nc", "-"), ("nc_lr", "--"), ("rrt_nc", "-"), ("rrt", "-."))]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(out / "convergence_all.png", dpi=110)
        plt.close(fig)
        L.append("![convergence](convergence_all.png)")
        L.append("")
        for p in paths:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
            ref = None
            best = {}
            for r in [r for r in runs if r["path"] == p]:
                if r["_best_ep"] is None:
                    continue
                cur = best.get(r["_v"])
                if cur is None or r["_best"] < cur["_best"]:
                    best[r["_v"]] = r
            z = None
            for v, r in best.items():
                stem = out / (f"{p}_{r['method']}_s{r['seed']}"
                              + (f"_p{r['plan_seed']}" if v == "rrt_nc" else "")
                              + (f"_lr{r['lr']:g}" if v == "nc_lr" else ""))
                z = np.load(f"{stem}.npz")
                ref = z["ref"]
                mp = z["epoch_mid_paths"][r["_best_ep"]]
                for ax, (i, j) in zip(axes, ((0, 1), (0, 2), (1, 2))):
                    ax.plot(mp[:, i], mp[:, j], color=COLORS[v], lw=1.6,
                            ls="--" if v == "nc_lr" else "-",
                            label=f"{LABELS[v]} (best, {r['_best'] * 1e3:.1f} mm)")
            pz = sorted((out / "plans").glob(f"{p}_p*.npz"))
            if pz:
                zz = np.load(pz[0])
                ref = zz["ref"]
                fm = np.vstack([ref[:1], zz["fine_mid"]])
                for ax, (i, j) in zip(axes, ((0, 1), (0, 2), (1, 2))):
                    ax.plot(fm[:, i], fm[:, j], color=COLORS["rrt"], ls="-.", lw=1.2,
                            label=LABELS["rrt"] + f" ({pz[0].stem.split('_')[-1]})")
            if ref is not None:
                for ax, (i, j), nm in zip(axes, ((0, 1), (0, 2), (1, 2)), ("x-y", "x-z", "y-z")):
                    ax.plot(ref[:, i], ref[:, j], color="black", lw=2.2, alpha=0.5,
                            label="reference")
                    ax.plot(ref[0, i], ref[0, j], "ko", ms=5)
                    ax.set_title(nm)
                    ax.set_aspect("equal", adjustable="datalim")
                    ax.grid(alpha=0.3)
            axes[0].legend(fontsize=8, loc="best")
            fig.suptitle(f"{p}: mid-node path, best valid epoch per method")
            fig.tight_layout()
            fig.savefig(out / f"paths_{p}.png", dpi=100)
            plt.close(fig)
            L.append(f"![{p}](paths_{p}.png)")
            L.append("")
    (out / "report.md").write_text("\n".join(L))
    print(f"wrote {out / 'report.md'}")


# --------------------------------------------------------------------- jobs

def cmd_jobs(args):
    """Training commands for the verified paths, one per line.

    Per path: NC from ``--nc-seeds``; RRT + NC once per *distinct* verified
    plan (planner seeds that produce the same commands count once), topped up
    with further policy seeds on the first plan until there are as many RRT +
    NC runs as NC runs; and NC at the fine-tuning learning rate from
    ``--nc-lr-seeds`` (the same-lr control: lr 1e-3 makes NC oscillate).
    """
    out = Path(args.out)
    run = f"python scripts/train_tracking.py --out {out}"
    for p in (args.paths or PATHS):
        f = out / "plans" / f"{p}_verify.json"
        if not f.exists() or not json.load(open(f))["feasible"]:
            continue
        v = json.load(open(f))
        for s in args.nc_seeds:
            print(f"{run} --path {p} --method nc --seed {s} > {out}/logs/{p}_nc_s{s}.log 2>&1")
        plans = []
        for q in v["plans"]:
            if q["success"] and q.get("fine_converged"):
                U = np.load(out / "plans" / f"{p}_p{q['seed']}.npz")["U_plan"]
                if all(np.max(np.abs(U - U2)) > 1e-6 for _, U2 in plans):
                    plans.append((q["seed"], U))
        gl = [(ps, 42) for ps, _ in plans]
        extra = [s for s in args.nc_seeds if s != 42]
        while plans and len(gl) < len(args.nc_seeds) and extra:
            gl.append((plans[0][0], extra.pop(0)))
        for ps, s in gl:
            print(f"{run} --path {p} --method rrt_nc --plan-seed {ps} --seed {s} "
                  f"> {out}/logs/{p}_rrt_nc_s{s}_p{ps}.log 2>&1")
        for s in args.nc_lr_seeds:
            print(f"{run} --path {p} --method nc --seed {s} --lr 0.0001 "
                  f"> {out}/logs/{p}_nc_s{s}_lr.log 2>&1")


def main():
    cmds = ("train", "verify", "jobs", "report")
    argv = sys.argv[1:]
    cmd = argv.pop(0) if argv and argv[0] in cmds else "train"

    ap = argparse.ArgumentParser(description=f"path tracking by neural control ({cmd})")
    ap.add_argument("--out", default=None,
                    help="directory the job's json/npz (and plans/) live in; "
                         "default for training: the run directory")
    if cmd in ("train", "verify"):
        ap.add_argument("--config", default=None, help="YAML config over configs/default.yaml")
        ap.add_argument("--path", choices=sorted(PATHS), default=None,
                        help="a named preset; default: the config's trajectory_type")
        ap.add_argument("--ready-dx", type=float, default=0.01,
                        help="[m] gripper move towards the fixed end before tracking starts")
    if cmd == "train":
        ap.add_argument("--method", choices=("nc", "rrt_nc"), default="nc")
        ap.add_argument("--grad-method", choices=GRAD_METHODS, default=None,
                        help="adjoint gradient: state (exact) or proxy (frozen tangent); "
                             "default: train.grad_method")
        ap.add_argument("--optimizer", choices=OPTIMIZERS, default=None,
                        help="adam (fixed lr, every step accepted) or linesearch (the paper's "
                             "Algorithm 1); default: train.optimizer")
        ap.add_argument("--seed", type=int, default=42, help="policy init seed")
        ap.add_argument("--plan-seed", type=int, default=0, help="rrt_nc: which verified plan")
        ap.add_argument("--epochs", type=int, default=None, help="default: train.epochs")
        ap.add_argument("--lr", type=float, default=None,
                        help="default: train.lr for nc, 1e-4 for rrt_nc")
        ap.add_argument("--device", default=None, help="default: train.device")
        ap.add_argument("--fit-iters", type=int, default=5000)
        ap.add_argument("--ws-weight", type=float, default=100.0)
    if cmd == "verify":
        ap.add_argument("--plan-seeds", type=int, nargs="+", default=[0, 1, 2])
        ap.add_argument("--segments", type=int, default=20, help="waypoints along the path")
        ap.add_argument("--wp-tol", type=float, default=0.005, help="[m] waypoint tolerance")
        ap.add_argument("--path-tol", type=float, default=0.015,
                        help="[m] largest allowed distance to the reference within a segment")
        ap.add_argument("--seg-time", type=float, default=60.0, help="[s] planner time per segment")
        ap.add_argument("--seg-iter", type=int, default=2000)
        ap.add_argument("--seg-tries", type=int, default=4, help="planner restarts per segment")
        ap.add_argument("--shortcuts", type=int, default=10)
        ap.add_argument("--plan-nv", type=int, default=11)
        ap.add_argument("--plan-newton-iter", type=int, default=100)
    if cmd == "jobs":
        ap.add_argument("--paths", nargs="+", choices=sorted(PATHS), default=None,
                        help="default: every verified path")
        ap.add_argument("--nc-seeds", type=int, nargs="+", default=[0, 1, 42])
        ap.add_argument("--nc-lr-seeds", type=int, nargs="*", default=[0, 1, 42],
                        help="NC runs per path at the fine-tuning lr (control); none to skip")
    args = ap.parse_args(argv)

    if cmd in ("verify", "jobs", "report") and args.out is None:
        raise SystemExit(f"{cmd} needs --out, the experiment directory")
    if cmd == "verify":
        if args.path is None:
            raise SystemExit("verify needs --path, one of the named presets")
        cmd_verify(args)
    elif cmd == "jobs":
        cmd_jobs(args)
    elif cmd == "report":
        report(Path(args.out))
    else:
        if args.method == "rrt_nc" and args.path is None:
            raise SystemExit("rrt_nc needs --path: the plan is verified per named path")
        if args.method == "rrt_nc" and args.out is None:
            raise SystemExit("rrt_nc needs --out, the directory `verify` wrote plans/ into")
        cmd_train(args)


if __name__ == "__main__":
    main()
