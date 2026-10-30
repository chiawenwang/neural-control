"""Global planning + local neural control vs. neural control alone (point-to-point).

The idea comes from Yu et al., "Generalizable whole-body global manipulation of
deformable linear objects by dual-arm robot in 3-D constrained environments"
(reference/): a global planner on a *coarse* rod model finds a feasible path
quickly, and a local controller on the real system refines it.  Here the
planner is src/dlo_rrt.py (RRT on a 10-segment rod) and the local stage is the
point-to-point neural control of scripts/train_adjoint.py, unchanged.

Methods, one per ``run`` invocation:

  nc       train_adjoint.py's single-horizon training from a random init.
  rrt_nc   plan with RRT -> resample the path to the N control steps -> fit the
           policy to it (the warm start) -> the same adjoint training.
           Also records the planned path executed open-loop on the fine rod,
           which is the "RRT only" result.

The training loop is train_adjoint.py's single-horizon loop, line for line,
with recording added; rollout_segment / BoundaryVelocityNet /
final_midpoint_loss are imported from it, so the two cannot drift apart.

With ``workspace.enabled`` in the config (e.g. configs/workspace_3d.yaml) the
right gripper is confined to that workspace in both stages: the planner only
grows the tree inside it, and the neural control handles it as
train_pipe.py does (each command projected onto it before it is executed,
plus a penalty on what the policy asks for beyond it).

Usage
-----
    python scripts/compare_global_local.py run --target T0 --method nc --seed 42 --out DIR
    python scripts/compare_global_local.py run --target T0 --method rrt_nc --seed 42 --rrt-seed 0 --out DIR
    python scripts/compare_global_local.py report --out DIR
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'scripts'))

from blas_threads import limit_blas_threads
limit_blas_threads()

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.optim as optim

from config import load_config
from dlo_rrt import (RRTPlanner, PlannerParams, path_length, resample_path,
                     stable_step, min_stiffness_eig)
from env_factory import (
    build_env, reset_env, env_dim, env_twist, end_control_dim, initial_control,
    task_target, dloss_dq_free_node, node_position,
)
from implicit_grad_tools import multistep_adjoint_gradient
from train_adjoint import BoundaryVelocityNet, rollout_segment, final_midpoint_loss

#: Mid-node targets.  T0 is configs/default.yaml's task.target_xyz; the others
#: spread over the reachable set (|target| < L/2 = 0.5 from the fixed end),
#: from mild (T1, T2) to ones that need the rod to arch over, swing sideways
#: past the fixed gripper, or curl back behind it (T3-T5).
TARGETS = {
    "T0": None,
    "T1": (0.20, 0.25, 0.25),
    "T2": (0.30, -0.25, -0.10),
    "T3": (0.10, 0.00, 0.35),
    "T4": (0.00, 0.30, 0.05),
    "T5": (-0.10, -0.15, 0.15),
}


# ---------------------------------------------------------------- helpers

def resolve_target(name, cfg):
    if name in TARGETS:
        return task_target(cfg) if TARGETS[name] is None else np.array(TARGETS[name], float)
    return np.array([float(v) for v in name.split(",")])


def workspace_bounds(cfg, dim, k):
    """(lo, hi) over the right gripper's ``k`` controls, or None when the
    workspace is off.  3D reads the ``*_3d`` keys (x, y, z, theta, phi), 2D the
    plain ones (x, y, theta) as train_pipe.py does; a control neither lists
    (the twist yaw) is unbounded."""
    ws = cfg.get("workspace") or {}
    if not ws.get("enabled", False):
        return None
    sfx = "_3d" if dim == 3 else ""
    ang = np.array(ws[f"orientation_limits{sfx}"], dtype=float).reshape(-1, 2)
    lo, hi = np.full(k, -np.inf), np.full(k, np.inf)
    lo[:dim], hi[:dim] = ws[f"position_min{sfx}"], ws[f"position_max{sfx}"]
    lo[dim:dim + len(ang)], hi[dim:dim + len(ang)] = ang[:, 0], ang[:, 1]
    return lo, hi


def workspace_projection(bounds, u0, k):
    """train_pipe.py's projection, for this control layout: clamp the right
    gripper's command onto the workspace before it is executed.

    Straight-through: the backward pass is the identity, so the task gradient
    is not cut off at the boundary (the penalty in train_p2p, on the
    unprojected command, pulls the policy back in).
    """
    if bounds is None:
        return None
    lo = torch.full_like(u0, -float("inf"))
    hi = torch.full_like(u0, float("inf"))
    lo[k:] = torch.tensor(bounds[0], dtype=u0.dtype)
    hi[k:] = torch.tensor(bounds[1], dtype=u0.dtype)
    return lambda u: u + (torch.clamp(u, lo, hi) - u).detach()


def rollout_record(env, q_list, u_list, dim):
    """Mid-node path (rest state first) and right-end controls of a rollout."""
    mid0 = node_position(env.nodes.reshape(-1), dim, env.mid_node)
    mids = np.array([mid0] + [node_position(q, dim, env.mid_node) for q in q_list])
    U = np.array([u.detach().cpu().numpy() if torch.is_tensor(u) else u for u in u_list])
    return mids, U


def execute_controls(env, checkpoint, U, stable=False, J_out=None):
    """Open-loop: step the env through the given absolute controls.

    ``stable=True`` steps with dlo_rrt.stable_step (the planner's own model of
    the rod); the default is the plain env.step the controller trains on.
    ``J_out`` (a list) collects the equilibrium Jacobians.
    """
    env.set_state(checkpoint)
    q_list, ok = [], True
    for u in U:
        if stable:
            info = stable_step(env, u)
            conv = info["stable"]
        else:
            _, _, info = env.step(u, use_inertia=False)
            conv = info["converged"]
        q_list.append(info["q"])
        if J_out is not None:
            J_out.append(info["J"])
        if not conv:
            ok = False
            break
    return q_list, ok


def path_metrics(env, u0, U, q_list, target, dim, k, J_list=None, bounds=None):
    """Scalar descriptors of one rollout on ``env`` (the fine rod).

    With ``J_list`` it also counts the steps whose equilibrium is unstable
    (J_ff not positive definite): states a real rod would not stay in.  With
    workspace ``bounds``, the largest violation and the number of steps with
    the gripper on the workspace boundary.
    """
    ws = {}
    if bounds is not None:
        R = np.asarray(U)[:, k:]
        lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
        on = np.isclose(R, lo, rtol=0.0, atol=1e-6) | np.isclose(R, hi, rtol=0.0, atol=1e-6)
        ws = {"ws_violation": float(np.max(np.maximum(R - hi, 0.0) + np.maximum(lo - R, 0.0))),
              "ws_boundary_steps": int(np.sum(on.any(axis=1)))}
    mids, _ = rollout_record(env, q_list, [], dim)
    U_full = np.vstack([u0[None, :], U])              # include the start pose
    P = U_full[:, k:k + dim]
    if dim == 3:
        th, ph = U_full[:, k + 3], U_full[:, k + 4]
        D = np.stack([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)], 1)
    else:
        th = U_full[:, k + 2]
        D = np.stack([np.cos(th), np.sin(th)], 1)
    rot = np.arccos(np.clip(np.sum(D[1:] * D[:-1], 1), -1.0, 1.0))
    dlam = 1.0 / len(U)
    acc_p = np.diff(P, 2, axis=0)
    mid_len = float(np.sum(np.linalg.norm(np.diff(mids, axis=0), axis=1)))
    n_pos = dim * env.nv
    energies = [env.energy.energy(np.asarray(q)[:n_pos]) if not getattr(env, "twist", False)
                else env.energy.energy(np.asarray(q)) for q in q_list]
    gap = np.linalg.norm(P - U_full[:, :dim], axis=1) / env.rod_length
    stab = {}
    if J_list is not None:
        lam = np.array([min_stiffness_eig(env, {"J": J}) for J in J_list])
        stab = {"unstable_steps": int(np.sum(lam < -1e-8)), "min_stiffness_eig": float(lam.min())}
    return {**stab, **ws,
        "final_error": float(np.linalg.norm(mids[-1] - target)),
        "ee_translation": float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))),
        "ee_rotation": float(np.sum(rot)),
        "ee_net_translation": float(np.linalg.norm(P[-1] - P[0])),
        "mid_length": mid_len,
        "mid_straightness": float(np.linalg.norm(mids[-1] - mids[0]) / max(mid_len, 1e-12)),
        "peak_elastic_energy": float(np.max(energies)),
        "final_elastic_energy": float(energies[-1]),
        "max_gap_ratio": float(np.max(gap)),
        "ee_accel_cost": float(np.sum(acc_p ** 2) / dlam ** 3),
        "ee_peak_step": float(np.max(np.linalg.norm(np.diff(P, axis=0), axis=1))),
    }


def fit_policy(net, u0, U_target, k, iters=3000, lr=1e-2):
    """Warm start: make the policy's integrated control reproduce ``U_target``.

    The rollout integrates ``u_j = u_0 + dlam * sum_{i<=j} net(i/N)`` (see
    rollout_segment), so this fits that cumulative sum, not the rates: errors
    in the rates would otherwise pile up along the path.
    """
    N = len(U_target)
    dlam = 1.0 / N
    lam = torch.arange(N, dtype=torch.float32).unsqueeze(1) / N
    tgt = torch.tensor(U_target[:, k:], dtype=torch.float32)
    base = u0[k:].detach()
    opt = optim.Adam(net.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 1e-2)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        u = base + dlam * torch.cumsum(net(lam), dim=0)
        loss = ((u - tgt) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        u = base + dlam * torch.cumsum(net(lam), dim=0)
        err = (u - tgt).abs()
    return {"fit_mse": float(loss.detach()), "fit_max_pos": float(err[:, :3].max()),
            "fit_max_ang": float(err[:, 3:].max())}


# --------------------------------------------------------------- training

def train_p2p(env, checkpoint, net, u0, target, cfg, epochs, device, lr, log_every=10, tag="",
              track_ref=None, track_weight=0.0, bounds=None, ws_weight=0.0):
    """train_adjoint.py, single-horizon mode, with per-epoch recording.

    ``checkpoint`` is the rest state every rollout starts from (train_adjoint
    takes it right after reset_env).  ``track_ref`` (N, dim) with
    ``track_weight > 0`` adds the paper's path-tracking term,
    ``track_weight * mean_k |mid_k - track_ref_k|^2``, to the terminal loss:
    the local stage then follows the global plan instead of only its end.
    The recorded error stays the terminal one.

    Workspace ``bounds`` (see workspace_bounds) are handled as in
    train_pipe.py: every command is projected onto the workspace before it is
    executed, and ``ws_weight * mean_k |violation_k|^2`` of the unprojected
    command joins the surrogate.
    """
    dim, twist = env.dim, getattr(env, "twist", False)
    opt = optim.Adam(net.parameters(), lr=lr)
    N = env.Nsteps
    dlam = 1.0 / N
    n_end = net.out_dim
    project = workspace_projection(bounds, u0, n_end)
    if bounds is not None:
        lo_t = torch.tensor(bounds[0], dtype=u0.dtype, device=device)
        hi_t = torch.tensor(bounds[1], dtype=u0.dtype, device=device)
    hist = {"loss": [], "error": [], "converged": [], "time": [], "mid_paths": [], "u_paths": [],
            "ws_loss": []}
    t_start = time.time()
    for ep in range(epochs):
        env.set_state(checkpoint)
        opt.zero_grad(set_to_none=True)

        u_list, q_list, J_list, converged = rollout_segment(env, net, u0, N, dlam, device,
                                                            cfg.train.jac_reg, project=project)
        if not converged:
            # train_adjoint skips the update on a failed rollout, so the next
            # epoch replays the very same rollout and fails at the same step,
            # and so on to the end: every later epoch is this one again.  Record
            # them as failed without re-running them (minutes each).
            hist["stuck"] = {"epoch": ep, "step": len(u_list)}
            print(f"[{tag} ep {ep:03d}] Newton failed at step {len(u_list)}/{N}; no update "
                  f"follows, so all {epochs - ep} remaining epochs repeat it -- recorded as failed",
                  flush=True)
            for _ in range(ep, epochs):
                for key, v in (("loss", np.nan), ("error", np.nan), ("converged", False),
                               ("time", time.time() - t_start), ("ws_loss", np.nan)):
                    hist[key].append(v)
                hist["mid_paths"].append(np.full((N + 1, dim), np.nan))
                hist["u_paths"].append(np.full((N, len(u0)), np.nan))
            break

        q_final = q_list[-1]
        free_index, fixed_index = env.free_index, env.fixed_index
        L_np, mid_pos = final_midpoint_loss(q_final, dim, env.mid_node, target)

        a_free_per_step = np.zeros((len(q_list), len(free_index)), dtype=float)
        a_free_per_step[-1] = dloss_dq_free_node(q_final, free_index, dim, env.mid_node, target)
        if track_weight > 0.0:
            w = track_weight / len(q_list)
            for kk, qk in enumerate(q_list):
                a_free_per_step[kk] += w * dloss_dq_free_node(qk, free_index, dim, env.mid_node,
                                                              track_ref[kk])

        u_seq_np = np.stack([u.detach().cpu().numpy() for u in u_list], axis=0)
        g_u_seq = multistep_adjoint_gradient(
            u_seq_np, J_list, free_index, fixed_index, env.deltaL,
            a_free_per_step, jac_reg=cfg.train.jac_reg, dim=dim, twist=twist,
            boundary_rows=getattr(env, 'boundary_rows', None),
        )

        surrogate = torch.zeros((), device=device)
        for k, u_k in enumerate(u_list):
            g_k = torch.tensor(g_u_seq[k], dtype=u_k.dtype, device=device)
            surrogate = surrogate + torch.dot(u_k, g_k)
        L_ws = np.nan
        if bounds is not None:
            # train_pipe.py's control_losses: the violation of what the policy
            # asks for -- the previous command plus this step's rate, before
            # the projection -- so the penalty sees what the projection clipped.
            lams = torch.arange(len(u_list), dtype=u0.dtype, device=device)[:, None] / N
            prev = torch.stack([u0] + u_list[:-1])[:, n_end:]
            raw = prev + dlam * net(lams)
            viol = torch.relu(raw - hi_t) + torch.relu(lo_t - raw)
            L_ws_t = ws_weight * (viol ** 2).sum(dim=1).mean()
            surrogate = surrogate + L_ws_t
            L_ws = float(L_ws_t.detach())

        surrogate.backward()
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
        opt.step()

        mids, U = rollout_record(env, q_list, u_list, dim)
        hist["loss"].append(float(L_np))
        hist["error"].append(float(L_np ** 0.5))
        hist["converged"].append(True)
        hist["time"].append(time.time() - t_start)
        hist["mid_paths"].append(mids)
        hist["u_paths"].append(u_seq_np)
        hist["ws_loss"].append(L_ws)
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"[{tag} ep {ep:03d}] err={L_np ** 0.5 * 1e3:7.2f} mm  "
                  + ("" if bounds is None else f"ws_pen={L_ws:.2e}  ")
                  + f"t={time.time() - t_start:7.1f}s", flush=True)

    # Final policy (after the last update), as optimize_segment reports it.
    env.set_state(checkpoint)
    with torch.no_grad():
        u_list, q_list, J_list, converged = rollout_segment(env, net, u0, N, dlam, device,
                                                            cfg.train.jac_reg, project=project)
    return hist, u_list, q_list, J_list, converged, time.time() - t_start


def plan_once(cfg, args, rrt_seed, env, checkpoint, u0_np, target, bounds=None):
    """One RRT plan on the coarse rod, replayed on both rods.

    With ``args.rrt_select == "open_loop"`` the unshortened tree path is
    replayed as well, and whichever of the two ends nearer the target on the
    fine rod is kept (``ps["path"]``): a shortcut that is valid on the coarse
    rod can take the fine rod onto another equilibrium branch.

    Returns (summary dict, arrays dict).  The fine ``env`` is left in an
    arbitrary state; callers restore ``checkpoint`` themselves.
    """
    dim, twist = env_dim(cfg), env_twist(cfg)
    k = end_control_dim(dim, twist)
    N = env.Nsteps
    cfg_c = copy.deepcopy(cfg)
    cfg_c["rod"]["nv"] = args.plan_nv
    env_c = build_env(cfg_c, max_newton_iter=args.plan_newton_iter)
    reset_env(env_c, cfg_c)
    params = PlannerParams(seed=rrt_seed, goal_tol=args.plan_tol, time_limit=args.plan_time)
    if bounds is not None:
        # The workspace replaces the planner's own (wide) box; its angle limits
        # are the finite ones (theta and phi -- not a twist yaw).
        n_ang = int(np.sum(np.isfinite(bounds[0][dim:])))
        params.box_min, params.box_max = tuple(bounds[0][:dim]), tuple(bounds[1][:dim])
        params.ang_min = tuple(bounds[0][dim:dim + n_ang])
        params.ang_max = tuple(bounds[1][dim:dim + n_ang])
    planner = RRTPlanner(env_c, u0_np, target, params)
    plan = planner.plan(env_c.get_state())
    ps = {kk: v for kk, v in plan.items() if not isinstance(v, np.ndarray)}
    ps["seed"] = rrt_seed
    if not plan["success"]:
        return ps, {}
    t_replay = time.time()
    paths = [("short", plan["U"])]
    if args.rrt_select == "open_loop":
        paths.append(("raw", plan["U_raw"]))
    chosen, ps["candidates"] = None, []
    for kind, W in paths:
        c_ps = {"path": kind, "length": float(path_length(W, k, dim, params.w_rot)[-1]),
                "n_waypoints": int(len(W))}
        U_plan = resample_path(W, N, k, dim, params.w_rot)
        c_pa = {"plan_waypoints": W, "plan_waypoints_raw": plan["U_raw"], "U_plan": U_plan}

        # the planned mid-node path, as the planner simulated it, at the N steps
        tr = planner.trace(W)
        if tr is not None:
            s_tr, mid_tr = tr
            s_q = s_tr[-1] * np.arange(1, N + 1) / N
            c_pa["plan_mid_trace"] = np.vstack([mid_tr[:1]] + [np.stack(
                [np.interp(s_q, s_tr, mid_tr[:, c]) for c in range(dim)], axis=1)])

        # the resampled plan on its own (coarse) model ...
        reset_env(env_c, cfg_c)
        q_c, ok_c = execute_controls(env_c, env_c.get_state(), U_plan, stable=True)
        c_pa["plan_mid_coarse"] = rollout_record(env_c, q_c, [], dim)[0]
        c_ps["coarse_converged"] = bool(ok_c)
        c_ps["coarse_error"] = (float(np.linalg.norm(c_pa["plan_mid_coarse"][-1] - target))
                                if ok_c else None)

        # ... and open-loop on the fine rod: the "RRT only" result
        J_f = []
        q_f, ok_f = execute_controls(env, checkpoint, U_plan, J_out=J_f)
        c_ps["open_loop"] = {"converged": bool(ok_f)}
        if ok_f:
            c_ps["open_loop"].update(path_metrics(env, u0_np, U_plan, q_f, target, dim, k, J_f,
                                                  bounds))
            c_pa["rrt_mid"] = rollout_record(env, q_f, [], dim)[0]
            c_pa["rrt_shapes"] = np.array([np.asarray(q)[:dim * env.nv].reshape(-1, dim)
                                           for q in q_f[::25] + [q_f[-1]]])
        ps["candidates"].append({"path": kind, "length": c_ps["length"],
                                 "converged": bool(ok_f),
                                 "open_loop_error": c_ps["open_loop"].get("final_error")})
        # The raw path has to be better by a margin: when both end on the same
        # branch they end in the same place, and the shortened path is kept.
        err = c_ps["open_loop"].get("final_error", np.inf)
        if chosen is None or err < chosen[0]["open_loop"].get("final_error", np.inf) - 1e-3:
            chosen = (c_ps, c_pa)
    ps.update(chosen[0])
    ps["time_replay"] = time.time() - t_replay
    env.set_state(checkpoint)
    return ps, chosen[1]


# ------------------------------------------------------------------- run

def cmd_run(args):
    cfg = load_config(args.config)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    dim, twist = env_dim(cfg), env_twist(cfg)
    k = end_control_dim(dim, twist)
    target = resolve_target(args.target, cfg)
    env = build_env(cfg)
    reset_env(env, cfg)
    checkpoint = env.get_state()
    N = env.Nsteps
    u0_np = initial_control(cfg)
    u0 = torch.tensor(u0_np, dtype=torch.float32, device=device)
    net = BoundaryVelocityNet(hidden=cfg.train.hidden, out_dim=k).to(device)
    bounds = workspace_bounds(cfg, dim, k)
    if bounds is not None and (np.any(u0_np[k:] < bounds[0]) or np.any(u0_np[k:] > bounds[1])):
        raise SystemExit(f"the initial gripper pose {u0_np[k:]} is outside the workspace")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lr = cfg.train.lr if args.lr is None else args.lr
    name = (f"{args.target}_{args.method}_s{args.seed}"
            + (f"_r{args.rrt_seed}" + (f"k{args.rrt_k}" if args.rrt_k > 1 else "")
               if args.method == "rrt_nc" else "")
            + ("" if args.lr is None else f"_lr{args.lr:g}")
            + (f"_trk{args.track_weight:g}" if args.track_weight > 0 else ""))
    arrays, summary = {}, {"target_name": args.target, "target": target.tolist(),
                           "method": args.method, "seed": args.seed, "epochs": args.epochs,
                           "nv": int(env.nv), "N": int(N), "lr": float(lr),
                           "lr_default": float(cfg.train.lr),
                           "track_weight": float(args.track_weight),
                           "rrt_seed": args.rrt_seed if args.method == "rrt_nc" else None,
                           "rrt_k": args.rrt_k if args.method == "rrt_nc" else None,
                           "workspace": None if bounds is None else
                           {"lo": bounds[0].tolist(), "hi": bounds[1].tolist(),
                            "weight": float(args.ws_weight)}}

    if args.method == "rrt_nc":
        # ---- global: RRT on the coarse rod, K seeds, keep the shortest path
        plans = []
        for rs in range(args.rrt_seed, args.rrt_seed + args.rrt_k):
            ps, pa = plan_once(cfg, args, rs, env, checkpoint, u0_np, target, bounds)
            plans.append((ps, pa))
            ol = ps.get("open_loop", {})
            print(f"[{name}] RRT seed {rs}: success={ps['success']}  t={ps['time']:.1f}s  "
                  f"length={ps.get('length', np.nan):.3f}  open-loop err on fine rod="
                  f"{ol.get('final_error', np.nan) * 1e3:.2f} mm  (coarse rod: "
                  f"{(ps.get('coarse_error') or np.nan) * 1e3:.2f} mm)", flush=True)
        summary["plans"] = [ps for ps, _ in plans]
        summary["rrt_select"] = args.rrt_select
        ok = [(ps, pa) for ps, pa in plans if ps["success"] and ps["open_loop"].get("converged")]
        if not ok:
            summary["status"] = "plan_failed"
            json.dump(summary, open(out / f"{name}.json", "w"), indent=1, default=float)
            return
        if args.rrt_select == "open_loop":
            ps, pa = min(ok, key=lambda x: x[0]["open_loop"]["final_error"])
        else:
            ps, pa = min(ok, key=lambda x: x[0]["length"])
        summary["plan"] = dict(ps, time=float(sum(p["time"] for p, _ in plans)),
                               time_selected=ps["time"],
                               time_replay=float(sum(p.get("time_replay", 0.0) for p, _ in plans)))
        summary["rrt_seed_selected"] = ps["seed"]
        summary["rrt_open_loop"] = ps["open_loop"]
        summary["plan_coarse_error"] = ps.get("coarse_error")
        for rs_ps, rs_pa in plans:
            for key, v in rs_pa.items():
                arrays[f"{key}_seed{rs_ps['seed']}"] = v
        arrays.update(pa)
        U_plan = pa["U_plan"]

        # ---- warm start
        t_fit = time.time()
        summary["fit"] = fit_policy(net, u0, U_plan, k, iters=args.fit_iters)
        summary["fit"]["time"] = time.time() - t_fit
        print(f"[{name}] policy fit: {summary['fit']}", flush=True)

    # ---- local: adjoint neural control (identical for both methods)
    track_ref = None
    if args.track_weight > 0:
        if args.method != "rrt_nc":
            raise ValueError("--track-weight needs a plan to track (--method rrt_nc)")
        track_ref = arrays["plan_mid_trace"][1:]           # the plan's mid path, step by step
    env.n_newton_failures = 0
    hist, u_list, q_list, J_list, conv, t_train = train_p2p(env, checkpoint, net, u0, target, cfg,
                                                    args.epochs, device, lr, tag=name,
                                                    track_ref=track_ref,
                                                    track_weight=args.track_weight,
                                                    bounds=bounds, ws_weight=args.ws_weight)
    summary["train_time"] = t_train
    summary["epoch_error"] = hist["error"]
    summary["epoch_time"] = hist["time"]
    if bounds is not None:
        summary["epoch_ws_loss"] = hist["ws_loss"]
    summary["stuck"] = hist.get("stuck")
    summary["final_converged"] = bool(conv)
    summary["newton_failures_total"] = int(env.n_newton_failures)
    if conv:
        U_fin = np.array([u.detach().cpu().numpy() for u in u_list])
        summary["final"] = path_metrics(env, u0_np, U_fin, q_list, target, dim, k, J_list, bounds)
        arrays["final_mid"] = rollout_record(env, q_list, [], dim)[0]
        arrays["final_U"] = U_fin
        arrays["final_shapes"] = np.array([np.asarray(q)[:dim * env.nv].reshape(-1, dim)
                                           for q in q_list[::25] + [q_list[-1]]])
    # first converged epoch = the starting point of the local stage
    first = next((i for i, c in enumerate(hist["converged"]) if c), None)
    if first is not None:
        J0 = []
        q0, ok0 = execute_controls(env, checkpoint, hist["u_paths"][first], J_out=J0)
        if ok0:
            summary["start"] = path_metrics(env, u0_np, hist["u_paths"][first], q0, target, dim, k,
                                            J0, bounds)
    arrays["epoch_mid_paths"] = np.array(hist["mid_paths"], dtype=np.float32)
    arrays["epoch_u_paths"] = np.array(hist["u_paths"], dtype=np.float32)
    summary["status"] = "done"
    np.savez_compressed(out / f"{name}.npz", **arrays)
    json.dump(summary, open(out / f"{name}.json", "w"), indent=1, default=float)
    fe = summary.get("final", {}).get("final_error", np.nan)
    print(f"[{name}] done: final err={fe * 1e3:.2f} mm, train {t_train:.0f}s", flush=True)


# ----------------------------------------------------------------- report

#: Fixed categorical slots (dataviz reference palette, slots 1-3): the entity
#: keeps its colour in every figure.  The RRT+NC variants share one hue and
#: differ by line style, so no fourth hue is needed.
COLORS = {"nc": "#2a78d6", "nc_lr": "#2a78d6", "rrt": "#eb6834",
          "rrt_nc_best": "#1baf7a", "rrt_nc": "#1baf7a", "rrt_nc_same": "#1baf7a",
          "rrt_nc_track": "#1baf7a"}
STYLES = {"nc": "-", "nc_lr": "--", "rrt": "-.", "rrt_nc_best": "-", "rrt_nc": ":",
          "rrt_nc_same": "--", "rrt_nc_track": (0, (6, 2, 1, 2))}
LABELS = {"nc": "NC (direct)",
          "nc_lr": "NC (direct), fine-tune lr",
          "rrt": "RRT (open loop)",
          "rrt_nc_best": "RRT best-of-K + NC",
          "rrt_nc": "RRT first plan + NC",
          "rrt_nc_same": "RRT first plan + NC, NC's lr",
          "rrt_nc_track": "RRT best-of-K + NC tracking the plan"}
VARIANTS = ("nc", "nc_lr", "rrt_nc_best", "rrt_nc_track", "rrt_nc", "rrt_nc_same")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def frechet(P, Q):
    """Discrete Frechet distance between two polylines."""
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    C = np.full_like(D, np.inf)
    C[0, 0] = D[0, 0]
    for i in range(1, len(P)):
        C[i, 0] = max(C[i - 1, 0], D[i, 0])
    for j in range(1, len(Q)):
        C[0, j] = max(C[0, j - 1], D[0, j])
    for i in range(1, len(P)):
        for j in range(1, len(Q)):
            C[i, j] = max(min(C[i - 1, j], C[i - 1, j - 1], C[i, j - 1]), D[i, j])
    return float(C[-1, -1])


def _first_below(err, thr):
    for i, e in enumerate(err):
        if e is not None and np.isfinite(e) and e < thr:
            return i
    return None


def _failed(r):
    """Failed-epoch count, with where the run got stuck (see train_p2p)."""
    n = sum(1 for c in r["epoch_error"] if c is None or not np.isfinite(c))
    st = r.get("stuck")
    return f"{n} (stuck ep {st['epoch']}, step {st['step']})" if st else str(n)


def _jumps(err, thr=0.05):
    """Epoch-to-epoch error increases above ``thr`` [m]: branch switches
    (snap-through), not gradient noise.  Failed epochs are skipped."""
    e = [v for v in err if v is not None and np.isfinite(v)]
    return int(sum(1 for a, b in zip(e[:-1], e[1:]) if b - a > thr))


def _errors_mm(r):
    return np.array([np.nan if v is None else v for v in r["epoch_error"]], float) * 1e3


def load_runs(out):
    runs = []
    for js in sorted(out.glob("*.json")):
        if js.name.startswith("report"):
            continue
        s = json.load(open(js))
        npz = js.with_suffix(".npz")
        s["_arrays"] = dict(np.load(npz)) if npz.exists() else {}
        s["_name"] = js.stem
        runs.append(s)
    return runs


def variant(r, base_lr):
    same_lr = abs(r["lr"] - r.get("lr_default", base_lr)) < 1e-12
    if r["method"] == "nc":
        return "nc" if same_lr else "nc_lr"
    if r.get("track_weight"):
        return "rrt_nc_track"
    if (r.get("rrt_k") or 1) > 1:
        return "rrt_nc_best"
    return "rrt_nc_same" if same_lr else "rrt_nc"


def rrt_plans(rn):
    """seed -> plan record (summary + fine-rod open-loop path), deduplicated."""
    plans = {}
    for r in rn:
        a = r["_arrays"]
        recs = r.get("plans") or [dict(r.get("plan", {}), seed=r.get("rrt_seed"),
                                       open_loop=r.get("rrt_open_loop", {}))]
        for ps in recs:
            sd = ps.get("seed")
            sfx = f"_seed{sd}" if f"rrt_mid_seed{sd}" in a else ""
            if sd in plans or not ps.get("open_loop", {}).get("converged") \
                    or f"rrt_mid{sfx}" not in a:
                continue
            plans[sd] = {"ps": ps, "mid": a[f"rrt_mid{sfx}"], "U": a[f"U_plan{sfx}"],
                         "shape": a[f"rrt_shapes{sfx}"][-1],
                         "coarse": a.get(f"plan_mid_trace{sfx}", a.get(f"plan_mid_coarse{sfx}"))}
    return plans


def compare_paths(A, B, k):
    """Frechet distances and final-configuration differences of two paths."""
    da = _tangent(A["U"][-1, k + 3:k + 5]) @ _tangent(B["U"][-1, k + 3:k + 5])
    return {"mid": frechet(A["mid"], B["mid"]),
            "gripper": frechet(A["U"][:, k:k + 3], B["U"][:, k:k + 3]),
            "shape": float(np.max(np.linalg.norm(A["shape"] - B["shape"], axis=1))),
            "gripper_end": float(np.linalg.norm(A["U"][-1, k:k + 3] - B["U"][-1, k:k + 3])),
            "tangent_deg": float(np.degrees(np.arccos(np.clip(da, -1.0, 1.0))))}


def _style(ax):
    from matplotlib.ticker import FuncFormatter, NullFormatter
    if ax.get_yscale() == "log":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def report(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_runs = load_runs(out)
    runs = [r for r in all_runs if r.get("status") == "done"]
    failed_plans = [r["_name"] for r in all_runs if r.get("status") == "plan_failed"]
    base_lr = next((r.get("lr_default", r["lr"]) for r in runs), 1e-3)
    for r in runs:
        r["_v"] = variant(r, base_lr)
    targets = sorted({r["target_name"] for r in runs})
    lines = ["# Global planning + local neural control vs. neural control alone", "",
             f"NC learning rate {base_lr:g}.  Error = mid-node distance to the target on the "
             "fine rod.  'epoch k' = the rollout before the k-th update (epoch 0 = the initial "
             "policy: random for NC, fitted to the plan for RRT+NC); 'final' = the policy after "
             "the last update.  'time to <5mm' includes planning and fitting.", ""]
    if failed_plans:
        lines += [f"Planning failed: {', '.join(failed_plans)}", ""]
    ws = next((r["workspace"] for r in runs if r.get("workspace")), None)
    if ws:
        lo_ws, hi_ws = np.array(ws["lo"], float), np.array(ws["hi"], float)
        coords = ["x", "y", "z", "theta", "phi", "yaw"] if len(lo_ws) >= 5 else ["x", "y", "theta"]
        lims = [f"{n} in [{a:g}, {b:g}] m" if n in "xyz" else
                f"{n} in [{np.degrees(a):.0f}, {np.degrees(b):.0f}] deg"
                for n, a, b in zip(coords, lo_ws, hi_ws) if np.isfinite(a) or np.isfinite(b)]
        lines += [f"Gripper workspace: {', '.join(lims)}.  RRT plans inside it; NC projects "
                  f"every command onto it (penalty weight {ws['weight']:g} on the unprojected "
                  "command).", ""]
    if any(r.get("rrt_select") == "open_loop" for r in runs):
        lines += ["Best-of-K plan = of every plan's shortened path and raw tree path, the one "
                  "ending nearest the target when executed open loop on the fine rod (the "
                  "replays count toward the planning time).", ""]
    rows, pair_rows = [], []

    def fmt(v, scale=1.0, nd=1):
        return "-" if v is None or not np.isfinite(v) else f"{v * scale:.{nd}f}"

    # The representative path of a training run is its best epoch (the policy a
    # checkpoint would keep): with branch jumps and Newton dead ends, the last
    # epoch often says little about what the method found.  Its controls are
    # re-simulated on the fine rod for the full metrics, and cached.
    cfg = load_config(None)
    env = build_env(cfg)
    reset_env(env, cfg)
    checkpoint = env.get_state()
    u0_np = initial_control(cfg)
    dim = env.dim
    k = end_control_dim(dim, env_twist(cfg))
    cache = out / "report_cache"
    cache.mkdir(exist_ok=True)
    for r in runs:
        err = np.array([np.nan if v is None else v for v in r["epoch_error"]], float)
        if not np.any(np.isfinite(err)):
            r["_best"] = None
            continue
        b = int(np.nanargmin(err))
        f = cache / f"{r['_name']}_best.npz"
        if f.exists():
            c = np.load(f, allow_pickle=True)
            r["_best"] = {"epoch": b, "mid": c["mid"], "U": c["U"], "shape": c["shape"],
                          "metrics": json.loads(str(c["metrics"]))}
            continue
        U_b = r["_arrays"]["epoch_u_paths"][b].astype(float)
        J = []
        q_b, ok = execute_controls(env, checkpoint, U_b, J_out=J)
        b_ws = (np.array(r["workspace"]["lo"], float), np.array(r["workspace"]["hi"], float)) \
            if r.get("workspace") else None
        met = path_metrics(env, u0_np, U_b, q_b, np.array(r["target"]), dim, k, J, b_ws) if ok else {}
        mids = rollout_record(env, q_b, [], dim)[0]
        shape = np.asarray(q_b[-1])[:dim * env.nv].reshape(-1, dim)
        np.savez(f, mid=mids, U=U_b, shape=shape, metrics=json.dumps(met))
        r["_best"] = {"epoch": b, "mid": mids, "U": U_b, "shape": shape, "metrics": met}

    for t in targets:
        R = [r for r in runs if r["target_name"] == t]
        nc = sorted([r for r in R if r["_v"] == "nc"], key=lambda r: (r["seed"] != 42, r["seed"]))
        nc_lr = [r for r in R if r["_v"] == "nc_lr"]
        rn = sorted([r for r in R if r["method"] == "rrt_nc"],
                    key=lambda r: VARIANTS.index(r["_v"]))
        plans = rrt_plans(rn)
        selected = {r.get("rrt_seed_selected") for r in rn if r["_v"] == "rrt_nc_best"}
        tgt = np.array(R[0]["target"])

        def run_label(r):
            if r["_v"].startswith("nc"):
                return f"seed {r['seed']}" + (f", lr {r['lr']:g}" if r["_v"] == "nc_lr" else "")
            sd = (f"K={r['rrt_k']}, plan seed {r.get('rrt_seed_selected')}"
                  if (r.get("rrt_k") or 1) > 1 else f"plan seed {r['rrt_seed']}")
            return sd + f", lr {r['lr']:g}"

        trained = nc + nc_lr + rn
        lines += [f"## {t}: target {np.round(tgt, 3).tolist()}", "",
                  "Convergence (errors in mm; epochs counted from 0):", "",
                  "| method | run | start | best (epoch) | final | ep<10mm | ep<5mm | ep<2mm "
                  "| time to <5mm [s] | plan(+fit) [s] | jumps >50mm | failed epochs |",
                  "|" + "---|" * 12]
        for sd, p in sorted(plans.items()):
            ol = p["ps"]["open_loop"]
            lines.append(f"| {LABELS['rrt']} | plan seed {sd}"
                         + (", raw tree path" if p["ps"].get("path") == "raw" else "")
                         + (" (selected)" if sd in selected else "")
                         + f" | {fmt(ol.get('final_error'), 1e3, 1)} | - | - | - | - | - | - | "
                         f"{fmt(p['ps'].get('time'), 1, 0)} | - | - |")
        for r in trained:
            err = r["epoch_error"]
            e10, e5, e2 = (_first_below(err, x) for x in (0.01, 0.005, 0.002))
            pre = (r["plan"]["time"] + r["plan"].get("time_replay", 0.0) + r["fit"]["time"]) \
                if r["method"] == "rrt_nc" else None
            t5 = ((pre or 0.0) + r["epoch_time"][e5]) if e5 is not None else None
            bst = r["_best"]
            best_s = (f"{np.nanmin([np.nan if v is None else v for v in err]) * 1e3:.1f} "
                      f"({bst['epoch']})") if bst else "-"
            start = err[0] if err and err[0] is not None and np.isfinite(err[0]) else None
            lines.append(
                f"| {LABELS[r['_v']]} | {run_label(r)} | {fmt(start, 1e3, 1)} | {best_s} | "
                f"{fmt((r.get('final') or {}).get('final_error'), 1e3, 1)} | "
                f"{e10 if e10 is not None else '-'} | {e5 if e5 is not None else '-'} | "
                f"{e2 if e2 is not None else '-'} | {fmt(t5, 1, 0)} | {fmt(pre, 1, 0)} | "
                f"{_jumps(err)} | {_failed(r)} |")
            rows.append({"target": t, "method": r["_v"], "run": run_label(r),
                         "start_error": start,
                         "best_error": (bst["metrics"].get("final_error") if bst else None),
                         "best_epoch": bst["epoch"] if bst else None,
                         "final_error": (r.get("final") or {}).get("final_error"),
                         "epochs_lt_10mm": e10, "epochs_lt_5mm": e5, "epochs_lt_2mm": e2,
                         "time_lt_5mm": t5, "pre_time": pre, "jumps_50mm": _jumps(err),
                         "failed_epochs": sum(1 for c in err if c is None or not np.isfinite(c)),
                         "stuck": r.get("stuck"),
                         "best_path": bst["metrics"] if bst else None})
        lines.append("")

        lines += ["Paths on the fine rod (RRT: the plan executed open loop; trained methods: "
                  "the best epoch's rollout):", "",
                  "| method | run | err [mm] | gripper travel [m] | gripper rotation [rad] "
                  "| mid-node path [m] | straightness | peak elastic E [mJ] | max gap/L "
                  "| unstable steps |" + (" steps on ws boundary |" if ws else ""),
                  "|" + "---|" * (11 if ws else 10)]
        path_rows = [(LABELS["rrt"], f"plan seed {sd}", p["ps"]["open_loop"])
                     for sd, p in sorted(plans.items())]
        path_rows += [(LABELS[r["_v"]], run_label(r) + f", epoch {r['_best']['epoch']}",
                       r["_best"]["metrics"]) for r in trained if r["_best"]]
        for lab, run, met in path_rows:
            lines.append(
                f"| {lab} | {run} | {fmt(met.get('final_error'), 1e3, 1)} | "
                f"{fmt(met.get('ee_translation'), 1, 2)} | {fmt(met.get('ee_rotation'), 1, 2)} | "
                f"{fmt(met.get('mid_length'), 1, 2)} | {fmt(met.get('mid_straightness'), 1, 2)} | "
                f"{fmt(met.get('peak_elastic_energy'), 1e3, 0)} | {fmt(met.get('max_gap_ratio'), 1, 3)} | "
                f"{met.get('unstable_steps', '-')} |"
                + (f" {met.get('ws_boundary_steps', '-')} |" if ws else ""))
        lines.append("")

        # ---- path differences, grouped by what is being compared
        P = {f"RRT s{sd}": dict(p, kind="rrt", seed=sd) for sd, p in plans.items()}
        for r in nc + rn:
            if r["_best"]:
                key = (f"NC s{r['seed']}" if r["_v"] == "nc" else
                       f"{LABELS[r['_v']]} (plan s{r.get('rrt_seed_selected', r.get('rrt_seed'))})")
                P[key] = {"kind": r["_v"], "mid": r["_best"]["mid"], "U": r["_best"]["U"],
                          "shape": r["_best"]["shape"],
                          "plan_seed": r.get("rrt_seed_selected", r.get("rrt_seed"))}
        if len(P) > 1:
            names = list(P)
            groups = {}
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    A, B = P[names[i]], P[names[j]]
                    kinds = {A["kind"], B["kind"]}
                    if kinds == {"rrt"}:
                        g = "RRT vs RRT (different seeds)"
                    elif kinds == {"nc"}:
                        g = "NC vs NC (different seeds)"
                    elif kinds == {"rrt", "nc"}:
                        g = "RRT (open loop) vs NC"
                    elif "rrt" in kinds:
                        other = A if A["kind"] != "rrt" else B
                        plan = A if A["kind"] == "rrt" else B
                        if other.get("plan_seed") != plan["seed"]:
                            continue
                        g = f"{LABELS[other['kind']]} vs its own RRT plan"
                    elif "nc" in kinds:
                        other = A if A["kind"] != "nc" else B
                        g = f"{LABELS[other['kind']]} vs NC"
                    else:
                        continue
                    d = compare_paths(A, B, k)
                    groups.setdefault(g, []).append(d)
                    pair_rows.append({"target": t, "group": g, "a": names[i], "b": names[j], **d})
            lines += ["Path differences (discrete Frechet distance between mid-node paths / "
                      "between right-gripper paths; final shape = max node distance between the "
                      "final rods).  mean (min-max) over pairs, mm / deg:", "",
                      "| comparison | pairs | mid path | gripper path | final shape "
                      "| final gripper pos | final tangent [deg] |", "|---|---|---|---|---|---|---|"]
            for g, ds in groups.items():
                cells = []
                for key, sc in (("mid", 1e3), ("gripper", 1e3), ("shape", 1e3),
                                ("gripper_end", 1e3), ("tangent_deg", 1.0)):
                    v = np.array([d[key] for d in ds]) * sc
                    cells.append(f"{v.mean():.0f}" + (f" ({v.min():.0f}-{v.max():.0f})" if len(v) > 1 else ""))
                lines.append(f"| {g} | {len(ds)} | " + " | ".join(cells) + " |")
            lines.append("")

        # ---- figure: top view, side view, convergence
        main_rn = next((r for r in rn if r["_v"] == "rrt_nc_best"), rn[0] if rn else None)
        shown_seed = (main_rn.get("rrt_seed_selected", main_rn.get("rrt_seed"))
                      if main_rn else None)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
        for ax, (a0, a1, lab) in zip(axes[:2], ((0, 1, ("x [m]", "y [m]")), (0, 2, ("x [m]", "z [m]")))):
            items = []
            if shown_seed in plans:
                p = plans[shown_seed]
                items.append(("rrt", p["mid"], p["U"], p["shape"]))
            if nc and nc[0]["_best"]:
                b = nc[0]["_best"]
                items.append(("nc", b["mid"], b["U"], b["shape"]))
            if main_rn is not None and main_rn["_best"]:
                b = main_rn["_best"]
                items.append((main_rn["_v"], b["mid"], b["U"], b["shape"]))
            for m, mid, U, S in items:
                c = COLORS[m]
                kk = U.shape[1] // 2
                ax.plot(mid[:, a0], mid[:, a1], color=c, lw=2.2 if m != "rrt" else 1.8,
                        label=f"{LABELS[m]}: mid-node path" + (" (best epoch)" if m != "rrt" else ""))
                ax.plot(U[:, kk + a0], U[:, kk + a1], color=c, lw=1.2, ls=":",
                        label=f"{LABELS[m]}: right-gripper path")
                ax.plot(S[:, a0], S[:, a1], color=c, lw=1.0, alpha=0.5)
                ax.plot(S[-1, a0], S[-1, a1], "s", color=c, ms=6, mec="white", mew=1)
            if shown_seed in plans and plans[shown_seed]["coarse"] is not None:
                pm = plans[shown_seed]["coarse"]
                ax.plot(pm[:, a0], pm[:, a1], color=COLORS["rrt"], lw=1.1, ls="--",
                        label="RRT: planned mid-node path (coarse rod)")
            if ws:
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((lo_ws[a0], lo_ws[a1]), hi_ws[a0] - lo_ws[a0],
                                       hi_ws[a1] - lo_ws[a1], fill=False, ec=INK2, lw=0.9,
                                       ls="--", label="gripper workspace"))
            ax.plot([0], [0], "s", color=INK2, ms=6, label="fixed left gripper")
            ax.plot(tgt[a0], tgt[a1], "x", color=INK, ms=11, mew=2.5, label="target")
            ax.set_xlabel(lab[0], color=INK2)
            ax.set_ylabel(lab[1], color=INK2)
            ax.set_aspect("equal", adjustable="datalim")
            _style(ax)
        axes[0].set_title("top view (x-y); faint line = final rod", color=INK, fontsize=9)
        axes[1].set_title("side view (x-z)", color=INK, fontsize=9)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)

        ax = axes[2]
        for r in nc + nc_lr + rn:
            v = r["_v"]
            ax.semilogy(_errors_mm(r), color=COLORS[v], ls=STYLES[v], lw=1.5,
                        alpha=0.4 if (v == "nc" and r is not nc[0]) else 0.95)
        if shown_seed in plans:
            ax.axhline(plans[shown_seed]["ps"]["open_loop"]["final_error"] * 1e3,
                       color=COLORS["rrt"], ls=STYLES["rrt"], lw=1.2)
        for v in VARIANTS:
            if any(r["_v"] == v for r in R):
                ax.plot([], [], color=COLORS[v], ls=STYLES[v], lw=1.5, label=LABELS[v])
        if shown_seed in plans:
            ax.plot([], [], color=COLORS["rrt"], ls=STYLES["rrt"], lw=1.2,
                    label="RRT open loop (final error)")
        ax.set_xlabel("epoch", color=INK2)
        ax.set_ylabel("mid-node error [mm]", color=INK2)
        ax.set_title("convergence (faint = other NC seeds)", color=INK, fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        _style(ax)
        fig.suptitle(f"{t}: mid-node target {np.round(tgt, 2).tolist()}", color=INK, fontsize=11)
        fig.tight_layout(rect=(0, 0.14, 1, 0.95))
        fig.savefig(out / f"paths_{t}.png", dpi=140)
        plt.close(fig)

    # ---- summary: convergence small multiples
    if targets:
        n = len(targets)
        cols = min(3, n)
        nrows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(nrows, cols, figsize=(4.6 * cols, 3.3 * nrows), squeeze=False)
        for ax, t in zip(axes.flat, targets):
            R = [r for r in runs if r["target_name"] == t]
            for r in R:
                v = r["_v"]
                ax.semilogy(_errors_mm(r), color=COLORS[v], ls=STYLES[v], lw=1.4,
                            alpha=0.4 if (v == "nc" and r["seed"] != 42) else 0.9)
            best = next((r for r in R if r["_v"] == "rrt_nc_best"), None)
            if best is not None:
                ax.axhline(best["rrt_open_loop"]["final_error"] * 1e3, color=COLORS["rrt"],
                           ls=STYLES["rrt"], lw=1.0)
            ax.set_title(f"{t}  {np.round(R[0]['target'], 2).tolist()}", color=INK, fontsize=9)
            ax.set_xlabel("epoch", color=INK2, fontsize=8)
            ax.set_ylabel("error [mm]", color=INK2, fontsize=8)
            _style(ax)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        vs = [v for v in VARIANTS if any(r["_v"] == v for r in runs)]
        handles = [plt.Line2D([], [], color=COLORS[v], ls=STYLES[v], lw=1.6) for v in vs]
        handles.append(plt.Line2D([], [], color=COLORS["rrt"], ls=STYLES["rrt"], lw=1.4))
        fig.legend(handles, [LABELS[v] for v in vs] + ["RRT open loop (selected plan)"],
                   loc="lower center", ncol=3, fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(out / "convergence_all.png", dpi=140)
        plt.close(fig)

    (out / "report.md").write_text("\n".join(lines))
    json.dump({"rows": rows, "pairs": pair_rows}, open(out / "report_rows.json", "w"),
              indent=1, default=float)
    print("\n".join(lines))
    print(f"\nreport -> {out / 'report.md'}")


def _tangent(angles):
    th, ph = angles
    return np.array([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)])


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", default=None)
    r.add_argument("--target", default="T0", help="T0..T5 or 'x,y,z'")
    r.add_argument("--method", choices=["nc", "rrt_nc"], required=True)
    r.add_argument("--seed", type=int, default=42, help="policy init seed (train.seed)")
    r.add_argument("--rrt-seed", type=int, default=0)
    r.add_argument("--rrt-k", type=int, default=1,
                   help="plan with seeds rrt_seed..rrt_seed+K-1 and keep one (--rrt-select)")
    r.add_argument("--rrt-select", choices=["length", "open_loop"], default="length",
                   help="length: the shortest plan; open_loop: of every plan's shortened and "
                        "raw tree path, the one ending nearest the target on the fine rod")
    r.add_argument("--epochs", type=int, default=None, help="default: train.epochs")
    r.add_argument("--lr", type=float, default=None,
                   help="learning rate of the adjoint stage (default: train.lr)")
    r.add_argument("--fit-iters", type=int, default=5000, help="warm-start fitting iterations")
    r.add_argument("--track-weight", type=float, default=0.0,
                   help="rrt_nc: weight of the term tracking the planned mid-node path")
    r.add_argument("--plan-nv", type=int, default=11, help="nodes of the planner's coarse rod")
    r.add_argument("--plan-tol", type=float, default=0.02, help="[m] planner goal tolerance")
    r.add_argument("--plan-time", type=float, default=600.0, help="[s] planner time limit")
    r.add_argument("--plan-newton-iter", type=int, default=100)
    r.add_argument("--ws-weight", type=float, default=100.0,
                   help="workspace penalty weight (task.pipe.weights.workspace's default)")
    r.add_argument("--out", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        if args.epochs is None:
            args.epochs = int(load_config(args.config).train.epochs)
        cmd_run(args)
    else:
        report(Path(args.out))


if __name__ == "__main__":
    main()
