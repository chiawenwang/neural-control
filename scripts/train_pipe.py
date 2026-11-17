"""90-degree pipe insertion: push the free tip of the rod round the bend to a target.

    python scripts/train_pipe.py --config configs/task_90deg_pipe.yaml
    python scripts/train_pipe.py --config configs/task_90deg_pipe.yaml --grad-method proxy
    python scripts/train_pipe.py --config configs/task_90deg_pipe.yaml --optimizer linesearch
    python scripts/train_pipe.py --config configs/task_90deg_pipe_3d.yaml --optimizer linesearch

In 3D (configs/task_90deg_pipe_3d.yaml) the pipe is two cylindrical bores
meeting at a right angle (contact.LPipe3D) and the target sits on the far
wall of the corner: once the tip touches that wall it cannot leave it again,
so a point on the centerline of the second bore is not where a rod ends up.

The controller is src/neural_control.py's, unchanged -- the same one the other
two tasks use: a BoundaryVelocityNet (lambda -> right-end control rate), the
same rollout (open loop: the whole control trajectory is generated, then
simulated), the same multi-step adjoint gradient through the equilibrium
Jacobians (train.grad_method, or --grad-method, picks the exact `state`
gradient or the frozen-tangent `proxy`). What this script adds is the task:

  * the rod is gripped at the right end only (env.grasp_mode: right, 2D or 3D
    without twist); node 0 is a free tip that moves only through the rod's
    response;
  * the pipe (pipe:, contact:) acts inside every equilibrium solve, so it shapes
    both the rollout and -- through the same Jacobian -- the gradient;
  * the loss is over the whole trajectory, not just the final pose:
        goal       |p_tip,T - target|^2
        path       mean_k (s_tip,k - s_ref,k)^2, s = arc length along the pipe
        contact    penetration beyond task.pipe.contact_margin, every step
        control    mean_k |u_dot_k|^2
        workspace  mean_k |workspace violation_k|^2
    The first three reach the network through the adjoint (dL/dq_free per
    step); the last two are functions of u alone and go straight into autograd.
  * the gripper command is projected onto the workspace before it is executed,
    so the robot never leaves it; the penalty acts on the unprojected command.
  * ``--optimizer`` (train.optimizer): ``adam`` as above, or ``linesearch``,
    the paper's Algorithm 1 (the Adam direction, a backtracking step length
    on full rollouts). The pipe loss is a trajectory objective, so in that
    mode a trial with a snap-through is rejected, like one with a Newton
    failure, an unstable state or a command outside the workspace (no
    projection then: the policy has to keep the gripper inside).

The rollouts start from the equilibrium at the initial gripper pose
(``newton.initial_equilibrium``), solved before the checkpoint is taken.

Before training it checks that the workspace can feed enough rod to reach the
target at all, and prints the buckling-vs-bending estimate for the configured
rod. After training it re-runs the final policy and reports the success
criteria of task.pipe.
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
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from config import load_config
from env_factory import (
    build_env,
    reset_env,
    settle_initial,
    end_control_dim,
    initial_control,
    task_target,
    material_summary,
    node_position,
)
from neural_control import (
    BoundaryVelocityNet, resolve_device, rollout, adjoint_gradient, surrogate_loss,
    workspace_bounds, workspace_projection, workspace_penalty, GRAD_METHODS, OPTIMIZERS,
    LineSearch, trial_rule, flat_gradient, get_parameters, set_parameters,
)
from runlog import RunLogger
from utils import maybe_save_trajectory_gif, visualize_pipe_task


# --------------------------------------------------------------- setup checks

def check_setup(cfg, env):
    """Refuse configurations this task cannot run, with the reason."""
    if getattr(env, "grasp_mode", "both") != "right":
        raise SystemExit("train_pipe.py needs env.grasp_mode: right -- with both ends "
                         "clamped there is no free tip to steer")
    if env.pipe is None:
        raise SystemExit("train_pipe.py needs pipe.enabled: true")
    tip = int(cfg.task.pipe.tip_node)
    if env.dim * tip not in set(env.free_index.tolist()):
        raise SystemExit(f"task.pipe.tip_node = {tip} is a clamped node")
    return tip


def feasibility_report(cfg, env, pipe, tip0, target):
    """Can the gripper feed enough rod, and will the free span hold the push?

    Feed: the rod is (nearly) inextensible, so the tip advances along the pipe
    by about as much as the gripper advances along the entry direction. The
    advance the workspace allows is ``max over the box of (p - p0) . d1``.

    Buckling: frictionless, forcing the rod round the bend costs about
    EI / (2 R_bend^2) of push (the bending energy per unit length fed in). The
    free span between the gripper and the entry face is clamped at the
    gripper and roughly pinned at the entry, so it buckles at about
    2.046 pi^2 EI / L_free^2. Both are proportional to EI: stiffening the rod
    does not change which one wins, only the geometry does.
    """
    u0 = initial_control(cfg)
    dim = env.dim
    k = end_control_dim(dim)
    g0 = u0[k:k + dim]
    s0 = pipe.arclength(tip0)[0]
    s_goal = pipe.arclength(target)[0]
    need = s_goal - s0

    bounds = workspace_bounds(cfg, dim, k)
    if bounds is None:
        avail = float("inf")
    else:
        lo, hi = bounds
        d = pipe.d1
        avail = float(sum(max(d[c] * (hi[c] - g0[c]), d[c] * (lo[c] - g0[c])) for c in range(dim)))

    L_free = float((g0 - pipe.P0) @ -pipe.d1)
    EI = env.EI
    P_bend = EI / (2.0 * pipe.turn_radius ** 2)
    P_euler = 2.046 * np.pi ** 2 * EI / L_free ** 2 if L_free > 0 else float("inf")
    strain = env.R_outer / pipe.turn_radius

    rep = {
        "tip_start_s": s0, "target_s": s_goal, "feed_required": need,
        "feed_available": avail, "free_span": L_free, "push_to_bend": P_bend,
        "euler_load": P_euler, "bend_strain": strain,
        "target_piece": pipe.piece_of(target),
        "target_clearance": float(pipe.compute_signed_clearance(target, smooth=False)[0]),
        "lateral_play": pipe.r_eff,
    }
    print("Feasibility:")
    print(f"  tip starts at s = {s0:.3f} m, target at s = {s_goal:.3f} m "
          f"({rep['target_piece']}); bend spans s = [{pipe.s_bend_start:.3f}, {pipe.s_bend_end:.3f}]")
    print(f"  feed required {need:.3f} m, workspace allows {avail:.3f} m")
    print(f"  free span {L_free:.3f} m: Euler load ~{P_euler:.2f} N vs ~{P_bend:.2f} N "
          f"to force the bend (ratio {P_euler / P_bend:.1f})")
    turn = "bend radius" if not pipe.corner else "corner (radius ~ inner_radius, a rough estimate)"
    print(f"  bending strain at the {turn}: {100 * strain:.2f}%; "
          f"lateral play in the pipe: +/-{1e3 * pipe.r_eff:.1f} mm")

    problems = []
    if rep["target_clearance"] < 0:
        problems.append("the target is outside the region the rod centerline can reach "
                        "(inside a wall, or closer to it than the rod radius)")
    if need > avail:
        problems.append(f"the workspace feeds at most {avail:.3f} m of rod but the tip has to "
                        f"travel {need:.3f} m: extend workspace.position_min/max along the "
                        "entry direction, start with more rod inserted, or move the target")
    if bounds is not None:
        lo, hi = bounds
        if np.any(u0[k:] < lo) or np.any(u0[k:] > hi):
            problems.append(f"the initial gripper pose {u0[k:]} is outside the workspace")
    if P_euler < P_bend:
        print("  WARNING: the free span will likely buckle before the tip is forced round "
              "the bend. Grip closer to the entry or enlarge the bend radius; a stiffer "
              "rod does not help (both loads scale with EI).")
    if strain > 0.03:
        print("  WARNING: bending strain above 3% -- beyond the elastic range of most "
              "polymers; the elastic rod model is optimistic here.")
    rep["problems"] = problems
    for p in problems:
        print(f"  INFEASIBLE: {p}")
    return rep


# --------------------------------------------------------------- losses

def free_grad(env, full):
    """A gradient over all DOFs restricted to the free ones."""
    return np.asarray(full)[env.free_index]


def node_grad(env, node, vec):
    full = np.zeros(env.q.size)
    full[env.dim * node:env.dim * node + env.dim] = vec
    return free_grad(env, full)


def task_losses(env, pipe, q_list, tip, target, s_ref, w, margin, goal=True):
    """Trajectory loss and its per-step dL/dq_free (for the adjoint).

    ``goal=False`` leaves out the final-tip term, for a rollout cut short by a
    failed solve: its prefix still has path and contact terms with exact
    Jacobians, but no final state.
    """
    T = len(q_list)
    N = len(s_ref)
    a_free = np.zeros((T, len(env.free_index)))
    parts = {"goal": 0.0, "path": 0.0, "contact": 0.0}

    dim = env.dim
    if goal:
        p_T = node_position(q_list[-1], dim, tip)
        d = p_T - target
        parts["goal"] = w["goal"] * float(d @ d)
        a_free[-1] += node_grad(env, tip, 2.0 * w["goal"] * d)

    for k, q in enumerate(q_list):
        s_k, ds_dp = pipe.arclength(node_position(q, dim, tip))
        e = s_k - s_ref[k]
        parts["path"] += w["path"] / N * e * e
        a_free[k] += node_grad(env, tip, 2.0 * w["path"] / N * e * ds_dp)

        Lc, dLc = pipe.compute_contact_loss(q, margin)
        if Lc > 0.0:
            parts["contact"] += w["contact"] * Lc
            full = np.zeros(q.size)
            full[:dLc.size] = w["contact"] * dLc
            a_free[k] += free_grad(env, full)
    return parts, a_free


def control_losses(net, u_list, u0, dlam, w, bounds, device):
    """Control effort and workspace violation, in torch (functions of u alone).

    Both are measured on what the policy *asks for* -- its rates, and the
    command before projection onto the workspace -- so the penalty still sees
    a violation that the projection kept from being executed.
    """
    T = len(u_list)
    lams = (torch.arange(T, dtype=u0.dtype, device=device) / round(1.0 / dlam))[:, None]
    L_ctrl = w["control"] * (net(lams) ** 2).sum(dim=1).mean()
    L_ws = workspace_penalty(net, u0, u_list, net.out_dim, bounds, w["workspace"], device, dlam)
    return L_ctrl, L_ws


# --------------------------------------------------------------- evaluation

def max_bending_strain(env, q):
    """Largest surface bending strain R * kappa along the rod (DER curvature)."""
    dim = env.dim
    X = np.asarray(q[:dim * env.nv]).reshape(-1, dim)
    e = np.diff(X, axis=0)
    t = e / np.linalg.norm(e, axis=1, keepdims=True)
    if dim == 2:
        cross = t[:-1, 0] * t[1:, 1] - t[:-1, 1] * t[1:, 0]
    else:
        cross = np.linalg.norm(np.cross(t[:-1], t[1:]), axis=1)
    dot = np.sum(t[:-1] * t[1:], axis=1)
    kappa = 2.0 * np.abs(cross) / (1.0 + dot) / env.deltaL      # 2 tan(phi/2) / l
    return float(env.R_outer * kappa.max())


def rollout_metrics(env, pipe, q_list, u_list, tip, target, bounds, n_steps):
    """Everything the success criteria and the log need, for one rollout."""
    dim = env.dim
    k = end_control_dim(dim)
    U = np.stack([u.detach().cpu().numpy() for u in u_list])
    tips = np.array([node_position(q, dim, tip) for q in q_list])
    s = np.array([pipe.arclength(p)[0] for p in tips])
    # Energy/force only when the pipe actually acts; penetration either way
    # (with contact off it measures how far the rod goes through the wall).
    acting = pipe in env.obstacles
    pen = e_max = f_max = 0.0
    for q in q_list:
        pen = max(pen, pipe.detect_dlo_pipe_contact(q)["max_penetration"])
        if acting:
            e_max = max(e_max, pipe.compute_contact_energy(q))
            F = pipe.compute_contact_force(q).reshape(-1, dim)
            f_max = max(f_max, float(np.linalg.norm(F, axis=1).max()))
    viol = 0.0
    if bounds is not None:
        lo, hi = bounds
        viol = float(np.maximum(np.maximum(U[:, k:] - hi, lo - U[:, k:]), 0.0).max())
    return {
        "tip_error": float(np.linalg.norm(tips[-1] - target)),
        "tip_final": tips[-1],
        "tip_s_final": float(s[-1]),
        "tip_s_max": float(s.max()),
        "tip_piece_final": pipe.piece_of(tips[-1]),
        "max_penetration": pen,
        "max_contact_energy": e_max,
        "max_contact_force": f_max,
        "max_bending_strain": max(max_bending_strain(env, q) for q in q_list),
        "progress_error": abs(float(s[-1]) - pipe.arclength(target)[0]),
        "workspace_violation": viol,
        "steps_completed": len(q_list),
        "all_converged": len(q_list) == n_steps,
        "gripper_final": U[-1, k:],
    }


def success_criteria(m, pipe, cfg):
    tp = cfg.task.pipe
    crit = {
        "all_steps_converged": m["all_converged"],
        "bend_crossed": (m["tip_s_max"] > pipe.s_bend_end) or not tp.require_bend_crossing,
        "tip_in_second_segment": m["tip_piece_final"] == "second_segment",
        "within_tolerance": m["tip_error"] < float(tp.success_tolerance),
        "no_penetration": m["max_penetration"] <= float(tp.max_penetration),
        "workspace_ok": m["workspace_violation"] <= float(tp.workspace_tolerance),
    }
    return crit, all(crit.values())


# --------------------------------------------------------------- line search

def train_linesearch(env, checkpoint, net, u0, N, dlam, device, cfg, pipe, tip, target, s_ref,
                     w, margin, bounds, run, epochs, lr, pbar):
    """Algorithm 1 on the pipe loss: :class:`neural_control.LineSearch`.

    The trial rule is the trajectory one: every step converged and stable,
    every command inside the workspace, and no snap-through.  Returns the
    per-epoch losses and the line-search records; the policy is left at the
    last accepted weights.
    """
    gm = cfg.train.grad_method
    ls = LineSearch(net, lr, c=float(cfg.train.linesearch.c),
                    tau_min=float(cfg.train.linesearch.tau_min))
    losses, records = [], []

    def evaluate(xi):
        set_parameters(net, xi)
        env.set_state(checkpoint)
        infos = []
        u_list, r_list, q_list, J_list, conv = rollout(env, net, u0, N, dlam, device,
                                                       project=None, infos=infos)
        ok, reason = trial_rule(infos, conv, u_list, net.out_dim, bounds, snap_ok=False)
        tr = {"ok": ok, "reason": reason, "loss": np.inf, "u_list": u_list, "r_list": r_list,
              "q_list": q_list, "J_list": J_list, "conv": conv, "infos": infos}
        if conv.all():
            parts, a_free = task_losses(env, pipe, q_list, tip, target, s_ref, w, margin)
            L_ctrl, L_ws = control_losses(net, u_list, u0, dlam, w, bounds, device)
            parts["control"] = float(L_ctrl.detach())
            parts["workspace"] = float(L_ws.detach())
            tr.update(loss=float(sum(parts.values())), parts=parts, a_free=a_free,
                      L_ctrl=L_ctrl, L_ws=L_ws)
        return tr

    def gradient(tr):
        g_seq = adjoint_gradient(env, tr["u_list"], tr["J_list"], tr["a_free"], cfg.train.jac_reg,
                                 grad_method=gm)
        surrogate = tr["L_ctrl"] + tr["L_ws"] + surrogate_loss(
            tr["u_list"], tr["r_list"], g_seq, dlam, device, grad_method=gm)
        return flat_gradient(net, surrogate)

    def record(ep, tr, rec, fallback):
        m = rollout_metrics(env, pipe, tr["q_list"], tr["u_list"], tip, target, bounds, N)
        n_snap = int(sum(1 for i in tr["infos"] if i.get("snap")))
        losses.append(tr["loss"])
        records.append({"epoch": ep, "loss": tr["loss"], "tau": rec["tau"],
                        "rejected": rec["n_rejected"],
                        "rejected_reasons": [r["reason"] for r in rec["rejected"]],
                        "rollouts": ls.n_rollouts, "snaps": n_snap, "fallback": fallback})
        pbar.set_postfix({"loss": f"{tr['loss']:.4g}", "tip_err": f"{m['tip_error']:.4f}",
                          "tau": "-" if rec["tau"] is None else f"{rec['tau']:g}",
                          "rej": rec["n_rejected"]})
        run.log(epoch=ep, total_loss=tr["loss"],
                **{f"loss_{k}": v for k, v in tr["parts"].items()},
                error=m["tip_error"], tip=m["tip_final"], target=target,
                tip_s=m["tip_s_final"], max_penetration=m["max_penetration"],
                max_contact_energy=m["max_contact_energy"],
                max_contact_force=m["max_contact_force"],
                workspace_violation=m["workspace_violation"],
                tau=rec["tau"], rejected=rec["n_rejected"], rollouts=ls.n_rollouts,
                snaps=n_snap, fallback=fallback,
                converged=True, steps=len(tr["u_list"]), newton_failures=env.n_newton_failures,
                u_final=tr["u_list"][-1].detach().cpu().numpy())
        if ep % 10 == 0 or rec["tau"] is None:
            print(f"[ep {ep:04d}] L={tr['loss']:.5g} "
                  f"({', '.join(f'{k}={v:.3g}' for k, v in tr['parts'].items())})  "
                  f"tip=({','.join(f'{v:.3f}' for v in m['tip_final'])}) err={m['tip_error']:.4f}  "
                  f"tau={'-' if rec['tau'] is None else f'{rec['tau']:g}'} rejected={rec['n_rejected']} "
                  f"rollouts={ls.n_rollouts}")

    xi = get_parameters(net)
    ls.n_rollouts += 1
    cur = evaluate(xi)
    stop = None
    if not cur["ok"]:
        stop = {"epoch": 0, "reason": f"initial rollout rejected: {cur['reason']}"}
        print(f"the initial rollout is rejected ({cur['reason']}): Algorithm 1 needs a feasible "
              "start, no update is made")
        run.log(epoch=0, converged=bool(cur["conv"].all()), steps=len(cur["u_list"]),
                newton_failures=env.n_newton_failures, note=stop["reason"])
        return losses, records, stop, ls
    g = gradient(cur)
    for ep in pbar:
        d, fallback = ls.direction(g)
        acc, rec = ls.search(xi, g, cur["loss"], d, evaluate)
        record(ep, cur, rec, fallback)
        if acc is None:
            stop = {"epoch": ep, "reason": rec["stop"]}
            print(f"[ep {ep:04d}] line search stopped: {rec['stop']}")
            break
        xi = get_parameters(net)
        cur = acc
        g = gradient(cur)
    return losses, records, stop, ls


# --------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config overriding configs/default.yaml")
    parser.add_argument("--grad-method", choices=GRAD_METHODS, default=None,
                        help="adjoint gradient: state (exact) or proxy (frozen tangent); "
                             "default: train.grad_method")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default=None,
                        help="adam (fixed lr, every step accepted) or linesearch (the paper's "
                             "Algorithm 1); default: train.optimizer")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.grad_method is not None:        # the command line wins over the YAML
        cfg["train"]["grad_method"] = args.grad_method
    if args.optimizer is not None:
        cfg["train"]["optimizer"] = args.optimizer
    grad_method = cfg.train.grad_method
    optimizer = cfg.train.optimizer

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    env = build_env(cfg)
    reset_env(env, cfg)
    tip = check_setup(cfg, env)
    pipe = env.pipe
    target = task_target(cfg)
    dim = env.dim
    k_end = end_control_dim(dim)
    N = env.Nsteps
    dlam = 1.0 / N
    tp = cfg.task.pipe
    w = {k: float(v) for k, v in tp.weights.items()}
    margin = float(tp.contact_margin)
    bounds = workspace_bounds(cfg, dim, k_end)

    print("Task: 90-degree pipe insertion, tip reaching  (adjoint gradients)")
    print(f"Adjoint gradient: {grad_method}  "
          f"({'--grad-method' if args.grad_method else 'train.grad_method'})")
    print(f"Optimizer: {optimizer}  "
          f"({'--optimizer' if args.optimizer else 'train.optimizer'})")
    print(f"Environment: {type(env).__name__}  (grasp_mode={env.grasp_mode}, "
          f"gravity={env.gravity.tolist()}, contact={'ON' if pipe in env.obstacles else 'OFF'})")
    print(f"Rod material: {material_summary(cfg)}")

    # The rollouts start from the equilibrium at the initial gripper pose
    # (newton.initial_equilibrium); without it, from the straight rod as before.
    settle_initial(env, cfg)
    checkpoint = env.get_state()
    # Where the tip actually starts: one step at the initial control.
    u0_np = initial_control(cfg)
    _, _, info0 = env.step(u0_np)
    tip0 = node_position(info0["q"], dim, tip)
    q_init = info0["q"].copy()
    env.set_state(checkpoint)

    feas = feasibility_report(cfg, env, pipe, tip0, target)
    if feas["problems"] and bool(tp.require_workspace_feasibility):
        raise SystemExit("Not training: the task is infeasible as configured (see above).")

    s0, s_goal = feas["tip_start_s"], feas["target_s"]
    s_ref = s0 + (np.arange(1, N + 1) / N) * (s_goal - s0)
    reference = np.array([pipe.centerline_point(v) for v in s_ref])

    device = resolve_device(cfg.train.device)
    u0 = torch.tensor(u0_np, dtype=torch.float32, device=device)
    net = BoundaryVelocityNet(hidden=cfg.train.hidden, out_dim=k_end).to(device)
    # Start from "hold still": zero output layer, same architecture. A randomly
    # initialized policy sweeps the gripper tens of cm out of the workspace and
    # levers the rod over the entry lip, where the solves fail and no gradient
    # flows; from rest the first rollout is feasible and the adjoint can move it.
    torch.nn.init.zeros_(net.mlp[-1].weight)
    torch.nn.init.zeros_(net.mlp[-1].bias)
    opt = optim.Adam(net.parameters(), lr=cfg.train.lr)
    # Adam moves u at a roughly constant rate, which carries it straight past a
    # target window that is only millimetres wide; halve the rate whenever the
    # loss goes up instead of down -- at the first rise (patience 0), floored so
    # a run that has overshot can still walk back.
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=0,
                                                 min_lr=1e-2 * float(cfg.train.lr))
    # No projection under the line search: a command outside the workspace
    # rejects the trial instead.
    project = None if optimizer == "linesearch" else workspace_projection(bounds, u0, k_end)
    # The policy that is finally executed: the best converged epoch whose
    # gripper stayed in the workspace.
    best = {"loss": float("inf"), "epoch": None, "state": None}

    # Tagged by the optimizer and gradient, so two runs started in the same
    # second (adam and linesearch side by side) get their own directories.
    run = RunLogger(cfg, env=env, root=_ROOT,
                    tag=f"pipe_{optimizer}" + ("_proxy" if grad_method == "proxy" else ""),
                    enabled=bool((cfg.get("output") or {}).get("run_log", True)))

    losses = []
    ls_records, ls_stop, ls = [], None, None
    pbar = tqdm(range(cfg.train.epochs), desc="Training", dynamic_ncols=True)
    if optimizer == "linesearch":
        losses, ls_records, ls_stop, ls = train_linesearch(
            env, checkpoint, net, u0, N, dlam, device, cfg, pipe, tip, target, s_ref, w,
            margin, bounds, run, cfg.train.epochs, float(cfg.train.lr), pbar)
        if losses:
            best.update(loss=losses[-1], epoch=len(losses) - 1, state=None)
    for ep in (pbar if optimizer == "adam" else ()):
        env.set_state(checkpoint)
        opt.zero_grad(set_to_none=True)
        u_list, r_list, q_list, J_list, conv = rollout(env, net, u0, N, dlam, device,
                                                       project=project)
        if not conv.all():
            # A failed solve has no usable state or Jacobian, and with no step
            # taken the next epoch would repeat the same rollout forever. The
            # converged prefix is still exact, though: step on its path and
            # contact terms (deep contact / jamming is usually what made the
            # solve fail) and on the workspace and control terms, which depend
            # on u alone.
            u_list, r_list, q_list, J_list = u_list[:-1], r_list[:-1], q_list[:-1], J_list[:-1]
            if not u_list:
                run.log(epoch=ep, converged=False, steps=0,
                        newton_failures=env.n_newton_failures,
                        note="first step failed, no gradient step")
                continue
            parts, a_free = task_losses(env, pipe, q_list, tip, target, s_ref, w,
                                        margin, goal=False)
        else:
            parts, a_free = task_losses(env, pipe, q_list, tip, target, s_ref, w, margin)
        g_seq = adjoint_gradient(env, u_list, J_list, a_free, cfg.train.jac_reg,
                                 grad_method=grad_method)

        L_ctrl, L_ws = control_losses(net, u_list, u0, dlam, w, bounds, device)
        surrogate = L_ctrl + L_ws + surrogate_loss(u_list, r_list, g_seq, dlam, device,
                                                   grad_method=grad_method)
        surrogate.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        pre_step_state = copy.deepcopy(net.state_dict())
        opt.step()

        parts["control"] = float(L_ctrl.detach())
        parts["workspace"] = float(L_ws.detach())
        if not conv.all():
            pbar.set_postfix({"status": "Newton failed", "steps": len(u_list)})
            run.log(epoch=ep, converged=False, steps=len(u_list),
                    newton_failures=env.n_newton_failures,
                    **{f"loss_{k}": v for k, v in parts.items()},
                    note="rollout aborted; stepped on the converged prefix (no goal term)")
            continue
        L = sum(parts.values())
        losses.append(L)
        m = rollout_metrics(env, pipe, q_list, u_list, tip, target, bounds, N)
        sched.step(L)
        if L < best["loss"] and m["workspace_violation"] <= float(tp.workspace_tolerance):
            # Weights from before this epoch's update: they produced this rollout.
            best.update(loss=L, epoch=ep, state=pre_step_state)
        pbar.set_postfix({"loss": f"{L:.4g}", "tip_err": f"{m['tip_error']:.4f}",
                          "pen_mm": f"{1e3 * m['max_penetration']:.3f}"})
        run.log(epoch=ep, total_loss=L, **{f"loss_{k}": v for k, v in parts.items()},
                error=m["tip_error"], tip=m["tip_final"], target=target,
                tip_s=m["tip_s_final"], max_penetration=m["max_penetration"],
                max_contact_energy=m["max_contact_energy"],
                max_contact_force=m["max_contact_force"],
                workspace_violation=m["workspace_violation"],
                lr=opt.param_groups[0]["lr"],
                converged=True, steps=len(u_list), newton_failures=env.n_newton_failures,
                u_final=u_list[-1].detach().cpu().numpy())
        if ep % 10 == 0:
            print(f"[ep {ep:04d}] L={L:.5g} ({', '.join(f'{k}={v:.3g}' for k, v in parts.items())})"
                  f"  tip=({','.join(f'{v:.3f}' for v in m['tip_final'])}) "
                  f"err={m['tip_error']:.4f}  pen={1e3 * m['max_penetration']:.3f}mm")

    # Final open-loop trajectory from the trained policy: generate, then execute.
    if best["state"] is not None:
        net.load_state_dict(best["state"])
        print(f"\nExecuting the policy from epoch {best['epoch']} (best loss {best['loss']:.5g})")
    env.set_state(checkpoint)
    with torch.no_grad():
        u_list, _, q_list, _, _ = rollout(env, net, u0, N, dlam, device, project=project)
    m = rollout_metrics(env, pipe, q_list, u_list, tip, target, bounds, N)
    crit, success = success_criteria(m, pipe, cfg)

    print("\n" + "=" * 72)
    print(f"Final tip error      {m['tip_error']:.4f} m   (tolerance {float(tp.success_tolerance):g})"
          f"  -- along the pipe {m['progress_error']:.4f} m, the rest is lateral "
          f"(the wall sets it; up to {pipe.r_eff:.4f} m)")
    print(f"Tip                  ({', '.join(f'{v:.4f}' for v in m['tip_final'])}) in "
          f"{m['tip_piece_final']}, s = {m['tip_s_final']:.3f} (bend ends at {pipe.s_bend_end:.3f})")
    print(f"Max penetration      {1e3 * m['max_penetration']:.3f} mm  "
          f"(limit {1e3 * float(tp.max_penetration):.3f} mm)")
    print(f"Max contact energy   {m['max_contact_energy']:.4g} J")
    print(f"Max contact force    {m['max_contact_force']:.4g} N  (largest per node)")
    print(f"Workspace violation  {m['workspace_violation']:.3g}")
    print(f"Max bending strain   {100 * m['max_bending_strain']:.2f}%  (R * kappa, any step)")
    print(f"Gripper final pose   {np.round(m['gripper_final'], 4).tolist()}")
    for k, v in crit.items():
        print(f"  [{'ok ' if v else 'FAIL'}] {k}")
    print(f"RESULT: {'SUCCESS' if success else 'FAILURE'}")
    print("=" * 72)

    out_dir = str(run.artifact_dir)      # the run's own directory
    visualize_pipe_task(
        q_list, env, target, losses, tip_node=tip, q_init=q_init, reference=reference,
        metrics={"err": f"{m['tip_error'] * 1e3:.1f}mm",
                 "pen": f"{m['max_penetration'] * 1e3:.2f}mm",
                 "ws_viol": f"{m['workspace_violation']:.2g}",
                 "result": "SUCCESS" if success else "FAILURE"},
        save_path=out_dir)
    maybe_save_trajectory_gif(cfg, q_list, env, target, save_path=out_dir,
                              filename='pipe_task_animation.gif',
                              title='90-degree pipe insertion',
                              track_node=tip, track_label='Tip', gripper_node=env.nv - 1)

    extra = {}
    if optimizer == "linesearch":
        extra = {"linesearch": ls_records, "stop": ls_stop,
                 "rollouts_total": ls.n_rollouts if ls is not None else 0,
                 "rejected_total": ls.n_rejected if ls is not None else 0}
    run.finish(
        task="pipe_tip_reaching", mode="single_horizon", grad_method=grad_method,
        optimizer=optimizer,
        final_loss=losses[-1] if losses else None, final_error=m["tip_error"],
        target=target, losses=losses, success=bool(success), best_epoch=best["epoch"],
        criteria={k: bool(v) for k, v in crit.items()},
        max_penetration=m["max_penetration"], workspace_violation=m["workspace_violation"],
        max_contact_force=m["max_contact_force"], tip_final=m["tip_final"],
        progress_error=m["progress_error"], max_bending_strain=m["max_bending_strain"],
        feasibility={k: v for k, v in feas.items() if k != "problems"},
        newton_failures_total=env.n_newton_failures, unstable_steps_total=env.n_unstable,
        snaps_total=env.n_snaps, device=str(device), **extra,
    )


if __name__ == "__main__":
    main()
