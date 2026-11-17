"""Task 1: point-to-point boundary control of a deformable linear object.

The rod's mid node has to reach a target.  The controller is the adjoint
neural control of :mod:`neural_control`: a network maps the continuation
parameter lambda to the right gripper's control rate, the whole command
trajectory is rolled out open loop, and the terminal loss is pushed back
through every step's equilibrium Jacobian by the multi-step adjoint.

Two ways of starting it, both ending in that same training:

  nc       from a random policy -- neural control on its own.
  rrt_nc   global planning first: an RRT on a coarse rod (src/dlo_rrt.py)
           finds a feasible path, the path is resampled to the N control steps
           and the policy is fitted to it, and training then refines it.  The
           plan executed open loop on the fine rod is recorded as well, which
           is the "RRT only" result.  The idea is Yu et al.'s (reference/
           Generalizable whole-body global manipulation ...): plan globally on
           a coarse model, refine locally on the real one.

With ``workspace.enabled`` in the config the right gripper is confined to that
box in both stages: the planner only grows the tree inside it, and the neural
control projects every command onto it before executing it (the penalty acts
on the unprojected command, so the policy is still pulled back in).

Receding horizon (``rhc.enabled``) splits the horizon into segments that are
optimized and committed one at a time; it is the scheme of the original
submission code.  Single-horizon training is the default and the one the
comparison uses.

``--grad-method`` (default ``train.grad_method``, ``state``) picks the adjoint
gradient: ``state`` is exact, ``proxy`` the frozen-tangent rate formulation
(see src/implicit_grad_tools.py).  ``--optimizer`` (default
``train.optimizer``, ``adam``) picks the update: ``adam`` is the fixed-rate
Adam with gradient clipping, ``linesearch`` the paper's Algorithm 1 -- the
Adam direction with a backtracking step length on full rollouts, where a
trial is rejected on a Newton failure, an unstable state or a command outside
the workspace, and a snap-through is admitted when it lowers the loss (a
terminal objective).  Single-horizon training only.

Every rollout starts from the equilibrium at the initial control
(``newton.initial_equilibrium``), solved before the checkpoint is taken.

Usage
-----
    python scripts/train_point_to_point.py                      # the config's target
    python scripts/train_point_to_point.py --grad-method proxy
    python scripts/train_point_to_point.py --optimizer linesearch
    python scripts/train_point_to_point.py --target T1 --method rrt_nc --out DIR
    python scripts/train_point_to_point.py report --out DIR
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

# Must run before numpy is imported: OpenBLAS goes multithreaded at about
# n = 100, and for the small dense solves here (94x94 in 2D, 141x141 in 3D)
# the threading overhead costs ~2000x. See src/blas_threads.py.
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

from compute_audit import from_config as audit_from_config
from config import load_config
from dlo_rrt import (RRTPlanner, PlannerParams, path_length, resample_path,
                     min_stiffness_eig)
from env_factory import (
    build_env, reset_env, settle_initial, env_dim, env_twist, control_dim,
    end_control_dim, initial_control, task_target, material_summary,
    node_tracking_loss, dloss_dq_free_node, node_position,
)
from neural_control import (
    BoundaryVelocityNet, reinit_net_, resolve_device, segment_lengths,
    rollout, execute_controls, adjoint_gradient, surrogate_loss, apply_update,
    fit_policy, workspace_bounds, workspace_projection, workspace_penalty,
    set_audit, GRAD_METHODS, OPTIMIZERS, LineSearch, trial_rule, flat_gradient,
    get_parameters, set_parameters,
)
from runlog import RunLogger
from utils import visualize_results, maybe_save_trajectory_gif

#: Redundancy audit for the inner-loop kernels.  Disabled (and unpatched)
#: unless `audit.enabled` is set in the config or --audit is passed.
AUDIT = set_audit(audit_from_config({}))


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


def final_midpoint_loss(q, dim, mid_node, target):
    """Squared distance of the mid node to the target. Returns (loss, position)."""
    return node_tracking_loss(q, dim, mid_node, target)


def rollout_record(env, q_list, u_list, dim):
    """Mid-node path (rest state first) and right-end controls of a rollout."""
    mid0 = node_position(env.nodes.reshape(-1), dim, env.mid_node)
    mids = np.array([mid0] + [node_position(q, dim, env.mid_node) for q in q_list])
    U = np.array([u.detach().cpu().numpy() if torch.is_tensor(u) else u for u in u_list])
    return mids, U


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


# --------------------------------------------------------------- training

def p2p_loss_terms(env, q_list, target, track_ref=None, track_weight=0.0):
    """Terminal loss, the full loss, the final mid node, and ``dL/dq_free`` per step.

    The full loss adds the optional plan-tracking term
    ``track_weight * mean_k |mid_k - track_ref_k|^2``; the recorded error
    stays the terminal one.
    """
    dim, free_index = env.dim, env.free_index
    q_final = q_list[-1]
    L_np, mid_pos = final_midpoint_loss(q_final, dim, env.mid_node, target)
    a_free_per_step = np.zeros((len(q_list), len(free_index)), dtype=float)
    a_free_per_step[-1] = dloss_dq_free_node(q_final, free_index, dim, env.mid_node, target)
    L_total = float(L_np)
    if track_weight > 0.0:
        w = track_weight / len(q_list)
        for kk, qk in enumerate(q_list):
            a_free_per_step[kk] += w * dloss_dq_free_node(qk, free_index, dim, env.mid_node,
                                                          track_ref[kk])
            L_total += w * node_tracking_loss(qk, dim, env.mid_node, track_ref[kk])[0]
    return L_np, L_total, mid_pos, a_free_per_step


def count_snaps(infos):
    return int(sum(1 for i in infos if i.get("snap")))


def train_single_horizon_linesearch(env, checkpoint, net, u0, target, cfg, epochs, device, lr,
                                    run=None, log_every=10, tag="", track_ref=None,
                                    track_weight=0.0, bounds=None):
    """Algorithm 1 over the whole horizon: :class:`neural_control.LineSearch`.

    Same inputs and outputs as :func:`train_single_horizon`, with the
    optimizer replaced.  Every round rolls out trials ``xi + tau d`` until one
    satisfies the Armijo condition and the point-to-point rule (no Newton
    failure, no unstable state, every command inside the workspace; a
    snap-through is admitted since only the final shape counts), reuses that
    rollout for the next gradient, and stops when no step is accepted.  There
    is no projection onto the workspace in this mode and no penalty is needed:
    a command outside it rejects the trial.  ``hist`` additionally records
    the accepted ``tau``, the rejected trials and their reasons, the
    cumulative number of rollouts and the snaps of each accepted rollout.
    """
    dim = env.dim
    N = env.Nsteps
    dlam = 1.0 / N
    n_end = net.out_dim
    gm = cfg.train.grad_method
    ls = LineSearch(net, lr, c=float(cfg.train.linesearch.c),
                    tau_min=float(cfg.train.linesearch.tau_min))
    hist = {"loss": [], "error": [], "converged": [], "time": [], "mid_paths": [], "u_paths": [],
            "ws_loss": [], "tau": [], "rejected": [], "rejected_reasons": [], "rollouts": [],
            "snaps": [], "fallback": [], "stop": None}
    t_start = time.time()

    def evaluate(xi):
        set_parameters(net, xi)
        env.set_state(checkpoint)
        infos = []
        u_list, r_list, q_list, J_list, conv = rollout(env, net, u0, N, dlam, device,
                                                       project=None, infos=infos)
        ok, reason = trial_rule(infos, conv, u_list, n_end, bounds, snap_ok=True)
        tr = {"ok": ok, "reason": reason, "loss": np.inf, "u_list": u_list, "r_list": r_list,
              "q_list": q_list, "J_list": J_list, "conv": conv, "infos": infos}
        if conv.all():
            L_np, L_total, mid_pos, a = p2p_loss_terms(env, q_list, target, track_ref, track_weight)
            tr.update(loss=L_total, task_loss=float(L_np), mid=mid_pos, a_free=a)
        return tr

    def gradient(tr):
        g_seq = adjoint_gradient(env, tr["u_list"], tr["J_list"], tr["a_free"], cfg.train.jac_reg,
                                 grad_method=gm)
        return flat_gradient(net, surrogate_loss(tr["u_list"], tr["r_list"], g_seq, dlam, device,
                                                 grad_method=gm))

    def record(ep, tr, rec, fallback):
        mids, u_seq_np = rollout_record(env, tr["q_list"], tr["u_list"], dim)
        hist["loss"].append(tr["task_loss"])
        hist["error"].append(tr["task_loss"] ** 0.5)
        hist["converged"].append(True)
        hist["time"].append(time.time() - t_start)
        hist["mid_paths"].append(mids)
        hist["u_paths"].append(u_seq_np)
        hist["ws_loss"].append(0.0 if bounds is not None else np.nan)
        hist["tau"].append(rec["tau"])
        hist["rejected"].append(rec["n_rejected"])
        hist["rejected_reasons"].append([r["reason"] for r in rec["rejected"]])
        hist["rollouts"].append(ls.n_rollouts)
        hist["snaps"].append(count_snaps(tr["infos"]))
        hist["fallback"].append(fallback)
        if run is not None:
            run.log(epoch=ep, loss=tr["task_loss"], error=tr["task_loss"] ** 0.5,
                    total_loss=tr["loss"], midpoint=tr["mid"], target=target, converged=True,
                    steps=len(tr["u_list"]), newton_failures=env.n_newton_failures,
                    tau=rec["tau"], rejected=rec["n_rejected"], rollouts=ls.n_rollouts,
                    snaps=count_snaps(tr["infos"]), fallback=fallback,
                    u_final=tr["u_list"][-1].detach().cpu().numpy())
        if ep % log_every == 0 or ep == epochs - 1 or rec["tau"] is None:
            print(f"[{tag} ep {ep:03d}] err={tr['task_loss'] ** 0.5 * 1e3:7.2f} mm  "
                  f"tau={'-' if rec['tau'] is None else f'{rec['tau']:g}'}  "
                  f"rejected={rec['n_rejected']}  rollouts={ls.n_rollouts}  "
                  f"snaps={count_snaps(tr['infos'])}  t={time.time() - t_start:7.1f}s", flush=True)

    xi = get_parameters(net)
    ls.n_rollouts += 1
    cur = evaluate(xi)
    if not cur["ok"]:
        hist["stop"] = {"epoch": 0, "reason": f"initial rollout rejected: {cur['reason']}"}
        print(f"[{tag}] the initial rollout is rejected ({cur['reason']}): Algorithm 1 needs a "
              "feasible start, no update is made", flush=True)
        if run is not None:
            run.log(epoch=0, converged=bool(cur["conv"].all()), steps=len(cur["u_list"]),
                    newton_failures=env.n_newton_failures, note=hist["stop"]["reason"])
        return (hist, cur["u_list"], cur["q_list"], cur["J_list"], bool(cur["conv"].all()),
                time.time() - t_start)
    g = gradient(cur)
    for ep in range(epochs):
        AUDIT.set_context(epoch=ep)
        d, fallback = ls.direction(g)
        acc, rec = ls.search(xi, g, cur["loss"], d, evaluate)
        record(ep, cur, rec, fallback)
        if acc is None:
            hist["stop"] = {"epoch": ep, "reason": rec["stop"]}
            print(f"[{tag} ep {ep:03d}] line search stopped: {rec['stop']}", flush=True)
            break
        xi = get_parameters(net)          # the accepted trial's weights, as stored
        cur = acc
        g = gradient(cur)
    # The policy after the last accepted update is the one `cur` was rolled out with.
    return (hist, cur["u_list"], cur["q_list"], cur["J_list"], bool(cur["conv"].all()),
            time.time() - t_start)


def train_single_horizon(env, checkpoint, net, u0, target, cfg, epochs, device, lr,
                         run=None, log_every=10, tag="", track_ref=None, track_weight=0.0,
                         bounds=None, ws_weight=0.0):
    """One continuation over the whole horizon, trained for ``epochs`` updates.

    ``checkpoint`` is the state every rollout starts from (taken right after
    reset_env).  ``track_ref`` (N, dim) with ``track_weight > 0`` adds the
    paper's path-tracking term, ``track_weight * mean_k |mid_k - track_ref_k|^2``,
    to the terminal loss: the local stage then follows the global plan instead
    of only its end.  The recorded error stays the terminal one.

    Workspace ``bounds`` are handled as in train_pipe.py: every command is
    projected onto the workspace before it is executed, and
    ``ws_weight * mean_k |violation_k|^2`` of the *unprojected* command joins
    the surrogate.

    This is the ``adam`` optimizer; ``train.optimizer: linesearch`` goes to
    :func:`train_single_horizon_linesearch` instead.
    """
    if cfg.train.optimizer == "linesearch":
        return train_single_horizon_linesearch(
            env, checkpoint, net, u0, target, cfg, epochs, device, lr, run=run,
            log_every=log_every, tag=tag, track_ref=track_ref, track_weight=track_weight,
            bounds=bounds)
    dim = env.dim
    opt = optim.Adam(net.parameters(), lr=lr)
    N = env.Nsteps
    dlam = 1.0 / N
    n_end = net.out_dim
    project = workspace_projection(bounds, u0, n_end)
    hist = {"loss": [], "error": [], "converged": [], "time": [], "mid_paths": [], "u_paths": [],
            "ws_loss": []}
    t_start = time.time()
    for ep in range(epochs):
        AUDIT.set_context(epoch=ep)
        AUDIT.checkpoint_report(None if run is None else run.artifact_dir, ep)
        env.set_state(checkpoint)
        opt.zero_grad(set_to_none=True)

        u_list, r_list, q_list, J_list, conv = rollout(env, net, u0, N, dlam, device,
                                                       project=project)
        if not conv.all():
            # A failed rollout gets no update, so the next epoch replays the very
            # same rollout and fails at the same step, and so on to the end:
            # every later epoch is this one again.  Record them as failed without
            # re-running them (minutes each).
            hist["stuck"] = {"epoch": ep, "step": len(u_list)}
            print(f"[{tag} ep {ep:03d}] Newton failed at step {len(u_list)}/{N}; no update "
                  f"follows, so all {epochs - ep} remaining epochs repeat it -- recorded as failed",
                  flush=True)
            if run is not None:
                run.log(epoch=ep, converged=False, steps=len(u_list),
                        newton_failures=env.n_newton_failures,
                        note="rollout aborted, no gradient step")
            for _ in range(ep, epochs):
                for key, v in (("loss", np.nan), ("error", np.nan), ("converged", False),
                               ("time", time.time() - t_start), ("ws_loss", np.nan)):
                    hist[key].append(v)
                hist["mid_paths"].append(np.full((N + 1, dim), np.nan))
                hist["u_paths"].append(np.full((N, len(u0)), np.nan))
            break

        L_np, _, mid_pos, a_free_per_step = p2p_loss_terms(env, q_list, target, track_ref,
                                                           track_weight)

        g_seq = adjoint_gradient(env, u_list, J_list, a_free_per_step, cfg.train.jac_reg,
                                 grad_method=cfg.train.grad_method)
        surrogate = surrogate_loss(u_list, r_list, g_seq, dlam, device,
                                   grad_method=cfg.train.grad_method)
        L_ws_t = workspace_penalty(net, u0, u_list, n_end, bounds, ws_weight, device, dlam)
        surrogate = surrogate + L_ws_t
        L_ws = float(L_ws_t.detach()) if bounds is not None else np.nan
        apply_update(net, opt, surrogate)

        mids, u_seq_np = rollout_record(env, q_list, u_list, dim)
        hist["loss"].append(float(L_np))
        hist["error"].append(float(L_np ** 0.5))
        hist["converged"].append(True)
        hist["time"].append(time.time() - t_start)
        hist["mid_paths"].append(mids)
        hist["u_paths"].append(u_seq_np)
        hist["ws_loss"].append(L_ws)
        if run is not None:
            run.log(epoch=ep, loss=float(L_np), error=float(L_np ** 0.5),
                    midpoint=mid_pos, target=target, converged=True, steps=len(u_list),
                    newton_failures=env.n_newton_failures,
                    u_final=u_list[-1].detach().cpu().numpy())
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"[{tag} ep {ep:03d}] err={L_np ** 0.5 * 1e3:7.2f} mm  "
                  + ("" if bounds is None else f"ws_pen={L_ws:.2e}  ")
                  + f"t={time.time() - t_start:7.1f}s", flush=True)

    # Final policy (after the last update).
    env.set_state(checkpoint)
    with torch.no_grad():
        u_list, _, q_list, J_list, conv = rollout(env, net, u0, N, dlam, device,
                                                  project=project)
    return hist, u_list, q_list, J_list, bool(conv.all()), time.time() - t_start


def train_rhc(env, checkpoint, net, u0, target, cfg, device, lr, run=None, bounds=None,
              ws_weight=0.0):
    """Receding horizon: optimize one segment, commit it, warm-start the next.

    Each segment gets its own (optionally reinitialized) policy, is optimized
    for ``train.iterations_per_segment`` gradient steps against the same
    overall target, and the optimized rollout is then committed -- the
    simulator state and the control carry over to the next segment.  Mirrors
    the MPC loop of the original adjoint_RHC_any_node.py.
    """
    N = env.Nsteps
    dim = env.dim
    seg_lens = segment_lengths(N, cfg.rhc.num_segments)
    n_end = net.out_dim
    project = workspace_projection(bounds, u0, n_end)
    u_current = u0.clone()
    opt = optim.Adam(net.parameters(), lr=lr)
    losses, q_traj, q_last = [], [], None
    t_start = time.time()

    for seg_idx, T_seg in enumerate(seg_lens):
        if cfg.rhc.reinit_policy_each_segment:
            reinit_net_(net)
            opt = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
        AUDIT.set_context(seg=seg_idx)
        dlam = 1.0 / T_seg
        print(f"\n{'=' * 70}\nRHC segment {seg_idx + 1}/{len(seg_lens)}  (T={T_seg})\n{'=' * 70}")

        best = float("inf")
        for it in range(cfg.train.iterations_per_segment):
            AUDIT.set_context(epoch=it)
            env.set_state(checkpoint)
            opt.zero_grad(set_to_none=True)
            u_list, r_list, q_list, J_list, conv = rollout(env, net, u_current, T_seg, dlam,
                                                           device, project=project)
            if not conv.all():
                continue
            L, _ = final_midpoint_loss(q_list[-1], dim, env.mid_node, target)
            a_free = np.zeros((len(q_list), len(env.free_index)), dtype=float)
            a_free[-1] = dloss_dq_free_node(q_list[-1], env.free_index, dim, env.mid_node, target)
            g_seq = adjoint_gradient(env, u_list, J_list, a_free, cfg.train.jac_reg,
                                     grad_method=cfg.train.grad_method)
            surrogate = surrogate_loss(u_list, r_list, g_seq, dlam, device,
                                       grad_method=cfg.train.grad_method)
            surrogate = surrogate + workspace_penalty(net, u_current, u_list, n_end, bounds,
                                                      ws_weight, device, dlam)
            apply_update(net, opt, surrogate)
            best = min(best, L)

        # Commit: advance the persistent simulator and control state.
        env.set_state(checkpoint)
        with torch.no_grad():
            u_list, _, q_list, _, conv = rollout(env, net, u_current, T_seg, dlam, device,
                                                 project=project)
        checkpoint = env.get_state()
        u_current = u_list[-1].detach().clone()
        q_last = q_list[-1]
        q_traj.extend(q_list)
        losses.append(best)
        mid_pos = node_position(q_last, dim, env.mid_node)
        print(f"[segment {seg_idx}] best_loss={best:.6e}  "
              f"midpoint=({','.join(f'{v:.3f}' for v in mid_pos)})")
        if run is not None:
            run.log(segment=seg_idx, horizon=int(T_seg), best_loss=float(best),
                    error=float(best ** 0.5), midpoint=mid_pos, target=target,
                    converged=bool(conv.all()), newton_failures=env.n_newton_failures,
                    u_final=u_current.detach().cpu().numpy())
    return losses, q_traj, q_last, time.time() - t_start


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
    if args.audit:
        cfg.setdefault("audit", {})["enabled"] = True
    if args.audit_kernels:
        cfg.setdefault("audit", {})["trace_kernels"] = True
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dim, twist = env_dim(cfg), env_twist(cfg)
    k = end_control_dim(dim, twist)
    target = resolve_target(args.target, cfg)
    env = build_env(cfg)
    reset_env(env, cfg)

    global AUDIT
    AUDIT = set_audit(audit_from_config(cfg).install(env))

    # The rollouts start from the equilibrium at u0, not from the straight rod.
    settle_initial(env, cfg)
    checkpoint = env.get_state()
    N = env.Nsteps
    device = resolve_device(cfg.train.device if args.device is None else args.device)
    u0_np = initial_control(cfg)
    u0 = torch.tensor(u0_np, dtype=torch.float32, device=device)
    net = BoundaryVelocityNet(hidden=cfg.train.hidden, out_dim=k).to(device)
    bounds = workspace_bounds(cfg, dim, k)
    if bounds is not None and (np.any(u0_np[k:] < bounds[0]) or np.any(u0_np[k:] > bounds[1])):
        raise SystemExit(f"the initial gripper pose {u0_np[k:]} is outside the workspace")

    lr = cfg.train.lr if args.lr is None else args.lr
    epochs = cfg.train.epochs if args.epochs is None else args.epochs
    name = (f"{args.target}_{args.method}_s{args.seed}"
            + (f"_r{args.rrt_seed}" + (f"k{args.rrt_k}" if args.rrt_k > 1 else "")
               if args.method == "rrt_nc" else "")
            + ("" if args.lr is None else f"_lr{args.lr:g}")
            + (f"_trk{args.track_weight:g}" if args.track_weight > 0 else "")
            + ("_proxy" if grad_method == "proxy" else "")
            + ("_ls" if optimizer == "linesearch" else ""))

    # RunLogger owns the TensorBoard writer: every result goes through
    # run.log(), so the JSON record and the scalars cannot drift apart.
    run = RunLogger(cfg, env=env, root=_ROOT, tag=name,
                    enabled=bool((cfg.get("output") or {}).get("run_log", True)))
    out = Path(args.out) if args.out else Path(run.artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Task: point-to-point  (adjoint neural control, method {args.method})")
    print(f"Adjoint gradient: {grad_method}  "
          f"({'--grad-method' if args.grad_method else 'train.grad_method'})")
    print(f"Optimizer: {optimizer}  "
          f"({'--optimizer' if args.optimizer else 'train.optimizer'})"
          + (f", Armijo c={float(cfg.train.linesearch.c):g}, "
             f"tau_min={float(cfg.train.linesearch.tau_min):g}" if optimizer == "linesearch" else ""))
    print(f"Environment: {type(env).__name__}  (dim={dim}, twist={twist}, "
          f"control={control_dim(dim, twist)}D)")
    print(f"Rod material: {material_summary(cfg)}")
    budget = (f"receding horizon, {cfg.rhc.num_segments} segments x "
              f"{cfg.train.iterations_per_segment} updates" if cfg.rhc.enabled
              else f"single horizon, {epochs} epochs")
    print(f"Target: {np.round(target, 4)}   device: {device}   lr {lr:g}")
    print(f"Horizon: {N} steps, {budget}")
    print(f"Workspace: {'off' if bounds is None else 'on, ' + ('hard constraint (trials outside it are rejected)' if optimizer == 'linesearch' else 'penalty ' + str(args.ws_weight))}")
    nw = cfg.newton
    print(f"Rollout physics: initial equilibrium {'on' if nw.get('initial_equilibrium') else 'off'}, "
          f"continuation {'on (max ' + str(nw.get('max_substeps')) + ' substeps)' if nw.get('continuation') else 'off'}, "
          f"stability check {'on' if nw.get('stability_check') else 'off'}, "
          f"snap rule {'on' if nw.get('snap_rule') else 'off'}")
    if AUDIT.enabled:
        print("Compute audit: ON  (instrumented kernels are wrapped; expect some overhead)")

    arrays, summary = {}, {"target_name": args.target, "target": target.tolist(),
                           "method": args.method, "grad_method": grad_method,
                           "optimizer": optimizer,
                           "seed": args.seed, "epochs": epochs,
                           "nv": int(env.nv), "N": int(N), "lr": float(lr),
                           "lr_default": float(cfg.train.lr),
                           "track_weight": float(args.track_weight),
                           "rrt_seed": args.rrt_seed if args.method == "rrt_nc" else None,
                           "rrt_k": args.rrt_k if args.method == "rrt_nc" else None,
                           "workspace": None if bounds is None else
                           {"lo": bounds[0].tolist(), "hi": bounds[1].tolist(),
                            "weight": float(args.ws_weight)}}

    if args.method == "rrt_nc":
        # ---- global: RRT on the coarse rod, K seeds, keep the best path
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
            run.finish(status="plan_failed", task="point_to_point", target=target)
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

        # ---- warm start
        t_fit = time.time()
        summary["fit"] = fit_policy(net, u0, pa["U_plan"], k, dim=dim, iters=args.fit_iters)
        summary["fit"]["time"] = time.time() - t_fit
        print(f"[{name}] policy fit: {summary['fit']}", flush=True)

    # ---- local: adjoint neural control (identical for both methods)
    track_ref = None
    if args.track_weight > 0:
        if args.method != "rrt_nc":
            raise ValueError("--track-weight needs a plan to track (--method rrt_nc)")
        track_ref = arrays["plan_mid_trace"][1:]           # the plan's mid path, step by step
    env.n_newton_failures = 0

    if cfg.rhc.enabled:
        losses, q_traj, q_last, t_train = train_rhc(env, checkpoint, net, u0, target, cfg,
                                                    device, lr, run=run, bounds=bounds,
                                                    ws_weight=args.ws_weight)
        final_loss = losses[-1] if losses else None
        summary["segment_loss"] = losses
        summary["final_converged"] = q_last is not None
        q_list = q_traj
        results = {'final_q': q_last, 'all_losses': losses,
                   'final_error': float(final_loss ** 0.5) if final_loss is not None else float('nan')}
    else:
        hist, u_list, q_list, J_list, conv, t_train = train_single_horizon(
            env, checkpoint, net, u0, target, cfg, epochs, device, lr, run=run, tag=name,
            track_ref=track_ref, track_weight=args.track_weight,
            bounds=bounds, ws_weight=args.ws_weight)
        losses = hist["loss"]
        summary["epoch_error"] = hist["error"]
        summary["epoch_time"] = hist["time"]
        if bounds is not None:
            summary["epoch_ws_loss"] = hist["ws_loss"]
        summary["stuck"] = hist.get("stuck")
        if optimizer == "linesearch":
            for key in ("tau", "rejected", "rejected_reasons", "rollouts", "snaps", "fallback"):
                summary[f"epoch_{key}"] = hist[key]
            summary["stop"] = hist.get("stop")
            summary["rollouts_total"] = int(hist["rollouts"][-1]) if hist["rollouts"] else 0
            summary["rejected_total"] = int(sum(hist["rejected"]))
        summary["final_converged"] = bool(conv)
        if conv:
            U_fin = np.array([u.detach().cpu().numpy() for u in u_list])
            summary["final"] = path_metrics(env, u0_np, U_fin, q_list, target, dim, k, J_list,
                                            bounds)
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
                summary["start"] = path_metrics(env, u0_np, hist["u_paths"][first], q0, target,
                                                dim, k, J0, bounds)
        arrays["epoch_mid_paths"] = np.array(hist["mid_paths"], dtype=np.float32)
        arrays["epoch_u_paths"] = np.array(hist["u_paths"], dtype=np.float32)
        final_loss = next((l for l in reversed(losses) if np.isfinite(l)), None)
        results = {'final_q': q_list[-1] if q_list else None,
                   'final_error': float(final_loss ** 0.5) if final_loss is not None else float('nan'),
                   'all_losses': losses}

    summary["train_time"] = t_train
    summary["newton_failures_total"] = int(env.n_newton_failures)
    summary["unstable_steps_total"] = int(env.n_unstable)
    summary["snaps_total"] = int(env.n_snaps)
    summary["status"] = "done"
    np.savez_compressed(out / f"{name}.npz", **arrays)
    json.dump(summary, open(out / f"{name}.json", "w"), indent=1, default=float)

    # Before any plotting, so the audit's wall time is the training's.
    if AUDIT.enabled:
        AUDIT.restore()
        print()
        print(AUDIT.save(run.artifact_dir,
                         save_records=bool(cfg.audit.get("save_records", True))))
        print("Compute audit written to: "
              + os.path.join(run.artifact_dir, 'compute_audit.txt'))

    if results['final_q'] is not None:
        visualize_results(results, env, target, save_path=str(run.artifact_dir))
        maybe_save_trajectory_gif(cfg, q_list, env, target, save_path=str(run.artifact_dir),
                                  title='DLO point-to-point')
    if losses:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(len(losses)), losses, linewidth=1.5, color="steelblue")
        ax.set_xlabel("RHC segment" if cfg.rhc.enabled else "Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curve")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(run.artifact_dir, 'loss_curve.png'), dpi=150)
        plt.close(fig)

    run.finish(task="point_to_point",
               mode="rhc" if cfg.rhc.enabled else "single_horizon",
               method=args.method, grad_method=grad_method, optimizer=optimizer,
               unstable_steps_total=env.n_unstable, snaps_total=env.n_snaps,
               final_loss=None if final_loss is None else float(final_loss),
               final_error=results['final_error'],
               target=target, losses=[float(l) for l in losses],
               best_loss=float(np.nanmin(losses)) if losses else None,
               epochs_recorded=len(losses),
               newton_failures_total=env.n_newton_failures,
               device=str(device))
    fe = summary.get("final", {}).get("final_error", results['final_error'])
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
    """Failed-epoch count, with where the run got stuck (see train_single_horizon)."""
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
    """Every point-to-point job record in ``out``.

    A job is recognised by its own fields, so a directory that also holds a
    run's own files (summary.json, compute_audit.json, another task's job) is
    read without tripping over them.
    """
    runs = []
    for js in sorted(out.glob("*.json")):
        try:
            s = json.load(open(js))
        except (ValueError, OSError):
            continue
        if not (isinstance(s, dict) and {"target_name", "method"} <= set(s)):
            continue
        if s.get("status") == "done" and "epoch_error" not in s:
            continue      # an RHC run: per-segment, so there is no epoch curve
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


def report(out, config=None):
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
    cfg = load_config(config)      # the rod the jobs ran on, for the replays
    env = build_env(cfg)
    reset_env(env, cfg)
    settle_initial(env, cfg)
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
    cmds = ("train", "report")
    argv = sys.argv[1:]
    cmd = argv.pop(0) if argv and argv[0] in cmds else "train"

    ap = argparse.ArgumentParser(description=f"point-to-point neural control ({cmd})")
    ap.add_argument("--config", default=None, help="YAML config over configs/default.yaml")
    ap.add_argument("--out", default=None,
                    help="directory the job's json/npz go to (default: the run directory); "
                         "for `report`, the directory holding them")
    if cmd == "train":
        ap.add_argument("--method", choices=("nc", "rrt_nc"), default="nc")
        ap.add_argument("--grad-method", choices=GRAD_METHODS, default=None,
                        help="adjoint gradient: state (exact) or proxy (frozen tangent); "
                             "default: train.grad_method")
        ap.add_argument("--optimizer", choices=OPTIMIZERS, default=None,
                        help="adam (fixed lr, every step accepted) or linesearch (the paper's "
                             "Algorithm 1); default: train.optimizer")
        ap.add_argument("--target", default="T0",
                        help="a name from TARGETS, or 'x,y,z' (T0 = the config's target)")
        ap.add_argument("--seed", type=int, default=0, help="policy seed")
        ap.add_argument("--epochs", type=int, default=None, help="default: train.epochs")
        ap.add_argument("--lr", type=float, default=None,
                        help="default: train.lr; fine-tuning a plan wants ~1e-4")
        ap.add_argument("--device", default=None, help="default: train.device")
        ap.add_argument("--ws-weight", type=float, default=100.0,
                        help="weight of the workspace-violation penalty")
        ap.add_argument("--track-weight", type=float, default=0.0,
                        help="weight of the term tracking the planned mid path (rrt_nc only)")
        ap.add_argument("--rrt-seed", type=int, default=0, help="first planner seed")
        ap.add_argument("--rrt-k", type=int, default=1, help="how many planner seeds to try")
        ap.add_argument("--rrt-select", choices=("length", "open_loop"), default="length",
                        help="pick the plan by path length, or by its open-loop error on the "
                             "fine rod (which catches shortcuts that change branch)")
        ap.add_argument("--plan-nv", type=int, default=11, help="nodes of the planner's rod")
        ap.add_argument("--plan-tol", type=float, default=0.01, help="[m] planner goal tolerance")
        ap.add_argument("--plan-time", type=float, default=600.0, help="[s] per planner seed")
        ap.add_argument("--plan-newton-iter", type=int, default=200)
        ap.add_argument("--fit-iters", type=int, default=3000,
                        help="Adam iterations fitting the policy to the plan")
        ap.add_argument("--audit", action="store_true",
                        help="audit the inner-loop kernels for repeated computation "
                             "(src/compute_audit.py)")
        ap.add_argument("--audit-kernels", action="store_true",
                        help="with --audit, also instrument the per-stencil kernels")
    args = ap.parse_args(argv)

    if cmd == "report":
        if args.out is None:
            raise SystemExit("report needs --out, the directory holding the job files")
        report(Path(args.out), args.config)
    else:
        cmd_train(args)


if __name__ == "__main__":
    main()
