import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

# Must run before numpy is imported: OpenBLAS goes multithreaded at about
# n = 100, and for the small dense solves here (94x94 in 2D, 141x141 in 3D)
# the threading overhead costs ~2000x. See src/blas_threads.py.
from blas_threads import limit_blas_threads
limit_blas_threads()

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from implicit_grad_tools import multistep_adjoint_gradient
from utils import visualize_results, maybe_save_trajectory_gif
from runlog import RunLogger
from config import load_config
from env_factory import (
    build_env,
    reset_env,
    env_dim,
    env_twist,
    control_dim,
    end_control_dim,
    initial_control,
    task_target,
    material_summary,
    node_tracking_loss,
    dloss_dq_free_node,
)


class BoundaryVelocityNet(nn.Module):
    """Maps the continuation parameter lambda to the right end's control rate.

    ``out_dim`` is 3 in 2D ((x, y, theta) rate) and 5 in 3D
    ((x, y, z, theta, phi) rate); the left end is held fixed.
    """

    def __init__(self, hidden=64, out_dim=3):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            # for right end
            nn.Linear(hidden, out_dim),
        )

    def forward(self, lam):
        return self.mlp(lam)


def reinit_net_(net: nn.Module):
    """Reinitialize network weights (used at the start of every RHC segment)."""
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
            nn.init.zeros_(m.bias)
    net.apply(_init)


def final_midpoint_loss(q, dim, mid_node, target):
    """Squared distance of the mid node to the target. Returns (loss, position)."""
    return node_tracking_loss(q, dim, mid_node, target)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def segment_lengths(N: int, num_segments: int):
    """Split N steps into num_segments chunks, as evenly as possible."""
    base, rem = divmod(N, num_segments)
    return [base + 1 if i < rem else base for i in range(num_segments)]


def rollout_segment(env, net, u_start, T, dlam, device, jac_reg):
    """
    Roll out `T` steps of the (right-end) policy starting from the current
    env state, with the continuation parameter lambda reset to [0, 1] over
    this segment (matches the original adjoint_RHC_*.py convention).

    Returns u_list (torch, grad-tracking), q_list, J_list, converged (bool).
    """
    u_list = []
    q_list = []
    J_list = []
    converged = True

    u = u_start.clone()
    n_left = net.out_dim          # left end is held fixed -> zero rate
    for k in range(T):
        lam = torch.tensor([[k / T]], dtype=torch.float32, device=device)
        u_dot_full = torch.cat([torch.zeros(n_left, device=device), net(lam).squeeze(0)])
        u = u + dlam * u_dot_full
        u_list.append(u)

        u_np = u.detach().cpu().numpy()
        obs, done, info = env.step(u_np, use_inertia=False)
        q_list.append(info["q"])
        J_list.append(info["J"])

        if not info["converged"]:
            converged = False
            break

    return u_list, q_list, J_list, converged


def optimize_segment(env, net, opt, u_start, T, dlam, device, target_xy, cfg,
                      n_iterations, checkpoint):
    """Run `n_iterations` exact-gradient updates for one horizon/segment,
    always rolling out from the same `checkpoint` state. Returns the final
    (post-optimization) rollout so the caller can commit it and advance."""
    best_loss = float("inf")

    for it in range(n_iterations):
        env.set_state(checkpoint)
        opt.zero_grad(set_to_none=True)

        u_list, q_list, J_list, converged = rollout_segment(env, net, u_start, T, dlam, device, cfg.train.jac_reg)
        if not converged:
            continue

        q_final = q_list[-1]
        free_index = env.free_index
        fixed_index = env.fixed_index

        dim = env.dim
        twist = getattr(env, 'twist', False)
        L, _ = final_midpoint_loss(q_final, dim, env.mid_node, target_xy)

        a_free_per_step = np.zeros((len(q_list), len(free_index)), dtype=float)
        a_free_per_step[-1] = dloss_dq_free_node(q_final, free_index, dim, env.mid_node, target_xy)

        u_seq_np = np.stack([u.detach().cpu().numpy() for u in u_list], axis=0)
        g_u_seq = multistep_adjoint_gradient(
            u_seq_np, J_list, free_index, fixed_index, env.deltaL,
            a_free_per_step, jac_reg=cfg.train.jac_reg, dim=dim, twist=twist,
        )

        surrogate = torch.zeros((), device=device)
        for k, u_k in enumerate(u_list):
            g_k = torch.tensor(g_u_seq[k], dtype=u_k.dtype, device=device)
            surrogate = surrogate + torch.dot(u_k, g_k)

        surrogate.backward()
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
        opt.step()

        best_loss = min(best_loss, L)

    # Final rollout with the optimized policy, to commit / report.
    env.set_state(checkpoint)
    with torch.no_grad():
        u_list, q_list, J_list, converged = rollout_segment(env, net, u_start, T, dlam, device, cfg.train.jac_reg)

    return {
        "best_loss": best_loss,
        "q_list": q_list,
        "u_final": u_list[-1] if u_list else u_start,
        "converged": converged,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config overriding configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    # -----------------------------
    # 1) Environment
    # -----------------------------
    dim = env_dim(cfg)
    twist = env_twist(cfg)
    env = build_env(cfg)
    reset_env(env, cfg)

    N = env.Nsteps
    target_xy = task_target(cfg)

    device = resolve_device(cfg.train.device)
    u0 = torch.tensor(initial_control(cfg), dtype=torch.float32, device=device)

    net = BoundaryVelocityNet(hidden=cfg.train.hidden, out_dim=end_control_dim(dim, twist)).to(device)
    opt = optim.Adam(net.parameters(), lr=cfg.train.lr)

    _log_dir = str(_ROOT / 'experiments' / cfg.output.experiment_name)
    os.makedirs(_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=_log_dir)
    run = RunLogger(cfg, env=env, root=_ROOT,
                    enabled=bool((cfg.get("output") or {}).get("run_log", True)))

    print("Network structure:")
    print(net)
    print(f"Environment: {type(env).__name__}  (dim={dim}, twist={twist}, control={control_dim(dim, twist)}D)")
    print(f"Rod material: {material_summary(cfg)}")
    print(f"Training on device: {device}")
    print(f"RHC: {'enabled (' + str(cfg.rhc.num_segments) + ' segments)' if cfg.rhc.enabled else 'disabled (single horizon)'}")

    loss_history = []

    if not cfg.rhc.enabled:
        # -------------------------------------------------------------
        # Single-horizon mode: whole trajectory is one continuation
        # segment, lambda in [0, 1] over all N steps, trained for
        # `epochs` full-horizon gradient updates (persistent net/opt).
        # -------------------------------------------------------------
        dlam = 1.0 / N
        checkpoint = env.get_state()

        pbar = tqdm(range(cfg.train.epochs), desc="Training", dynamic_ncols=True)
        final_loss = None
        q_last = None
        q_traj = []
        for ep in pbar:
            env.set_state(checkpoint)
            opt.zero_grad(set_to_none=True)

            u_list, q_list, J_list, converged = rollout_segment(env, net, u0, N, dlam, device, cfg.train.jac_reg)
            if not converged:
                pbar.set_postfix({"status": "Newton failed"})
                writer.add_scalar("physics/newton_converged", 0.0, ep)
                run.log(epoch=ep, converged=False, steps=len(u_list),
                        newton_failures=env.n_newton_failures,
                        note="rollout aborted, no gradient step")
                continue

            q_final = q_list[-1]
            free_index, fixed_index = env.free_index, env.fixed_index
            L_np, mid_pos = final_midpoint_loss(q_final, dim, env.mid_node, target_xy)

            a_free_per_step = np.zeros((len(q_list), len(free_index)), dtype=float)
            a_free_per_step[-1] = dloss_dq_free_node(q_final, free_index, dim, env.mid_node, target_xy)

            u_seq_np = np.stack([u.detach().cpu().numpy() for u in u_list], axis=0)
            g_u_seq = multistep_adjoint_gradient(
                u_seq_np, J_list, free_index, fixed_index, env.deltaL,
                a_free_per_step, jac_reg=cfg.train.jac_reg, dim=dim, twist=twist,
            )

            surrogate = torch.zeros((), device=device)
            for k, u_k in enumerate(u_list):
                g_k = torch.tensor(g_u_seq[k], dtype=u_k.dtype, device=device)
                surrogate = surrogate + torch.dot(u_k, g_k)

            surrogate.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
            opt.step()

            writer.add_scalar("loss/final", L_np, ep)
            for c, axis_name in enumerate("xyz"[:dim]):
                writer.add_scalar(f"midpoint/{axis_name}", mid_pos[c], ep)
            writer.add_scalar("physics/newton_converged", 1.0, ep)

            pbar.set_postfix({"loss": f"{L_np:.6f}"})
            final_loss = L_np
            q_last = q_final
            q_traj = list(q_list)      # kept for the animation: last converged rollout
            loss_history.append((ep, L_np))
            run.log(epoch=ep, loss=float(L_np), error=float(L_np ** 0.5),
                    midpoint=mid_pos, target=target_xy, converged=True,
                    steps=len(u_list), newton_failures=env.n_newton_failures,
                    u_final=u_list[-1].detach().cpu().numpy())

            if ep % 10 == 0:
                mid_s = ",".join(f"{v:.3f}" for v in mid_pos)
                tgt_s = ",".join(f"{v:.3f}" for v in target_xy)
                print(f"[ep {ep:04d}] L(final)={L_np:.6f}  midpoint=({mid_s})  target=({tgt_s})")

        if final_loss is not None:
            print(f"\nTraining finished. Final loss: {final_loss:.6f}")
        else:
            print("\nTraining finished. No converged epoch recorded.")

        results = {
            'final_q': q_last,
            'final_error': float(final_loss ** 0.5) if final_loss is not None else float('nan'),
            'all_losses': [l for _, l in loss_history],
        }

    else:
        # -------------------------------------------------------------
        # RHC mode: split the N-step horizon into cfg.rhc.num_segments
        # chunks. Each segment gets its own (optionally reinitialized)
        # policy, is optimized for `iterations_per_segment` gradient
        # steps against the same overall target, then the optimized
        # rollout is committed and the next segment warm-starts from its
        # end state. Mirrors adjoint_RHC_any_node.py's MPC loop.
        # -------------------------------------------------------------
        seg_lens = segment_lengths(N, cfg.rhc.num_segments)
        checkpoint = env.get_state()
        u_current = u0.clone()

        q_last = None
        q_traj = []
        global_ep = 0
        for seg_idx, T_seg in enumerate(seg_lens):
            if cfg.rhc.reinit_policy_each_segment:
                reinit_net_(net)
                opt = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=cfg.train.lr)

            dlam = 1.0 / T_seg
            print(f"\n{'='*70}\nRHC segment {seg_idx + 1}/{len(seg_lens)}  (T={T_seg})\n{'='*70}")

            result = optimize_segment(
                env, net, opt, u_current, T_seg, dlam, device, target_xy, cfg,
                n_iterations=cfg.train.iterations_per_segment, checkpoint=checkpoint,
            )

            # Commit: advance the persistent simulator + control state.
            env.set_state(checkpoint)
            with torch.no_grad():
                u_list, q_list, J_list, converged = rollout_segment(env, net, u_current, T_seg, dlam, device, cfg.train.jac_reg)
            checkpoint = env.get_state()
            u_current = u_list[-1].detach().clone()
            q_last = q_list[-1]
            q_traj.extend(q_list)      # committed rollout, concatenated across segments

            loss_history.append((seg_idx, result["best_loss"]))
            writer.add_scalar("loss/segment_best", result["best_loss"], seg_idx)
            mid_pos = q_last[dim * env.mid_node: dim * env.mid_node + dim]
            for c, axis_name in enumerate("xyz"[:dim]):
                writer.add_scalar(f"midpoint/{axis_name}", mid_pos[c], seg_idx)
            mid_s = ",".join(f"{v:.3f}" for v in mid_pos)
            print(f"[segment {seg_idx}] best_loss={result['best_loss']:.6e}  midpoint=({mid_s})")
            run.log(segment=seg_idx, horizon=int(T_seg),
                    best_loss=float(result["best_loss"]),
                    error=float(result["best_loss"] ** 0.5),
                    midpoint=mid_pos, target=target_xy,
                    converged=bool(result["converged"]),
                    newton_failures=env.n_newton_failures,
                    u_final=u_current.detach().cpu().numpy())
            global_ep += 1

        final_loss = loss_history[-1][1] if loss_history else None
        results = {
            'final_q': q_last,
            'final_error': float(final_loss ** 0.5) if final_loss is not None else float('nan'),
            'all_losses': [l for _, l in loss_history],
        }

    visualize_results(results, env, target_xy, save_path=str(_ROOT / 'outputs'))
    maybe_save_trajectory_gif(cfg, q_traj, env, target_xy,
                              save_path=str(_ROOT / 'outputs'),
                              title='DLO point-to-point')

    if loss_history:
        eps, losses = zip(*loss_history)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(eps, losses, linewidth=1.5, color="steelblue")
        ax.set_xlabel("Epoch" if not cfg.rhc.enabled else "RHC segment")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curve")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        save_path = str(_ROOT / 'experiments' / cfg.output.experiment_name / 'loss_curve.png')
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Loss curve saved to: {save_path}")

    run.finish(
        task="point_to_point",
        mode="rhc" if cfg.rhc.enabled else "single_horizon",
        final_loss=None if final_loss is None else float(final_loss),
        final_error=results['final_error'],
        final_midpoint=(None if q_last is None else
                        q_last[dim * env.mid_node: dim * env.mid_node + dim]),
        target=target_xy,
        losses=[l for _, l in loss_history],
        best_loss=(min(l for _, l in loss_history) if loss_history else None),
        epochs_recorded=len(loss_history),
        newton_failures_total=env.n_newton_failures,
        device=str(device),
    )
    writer.close()


if __name__ == "__main__":
    main()
