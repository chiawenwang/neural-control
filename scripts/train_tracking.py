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
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from implicit_grad_tools import multistep_adjoint_gradient
from config import load_config
from runlog import RunLogger
from utils import maybe_save_trajectory_gif
from env_factory import (
    build_env,
    reset_env,
    env_dim,
    env_twist,
    control_dim,
    end_control_dim,
    initial_control,
    material_summary,
    make_target_trajectory,
    node_tracking_loss,
    node_position,
    dloss_dq_free_node,
)


# ─────────────────────────────────────────────
# Network architecture
# ─────────────────────────────────────────────
class BoundaryVelocityNet(nn.Module):
    """lambda -> control rate of the right end (3 DOFs in 2D, 5 in 3D)."""

    def __init__(self, hidden=64, out_dim=3):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),   # controls the right end's DOFs
        )

    def forward(self, lam):
        return self.mlp(lam)


def reinit_net_(net: nn.Module):
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
            nn.init.zeros_(m.bias)
    net.apply(_init)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def segment_lengths(N: int, num_segments: int):
    base, rem = divmod(N, num_segments)
    return [base + 1 if i < rem else base for i in range(num_segments)]




def rollout_segment(env, net, u_start, targets_seg, dlam, device, track_w):
    """Roll out len(targets_seg) steps, lambda reset to [0,1] over this segment.
    Returns per-step data needed for the exact multi-step adjoint gradient."""
    T = len(targets_seg)
    u_list, q_list, J_list = [], [], []
    converged_mask = np.zeros(T, dtype=bool)

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
        converged_mask[k] = info["converged"]

    return u_list, q_list, J_list, converged_mask


def compute_loss_and_grad(env, u_list, q_list, J_list, converged_mask, targets_seg, track_w, jac_reg):
    free_index, fixed_index = env.free_index, env.fixed_index
    dim = env.dim
    twist = getattr(env, 'twist', False)
    T = len(q_list)

    total_loss = 0.0
    a_free_per_step = np.zeros((T, len(free_index)), dtype=float)
    for k in range(T):
        if not converged_mask[k]:
            continue
        L_k, _ = node_tracking_loss(q_list[k], dim, env.mid_node, targets_seg[k])
        total_loss += track_w * L_k
        a_free_per_step[k] = track_w * dloss_dq_free_node(q_list[k], free_index, dim, env.mid_node, targets_seg[k])

    if not converged_mask.any():
        return None, total_loss

    u_seq_np = np.stack([u.detach().cpu().numpy() for u in u_list], axis=0)
    g_u_seq = multistep_adjoint_gradient(
        u_seq_np, J_list, free_index, fixed_index, env.deltaL, a_free_per_step,
        jac_reg=jac_reg, dim=dim, twist=twist,
    )
    return g_u_seq, total_loss


def optimize_segment(env, net, opt, u_start, targets_seg, dlam, device, cfg, n_iterations, checkpoint):
    best_loss = float("inf")
    q_list = None

    for it in range(n_iterations):
        env.set_state(checkpoint)
        opt.zero_grad(set_to_none=True)

        u_list, q_list, J_list, converged_mask = rollout_segment(env, net, u_start, targets_seg, dlam, device, cfg.task.track_w)
        g_u_seq, total_loss = compute_loss_and_grad(env, u_list, q_list, J_list, converged_mask, targets_seg, cfg.task.track_w, cfg.train.jac_reg)
        if g_u_seq is None:
            continue

        surrogate = torch.zeros((), device=device)
        for k, u_k in enumerate(u_list):
            g_k = torch.tensor(g_u_seq[k], dtype=u_k.dtype, device=device)
            surrogate = surrogate + torch.dot(u_k, g_k)

        surrogate.backward()
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
        opt.step()

        best_loss = min(best_loss, total_loss)

    env.set_state(checkpoint)
    with torch.no_grad():
        u_list, q_list, J_list, converged_mask = rollout_segment(env, net, u_start, targets_seg, dlam, device, cfg.task.track_w)

    return {"best_loss": best_loss, "q_list": q_list, "u_final": u_list[-1], "converged_mask": converged_mask}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config overriding configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    dim = env_dim(cfg)
    twist = env_twist(cfg)
    env = build_env(cfg)
    reset_env(env, cfg)

    N = env.Nsteps
    targets = make_target_trajectory(N, env.rod_length,
                                     traj_type=cfg.task.trajectory_type, dim=dim)

    device = resolve_device(cfg.train.device)
    u0 = torch.tensor(initial_control(cfg), dtype=torch.float32, device=device)

    net = BoundaryVelocityNet(hidden=cfg.train.hidden, out_dim=end_control_dim(dim, twist)).to(device)
    opt = optim.Adam(net.parameters(), lr=cfg.train.lr)

    # RunLogger owns the TensorBoard writer: every result goes through
    # run.log(), so the JSON record and the scalars cannot drift apart.
    run = RunLogger(cfg, env=env, root=_ROOT,
                    enabled=bool((cfg.get("output") or {}).get("run_log", True)))

    print("Network structure:")
    print(net)
    print(f"Environment: {type(env).__name__}  (dim={dim}, twist={twist}, control={control_dim(dim, twist)}D)")
    print(f"Rod material: {material_summary(cfg)}")
    print(f"Device: {device}")
    print(f"Trajectory type: {cfg.task.trajectory_type}")
    print(f"RHC: {'enabled (' + str(cfg.rhc.num_segments) + ' segments)' if cfg.rhc.enabled else 'disabled (single horizon)'}")

    loss_history = []

    if not cfg.rhc.enabled:
        dlam = 1.0 / N
        checkpoint = env.get_state()

        pbar = tqdm(range(cfg.train.epochs), desc="Training", dynamic_ncols=True)
        for ep in pbar:
            env.set_state(checkpoint)
            opt.zero_grad(set_to_none=True)

            u_list, q_list, J_list, converged_mask = rollout_segment(env, net, u0, targets, dlam, device, cfg.task.track_w)
            g_u_seq, total_loss = compute_loss_and_grad(env, u_list, q_list, J_list, converged_mask, targets, cfg.task.track_w, cfg.train.jac_reg)

            converged_ratio = converged_mask.mean()

            if g_u_seq is None:
                pbar.set_postfix({"status": "Newton failed all steps"})
                run.log(epoch=ep, converged_ratio=float(converged_ratio),
                        newton_failures=env.n_newton_failures,
                        note="no converged step, no gradient step")
                continue

            surrogate = torch.zeros((), device=device)
            for k, u_k in enumerate(u_list):
                g_k = torch.tensor(g_u_seq[k], dtype=u_k.dtype, device=device)
                surrogate = surrogate + torch.dot(u_k, g_k)

            surrogate.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
            opt.step()

            loss_history.append((ep, total_loss))

            pbar.set_postfix({"total_loss": f"{total_loss:.4f}", "conv%": f"{converged_ratio*100:.0f}"})
            # Mean per-step distance to the target is the number to read; the
            # total loss is a sum of squares and scales with the horizon.
            errs = [float(np.linalg.norm(node_position(q, dim, env.mid_node) - targets[k]))
                    for k, q in enumerate(q_list) if converged_mask[k]]
            run.log(epoch=ep, total_loss=float(total_loss),
                    mean_track_error=float(np.mean(errs)) if errs else None,
                    max_track_error=float(np.max(errs)) if errs else None,
                    converged_ratio=float(converged_ratio),
                    newton_failures=env.n_newton_failures)
            if ep % 20 == 0:
                print(f"[ep {ep:04d}]  total_loss={total_loss:.6f}  converged={converged_ratio*100:.0f}%")

    else:
        seg_lens = segment_lengths(N, cfg.rhc.num_segments)
        seg_bounds = np.cumsum([0] + seg_lens)
        checkpoint = env.get_state()
        u_current = u0.clone()

        for seg_idx, T_seg in enumerate(seg_lens):
            if cfg.rhc.reinit_policy_each_segment:
                reinit_net_(net)
                opt = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=cfg.train.lr)

            targets_seg = targets[seg_bounds[seg_idx]:seg_bounds[seg_idx + 1]]
            dlam = 1.0 / T_seg
            print(f"\n{'='*70}\nRHC segment {seg_idx + 1}/{len(seg_lens)}  (T={T_seg})\n{'='*70}")

            result = optimize_segment(env, net, opt, u_current, targets_seg, dlam, device, cfg,
                                       n_iterations=cfg.train.iterations_per_segment, checkpoint=checkpoint)

            env.set_state(checkpoint)
            with torch.no_grad():
                u_list, q_list, J_list, converged_mask = rollout_segment(env, net, u_current, targets_seg, dlam, device, cfg.task.track_w)
            checkpoint = env.get_state()
            u_current = u_list[-1].detach().clone()

            loss_history.append((seg_idx, result["best_loss"]))
            print(f"[segment {seg_idx}] best_loss={result['best_loss']:.6e}")
            run.log(segment=seg_idx, horizon=int(T_seg),
                    best_loss=float(result["best_loss"]),
                    converged_ratio=float(result["converged_mask"].mean()),
                    newton_failures=env.n_newton_failures)

    if loss_history:
        eps, losses = zip(*loss_history)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(eps, losses, linewidth=1.5, color="steelblue")
        ax.set_xlabel("Epoch" if not cfg.rhc.enabled else "RHC segment")
        ax.set_ylabel("Total Tracking Loss")
        ax.set_title(f"Tracking Loss ({cfg.task.trajectory_type})")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        out = os.path.join(run.artifact_dir, 'loss_curve.png')
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Loss curve → {out}")

    _visualize_tracking(env, net, u0, N, targets, device, cfg, run.artifact_dir)

    run.finish(
        task="tracking",
        trajectory=cfg.task.trajectory_type,
        mode="rhc" if cfg.rhc.enabled else "single_horizon",
        final_loss=(loss_history[-1][1] if loss_history else None),
        best_loss=(min(l for _, l in loss_history) if loss_history else None),
        losses=[l for _, l in loss_history],
        epochs_recorded=len(loss_history),
        newton_failures_total=env.n_newton_failures,
        device=str(device),
    )


def _visualize_tracking(env, net, u0, N, targets, device, cfg, out_dir):
    reset_env(env, cfg)
    dim = env.dim
    twist = getattr(env, 'twist', False)
    actual = []
    q_traj = []       # full configuration per step, for the animation

    if cfg.rhc.enabled:
        seg_lens = segment_lengths(N, cfg.rhc.num_segments)
    else:
        seg_lens = [N]

    u = u0.clone()
    n_left = net.out_dim
    with torch.no_grad():
        for T_seg in seg_lens:
            for k in range(T_seg):
                lam = torch.tensor([[k / T_seg]], dtype=torch.float32, device=device)
                u_dot_full = torch.cat([torch.zeros(n_left, device=device), net(lam).squeeze(0)])
                u = u + (1.0 / T_seg) * u_dot_full
                u_np = u.cpu().numpy()
                obs, done, info = env.step(u_np, use_inertia=False)
                if info["converged"]:
                    a = dim * env.mid_node
                    actual.append(info["q"][a:a + dim])
                    q_traj.append(info["q"].copy())
    actual = np.array(actual).reshape(-1, dim)

    fig = plt.figure(figsize=(6, 6))
    if dim == 2:
        ax = fig.add_subplot(111)
        ax.plot(targets[:, 0], targets[:, 1], 'b--', linewidth=2, label="Target trajectory")
        ax.plot(actual[:, 0], actual[:, 1], 'r-o', markersize=3, linewidth=1.5,
                label="Actual midpoint")
        ax.set_aspect('equal')
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(targets[:, 0], targets[:, 1], targets[:, 2], 'b--', linewidth=2,
                label="Target trajectory")
        ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], 'r-o', markersize=3,
                linewidth=1.5, label="Actual midpoint")
        ax.set_zlabel("z")
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:      # matplotlib < 3.3
            pass
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Midpoint Tracking ({cfg.task.trajectory_type})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(out_dir, 'tracking_result.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Tracking result → {out}")

    maybe_save_trajectory_gif(cfg, q_traj, env, targets,
                              save_path=str(_ROOT / 'outputs'),
                              title=f'DLO tracking ({cfg.task.trajectory_type})')


if __name__ == "__main__":
    main()
