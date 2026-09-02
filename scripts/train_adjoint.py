import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from SimulatorEnv_2D import SimulatorEnv_2D
from implicit_grad_tools import multistep_adjoint_gradient
from utils import visualize_results
from config import load_config


class BoundaryVelocityNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            # for right end
            nn.Linear(hidden, 3),
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


def final_midpoint_loss(q, mid_node, target_xy):
    mx = q[2 * mid_node]
    my = q[2 * mid_node + 1]
    dx = mx - target_xy[0]
    dy = my - target_xy[1]
    return dx * dx + dy * dy, (mx, my)


def dloss_dq_free_midpoint(q, free_index, mid_node, target_xy):
    ix = 2 * mid_node
    iy = 2 * mid_node + 1
    mx, my = q[ix], q[iy]

    a_free = np.zeros(len(free_index), dtype=float)
    pos_x = np.where(free_index == ix)[0]
    pos_y = np.where(free_index == iy)[0]
    if len(pos_x) == 1:
        a_free[pos_x[0]] = 2.0 * (mx - target_xy[0])
    if len(pos_y) == 1:
        a_free[pos_y[0]] = 2.0 * (my - target_xy[1])
    return a_free


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
    for k in range(T):
        lam = torch.tensor([[k / T]], dtype=torch.float32, device=device)
        u_dot_full = torch.cat([torch.zeros(3, device=device), net(lam).squeeze(0)])
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

        L, (mx, my) = final_midpoint_loss(q_final, env.mid_node, target_xy)

        a_free_per_step = np.zeros((len(q_list), len(free_index)), dtype=float)
        a_free_per_step[-1] = dloss_dq_free_midpoint(q_final, free_index, env.mid_node, target_xy)

        u_seq_np = np.stack([u.detach().cpu().numpy() for u in u_list], axis=0)
        g_u_seq = multistep_adjoint_gradient(
            u_seq_np, J_list, free_index, fixed_index, env.deltaL,
            a_free_per_step, jac_reg=cfg.train.jac_reg,
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
    env = SimulatorEnv_2D(
        nv=cfg.rod.nv,
        dt=cfg.rod.dt,
        rod_length=cfg.rod.rod_length,
        total_time=cfg.rod.total_time,
        R_outer=cfg.rod.R_outer,
        r_inner=cfg.rod.r_inner,
        E_al=cfg.rod.E_al,
        rho_al=cfg.rod.rho_al,
        max_newton_iter=cfg.rod.max_newton_iter,
        save_history=False,
    )
    env.reset(init_x=cfg.start.init_x, init_y=cfg.start.init_y, init_theta=cfg.start.init_theta)

    N = env.Nsteps
    target_xy = np.array(cfg.task.target_xy, dtype=float)

    device = resolve_device(cfg.train.device)
    u0 = torch.tensor(list(cfg.start.u_left) + list(cfg.start.u_right), dtype=torch.float32, device=device)

    net = BoundaryVelocityNet(hidden=cfg.train.hidden).to(device)
    opt = optim.Adam(net.parameters(), lr=cfg.train.lr)

    _log_dir = str(_ROOT / 'experiments' / cfg.output.experiment_name)
    os.makedirs(_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=_log_dir)

    print("Network structure:")
    print(net)
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
        for ep in pbar:
            env.set_state(checkpoint)
            opt.zero_grad(set_to_none=True)

            u_list, q_list, J_list, converged = rollout_segment(env, net, u0, N, dlam, device, cfg.train.jac_reg)
            if not converged:
                pbar.set_postfix({"status": "Newton failed"})
                writer.add_scalar("physics/newton_converged", 0.0, ep)
                continue

            q_final = q_list[-1]
            free_index, fixed_index = env.free_index, env.fixed_index
            L_np, (mx, my) = final_midpoint_loss(q_final, env.mid_node, target_xy)

            a_free_per_step = np.zeros((len(q_list), len(free_index)), dtype=float)
            a_free_per_step[-1] = dloss_dq_free_midpoint(q_final, free_index, env.mid_node, target_xy)

            u_seq_np = np.stack([u.detach().cpu().numpy() for u in u_list], axis=0)
            g_u_seq = multistep_adjoint_gradient(
                u_seq_np, J_list, free_index, fixed_index, env.deltaL,
                a_free_per_step, jac_reg=cfg.train.jac_reg,
            )

            surrogate = torch.zeros((), device=device)
            for k, u_k in enumerate(u_list):
                g_k = torch.tensor(g_u_seq[k], dtype=u_k.dtype, device=device)
                surrogate = surrogate + torch.dot(u_k, g_k)

            surrogate.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
            opt.step()

            writer.add_scalar("loss/final", L_np, ep)
            writer.add_scalar("midpoint/x", mx, ep)
            writer.add_scalar("midpoint/y", my, ep)
            writer.add_scalar("physics/newton_converged", 1.0, ep)

            pbar.set_postfix({"loss": f"{L_np:.6f}"})
            final_loss = L_np
            q_last = q_final
            loss_history.append((ep, L_np))

            if ep % 10 == 0:
                print(f"[ep {ep:04d}] L(final)={L_np:.6f}  midpoint=({mx:.3f},{my:.3f})  target=({target_xy[0]:.3f},{target_xy[1]:.3f})")

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

            loss_history.append((seg_idx, result["best_loss"]))
            writer.add_scalar("loss/segment_best", result["best_loss"], seg_idx)
            mx, my = q_last[2 * env.mid_node], q_last[2 * env.mid_node + 1]
            writer.add_scalar("midpoint/x", mx, seg_idx)
            writer.add_scalar("midpoint/y", my, seg_idx)
            print(f"[segment {seg_idx}] best_loss={result['best_loss']:.6e}  midpoint=({mx:.3f},{my:.3f})")
            global_ep += 1

        final_loss = loss_history[-1][1] if loss_history else None
        results = {
            'final_q': q_last,
            'final_error': float(final_loss ** 0.5) if final_loss is not None else float('nan'),
            'all_losses': [l for _, l in loss_history],
        }

    visualize_results(results, env, target_xy, save_path=str(_ROOT / 'outputs'))

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

    writer.close()


if __name__ == "__main__":
    main()
