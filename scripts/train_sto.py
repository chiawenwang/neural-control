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
from implicit_grad_tools import implicit_final_control_grad
from sto_simulation import StatefulTangentOperator
from utils import visualize_results


class BoundaryVelocityNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            # for right end only
            nn.Linear(hidden, 3),
        )

    def forward(self, lam):
        return self.mlp(lam)


def final_midpoint_loss(q, nv, mid_node, target_xy):
    x_arr = q[::2]
    y_arr = q[1::2]
    mx = x_arr[mid_node]
    my = y_arr[mid_node]
    dx = mx - target_xy[0]
    dy = my - target_xy[1]
    L = dx * dx + dy * dy
    return L, (mx, my)


def dloss_dq_free_midpoint(q, free_index, nv, mid_node, target_xy):
    ix = 2 * mid_node
    iy = 2 * mid_node + 1

    mx = q[ix]
    my = q[iy]

    dL_dmx = 2.0 * (mx - target_xy[0])
    dL_dmy = 2.0 * (my - target_xy[1])

    a_free = np.zeros(len(free_index), dtype=float)

    pos_x = np.where(free_index == ix)[0]
    pos_y = np.where(free_index == iy)[0]

    if len(pos_x) == 1:
        a_free[pos_x[0]] = dL_dmx
    if len(pos_y) == 1:
        a_free[pos_y[0]] = dL_dmy

    return a_free


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="epochs")
    parser.add_argument("--use_sto", action="store_true", default=True,
                        help="use STO(default: True)")
    parser.add_argument("--no_sto", action="store_true",
                        help="no STO，exact solve every epoch")
    parser.add_argument("--kappa_abs", type=float, default=1e6,
                        help="STO invalidation: absolute kappa threshold")
    parser.add_argument("--kappa_ratio", type=float, default=3.0,
                        help="STO invalidation: kappa ratio threshold")
    parser.add_argument("--max_reuse", type=int, default=1000,
                        help="STO invalidation: max cache reuse count")
    args = parser.parse_args()
    epochs = args.epochs
    use_sto = args.use_sto and (not args.no_sto)

    # -----------------------------
    # 1) env
    # -----------------------------
    env = SimulatorEnv_2D(
        nv=51,
        dt=0.01,
        total_time=2.0,
        save_history=False
    )

    _ = env.reset()

    N = env.Nsteps
    dlam = 1.0 / N

    # -----------------------------
    # 2) target
    # -----------------------------
    target_xy = np.array([0.2, 0.4], dtype=float)

    # -----------------------------
    # 3) nn
    # -----------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    u0 = torch.tensor([0.0, 0.0, 0.0, env.rod_length, 0.0, 0.0],
                       dtype=torch.float32, device=device)

    net = BoundaryVelocityNet(hidden=64).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)

    log_dir = str(_ROOT / 'experiments' / 'dlo_final_shape')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # -----------------------------
    # 4) STO
    # -----------------------------
    sto = None
    if use_sto:
        sto = StatefulTangentOperator(
            kappa_abs=args.kappa_abs,
            kappa_ratio=args.kappa_ratio,
            max_reuse=args.max_reuse,
        )

    print("Network structure:")
    print(net)
    print(f"Training on device: {device}")
    print(f"STO enabled: {use_sto}")
    if use_sto:
        print(f"  kappa_abs={args.kappa_abs:.0e}  kappa_ratio={args.kappa_ratio}  max_reuse={args.max_reuse}")
    print("Starting training...")

    final_loss = None
    loss_history = []
    sto_events = []  # (epoch, used_cache, kappa, reason)

    pbar = tqdm(range(epochs), desc="Training", dynamic_ncols=True)
    for ep in pbar:
        u_list = []
        q_last = None
        J_last = None
        info_last = None

        env.reset()

        u = u0.clone()

        for k in range(N):
            lam = torch.tensor([[k / N]], dtype=torch.float32, device=device)

            # right end only: left 3 dims are zero
            u_dot_full = torch.cat([torch.zeros(3, device=device), net(lam).squeeze(0)])
            u = u + dlam * u_dot_full
            u_list.append(u)

            u_np = u.detach().cpu().numpy()
            obs, done, info = env.step(u_np, use_inertia=False)

            q_last = info["q"]
            J_last = info["J"]
            info_last = info

            if done and (k < N - 1):
                break

        if (info_last is None) or (not info_last["converged"]):
            pbar.set_postfix({"status": "Newton failed"})
            writer.add_scalar("physics/newton_converged", 0.0, ep)
            continue

        # ---------------------------------------------------------
        # Loss
        # ---------------------------------------------------------
        L_np, (mx, my) = final_midpoint_loss(
            q_last, env.nv, env.mid_node, target_xy
        )

        free_index = info_last["free_index"]
        fixed_index = info_last["fixed_index"]

        a_free = dloss_dq_free_midpoint(
            q_last, free_index, env.nv, env.mid_node, target_xy
        )

        u_final_np = u_list[-1].detach().cpu().numpy()

        # ---------------------------------------------------------
        # Gradient: STO or exact
        # ---------------------------------------------------------
        if use_sto:
            g_u_final, used_cache, kappa, reason = sto.query(
                J_full=J_last,
                free_index=free_index,
                fixed_index=fixed_index,
                a_free=a_free,
                control_6d=u_final_np,
                deltaL=env.deltaL,
            )
            sto_events.append((ep, used_cache, kappa, reason))

            # TensorBoard: STO-specific metrics
            writer.add_scalar("sto/used_cache", float(used_cache), ep)
            writer.add_scalar("sto/kappa", kappa, ep)
        else:
            g_u_final = implicit_final_control_grad(
                J=J_last,
                free_index=free_index,
                fixed_index=fixed_index,
                a_free=a_free,
                control_6d=u_final_np,
                deltaL=env.deltaL,
            )

        # ---------------------------------------------------------
        # Surrogate loss for backprop through network
        # ---------------------------------------------------------
        g = torch.tensor(g_u_final, dtype=torch.float32, device=device)

        opt.zero_grad()

        loss_torch = 0.0
        for u_k in u_list:
            loss_torch = loss_torch + dlam * torch.dot(u_k, g)

        loss_torch.backward()
        opt.step()

        # ---------------------------------------------------------
        # Logging
        # ---------------------------------------------------------
        writer.add_scalar("loss/final", L_np, ep)
        writer.add_scalar("midpoint/x", mx, ep)
        writer.add_scalar("midpoint/y", my, ep)
        writer.add_scalar("physics/newton_converged", float(info_last["converged"]), ep)

        g_norm = np.linalg.norm(g_u_final)
        writer.add_scalar("grad/gu_final_norm", g_norm, ep)

        u_final_norm = torch.norm(u_list[-1]).item()
        writer.add_scalar("control/u_final_norm", u_final_norm, ep)

        pbar.set_postfix({"loss": f"{L_np:.6f}"})

        final_loss = L_np
        loss_history.append((ep, L_np))

        if ep % 10 == 0:
            sto_tag = ""
            if use_sto and sto_events:
                last_event = sto_events[-1]
                sto_tag = f"  [STO: {'cache' if last_event[1] else 'exact'}  κ={last_event[2]:.1e}]"
            print(f"[ep {ep:04d}] L={L_np:.6f}  mid=({mx:.3f},{my:.3f})  "
                  f"target=({target_xy[0]:.3f},{target_xy[1]:.3f}){sto_tag}")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    if final_loss is not None:
        print(f"\nTraining finished. Final loss: {final_loss:.6f}")
    else:
        print("\nTraining finished. No converged epoch recorded.")

    if use_sto and sto is not None:
        print(sto.stats_str())

    # ---------------------------------------------------------
    # Visualize results
    # ---------------------------------------------------------
    results = {
        'final_q': q_last,
        'final_error': float(L_np ** 0.5) if L_np is not None else float('inf'),
        'all_losses': [l for _, l in loss_history],
    }
    visualize_results(results, env, target_xy, save_path=str(_ROOT / 'outputs'))

    # Loss curve
    if loss_history:
        eps_list, losses = zip(*loss_history)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(eps_list, losses, linewidth=1.5, color="steelblue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curve")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        save_path = f"{log_dir}/loss_curve.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Loss curve saved to: {save_path}")

    # STO diagnostics plot (if STO was used)
    if use_sto and sto_events:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("STO Diagnostics", fontsize=14, fontweight='bold')

        # (a) Cache hit / miss timeline
        ax = axes[0]
        for (ep_i, used_cache, kp, reason) in sto_events:
            color = '#2ca02c' if used_cache else '#d62728'
            ax.bar(ep_i, 1, width=0.8, color=color, alpha=0.8)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color='#2ca02c', label='Cache hit (reuse STO)'),
            Patch(color='#d62728', label='Exact solve + (re-)init'),
        ], fontsize=8)
        ax.set_xlabel('Epoch')
        ax.set_yticks([])
        ax.set_title('STO Cache Lifecycle')
        ax.text(0.02, 0.7,
                f"Inits: {sto.n_init}\nCached: {sto.n_cached}\n"
                f"Fallback: {sto.n_fallback}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        # (b) Condition number over epochs
        ax = axes[1]
        kappas = [kp for (_, _, kp, _) in sto_events]
        ep_vals = [ep_i for (ep_i, _, _, _) in sto_events]
        ax.semilogy(ep_vals, kappas, 'o-', color='#d62728', ms=3, lw=1.5)
        ax.axhline(args.kappa_abs, color='red', ls='--', lw=1.5,
                    label=f'kappa_abs = {args.kappa_abs:.0e}')
        # mark invalidation points
        for (ep_i, used_cache, kp, reason) in sto_events:
            if not used_cache and ep_i > 0:
                ax.axvline(ep_i, color='red', alpha=0.2, lw=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('κ(J_ff)')
        ax.set_title('Condition Number')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (c) Cumulative solve cost comparison
        ax = axes[2]
        n_ep = len(sto_events)
        cum_exact = np.arange(1, n_ep + 1)  # 1 adjoint solve per epoch
        cum_sto = np.cumsum([0 if uc else 1 for (_, uc, _, _) in sto_events])
        ax.plot(cum_exact, lw=2.5, color='#1f77b4', label='Exact (solve every epoch)')
        ax.plot(cum_sto, lw=2.5, color='#d62728', label='STO (cache when valid)')
        ax.fill_between(range(n_ep), cum_sto, cum_exact, alpha=0.15, color='green',
                         label='Savings')
        if cum_exact[-1] > 0:
            savings_pct = (1 - cum_sto[-1] / cum_exact[-1]) * 100
            ax.text(0.5, 0.3,
                    f"STO saves {savings_pct:.1f}% of\nadjoint linear solves",
                    transform=ax.transAxes, fontsize=11, fontweight='bold',
                    ha='center', color='darkgreen',
                    bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.9))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cumulative adjoint solves')
        ax.set_title('Computational Cost')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        diag_path = f"{log_dir}/sto_diagnostics.png"
        fig.savefig(diag_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"STO diagnostics saved to: {diag_path}")

    writer.close()


"""
Usage:
  python main.py                          # STO enabled (default)
  python main.py --epochs 500             # more epochs
  python main.py --no_sto                 # disable STO, exact every step
  python main.py --kappa_abs 1e5          # stricter invalidation
  python main.py --kappa_ratio 2.0        # more frequent re-init
"""

if __name__ == "__main__":
    main()
