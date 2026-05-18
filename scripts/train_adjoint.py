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
from utils import visualize_results


class BoundaryVelocityNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            # for right end
            # nn.Linear(hidden, 6),
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
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    args = parser.parse_args()
    epochs = args.epochs

    # -----------------------------
    # 1) 创建环境
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
    # 2) 目标
    # -----------------------------
    target_xy = np.array([0.2, 0.4], dtype=float)

    # -----------------------------
    # 3) 网络与优化器
    # -----------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    u0 = torch.tensor([0.0, 0.0, 0.0, env.rod_length, 0.0, 0.0], dtype=torch.float32, device=device)

    net = BoundaryVelocityNet(hidden=64).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)

    _log_dir = str(_ROOT / 'experiments' / 'dlo_final_shape')
    os.makedirs(_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=_log_dir)

    print("Network structure:")
    print(net)
    print(f"Training on device: {device}")
    print("Starting training...")

    final_loss = None
    loss_history = []  # ← 新增：记录 (epoch, loss) 用于绘图

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

            # u_dot = net(lam).squeeze(0)
            # u = u + dlam * u_dot
            # for right end
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

        L_np, (mx, my) = final_midpoint_loss(
            q_last, env.nv, env.mid_node, target_xy
        )

        free_index = info_last["free_index"]
        fixed_index = info_last["fixed_index"]

        a_free = dloss_dq_free_midpoint(q_last, free_index, env.nv, env.mid_node, target_xy)

        u_final_np = u_list[-1].detach().cpu().numpy()

        g_u_final = implicit_final_control_grad(
            J=J_last,
            free_index=free_index,
            fixed_index=fixed_index,
            a_free=a_free,
            control_6d=u_final_np,
            deltaL=env.deltaL
        )

        g = torch.tensor(g_u_final, dtype=torch.float32, device=device)

        opt.zero_grad()

        loss_torch = 0.0
        for u_k in u_list:
            loss_torch = loss_torch + dlam * torch.dot(u_k, g)

        loss_torch.backward()
        opt.step()

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
        loss_history.append((ep, L_np))  # ← 新增：记录本轮 loss

        if ep % 10 == 0:
            print(f"[ep {ep:04d}] L(final)={L_np:.6f}  midpoint=({mx:.3f},{my:.3f})  target=({target_xy[0]:.3f},{target_xy[1]:.3f})")

    if final_loss is not None:
        print(f"\nTraining finished. Final loss: {final_loss:.6f}")
    else:
        print("\nTraining finished. No converged epoch recorded.")

    results = {
        'final_q': q_last,
        'final_error': float(L_np ** 0.5),
        'all_losses': [l for _, l in loss_history],
    }
    # visualize the results
    visualize_results(results, env, target_xy, save_path=str(_ROOT / 'outputs'))

    if loss_history:
        eps, losses = zip(*loss_history)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(eps, losses, linewidth=1.5, color="steelblue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curve")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        save_path = str(_ROOT / 'experiments' / 'dlo_final_shape' / 'loss_curve.png')
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Loss curve saved to: {save_path}")

    writer.close()

"""
args:
epochs: default is 200
"""

if __name__ == "__main__":
    main()