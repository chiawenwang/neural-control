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


# ─────────────────────────────────────────────
# 网络结构（与原始相同）
# ─────────────────────────────────────────────
class BoundaryVelocityNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),   # 控制右端 3 DOF
        )

    def forward(self, lam):
        return self.mlp(lam)


# ─────────────────────────────────────────────
# 目标轨迹生成（可自定义）
# ─────────────────────────────────────────────
def make_target_trajectory(N, rod_length, traj_type="circle"):
    """
    生成 N 步的目标中点轨迹，返回 shape (N, 2) 的数组。

    traj_type:
      "circle"  : 圆形
      "figure8" : 8 字形
      "line"    : 直线平移
    """
    lams = np.linspace(0, 1, N)
    targets = np.zeros((N, 2))

    if traj_type == "circle":
        cx, cy, r = rod_length / 2, 0.25, 0.15
        for i, lam in enumerate(lams):
            theta = 2 * np.pi * lam
            targets[i] = [cx + r * np.cos(theta), cy + r * np.sin(theta)]

    elif traj_type == "figure8":
        cx, cy, r = rod_length / 2, 0.0, 0.20
        for i, lam in enumerate(lams):
            theta = 2 * np.pi * lam
            targets[i] = [cx + r * np.sin(theta), cy + r * np.sin(2 * theta) / 2]

    elif traj_type == "line":
        for i, lam in enumerate(lams):
            targets[i] = [rod_length / 2, -0.20 + 0.40 * lam]

    else:
        raise ValueError(f"Unknown traj_type: {traj_type}")

    return targets  # (N, 2)


# ─────────────────────────────────────────────
# 每步的 loss 与 dL/dq_free
# ─────────────────────────────────────────────
def step_midpoint_loss(q, mid_node, target_xy):
    """L_k = ||midpoint - target||^2"""
    mx = q[2 * mid_node]
    my = q[2 * mid_node + 1]
    dx = mx - target_xy[0]
    dy = my - target_xy[1]
    return dx * dx + dy * dy, (mx, my)


def dloss_dq_free_midpoint(q, free_index, mid_node, target_xy):
    """dL_k / dq_free  (稀疏，只有中点两个分量非零)"""
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


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=300)
    parser.add_argument("--traj",      type=str,   default="circle",
                        choices=["circle", "figure8", "line"],
                        help="目标轨迹类型")
    parser.add_argument("--track_w",   type=float, default=1.0,
                        help="tracking loss 权重（每步均匀）")
    args = parser.parse_args()

    # ── 1) 环境 ──────────────────────────────
    env = SimulatorEnv_2D(nv=51, dt=0.01, total_time=2.0, save_history=False)
    env.reset()

    N    = env.Nsteps
    dlam = 1.0 / N

    # ── 2) 目标轨迹 ───────────────────────────
    targets = make_target_trajectory(N, env.rod_length, traj_type=args.traj)
    # targets[k] = (x_target, y_target) at step k,  shape (N, 2)

    # ── 3) 网络与优化器 ──────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    u0 = torch.tensor(
        [0.0, 0.0, 0.0, env.rod_length, 0.0, 0.0],
        dtype=torch.float32, device=device
    )

    net = BoundaryVelocityNet(hidden=64).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)

    os.makedirs("runs/dlo_tracking", exist_ok=True)
    writer = SummaryWriter(log_dir="runs/dlo_tracking")

    print("Network structure:")
    print(net)
    print(f"Device: {device}")
    print(f"Trajectory type: {args.traj}")
    print("Starting tracking training...\n")

    loss_history = []
    pbar = tqdm(range(args.epochs), desc="Training", dynamic_ncols=True)

    for ep in pbar:
        env.reset()
        u = u0.clone()

        # 储存每一步的数据，用于事后反向传播
        u_list    = []          # list of torch tensors (6D)
        q_list    = []          # list of numpy arrays  (2*nv)
        J_list    = []          # list of numpy matrices
        info_list = []          # list of info dicts
        converged_steps = []    # 成功收敛的步索引

        # ── 前向仿真 ──────────────────────────
        for k in range(N):
            lam = torch.tensor([[k / N]], dtype=torch.float32, device=device)

            u_dot_full = torch.cat([
                torch.zeros(3, device=device),
                net(lam).squeeze(0)
            ])
            u = u + dlam * u_dot_full
            u_list.append(u)

            u_np = u.detach().cpu().numpy()
            obs, done, info = env.step(u_np, use_inertia=False)

            q_list.append(info["q"])
            J_list.append(info["J"])
            info_list.append(info)

            if info["converged"]:
                converged_steps.append(k)

            if done and (k < N - 1):
                break

        if len(converged_steps) == 0:
            pbar.set_postfix({"status": "Newton failed all steps"})
            writer.add_scalar("physics/converged_ratio", 0.0, ep)
            continue

        converged_ratio = len(converged_steps) / N
        writer.add_scalar("physics/converged_ratio", converged_ratio, ep)

        # ── 计算累积 tracking loss & 每步梯度 ──
        # 
        # 总 loss = sum_k  w * ||mid(q_k) - target_k||^2
        #
        # 对每个收敛步 k，用伴随法算 dL_k/du_k，
        # 然后将梯度沿控制积分路径叠加（类似 adjoint rollout）。
        #
        # 简化处理：每步独立用 implicit_final_control_grad，
        # 再对时间步加权求和，得到总的 g_u（以 u_k 作为"当前步控制"）。

        total_loss_np = 0.0
        # 每步对 u_k 的梯度累加到此向量（6D numpy）
        # grad_sum[k] = dL/du_k  (从步 k 往后所有 loss 对 u_k 的梯度之和)
        # 这里采用"当步即时梯度"近似（忽略未来步的 implicit 依赖，
        # 若需严格递推可扩展为完整轨迹伴随法，见注释）
        grad_per_step = []  # list of numpy 6D vectors

        for k in converged_steps:
            q_k    = q_list[k]
            J_k    = J_list[k]
            info_k = info_list[k]

            L_k, _ = step_midpoint_loss(q_k, env.mid_node, targets[k])
            total_loss_np += args.track_w * L_k

            a_free_k = dloss_dq_free_midpoint(
                q_k, info_k["free_index"], env.mid_node, targets[k]
            )

            g_uk = implicit_final_control_grad(
                J=J_k,
                free_index=info_k["free_index"],
                fixed_index=info_k["fixed_index"],
                a_free=a_free_k,
                control_6d=u_list[k].detach().cpu().numpy(),
                deltaL=env.deltaL
            )
            grad_per_step.append((k, g_uk))

        # ── 反向传播 ──────────────────────────
        # loss_torch = sum_k <u_k , g_uk>  （与原始 final 版本类似）
        opt.zero_grad()

        loss_torch = torch.tensor(0.0, device=device)
        for (k, g_uk) in grad_per_step:
            g = torch.tensor(g_uk * args.track_w, dtype=torch.float32, device=device)
            # u_k 对网络参数的梯度通过 dot product 传播
            loss_torch = loss_torch + dlam * torch.dot(u_list[k], g)

        loss_torch.backward()
        opt.step()

        # ── 日志 ──────────────────────────────
        avg_loss = total_loss_np / max(len(converged_steps), 1)
        writer.add_scalar("loss/total",   total_loss_np, ep)
        writer.add_scalar("loss/avg",     avg_loss,      ep)
        loss_history.append((ep, total_loss_np))

        # 记录最后一步中点位置
        if len(q_list) > 0:
            last_k = converged_steps[-1]
            mx = q_list[last_k][2 * env.mid_node]
            my = q_list[last_k][2 * env.mid_node + 1]
            writer.add_scalar("midpoint/x_last", mx, ep)
            writer.add_scalar("midpoint/y_last", my, ep)

        pbar.set_postfix({
            "total_loss": f"{total_loss_np:.4f}",
            "conv%": f"{converged_ratio*100:.0f}"
        })

        if ep % 20 == 0:
            print(f"[ep {ep:04d}]  total_loss={total_loss_np:.6f}"
                  f"  converged={converged_ratio*100:.0f}%")

    # ── 训练结束：绘图 ────────────────────────
    writer.close()

    # Loss 曲线
    if loss_history:
        eps, losses = zip(*loss_history)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(eps, losses, linewidth=1.5, color="steelblue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Tracking Loss")
        ax.set_title(f"Tracking Loss ({args.traj})")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig("runs/dlo_tracking/loss_curve.png", dpi=150)
        plt.close(fig)
        print("Loss curve → runs/dlo_tracking/loss_curve.png")

    # 最终轨迹可视化
    _visualize_tracking(env, net, u0, N, dlam, targets, device, args.traj)


# ─────────────────────────────────────────────
# 可视化：训练后跑一遍，画实际轨迹 vs 目标轨迹
# ─────────────────────────────────────────────
def _visualize_tracking(env, net, u0, N, dlam, targets, device, traj_type):
    env.reset()
    u = u0.clone()
    mid_xs, mid_ys = [], []

    with torch.no_grad():
        for k in range(N):
            lam = torch.tensor([[k / N]], dtype=torch.float32, device=device)
            u_dot_full = torch.cat([torch.zeros(3, device=device),
                                    net(lam).squeeze(0)])
            u = u + dlam * u_dot_full
            u_np = u.cpu().numpy()
            obs, done, info = env.step(u_np, use_inertia=False)
            if info["converged"]:
                mid_xs.append(info["q"][2 * env.mid_node])
                mid_ys.append(info["q"][2 * env.mid_node + 1])
            if done:
                break

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(targets[:, 0], targets[:, 1], 'b--', linewidth=2, label="Target trajectory")
    ax.plot(mid_xs, mid_ys, 'r-o', markersize=3, linewidth=1.5, label="Actual midpoint")
    ax.set_aspect('equal')
    ax.set_xlabel("x");  ax.set_ylabel("y")
    ax.set_title(f"Midpoint Tracking ({traj_type})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("runs/dlo_tracking/tracking_result.png", dpi=150)
    plt.close(fig)
    print("Tracking result → runs/dlo_tracking/tracking_result.png")


if __name__ == "__main__":
    main()
