import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from SimulatorEnv_2D import SimulatorEnv_2D
from implicit_grad_tools import implicit_final_control_grad


class BoundaryVelocityNet(nn.Module):
    """
    论文里的关键设计：网络输出“边界变化率” du/dλ，而不是直接输出 u(λ)
    这样 u(λ) = u0 + ∫ du/dλ dλ  （离散就是累加）
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 6),   # 输出 6D: [x_l,y_l,th_l,x_r,y_r,th_r] 的“速度”
        )

    def forward(self, lam):
        """
        lam: shape (...,1)
        return: shape (...,6)
        """
        return self.mlp(lam)


def final_midpoint_loss(q, nv, mid_node, target_xy):
    """
    一个最小复现的 final shape loss：只约束中点坐标
    L = || midpoint - target ||^2
    """
    x_arr = q[::2]
    y_arr = q[1::2]
    mx = x_arr[mid_node]
    my = y_arr[mid_node]
    dx = mx - target_xy[0]
    dy = my - target_xy[1]
    L = dx * dx + dy * dy
    return L, (mx, my)


def dloss_dq_free_midpoint(q, free_index, nv, mid_node, target_xy):
    """
    计算 a_free = dL/dq_free （只在自由DOF上）
    对中点loss：
      L = (mx-tx)^2 + (my-ty)^2
      dL/dmx = 2(mx-tx), dL/dmy = 2(my-ty)
    """
    # mid_node 的 DOF 在 q 里的索引
    ix = 2 * mid_node     # mx 对应的 q 下标
    iy = 2 * mid_node + 1 # my 对应的 q 下标

    mx = q[ix]
    my = q[iy]

    dL_dmx = 2.0 * (mx - target_xy[0])
    dL_dmy = 2.0 * (my - target_xy[1])

    # a_free 的长度等于自由DOF数量
    a_free = np.zeros(len(free_index), dtype=float)

    # 找到 ix, iy 在 free_index 里的位置
    # （如果 mid 节点恰好是 fixed，则这里会找不到；你的 env 固定的是两端，所以 mid 是 free）
    pos_x = np.where(free_index == ix)[0]
    pos_y = np.where(free_index == iy)[0]

    if len(pos_x) == 1:
        a_free[pos_x[0]] = dL_dmx
    if len(pos_y) == 1:
        a_free[pos_y[0]] = dL_dmy

    return a_free


def main():
    # -----------------------------
    # 1) 创建环境（建议先准静态复现）
    # -----------------------------
    env = SimulatorEnv_2D(
        nv=51,
        dt=0.01,
        total_time=2.0,     # 这里 total_time 只影响 Nsteps = total_time/dt
        save_history=False
    )

    # reset
    _ = env.reset()

    # 同伦离散步数（你可以直接用 env.Nsteps）
    N = env.Nsteps
    dlam = 1.0 / N

    # -----------------------------
    # 2) 目标：final midpoint 到 target
    # -----------------------------
    # 你可以换成任何你论文里的 final shape target
    target_xy = np.array([0.5, 0.3], dtype=float)

    # -----------------------------
    # 3) 初始边界 u0（6D）
    # -----------------------------
    # 左端固定在原点附近，右端初始在 (rod_length,0,0)

    # -----------------------------
    # 4) 网络与优化器
    # -----------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    u0 = torch.tensor([0.0, 0.0, 0.0, env.rod_length, 0.0, 0.0], dtype=torch.float32, device=device)

    net = BoundaryVelocityNet(hidden=64).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)

    writer = SummaryWriter(log_dir="runs/dlo_final_shape")
    # make sure the log_dir exists
    os.makedirs("runs/dlo_final_shape", exist_ok=True)

    # 训练轮数
    epochs = 200

    # print network basic info
    print("Network structure:")
    print(net)
    print(f"Training on device: {device}")
    print("Starting training...")

    for ep in range(epochs):
        # ---------------------------------------------------------
        # 前向：用网络输出 du/dλ，然后积分得到 u_k，并用 env.step 求平衡
        # ---------------------------------------------------------
        # 重要：env 是 numpy 物理；torch 只负责产生控制序列并保留计算图
        u_list = []         # torch tensor list, 每步控制 u_k (6D)
        q_last = None
        J_last = None
        info_last = None

        # reset, from λ=0 to 1
        env.reset()

        # 当前控制 u (torch)
        u = u0.clone()

        for k in range(N):
            lam = torch.tensor([[k / N]], dtype=torch.float32, device=device)  # shape (1,1)

            # 网络输出边界变化率（du/dλ）
            u_dot = net(lam).squeeze(0)  # shape (6,)

            # 同伦积分：u_{k+1} = u_k + Δλ * u_dot
            u = u + dlam * u_dot
            u_list.append(u)

            # 物理步进（准静态复现：use_inertia=False）
            u_np = u.detach().cpu().numpy()
            obs, done, info = env.step(u_np, use_inertia=False)

            # 保存最后一步（final shape）
            q_last = info["q"]
            J_last = info["J"]
            info_last = info

            if done and (k < N - 1):
                # 如果 Newton 失败提前终止，这个 epoch 直接跳过（也可以给惩罚）
                break

        if (info_last is None) or (not info_last["converged"]):
            print(f"[ep {ep}] Newton failed, skip.")
            writer.add_scalar("physics/newton_converged", 0.0, ep)
            continue

        # ---------------------------------------------------------
        # 计算 final loss（numpy）
        # ---------------------------------------------------------
        L_np, (mx, my) = final_midpoint_loss(
            q_last, env.nv, env.mid_node, target_xy
        )

        # ---------------------------------------------------------
        # 反向（关键）：隐式梯度 -> final dL/du_final (6D)
        # ---------------------------------------------------------
        free_index = info_last["free_index"]
        fixed_index = info_last["fixed_index"]

        # a_free = dL/dq_free
        a_free = dloss_dq_free_midpoint(q_last, free_index, env.nv, env.mid_node, target_xy)

        # final 控制的 numpy 版本
        u_final_np = u_list[-1].detach().cpu().numpy()

        # g_u_final = dL/du_final (6D)
        g_u_final = implicit_final_control_grad(
            J=J_last,
            free_index=free_index,
            fixed_index=fixed_index,
            a_free=a_free,
            control_6d=u_final_np,
            deltaL=env.deltaL
        )  # shape (6,)

        # ---------------------------------------------------------
        # 把物理梯度“注入”到 torch 计算图里
        #
        # 因为: u_final = u0 + Δλ * Σ_k u_dot_k
        # 所以对每步 u_dot_k:
        #    dL/du_dot_k = Δλ * dL/du_final
        #
        # 这里用一个 surrogate 标量 loss_torch 来触发 autograd:
        #    loss_torch = Σ_k <u_k, g_u_final>
        # 因为 d/d(u_k) loss_torch = g_u_final
        # 而 u_k 由网络产生（通过积分），能把梯度传回 Θ
        # ---------------------------------------------------------
        g = torch.tensor(g_u_final, dtype=torch.float32, device=device)  # shape (6,)

        opt.zero_grad()

        # 构造 surrogate loss（只用于反传，不代表真实物理loss）
        # 注意：每个 u_k 对最终边界都有 1 的系数（因为积分累加），因此这里直接对每步 u_k 累加
        # 等价于对每个 u_dot_k 乘以 dlam。
        loss_torch = 0.0
        for u_k in u_list:
            loss_torch = loss_torch + dlam * torch.dot(u_k, g)

        # 反传并更新
        loss_torch.backward()
        opt.step()

        # -----------------------------
        # TensorBoard logging
        # -----------------------------
        writer.add_scalar("loss/final", L_np, ep)

        writer.add_scalar("midpoint/x", mx, ep)
        writer.add_scalar("midpoint/y", my, ep)

        writer.add_scalar(
            "physics/newton_converged",
            float(info_last["converged"]),
            ep
        )

        # 隐式梯度范数（非常重要）
        g_norm = np.linalg.norm(g_u_final)
        writer.add_scalar("grad/gu_final_norm", g_norm, ep)

        # final 控制量大小
        u_final_norm = torch.norm(u_list[-1]).item()
        writer.add_scalar("control/u_final_norm", u_final_norm, ep)


        # 打印训练信息
        if ep % 10 == 0:
            print(f"[ep {ep:04d}] L(final)={L_np:.6f}  midpoint=({mx:.3f},{my:.3f})  target=({target_xy[0]:.3f},{target_xy[1]:.3f})")
    
    writer.close()

if __name__ == "__main__":
    main()
