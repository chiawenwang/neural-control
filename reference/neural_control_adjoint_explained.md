# Neural Control 讲解：怎样让 loss 穿过“平衡求解”反向传播

对照两份材料：
- 论文 *Neural Control: Adjoint Learning Through Equilibrium Constraints*（同目录下的 PDF）。下文公式编号都和论文一致。
- 本仓库代码，主要是 [scripts/train_adjoint.py](../scripts/train_adjoint.py) 和 [src/implicit_grad_tools.py](../src/implicit_grad_tools.py)。

文中的 `文件:行号` 都指当前代码。

---

## 0. 一句话和一张图

**一句话**：网络只负责输出“末端往哪动、动多快”。模拟器在每一步用 Newton 把杆解到平衡。反向时**不**穿过 Newton 的迭代，而是对“平衡条件 = 0”这个等式做隐式微分，得到 loss 对每一步控制的梯度，再交给 PyTorch 传回网络参数。

```
前向（每一步 k）
  λ_k ──网络──▶ r_k ──积分──▶ p_k ──c(·)──▶ q_b,k ──Newton 解平衡──▶ x_k ──▶ … ──▶ 损失 L
       （速率）       （末端位姿）  （被夹节点）  G(x, q_b) = 0     （自由节点）
  └──────── PyTorch 记录计算图 ────────┘ └──────── numpy，不在计算图里 ────────┘

反向
  ∂L/∂x ──伴随 Sᵀ──▶ ∂L/∂q_b ──Cᵀ──▶ ∂L/∂p_k ──surrogate.backward()──▶ ∂L/∂Θ ──Adam──▶ 更新 Θ
        └──────── 第 3–5 节手算（numpy）─────────┘ └──────── 第 6 节交给 PyTorch ────────┘
```

整个方法的核心就三件事：
1. **隐式微分**：不管 Newton 怎么走到平衡，只看平衡点随控制怎么变（第 3 节）。
2. **伴随递推**：loss 对状态的敏感度从最后一步往前传（第 5 节）。
3. **surrogate 技巧**：把手算出来的梯度“注入”PyTorch，让它接着往网络里传（第 6 节）。

RHC（第 9 节）是在这三件事外面再套一层：把长过程切成短段，一段一段地优化。

---

## 1. 符号对照（先看这张表）

| 含义 | 论文 | 代码 |
|---|---|---|
| 进度参数（相当于“时间”） | $\lambda\in[0,1]$，$\lambda_k=k/K$ | `lam = k / T`（train_adjoint.py:120） |
| 网络输出：末端运动的**速率** | $\mathbf u_k=\mathbf u_\Theta(\lambda_k)$，式 (3)(25) | `net(lam)`，`u_dot_full`（:121） |
| 可控量 | $\mathbf z$ = 边界自由度 $\mathbf q_b$ | 多一层：先积分出末端**位姿** $\mathbf p_k$（代码变量名是 `u`），再由 `_control_node_from_pose` 映射成被夹节点 $\mathbf q_b$ |
| 状态：自由节点 | $\mathbf x\equiv\mathbf q_f$ | `q[free_index]` |
| 平衡残差 | $\mathcal G=\nabla_{\mathbf x}E-\mathbf F^{\rm ext}$，式 (2) | newton.py:99 返回 $f=\nabla E-W$（$W$ 是重力，有接触时再减接触力）。`energy.force_and_jacobian` 给的是 $-\nabla E$，前面再取负号，所以 $f$ 就是论文的 $\mathcal G$ |
| 残差的雅可比 | $\mathbf G_x,\ \mathbf G_z$ | newton.py:89 的 `J` $=\partial f/\partial\mathbf q$：`J_ff` $=\mathbf G_x$，`J_fb` $=\mathbf G_z$ |
| 平衡敏感度 | $\mathbf S=\partial\mathbf x/\partial\mathbf z$，式 (5) | `local_dxf_dxb`（implicit_grad_tools.py:199） |
| 位姿 → 被夹节点的雅可比 | 论文没有这一层 | `_boundary_jacobian`，代码里叫 `B_i` / `J_map`；本文记作 $\mathbf C_k$ |
| 伴随（costate） | $\mathbf a_k=\partial\mathcal L/\partial\mathbf x_k$，式 (36) | `lam_f_cum` |
| 每一步的控制梯度 | $d\mathcal L/d\mathbf u_k$，式 (43) | `g_u_seq[k]`（对位姿 $\mathbf p_k$） |
| 网络参数 | $\Theta$ | `net.parameters()` |

> **名字撞车，注意别混。** 代码里的 `B_i` 是“位姿 → 被夹节点”的雅可比（本文写成 $\mathbf C_k$）。论文里的 $\mathbf B_k=\partial f/\partial\mathbf u$ 是另一个东西。在这个代码里网络直接输出速率，$f=\mathbf u$，所以论文的 $\mathbf B_k=\mathbf I$、$\mathbf A_k=\partial f/\partial\mathbf z=\mathbf 0$。

---

## 2. 反向传播补课

### 2.1 反向传播就是“从后往前用链式法则”

设 $y=f(x)$，$L=h(y)$，那么 $\dfrac{dL}{dx}=h'(y)\,f'(x)$。

做法是从 $L$ 出发，往回走一步就乘一个局部导数。多层、多步也一样，每经过一个运算就乘它的局部导数。

为什么要“从后往前”：loss 是一个**标量**。从它出发往回传一遍，就能同时得到它对所有参数的梯度。往回传的那个量就是“loss 对这一层的敏感度”。每一步做的运算叫**向量-雅可比积**（VJP）：$(\text{局部雅可比})^\top\times(\text{后面传来的敏感度})$。

PyTorch 的 autograd 就是自动做这件事：前向时记下每个运算，`backward()` 时按链式法则往回乘。

### 2.2 为什么这里不能直接 autograd

1. **平衡状态不在 PyTorch 的图里。** 每一步的 $\mathbf x_k$ 是 Newton 迭代几十次的结果，在 numpy 里算。送进模拟器之前控制已经 `detach()` 了（train_adjoint.py:127），PyTorch 的链条在 $\mathbf p_k$ 处就断了。
2. **就算放进 PyTorch，也不该展开 Newton 求导。** 每次 Newton 迭代都要存下来，内存随“迭代次数 × 步数”增长；而且在失稳附近 $\mathbf G_x$ 接近奇异，展开求导数值上很脆弱（论文第 1、2 节）。
3. **关键观察**：我们关心的不是“Newton 怎么走到平衡”，而是“**平衡点随控制怎么变**”。这只由平衡条件 $\mathcal G=0$ 本身决定，和用什么求解器、迭代几次都无关。所以对平衡条件做隐式微分就够了。

---

## 3. 核心一：单步隐式微分（论文 3.1 节，式 5–7、30–32）

### 3.1 推导

平衡条件对每一个控制 $\mathbf z$ 都成立：

$$\mathcal G(\mathbf x(\mathbf z),\mathbf z)=\mathbf 0$$

既然它恒等于 0，对 $\mathbf z$ 求导也是 0：

$$\mathbf G_x\,\frac{\partial\mathbf x}{\partial\mathbf z}+\mathbf G_z=\mathbf 0$$

于是得到论文式 (5)，离散形式是 (30)：

$$\mathbf S_k:=\frac{\partial\mathbf x_k}{\partial\mathbf z_k},\qquad \mathbf G_{x,k}\,\mathbf S_k=-\mathbf G_{z,k}\tag{30}$$

在弹性杆里，残差就是能量梯度（论文式 51）：

$$\mathcal G(\mathbf x,\mathbf z):=\nabla_{\mathbf q_f}E_{\rm tot}(\mathbf q_f,\mathbf q_b),\qquad \mathbf G_x=\nabla^2_{\mathbf q_f\mathbf q_f}E_{\rm tot},\qquad \mathbf G_z=\nabla^2_{\mathbf q_f\mathbf q_b}E_{\rm tot}\tag{51}$$

所以 $\mathbf G_x$ 就是自由节点那一块的刚度矩阵（Hessian），$\mathbf G_z$ 是自由节点和边界节点之间的耦合块。

### 3.2 一维例子

一个自由节点 $x$，左边用刚度 $k_1$ 的弹簧连到墙，右边用刚度 $k_2$ 的弹簧连到被控端 $z$：

$$E=\tfrac12k_1x^2+\tfrac12k_2(z-x)^2,\qquad \mathcal G=\frac{\partial E}{\partial x}=k_1x-k_2(z-x)$$

于是 $G_x=k_1+k_2$，$G_z=-k_2$，代入得 $S=-G_z/G_x=k_2/(k_1+k_2)$。

直接解平衡 $x=\dfrac{k_2}{k_1+k_2}z$ 也得到同一个 $S$。**整个过程完全不需要知道 $x$ 是怎么解出来的。**

### 3.3 伴随：只要 $\mathbf S^\top\mathbf v$，不必求出 $\mathbf S$

反向传播只需要“敏感度 $\times$ 向量”，即 $\mathbf S^\top\mathbf v$，其中 $\mathbf v$ 通常是 $\partial L/\partial\mathbf x$。

由 $\mathbf S^\top\mathbf v=-\mathbf G_z^\top\mathbf G_x^{-\top}\mathbf v$，先解一个线性方程，再乘一次（论文式 6–7、31–32）：

$$\mathbf G_{x,k}^\top\,\boldsymbol\eta_k=\mathbf v,\qquad \mathbf S_k^\top\mathbf v=-\mathbf G_{z,k}^\top\,\boldsymbol\eta_k\tag{31, 32}$$

这样每一步只解**一次**线性方程。如果显式求 $\mathbf S$，就要对每个边界自由度各解一次。这就是“**伴随法**”名字的由来：它是反向模式（VJP），而不是正向模式。

### 3.4 代码对应

- **`implicit_final_control_grad`**（implicit_grad_tools.py:166–196）就是式 (31)(32)：
  - `v = np.linalg.solve(J_ff.T, a_free)`（:187）
  - `g_qb = -(J_fb.T @ v)`（:190）
  - 最后再乘位姿雅可比：`g_u = J_map.T @ g_qb`（:194）
  - 代码里的 `J` 就是 $\partial\mathcal G/\partial\mathbf q$，所以前两行和式 (31)(32) 完全对应：`v` 就是 $\boldsymbol\eta$。
- **`local_dxf_dxb`**（:199–220）走的是另一条路：直接求出 $\mathbf S=-\mathbf J_{ff}^{-1}\mathbf J_{fb}$。
  - 边界自由度只有 12 个（3D），等于解一次带 12 个右端项的方程，代价不大。
  - 多步梯度 `multistep_adjoint_gradient` 用的是这个函数。
  - `jac_reg = 1e-6` 加在 $\mathbf J_{ff}$ 的对角线上，防止矩阵奇异。
- **雅可比从哪来**：每一步 Newton 收敛时顺手返回 `J`（newton.py:89；`env.step` 放进 `info["J"]`，SimulatorEnv_3D.py:447）。反向时直接用，不用重新组装 Hessian。

---

## 4. 核心二：多步，状态沿着路径一步步走（式 8–10、25–27、33–34）

### 4.1 前向 rollout

论文式 (25)–(27)：

$$\mathbf u_k=\mathbf u_\Theta(\lambda_k)\tag{25}$$

$$\mathbf z_{k+1}=\mathbf z_k+\Delta\lambda\,f(\mathbf z_k,\mathbf u_k)\tag{26}$$

$$\mathbf x_k={\rm SolveEq}(\mathbf z_k;\ {\rm init}=\mathbf x_{k-1}),\qquad \mathcal G(\mathbf x_k,\mathbf z_k)=\mathbf 0\tag{27}$$

代码 `rollout_segment`（train_adjoint.py:100–136）逐行对应：

| 论文 | 代码 |
|---|---|
| (25) $\mathbf u_k=\mathbf u_\Theta(\lambda_k)$ | `u_dot_full = cat([0, net(lam)])`（:121），左端固定，速率为 0 |
| (26) 积分 | `u = u + dlam * u_dot_full`（:122），得到位姿 $\mathbf p_k$ |
| (27) 从上一步出发解平衡 | `env.step(u_np)`（:128）→ Newton 从当前 `self.q` 出发（SimulatorEnv_3D.py:82） |

下标只差一位：代码先加第 $k$ 步的速率，再解出 $\mathbf x_k$；论文是 $\mathbf u_k\to\mathbf z_{k+1}\to\mathbf x_{k+1}$。

**为什么必须从上一步出发**：杆是多稳态的，同一个 $\mathbf z$ 可能对应好几个平衡形状（论文图 1B、图 3）。从上一步的解出发，Newton 才会落在“顺着走过来的那一支”上。这就是论文说的 *branch-consistent*。

### 4.2 代理动力学和冻结切线（只用于反向）

反向时需要知道“前面一步的小变化怎么影响后面”。论文用平衡敏感度把它线性化：

$$\delta\mathbf x_k\approx\mathbf S_k\,\delta\mathbf z_k\tag{33}$$

$$\mathbf x_{k+1}\approx\mathbf x_k+\Delta\lambda\,\mathbf S_k\,f(\mathbf z_k,\mathbf u_k)\tag{34}$$

它的连续形式是式 (10)：$\dfrac{d\mathbf x}{d\lambda}=\mathbf S(\lambda)\,f(\mathbf z(\lambda),\mathbf u(\lambda))$。

两点要分清：
- **前向不用 (34)**。前向每一步都老老实实解到收敛（式 27）。(34) 只是给反向用的近似模型。
- **冻结切线**：在每一小步里把 $\mathbf S_k$ 当常数，忽略 $\mathbf S$ 随 $\mathbf x$、$\mathbf z$ 的变化，这样就不用算平衡残差的二阶导。
  - 论文附录 C 的命题 C.2 给出误差界：$\|\nabla_\Theta\mathcal L_{\rm proxy}-\nabla_\Theta\mathcal L\|\le C\,\eta_S$，式 (60)。
  - 其中常数 $C$ 会随段长增大，这是要用 RHC 的原因之一（第 9 节）。

---

## 5. 核心三：反向传播详解：伴随、从后往前累加、一直传到 policy

这一节只讲反向，一步一步来。前向（第 4.1 节）你已经清楚了，这里只用它的结构。

### 5.1 要算的是什么

训练要的是 $d\mathcal L/d\Theta$：loss 对网络每一个权重的导数。有了它，就可以按 $\Theta\leftarrow\Theta-\eta\,d\mathcal L/d\Theta$ 更新（Adam 是它的改进版）。

先看规模：网络 1→64→64→5 一共约 4600 个权重，而 $\mathcal L$ 只是**一个数**。

再看 $\Theta$ 是**怎么影响** $\mathcal L$ 的。顺着前向画出依赖关系：

```
Θ ──▶ r_1, r_2, …, r_K          网络在每个 λ_k 输出一个速率
r_j ──▶ p_j, p_{j+1}, …, p_K     位姿是累加出来的：p_k = p_{k-1} + Δλ r_k
p_k ──▶ q_b,k ──▶ x_k            被夹节点 → Newton 解出平衡
x_k ──▶ ℓ_k                      每一步的 loss（点到点任务只有最后一步有）
L = ℓ_1 + ℓ_2 + … + ℓ_K
```

关键在第二行：**一个早期的速率 $\mathbf r_j$ 会影响它之后所有的位姿**，从而影响之后所有步的平衡和 loss。“从后往前累加”就是从这里来的。

### 5.2 为什么要“反向”算

**办法一：正向推扰动。** 把 $\mathbf r_j$ 加一点 $\varepsilon$，看它怎么一路影响到 $\mathcal L$：
- 位姿 $\mathbf p_j,\mathbf p_{j+1},\dots$ 都挪了 $\Delta\lambda\,\varepsilon$；
- 每一步的平衡 $\mathbf x_k$ 跟着变；
- 最后把各步 loss 的变化加起来。

问题是**每个要求导的量都得单独推一遍**：每一步、每个分量都要一遍，换成网络权重就要 4600 遍。第 8 节的有限差分就是这个办法的暴力版：每个数重跑两次完整的 Newton rollout。

**办法二：反向推敏感度。** 从 $\mathcal L$ 出发往回问：“$\mathcal L$ 对这个中间量有多敏感？”一路问到网络权重。

因为 $\mathcal L$ 只有一个数，**往回走一遍就能同时得到对所有量的导数**，代价和一次前向差不多。这就是反向传播，也叫反向模式。

### 5.3 伴随就是“loss 对某个中间量的敏感度”

给每个中间量 $\mathbf v$ 配一个同样大小的量

$$\bar{\mathbf v}:=\frac{\partial\mathcal L}{\partial\mathbf v}$$

它叫 $\mathbf v$ 的**伴随**（adjoint）。直白意思：把 $\mathbf v$ 挪一点 $\delta\mathbf v$，loss 大约变化 $\bar{\mathbf v}\cdot\delta\mathbf v$。

反向传播只用两条规则：

1. **穿过一个运算就乘它的转置。** 如果 $\mathbf y=\mathbf M\mathbf v$（或者局部线性化后是这样），那么 $\bar{\mathbf v}=\mathbf M^\top\bar{\mathbf y}$。
   理由：$\bar{\mathbf y}\cdot\delta\mathbf y=\bar{\mathbf y}\cdot\mathbf M\delta\mathbf v=(\mathbf M^\top\bar{\mathbf y})\cdot\delta\mathbf v$。
2. **一个量被用了几次，就把几条路的贡献加起来。** 如果 $\mathbf v$ 既流向 $\mathbf y_1$ 又流向 $\mathbf y_2$，那么 $\bar{\mathbf v}=\mathbf M_1^\top\bar{\mathbf y}_1+\mathbf M_2^\top\bar{\mathbf y}_2$。

**“从后往前累加”就是第 2 条规则。** 位姿 $\mathbf p_k$ 被用了两次：
- 这一步解平衡要用它；
- 下一步的位姿 $\mathbf p_{k+1}=\mathbf p_k+\Delta\lambda\,\mathbf r_{k+1}$ 是在它上面加出来的。

所以 $\mathbf p_k$ 的伴随 = 这一步的贡献 + $\mathbf p_{k+1}$ 的伴随。要知道 $\mathbf p_k$ 的，就得先知道 $\mathbf p_{k+1}$ 的，因此只能从最后一步开始往前算。

### 5.4 反向的五个环节

记号：$\bar{\mathbf x}_k$ 是 loss 对第 $k$ 步平衡状态的敏感度；$\bar{\mathbf p}_k$ 是“只通过第 $k$ 步的平衡”传到位姿的那部分；$\mathbf P_k$ 是 $\mathbf p_k$ 的**总**伴随（两条路都算上）。

**环节 A：loss → 平衡状态。** 对每一步，

$$\bar{\mathbf x}_k=\nabla_{\mathbf x}\ell_k$$

- 点到点任务只有最后一步有 loss：$\bar{\mathbf x}_K=2(\mathbf p_{\rm mid}-\mathbf p^\star)$，只放在中点那几个自由度上，其余步为 0。
- 轨迹跟踪任务每一步都有。

在这个准静态模型里，$\mathbf x_k$ 只被第 $k$ 步的 loss 用到（下一步的平衡由新的位姿决定，第 8 节的有限差分验证了这一点），所以这里不需要累加。

**环节 B：平衡状态 → 被夹节点（穿过 Newton，需要伴随方程）。**

$$\bar{\mathbf q}_{b,k}=\mathbf S_k^\top\,\bar{\mathbf x}_k$$

$\mathbf S_k=\partial\mathbf x_k/\partial\mathbf q_{b,k}$ 没有现成的公式，因为 $\mathbf x_k$ 是 Newton 解出来的。具体怎么算、为什么叫伴随，见第 5.6 节。

**环节 C：被夹节点 → 末端位姿。** 被夹节点是位姿的显式函数 $\mathbf q_b=c(\mathbf p)$，乘它的雅可比转置即可：

$$\bar{\mathbf p}_k=\mathbf C_k^\top\,\bar{\mathbf q}_{b,k},\qquad \mathbf C_k=\frac{\partial c}{\partial\mathbf p}\Big|_{\mathbf p_k}$$

**环节 D：位姿 → 速率（从后往前累加）。** 用第 2 条规则：

$$\mathbf P_K=\bar{\mathbf p}_K,\qquad \mathbf P_k=\bar{\mathbf p}_k+\mathbf P_{k+1}\quad(k=K-1,\dots,1)$$

因为 $\mathbf p_k=\mathbf p_{k-1}+\Delta\lambda\,\mathbf r_k$，$\mathbf r_k$ 只通过 $\mathbf p_k$ 进入，所以

$$\bar{\mathbf r}_k=\Delta\lambda\,\mathbf P_k$$

展开就是 $\bar{\mathbf r}_j=\Delta\lambda\sum_{k\ge j}\mathbf C_k^\top\mathbf S_k^\top\nabla\ell_k$：**第 $j$ 步的速率收到它之后每一步 loss 的贡献，每一份都用 loss 发生那一步自己的 $\mathbf S_k$**。

**环节 E：速率 → 网络权重。**

$$\frac{d\mathcal L}{d\Theta}=\sum_k\Big(\frac{\partial\mathbf r_k}{\partial\Theta}\Big)^{\!\top}\bar{\mathbf r}_k$$

$\partial\mathbf r_k/\partial\Theta$ 是网络在输入 $\lambda_k$ 处对权重的导数，这就是普通神经网络的反向传播（穿过 Linear、Tanh 层），由 PyTorch 自动完成。

```
前向：   r_1 ──▶ p_1 ──▶ p_2 ──▶ p_3            p_k = p_{k-1} + Δλ r_k
                  │        │        │
                  ▼        ▼        ▼
                 x_1      x_2      x_3           x_k = 平衡(p_k)
                  │        │        │
                  ▼        ▼        ▼
                 ℓ_1      ℓ_2      ℓ_3    ──▶ L = ℓ_1 + ℓ_2 + ℓ_3

反向：   r̄_1 ◀── P_1 ◀(+)── P_2 ◀(+)── P_3       D：P_k = p̄_k + P_{k+1}，r̄_k = Δλ P_k
                  ▲          ▲          ▲
                 p̄_1        p̄_2        p̄_3       C：p̄_k = C_kᵀ q̄_b,k
                  ▲          ▲          ▲
                 q̄_b,1      q̄_b,2      q̄_b,3     B：q̄_b,k = S_kᵀ x̄_k（伴随方程）
                  ▲          ▲          ▲
                 x̄_1        x̄_2        x̄_3       A：x̄_k = ∇ℓ_k
```

### 5.5 三步数值例子（可以自己跑）

全部取标量：$\mathbf C=1$，$\Delta\lambda=1/3$。

假设三步的平衡敏感度是 $S_1=0.5,\ S_2=0.4,\ S_3=0.3$（真实问题里它们由每一步的刚度矩阵决定，所以会随步变化），三步的 loss 梯度是 $\nabla\ell_1=0.2,\ \nabla\ell_2=-0.1,\ \nabla\ell_3=0.6$。

反向，从第 3 步开始：

| 步 k | A：$\bar x_k=\nabla\ell_k$ | B+C：$\bar p_k=S_k\bar x_k$ | D：$P_k=\bar p_k+P_{k+1}$ | $\bar r_k=\Delta\lambda P_k$ |
|---|---|---|---|---|
| 3 | 0.6 | 0.3 × 0.6 = 0.18 | 0.18 | 0.0600 |
| 2 | −0.1 | 0.4 × (−0.1) = −0.04 | −0.04 + 0.18 = 0.14 | 0.0467 |
| 1 | 0.2 | 0.5 × 0.2 = 0.10 | 0.10 + 0.14 = 0.24 | 0.0800 |

用正向推扰动检查 $\bar r_1$：$r_1$ 加 $\varepsilon$，会让 $p_1,p_2,p_3$ 都挪 $\varepsilon/3$，于是

$$\delta L=(0.2\times0.5-0.1\times0.4+0.6\times0.3)\,\varepsilon/3=0.08\,\varepsilon$$

和表里一致。脚本在 `/tmp/claude-1000/-home-sciarm20-neural-control/b509cc89-cdd3-4dd7-a7a6-b483396e4e85/scratchpad/backward_toy.py`，核心只有三行：

```python
pbar = S * dl                          # 环节 B+C：每一步自己的贡献
P = np.cumsum(pbar[::-1])[::-1]        # 环节 D：从最后一步往前累加
rbar = dlam * P                        # 速率的梯度
```

### 5.6 为什么穿过平衡这一步需要“伴随方程”

环节 B 要算 $\mathbf S_k^\top\bar{\mathbf x}_k$，但 $\mathbf x_k$ 是 Newton 迭代出来的，没有 $\mathbf x=F(\mathbf q_b)$ 这样的公式可以求导。能用的只有平衡条件本身。第 3.1 节已经推出：

$$\mathbf G_x\,\mathbf S=-\mathbf G_z$$

于是 $\mathbf S^\top\bar{\mathbf x}=-\mathbf G_z^\top\mathbf G_x^{-\top}\bar{\mathbf x}$。有两种算法：

1. **先求出 $\mathbf S$ 再乘**：$\mathbf G_x\mathbf S=-\mathbf G_z$ 的每一列都要解一次，边界有几个自由度就解几次。
2. **伴随法**：先解**一次**带转置的方程，再做一次乘法（论文式 31、32）：

$$\mathbf G_x^\top\,\boldsymbol\eta=\bar{\mathbf x},\qquad \bar{\mathbf q}_b=\mathbf S^\top\bar{\mathbf x}=-\mathbf G_z^\top\boldsymbol\eta$$

这个带转置的方程就叫**伴随方程**。“伴随”在线性代数里就是转置的意思：反向传播穿过 $\mathbf x=\mathbf S\mathbf q_b$ 要乘 $\mathbf S^\top$（第 5.3 节规则 1），而 $\mathbf S$ 里含 $\mathbf G_x^{-1}$，转置以后就变成解 $\mathbf G_x^\top$ 的方程。

所以“为什么反向需要伴随”，答案是两层：
- **伴随变量**（$\bar{\mathbf x},\bar{\mathbf p},\mathbf P$，论文的 $\mathbf a_k$）：反向传播本来就是在传“loss 对中间量的敏感度”。loss 只有一个数而参数很多，往回传最省。
- **伴随方程**：其中有一段（平衡）是隐式的，没法直接求导。往回穿过它时，只能借平衡条件解一个转置方程。它只用到收敛时的雅可比，所以不需要对 Newton 的迭代求导。

**物理直觉。** 弹性杆的 $\mathbf G_x$ 就是自由节点的刚度矩阵 $\mathbf K_{ff}$（对称，转置等于自己），$\mathbf G_z$ 是自由节点和边界节点之间的耦合刚度 $\mathbf K_{fb}$。于是伴随方程 $\mathbf K_{ff}\boldsymbol\eta=\bar{\mathbf x}$ 可以这样读：
- 把 loss 梯度 $\bar{\mathbf x}$ 当成一个力，加在自由节点上（边界夹住不动）；
- $\boldsymbol\eta$ 就是杆在这个力下的变形；
- $-\mathbf K_{fb}^\top\boldsymbol\eta$ 就是夹具为了夹住杆要提供的反力，它正好等于 loss 对边界位置的梯度。

用第 3.2 节的一维弹簧检查：$\eta=\bar x/(k_1+k_2)$，$-G_z\eta=k_2\bar x/(k_1+k_2)=S\,\bar x$，正确。

**代码里**，`implicit_final_control_grad`（implicit_grad_tools.py:187–190）就是这两行。`multistep_adjoint_gradient` 用的 `local_dxf_dxb`（:217）走的是第 1 种算法：一次解出 12 列的 $\mathbf S$ 再乘转置。结果相同，因为边界只有 12 个自由度，代价也差不多。

### 5.7 论文的写法：把累加放在状态上（式 35–44）

论文用同一个“多条路相加”的规则，只是把**记忆放在状态 $\mathbf x$ 上**。它的代理动力学 (34) 写成

$$\mathbf x_{k+1}\approx\mathbf x_k+\Delta\lambda\,\mathbf S_k\,f(\mathbf z_k,\mathbf u_k)$$

$\mathbf x_k$ 被下一步的 $\mathbf x_{k+1}$ 用到（$\partial\mathbf x_{k+1}/\partial\mathbf x_k=\mathbf I$），所以累加发生在状态的伴随 $\mathbf a_k$ 上，而不是位姿上。

先把损失写成统一形式（式 35），再定义伴随（式 36–37）：

$$\mathcal L=\phi(\mathbf x_K,\mathbf z_K)+\sum_{k=0}^{K-1}w_k\,\ell(\mathbf x_k,\mathbf z_k)\tag{35}$$

$$\mathbf a_k:=\frac{\partial\mathcal L}{\partial\mathbf x_k},\quad \mathbf g_k:=\frac{\partial\mathcal L}{\partial\mathbf z_k};\qquad \mathbf a_K=\nabla_{\mathbf x}\phi(\mathbf x_K,\mathbf z_K),\quad \mathbf g_K=\nabla_{\mathbf z}\phi(\mathbf x_K,\mathbf z_K)\tag{36, 37}$$

任务里的损失：

$$\mathcal L_{\rm pos}=\tfrac12\big\|\mathbf p_i(1)-\mathbf p_i^\star\big\|_2^2\tag{21}$$

$$\ell_{\rm traj}(\lambda)=\tfrac12\big\|\mathbf p_{\rm mid}(\lambda)-\mathbf p^\star_{\rm mid}(\lambda)\big\|_2^2,\qquad \mathcal L_{\rm traj}=\int_0^1\ell_{\rm traj}(\lambda)\,d\lambda\tag{22, 23}$$

代码里点到点的 loss 是 `node_tracking_loss`（env_factory.py:443），$L=\|\mathbf p_{\rm mid}-\mathbf p^\star\|^2$，没有 $\tfrac12$；梯度 `dloss_dq_free_node`（:450）。轨迹跟踪在 train_tracking.py:124–126。

一步的线性化（式 38–39）：

$$\frac{\partial\mathbf z_{k+1}}{\partial\mathbf z_k}=\mathbf I+\Delta\lambda\,\mathbf A_k,\qquad \frac{\partial\mathbf z_{k+1}}{\partial\mathbf u_k}=\Delta\lambda\,\mathbf B_k\tag{38}$$

$$\frac{\partial\mathbf x_{k+1}}{\partial\mathbf x_k}=\mathbf I,\qquad \frac{\partial\mathbf x_{k+1}}{\partial\mathbf z_k}\approx\Delta\lambda\,\mathbf S_k\mathbf A_k,\qquad \frac{\partial\mathbf x_{k+1}}{\partial\mathbf u_k}\approx\Delta\lambda\,\mathbf S_k\mathbf B_k\tag{39}$$

从 $k=K-1$ 往回到 $0$：

$$\mathbf a_k\approx\mathbf a_{k+1}+w_k\,\nabla_{\mathbf x}\ell(\mathbf x_k,\mathbf z_k)\tag{40}$$

$$\mathbf g_k\approx(\mathbf I+\Delta\lambda\,\mathbf A_k)^\top\mathbf g_{k+1}+\Delta\lambda\,\mathbf A_k^\top\big(\mathbf S_k^\top\mathbf a_{k+1}\big)+w_k\,\nabla_{\mathbf z}\ell(\mathbf x_k,\mathbf z_k)\tag{41}$$

$$\mathbf d_k:=\mathbf g_{k+1}+\mathbf S_k^\top\mathbf a_{k+1}\tag{42}$$

$$\frac{d\mathcal L}{d\mathbf u_k}\approx\Delta\lambda\,\mathbf B_k^\top\mathbf d_k\tag{43}$$

$$\frac{d\mathcal L}{d\Theta}=\sum_{k=0}^{K-1}\Big(\frac{\partial\mathbf u_\Theta(\lambda_k)}{\partial\Theta}\Big)^{\!\top}\frac{d\mathcal L}{d\mathbf u_k}\tag{44}$$

逐条对应第 5.4 节的五个环节：
- **(40)** 是环节 A 加上累加：$\mathbf a_k$ = 第 $k$ 步以后所有 loss 对状态的敏感度之和。
- **(42)** 的 $\mathbf S_k^\top\mathbf a_{k+1}$ 是环节 B，用伴随方程 (31)(32) 算。
- **(43)** 是环节 D 的最后一步（$\Delta\lambda$）。在这个代码里网络直接输出速率，$f=\mathbf u$，所以 $\mathbf A_k=\mathbf 0$、$\mathbf B_k=\mathbf I$；loss 不直接依赖 $\mathbf z$，所以 $\mathbf g\equiv\mathbf 0$，(43) 化简为 $\Delta\lambda\,\mathbf S_k^\top\mathbf a_{k+1}$。
- **(44)** 是环节 E。

**和第 5.4 节的区别只有一处。** 论文把后面所有 loss 先在状态上加成 $\mathbf a_{k+1}$，再统一乘**当前这一步**的 $\mathbf S_k$，这就是“冻结切线”。第 5.4 节则是每份 loss 各自乘**它发生那一步**的 $\mathbf S$。$\mathbf S$ 不随步变化时两者完全相同。

在第 5.5 节的例子里，$S$ 从 0.5 变到 0.3，变化很大：

| | $\bar r_1$ | $\bar r_2$ | $\bar r_3$ |
|---|---|---|---|
| 正向扰动（真值） | 0.0800 | 0.0467 | 0.0600 |
| 第 5.4 节（累加在位姿上） | 0.0800 | 0.0467 | 0.0600 |
| 论文冻结切线：$\Delta\lambda\,S_k\sum_{i\ge k}\nabla\ell_i$ | 0.1167 | 0.0667 | 0.0600 |
| 代码默认 branch：两边都累加 | 0.2433 | 0.1267 | 0.0600 |

真实的杆在控制平滑时 $\mathbf S$ 变化很慢，冻结切线只差约 1%（第 8.2 节）。

代码默认的 branch 模式在环节 B 里先像论文那样把 loss 累加到 $\mathbf a$ 上（`lam_f_cum`），环节 D 里 autograd 又在位姿上累加一次，所以多加了一遍（第 8 节）。

### 5.8 连续形式（论文 3.1–3.2 节，供对照）

把步长取到无穷小，求和变积分，递推变成往回积的常微分方程。终端目标：

$$\mathbf a(1)=\nabla_{\mathbf x}\phi,\quad \mathbf g(1)=\nabla_{\mathbf z}\phi\tag{12}$$

$$-\frac{d\mathbf a}{d\lambda}=\mathbf 0,\qquad -\frac{d\mathbf g}{d\lambda}=\mathbf A^\top\big(\mathbf g+\mathbf S^\top\mathbf a\big)\tag{13}$$

$$\frac{d\mathcal L}{d\mathbf u(\lambda)}=\mathbf B^\top\big(\mathbf g+\mathbf S^\top\mathbf a\big),\qquad \frac{d\mathcal L}{d\Theta}=\int_0^1\Big(\frac{\partial\mathbf u_\Theta(\lambda)}{\partial\Theta}\Big)^{\!\top}\frac{d\mathcal L}{d\mathbf u(\lambda)}\,d\lambda\tag{14, 15}$$

轨迹目标（式 16）$\mathcal L=\phi+\int_0^1w\,\ell\,d\lambda$ 只改 (13)，变成 (17)：

$$-\frac{d\mathbf a}{d\lambda}=w\,\nabla_{\mathbf x}\ell,\qquad -\frac{d\mathbf g}{d\lambda}=w\,\nabla_{\mathbf z}\ell+\mathbf A^\top\big(\mathbf g+\mathbf S^\top\mathbf a\big)\tag{17}$$

$-d\mathbf a/d\lambda=w\nabla_{\mathbf x}\ell$ 就是式 (40) 的连续版：从 $\lambda=1$ 往回积分，一路把 loss 加进来。

### 5.9 五个环节在代码里的位置

| 环节 | 做什么 | 谁来算 | 代码 |
|---|---|---|---|
| A | $\bar{\mathbf x}_k=\nabla\ell_k$ | numpy | `a_free_per_step`（train_adjoint.py:164–165；train_tracking.py:126） |
| B | $\bar{\mathbf q}_{b,k}=\mathbf S_k^\top\bar{\mathbf x}_k$ | numpy，要用每一步存下的 `J_list[k]` | `local_dxf_dxb` + `dxf_dxb_i.T @ …`（implicit_grad_tools.py:310、313） |
| C | $\bar{\mathbf p}_k=\mathbf C_k^\top\bar{\mathbf q}_{b,k}$ | numpy | `B_i.T @ g_qb_i`（:311、314），结果是 `g_u_seq[k]` |
| D | $\mathbf P_k=\bar{\mathbf p}_k+\mathbf P_{k+1}$，$\bar{\mathbf r}_k=\Delta\lambda\mathbf P_k$ | **PyTorch 自动做** | 反向穿过 `u = u + dlam * u_dot_full`（train_adjoint.py:122），由 surrogate 触发（:174–179） |
| E | $d\mathcal L/d\Theta=\sum(\partial\mathbf r_k/\partial\Theta)^\top\bar{\mathbf r}_k$ | PyTorch 自动做 | 反向穿过 `net(lam)`（:121），结果进各权重的 `.grad`；然后裁剪（:180）和 Adam（:181） |

两点补充：
- **环节 D 的“从后往前累加”在代码里不是手写的。** autograd 反向穿过 `u = u + dlam * u_dot_full` 这一串加法时，正好按 $\mathbf P_k=\bar{\mathbf p}_k+\mathbf P_{k+1}$ 逐步往前加。第 6 节的 surrogate 技巧就是为了让 autograd 接手 D 和 E。
- **反向需要前向存下的东西**：每一步的平衡状态和雅可比 `J_list`。这是论文表 1 里伴随法的内存开销；分段（RHC）能缩短要存的长度。

---

## 6. 把梯度交给 PyTorch：surrogate 技巧（train_adjoint.py:167–181）

手算部分（numpy）在第 5 节结束时，得到的是每一步 loss 对位姿的梯度 $\mathbf g_k\approx\partial L/\partial\mathbf p_k$（代码里的 `g_u_seq[k]`）。

剩下的链条是“位姿 ← 速率积分 ← 网络 ← 参数”，全在 PyTorch 的图里。代码用一个小技巧把手算的梯度接上去：

```python
surrogate = sum(torch.dot(u_k, g_k) for k ...)   # :174-177，u_k 带计算图，g_k 当常数
surrogate.backward()                              # :179
clip_grad_norm_(net.parameters(), 1.0)            # :180
opt.step()                                        # :181，Adam
```

**为什么这样就对**：对 $\Theta$ 求导时 $\mathbf g_k$ 是常数，所以

$$\frac{\partial}{\partial\Theta}\sum_k\mathbf p_k\cdot\mathbf g_k=\sum_k\Big(\frac{\partial\mathbf p_k}{\partial\Theta}\Big)^{\!\top}\mathbf g_k=\frac{dL}{d\Theta}$$

这正是链式法则，也就是式 (44)/(15)。

surrogate 的**数值本身没有意义**，只有它的梯度有用。

autograd 在这里自动完成两件事：
1. 因为 $\mathbf p_k=\mathbf p_0+\Delta\lambda\sum_{j\le k}\mathbf r_j$，它自动算出 $dL/d\mathbf r_j=\Delta\lambda\sum_{k\ge j}\mathbf g_k$。
2. 再穿过 MLP，得到对每个权重的梯度。

标量版：$L(\Theta)=h(p(\Theta))$，已知 $g=h'(p)$。令 surrogate $=p(\Theta)\,g$，则 $\dfrac{d}{d\Theta}=p'(\Theta)\,g=\dfrac{dL}{d\Theta}$。

---

## 7. 代码里的反向循环，逐行看（implicit_grad_tools.py:307–314）

```python
for i in range(T - 1, -1, -1):                       # 从最后一步往前
    lam_f_cum = lam_f_cum + a_free_per_step[i]        # a_i：把第 i 步的 ∇ℓ 累加进伴随（式 40）
    dxf_dxb_i = local_dxf_dxb(J_list[i], ...)         # S_i = −J_ff⁻¹ J_fb（式 30）
    B_i = _boundary_jacobian(u_seq[i], ...)           # C_i：位姿 → 被夹节点
    g_qb_i = dxf_dxb_i.T @ lam_f_cum                  # S_iᵀ a_i：换到边界节点（式 32/42）
    g_u_seq[i] = B_i.T @ g_qb_i                       # C_iᵀ(…)：换到末端位姿
```

参数 `accumulate=False` 时，第一行变成 `lam_f_cum = a_free_per_step[i]`：只用这一步自己的 loss，不往回累加。默认 `accumulate=True`，配置里叫 `branch` 模式。两者的区别见第 8 节。

一次完整更新（不分段模式，train_adjoint.py:269–303）：

| 步骤 | 做什么 | 行号 |
|---|---|---|
| 1 | 恢复起点 `env.set_state(checkpoint)` | :271 |
| 2 | rollout：λ → 速率 → 位姿 → Newton；存下每步的 `q` 和 `J` | :274 |
| 3 | 算 loss 和 $\mathbf a_K$ | :284–287 |
| 4 | 手算反向：`multistep_adjoint_gradient` | :290–294 |
| 5 | surrogate + `backward()` | :296–301 |
| 6 | 梯度裁剪 + Adam | :302–303 |

Newton 没收敛的 rollout 直接跳过，不更新（:275–280）。

---

## 8. 实测：这个梯度是不是真的梯度？

### 8.1 测试方法

用默认配置（3D、soft_strip、51 个节点），跑 10 步 rollout，速率直接固定，不经过网络。然后用三种方法算 $dL/d\mathbf r_k$，和**有限差分**（把第 $k$ 步的速率加减 0.01，重跑整个 rollout）比较：

- **local**：`accumulate=False`
- **branch**：`accumulate=True`，train_adjoint.py 的默认
- **论文冻结切线**：式 (43)，$\mathbf S$ 取第 $k$ 步自己的

模拟器和梯度代码都没改。测试脚本：`/tmp/claude-1000/-home-sciarm20-neural-control/b509cc89-cdd3-4dd7-a7a6-b483396e4e85/scratchpad/adjoint_check.py`。这是临时目录，需要的话可以挪进 `scripts/`。

### 8.2 平滑控制（每一步速率相同，接近刚初始化的网络）

| 步 k | 分量 | 有限差分 | branch（默认） | local | 论文冻结切线 |
|---|---|---|---|---|---|
| 0 | x 速率 | 1.6899e-02 | 1.6870e-01 | 1.6899e-02 | 1.6772e-02 |
| 0 | z 速率 | −2.2413e-04 | −2.2435e-03 | −2.2413e-04 | −2.2060e-04 |
| 5 | x 速率 | 1.6899e-02 | 8.4474e-02 | 1.6899e-02 | 1.6889e-02 |
| 5 | z 速率 | −2.2413e-04 | −1.1232e-03 | −2.2413e-04 | −2.2514e-04 |
| 9 | x 速率 | 1.6899e-02 | 1.6899e-02 | 1.6899e-02 | 1.6899e-02 |
| 9 | z 速率 | −2.2413e-04 | −2.2413e-04 | −2.2413e-04 | −2.2413e-04 |

每一步 branch / local 的比值：9.98, 8.99, 7.99, 7, 6, 5, 4, 3, 2, 1。

### 8.3 剧烈变化的控制（每一步速率随机，σ = 0.3）

| 步 k | 分量 | 有限差分 | branch（默认） | local | 论文冻结切线 |
|---|---|---|---|---|---|
| 0 | x 速率 | −4.5612e-02 | **+1.2365e-01** | −4.5612e-02 | +1.0848e-02 |
| 0 | z 速率 | −2.0422e-02 | −1.8924e-01 | −2.0422e-02 | −2.4041e-02 |
| 5 | x 速率 | −4.5612e-02 | −2.7463e-01 | −4.5612e-02 | −5.4062e-02 |
| 5 | z 速率 | −2.0422e-02 | −7.6326e-02 | −2.0422e-02 | −1.2678e-02 |
| 9 | x 速率 | −4.5612e-02 | −4.5612e-02 | −4.5612e-02 | −4.5612e-02 |
| 9 | z 速率 | −2.0422e-02 | −2.0422e-02 | −2.0422e-02 | −2.0422e-02 |

### 8.4 怎么理解

**有限差分对每一步都给出同一个数。** 说明在这一段里，终点形状只取决于最后的末端位姿，不取决于怎么走到那里：准静态、没有跳到别的分支，和第 5.4 节的小例子一样。

写成公式，三种算法的区别在于：用哪一步的 $\mathbf S$，以及 loss 被加了几次（$\mathbf C$ 是位姿雅可比）：

**local**：

$$\frac{dL}{d\mathbf r_j}=\Delta\lambda\sum_{k\ge j}\mathbf C_k^\top\mathbf S_k^\top\nabla\ell_k$$

**论文冻结切线**：

$$\frac{dL}{d\mathbf r_j}\approx\Delta\lambda\,\mathbf C_j^\top\mathbf S_j^\top\sum_{k\ge j}\nabla\ell_k$$

**branch（默认）**：

$$\frac{dL}{d\mathbf r_j}=\Delta\lambda\sum_{k\ge j}\mathbf C_k^\top\mathbf S_k^\top\sum_{i\ge k}\nabla\ell_i$$

- **local 和有限差分完全一致**（两个测试都一样）。它在每个 loss 发生的那一步用那一步的 $\mathbf S$，对这个准静态、无摩擦的模型是精确的。
- **论文的冻结切线**把后面所有步的 $\mathbf S_k$ 都换成第 $j$ 步的 $\mathbf S_j$。
  - 控制平滑时 $\mathbf S$ 变化慢，误差约 1%（8.2）。这和论文的结论一致。
  - 控制剧烈变化时 $\mathbf S$ 沿路变化大，早期步的误差就大了（8.3）。这也是论文要用短段 RHC 的原因。
- **branch 多加了一次**：
  - 反向循环里的 `lam_f_cum` 已经把后面所有 loss 累加进来了，对应式 (40)；
  - 结果又被当成对累积位姿的梯度，autograd 的积分 $\mathbf p_k=\mathbf p_0+\Delta\lambda\sum\mathbf r$ 再累加一次。
  - 于是第 $k$ 步大约被放大 $T-k$ 倍（8.2 里正好是 10, 9, …, 1）。
  - 控制平滑时符号还是对的，所以训练仍然能下降，只是各步的权重被扭曲了，早期步权重过大；控制剧烈变化时，早期步甚至会反号（8.3 第一行）。

### 8.5 代码现状

- [configs/default.yaml](../configs/default.yaml) 第 393–401 行的注释已经写明了两种模式：`train.adjoint_mode: branch | local`。
  - 但只有 [scripts/train_pipe.py](../scripts/train_pipe.py) 读这个设置（:340）。
  - `train_adjoint.py` 和 `train_tracking.py` 总是用 branch。
- 本文没有修改任何代码。

---

## 9. RHC：分段优化（论文 3.3 节，式 18–20，算法 1）

**为什么要分段**：
- 长过程里，冻结切线的误差会累积（第 8 节）；
- 一点误差就可能把热启动的 Newton 推到另一个平衡分支上。

所以把整个过程切成短段，每段单独优化，然后从**真实达到的**平衡出发开始下一段。

每段用一个新的网络，局部进度 $\lambda\in[0,1]$ 分成 $H$ 步：

$$\mathbf u(\lambda)=\mathbf u_{\Theta^{(m)}}(\lambda)\tag{18}$$

$$\min_{\Theta^{(m)}}\ \mathcal L^{(m)}=\phi\big(\mathbf x^{(m)}(1),\mathbf z^{(m)}(1)\big)+\int_0^1w(\lambda)\,\ell\big(\mathbf x^{(m)}(\lambda),\mathbf z^{(m)}(\lambda)\big)\,d\lambda\tag{19}$$

这一段优化完、真正执行一遍以后，把终点交给下一段：

$$\mathbf x_0^{(m+1)}\leftarrow\mathbf x^{(m)}(1),\qquad \mathbf z_0^{(m+1)}\leftarrow\mathbf z^{(m)}(1)\tag{20}$$

代码对应（train_adjoint.py:335–393，配置里打开 `rhc.enabled`）：

| 算法 1 | 代码 |
|---|---|
| 切段 | `segment_lengths(N, num_segments)`（:94）。默认 N = 200 步、5 段，每段 40 步 |
| 每段重新初始化 Θ | `reinit_net_(net)` + 新的 Adam（:352–354） |
| 段内 $\Delta\lambda=1/H$ | `dlam = 1.0 / T_seg`（:357）；λ 在每段内从 0 重新开始（:120） |
| 段内更新 $T_{\rm seg}$ 次 | `optimize_segment`（:139–195）：每次都从同一个 `checkpoint` 出发 rollout → loss → 伴随 → 更新 |
| 执行一次并交接，式 (20) | :366–371：重跑一遍、`checkpoint = env.get_state()`，并把末端位姿带到下一段 |

每一段的目标都是“这段结束时中点到达最终目标”（:162）。

分段还有两个附带好处：
- 反向只穿过一段，要存的 `J_list` 变短；
- 冻结切线的误差只在一段内累积（论文附录 C 末尾）。

---

## 10. 论文设置与本仓库默认的差别

| | 论文 | 本仓库默认 |
|---|---|---|
| 维度 | 2D 弹性条，不含扭转 | 3D（`env.dim: 3`），可开扭转 |
| 节点数 | 101 | `nv: 51` |
| 重力 | 关（g = 0） | 开 `[0, 0, −9.81]` |
| 平衡求解器 | 信赖域 Newton–CG（附录 B） | 非单调线搜索 + Tikhonov 正则的 Newton（newton.py） |
| 网络 | MLP [64,64] 或 [64,64,64]，ReLU，输出用 tanh 限幅 | Linear–Tanh–Linear–Tanh–Linear，宽 64，输出不限幅（train_adjoint.py:47） |
| 优化器 | Adam，lr 1e-2 | Adam，lr 1e-3，加梯度裁剪 1.0 |
| 步数 | 不分段 K = 101；RHC 每段 H = 10 | N = 200；RHC 默认关，打开后 5 段 |
| 求 $\mathbf S^\top\mathbf v$ | 每步一次伴随解，式 (31) | 直接求出 $\mathbf S$（12 列） |
| 点到点 loss | $\tfrac12\|\cdot\|^2$ | $\|\cdot\|^2$ |
| 多步梯度 | 式 (40)–(43)，冻结切线 | branch（`train_adjoint.py` 固定用它）；local 只有 `train_pipe.py` 可选 |

---

## 11. 回顾：一次训练迭代里梯度走过的路

1. **前向**：网络把 $\lambda_k$ 变成速率 $\mathbf r_k$，积分成位姿 $\mathbf p_k$，映射成被夹节点 $\mathbf q_b$，Newton 解出平衡 $\mathbf x_k$。每步存下 $\mathbf x_k$ 和雅可比 $\mathbf J_k$。
2. **loss**：只看平衡状态，比如中点和目标的距离。
3. **伴随（numpy）**：
   - 从最后一步往前，把每步的 $\nabla\ell$ 累加成 $\mathbf a$；
   - 用 $\mathbf S^\top$ 换到边界节点（这一步就是对平衡条件的隐式微分，只用到收敛时的 $\mathbf J$）；
   - 再用 $\mathbf C^\top$ 换到位姿，得到每步的 $\mathbf g_k$。
4. **注入 PyTorch**：`surrogate = Σ p_k·g_k`，`backward()` 让 autograd 完成“位姿 ← 积分 ← 网络 ← Θ”这一段。
5. **更新**：梯度裁剪 + Adam。分段（RHC）时，每段重复以上步骤若干次，然后执行并交接。

整个过程**从没有对 Newton 的迭代求导**。Newton 只负责把每一步解准；所有导数都来自“平衡条件 = 0”这一个等式。
