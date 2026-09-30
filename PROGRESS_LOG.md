# Progress Log

工作记录：框架理解、已完成的改造、验证结果。会议本身的讨论记在 `MEETING_NOTES.md`。

| 日期 | 内容 | 状态 |
|---|---|---|
| 2026-09-14 | 梳理现有框架运行逻辑（控制 → 物理 → 伴随法梯度） | 笔记 |
| 2026-09-15 | 自定义 energy 开关（基类 + 注册表 + 配置项） | 已完成 |
| 2026-09-21 | 3D 环境、全局化 Newton 求解、BLAS 线程修复 | 已完成 |
| 2026-09-22 | 训练结束输出整段控制过程的 GIF 动画（配置开关） | 已完成 |
| 2026-09-22 | 材料预设选择器（`rod.material` + `materials:` 注册表） | 已完成 |
| 2026-09-28 | 3D 杆的扭转能量 + 夹爪 yaw 控制（`rod.twist`） | 已完成 |
| 2026-09-29 | 计算冗余审计（`--audit`）：找出内层循环里重复算了多少 | 已完成 |

---

## 2026-09-14 — 现有框架运行逻辑

### 0. 一句话概括

> **让一根弹性杆的两端按某种方式移动，使杆的中点最终停在目标位置。**
> 两端怎么动，由一个小神经网络决定；杆被两端拽成什么形状，由物理求解器算出来。
> 训练要解决的核心问题是：**"中点差了 2 厘米"这个误差，怎么回传到神经网络的权重上？**

一个容易上手的类比：你两只手各捏住一根钢丝的一端，想让钢丝的**中间**那一点落在空中某个指定位置。
你不能直接去碰中点——只能动两只手，钢丝自己会弯成它该弯的样子。神经网络就是在学"两只手怎么动"。

### 1. 先搞清楚几个名词

| 名词 | 是什么 | 在代码里 |
|---|---|---|
| `q` | 杆的形状。51 个节点，每个节点 (x,y)，拉直成一个 102 维向量 | `env.q`，长度 `2*nv` |
| **边界 DOF** | 被两只手"钉死"的 8 个数（每端 2 个节点 × 2 坐标） | `fixed_index` |
| **自由 DOF** | 剩下 94 个数，由物理自己决定 | `free_index` |
| `u` | 控制量，6 维：`[x_l, y_l, th_l, x_r, y_r, th_r]`，即两端的位置+朝向 | 网络的输出积分而来 |
| `lambda` | 归一化时间，从 0 走到 1。网络的**唯一输入** | `lam = k / T` |
| `J` | 切线刚度阵（能量的二阶导）。既用于 Newton 迭代，又用于求梯度 | `objfun` 的返回值 |
| `mid_node` | 我们关心的那个中点，`nv // 2` | `env.mid_node` |

### 2. 整体只有三件事，循环往复

```
     ┌──────────────────────────────────────────────────┐
     │                                                  │
  ①  网络说"手该怎么动"  →  ②  物理算"杆弯成什么样"  →  ③  比对目标算误差
     │                                                  │
     └────────────  ③ 的误差回传去改网络权重  ←──────────┘
```

- ① 便宜、可微（torch）
- ② 贵、**不可微**（numpy 里的 Newton 迭代）  ← 难点就在这
- ③ 用**伴随法**（adjoint）绕过 ② 的不可微，见第 6 节

### 3. 第 ① 步：网络怎么产生控制

网络非常小：输入一个标量 `lambda`，输出 3 个数（只控右端，左端全程不动）。

```
u = u0                                    # 起始姿态，比如右端在 (1, 0, 0)
for k in range(T):                        # T 步，比如 200 步
    lam   = k / T                         # 0 → 1
    速度  = net(lam)                       # 3 维：右端的 (dx, dy, dtheta) 速率
    u     = u + (1/T) * [0,0,0, 速度]      # 积分成"绝对位置"，左端补 0
    存下 u
```

要点：
- 网络输出的是**速度**不是位置，位置靠累加得到 —— 这样轨迹天然是连续的。
- 这个累加发生在 **torch 计算图里**，所以 "权重 → u_k" 这一段 autograd 自己会求导，不用我们管。
- `lambda` 只是"进度条"。RHC 模式下每一段的进度条都重新从 0 开始。

### 4. 第 ② 步：6 个数怎么变成一根弯曲的杆

**4a. 6 维控制 → 8 个边界 DOF**（`_control_node_from_xytheta`）

杆是一串点，没有"角度"这个量。要表达右端朝向 `th_r`，办法是**多钉一个邻居节点**：

```
node_0    = (x_l, y_l)                                   ← 左端点
node_1    = (x_l + cos(th_l)*dL,  y_l + sin(th_l)*dL)    ← 左端邻居，编码朝向
node_n-2  = (x_r - cos(th_r)*dL,  y_r - sin(th_r)*dL)    ← 右端邻居，编码朝向
node_n-1  = (x_r, y_r)                                   ← 右端点
```

这 4 个节点 = 8 个数，就是被钉死的边界。这个 6→8 映射的雅可比矩阵（8×6）后面求梯度还要用，
就是 `boundary_node_jacobian()`。

**4b. 解平衡：杆自己弯成能量最低的形状**（`objfun`）

物理假设是**准静态**：每一步都当作杆已经完全静止在平衡态（惯性忽略）。
平衡 = 合力为零，用 Newton 迭代解：

```
q = 上一步的形状                     # 热启动，所以迭代通常两三步就收敛
q[边界 8 个 DOF] = control_node       # 手的位置，直接钉死

重复直到 ||残差|| < tol:
    弹性力   F_elastic = -dE/dq       # ← 【自定义 energy 唯一要插进来的地方】
    切线刚度 J_elastic = -d²E/dq²
    接触力   F_contact                 # 障碍物罚函数，没有障碍物时为 0

    残差 f = -F_elastic - F_contact - W       # W 是重力
    刚度 J = -J_elastic - J_contact

    # 只解自由 DOF —— 边界是被钉死的，不参与求解
    Δ = solve( J[自由,自由],  f[自由] )
    q[自由] -= Δ

返回 q（新形状）, J（完整的刚度阵，含边界那些行列）
```

> **为什么返回完整的 J 而不只是 `J[自由,自由]`？**
> 因为第 6 节求梯度时需要 `J[自由,边界]` 这一块 —— 它描述"手动一点点，杆的其余部分跟着动多少"。

### 5. 第 ③ 步：误差怎么定义

```
point-to-point（train_adjoint.py）：只有最后一步有误差
    L = ||q[mid_node] - target||²
    a_free = dL/dq[自由]  = 2*(中点 - 目标)，填在中点对应的那两个位置，其余全 0

tracking（train_tracking.py）：每一步都有误差，L = Σ_k ||q_k[mid] - target_k||²
    区别仅仅是 a_free 从"只有最后一行非零"变成"逐行填满"
```

`a_free` 是一个 (T, 94) 的矩阵：第 k 行 = 第 k 步的误差对第 k 步自由 DOF 的导数。
**这是唯一需要用户定义的东西**，剩下的推导全是通用的。

### 6. 关键难点：误差怎么穿过不可微的 Newton 迭代

Newton 迭代是一个 while 循环，里面有解线性方程组、有收敛判据，autograd 根本没法追踪。
解决办法是**不去追踪它**，而是用数学直接写出答案。

**6a. 隐函数定理**

平衡态满足 `残差(q_自由, q_边界) = 0`。两边对 `q_边界` 求导：

```
J_ff · (dq_自由/dq_边界) + J_fb = 0
=>   dq_自由/dq_边界 = -J_ff⁻¹ · J_fb            ← local_dxf_dxb()，形状 (94, 8)
```

**不需要知道 Newton 怎么迭代的**，只要知道它最后停在了平衡点上，这个导数就成立。
这是整个框架能训练起来的根本原因。

**6b. 链式法则串起来**

```
误差 L  ←  q_自由  ←  q_边界  ←  u（6维控制）
          └ a_free ┘ └ dxf_dxb ┘ └ boundary_node_jacobian ┘

dL/du_k = B_kᵀ · dxf_dxbᵀ · (累积的 a_free)
```

**6c. 为什么要"反向扫 + 累积"**

第 k 步的手位一动，不只影响第 k 步的形状 —— 后面每一步都是从它热启动的，影响会一直传下去。
所以第 k 步要为**它自己以及之后所有步**的误差负责：

```
lam_cum = 0
for i = T-1 downto 0:              # 从最后一步倒着扫
    lam_cum += a_free[i]           # 把第 i 步的误差累加进来
    dxf_dxb  = local_dxf_dxb(J_list[i])       # 用第 i 步【自己】的刚度阵
    B        = boundary_node_jacobian(u[i])
    g_u[i]   = Bᵀ @ (dxf_dxbᵀ @ lam_cum)
```

> 这里的物理假设（沿用原论文代码）：第 i 步的边界扰动 δq_b 会**近似不变地**传播到之后每一步。
> 所以误差往前累积，但折算时每一步用自己的局部刚度阵。

### 7. 最后一环：numpy 的梯度怎么接回 torch

第 6 节在 numpy 里算出了 `dL/du_k`（一堆数）。但我们要改的是**网络权重**，还差 `du_k/d权重` 这一段。
这段在 torch 图里是现成的，所以用一个技巧把两边缝起来：

```
surrogate = Σ_k  dot( u_k , g_u[k] )      # u_k 带梯度；g_u[k] 当常数塞进去
surrogate.backward()
```

因为 `g_u[k]` 是常数，`d(surrogate)/d权重 = Σ_k g_u[k] · du_k/d权重`，**正好就是真正的 `dL/d权重`**。
`surrogate` 这个数本身没有物理意义（不是损失值），它只是一个"梯度的搬运工"。

### 8. 训练循环：把上面全部串起来

```
cfg = load_config(yaml)
env = SimulatorEnv_2D(...);  env.reset()
net = MLP(1 -> 3);  opt = Adam(net.parameters())
checkpoint = env.get_state()                     # 存档

for ep in range(epochs):
    env.set_state(checkpoint)                    # ← 每个 epoch 都读档重来

    # --- 前向 ---
    u_list, q_list, J_list = [], [], []
    u = u0
    for k in range(T):
        u = u + (1/T) * net(k/T)                 # ① 见第 3 节
        obs, done, info = env.step(u.numpy())    # ② 见第 4 节
        存 u, info["q"], info["J"]
        if not info["converged"]: 本轮作废，跳过

    # --- 误差 ---
    L, a_free = 算误差(q_list, target)            # ③ 见第 5 节

    # --- 梯度 ---
    g_u = multistep_adjoint_gradient(J_list, a_free, ...)    # 见第 6 节
    surrogate = Σ dot(u_list[k], g_u[k])                     # 见第 7 节
    surrogate.backward();  clip_grad_norm_(1.0);  opt.step()
```

**RHC（滚动时域）模式**只是把 N 步切成几段，每段单独优化好再"提交"推进：

```
for 每一段:
    (可选) 重置网络权重
    for it in range(iterations_per_segment):
        env.set_state(checkpoint);  做一次上面的梯度更新     # 段内反复读档重练
    env.set_state(checkpoint);  用练好的策略跑一遍            # 提交
    checkpoint = env.get_state()                            # 存档，下一段从这继续
    u_current  = 这一段最后的控制量                           # 热启动
```

### 9. 数据流总图

```
  cfg.yaml
     │
     ▼
   env ──────────────────────── 常量：EI, EA, deltaL, W, free/fixed_index, tol
     │
  lambda ──► net ──► 速度 ──(累加, torch 图)──► u_k  [6维]
                                                │
                              _control_node_from_xytheta   (6 → 8)
                                                ▼
                                         control_node [8维] = 钉死的边界
                                                │
                            objfun: Newton 解 F_elastic + F_contact + W = 0
                                                ▼
                                    q_k [102维 形状]  +  J_k [刚度阵]
                                                │
                              L = ||q_k[mid] - target||²  ──►  a_free
                                                │
                       multistep_adjoint_gradient(J_list, a_free)  ──►  dL/du_k
                                                │
                          surrogate = Σ<u_k, dL/du_k> ──► backward ──► Adam ──► net
```

### 10. 这套逻辑和"自定义 energy"改造的关系

对照第 4b 节：整个框架里**能量只在一个地方出现** —— `objfun` 里算 `F_elastic / J_elastic` 的那两行
（现在是 `getFs` + `getFb`）。换成 `energy.force_and_jacobian(q)` 即可。

连带要处理的三处：
1. `objfun` 签名里的 `EI, EA` → 换成 energy 对象；
2. 两个调用点：`env.step` 和 `scripts/train_cmaes.py`（后者自己又调了一遍 objfun，容易漏）；
3. `tol = EI / L² * 1e-3` 对 `EI` 的隐式依赖 → 基类要暴露一个 `stiffness_scale`。

第 6 节**完全不用改**，但它对 `J` 有硬要求：必须是**精确**的二阶导。
如果自定义 energy 用近似 Hessian，Newton 照样能收敛，但梯度会**静默算错**，训练看起来在跑其实没在学。

## 2026-09-15 — 自定义 energy 开关（已实现）

### 怎么用

打开 `configs/default.yaml`，新增了一段 `energy:`：

```yaml
energy:
  custom: false      # 默认 false = 用原来的 DER，行为和以前一模一样
  type: der          # custom 为 true 时才看这一行
  params: {}         # 传给你的能量类的额外参数
```

- **什么都不改** → 还是原来的 DER，结果完全不变。
- **想换自己的能量** → 把 `custom` 改成 `true`，`type` 填你注册的名字。

### 怎么写自己的能量

在 `src/energy.py` 里加一个类，三步：

```python
@register_energy("my_energy")            # 1. 起个名字
class MyEnergy(RodEnergy):               # 2. 继承基类
    def __init__(self, deltaL, 你的参数=默认值, **_ignored):
        ...
    def energy(self, q):    return 能量值          # 3. 写这三个函数
    def gradient(self, q):  return dE/dq           #    形状 (2nv,)
    def hessian(self, q):   return d2E/dq2         #    形状 (2nv, 2nv)
    @property
    def stiffness_scale(self): return 一个刚度量级  #    用来定 Newton 容差
```

然后配置里写：

```yaml
energy:
  custom: true
  type: my_energy
  params: {你的参数: 123}
```

### 三个必须注意的点

**1. 符号别搞反。** 你写的是**能量**的导数（`dE/dq`），基类会自动取负号变成力。写反了 Newton 会发散。

**2. Hessian 必须是精确的，不能用近似。** 因为这个矩阵有两个用途：一是 Newton 迭代，二是训练时求梯度（笔记 §6）。
用近似 Hessian 的话，Newton 照样收敛、训练照样跑完、loss 曲线照样在动，**但梯度是错的，网络其实没在学**，而且完全不报错。
如果你的能量是神经网络，用 `torch.autograd.functional.hessian` 出精确二阶导。

**3. 写完先自检。** 基类带了一个有限差分检查，会告诉你梯度和 Hessian 对不对：

```python
env.energy.check_gradients(q)
# 输出 {'hessian_rel_err': 5.4e-10, 'gradient_rel_err': 4.9e-10}  <- 这种量级就是对的
```

### 具体改了哪些文件

| 文件 | 改了什么 |
|---|---|
| `src/energy.py` | **新文件**。基类 `RodEnergy`、`DERElasticEnergy`（把原来的 DER 包了一层，公式一个字没动）、注册表、`build_energy()` 工厂 |
| `src/SimulatorEnv_2D.py` | `objfun` 里算弹性力的 4 行 → 1 行 `energy.force_and_jacobian(q)`；函数签名的 `EI, EA` → `energy`；`__init__` 加 `energy_cfg` 参数；Newton 容差 `tol` 改用 `energy.stiffness_scale` 算 |
| `configs/default.yaml` | 新增 `energy:` 那一段 |
| `scripts/train_adjoint.py`<br>`scripts/train_tracking.py`<br>`scripts/train_cmaes.py` | 建环境时多传一个 `energy_cfg=cfg.get("energy")`。（train_cmaes 还单独调了一次 `objfun`，也一起改了） |
| `src/__init__.py` | 导出 `RodEnergy` 等 |

`env.EI / env.EA` 这两个属性**保留没删**，别的文件还在用。

### 已验证

- **结果没变**：改动前后跑同一个仿真，`q` 和 `J` 的哈希**完全一致**（bit-wise 相同）。纯重构，物理行为零变化。
- **两条路都跑通**：`custom: false` 走 DER，`custom: true` 走示例的 `linear_spring`，`train_adjoint.py` 都能正常训练。
- `src/energy.py` 里还留了一个 `LinearSpringEnergy`（一串线性弹簧，没有弯曲刚度）。它没什么物理意义，作用是：① 验证开关是通的；② 当作写自己能量时的抄写模板。

### 下一步

真正要做的就只剩**写 `MyEnergy` 这一个类**了，框架其余部分（Newton、伴随法、训练循环）一行都不用碰。

## 2026-09-21 — 3D 环境 / Newton 求解器 / BLAS 线程（已完成）

提交：`874fcda`。三件事互相关联，都由配置里新增的 `env.dim` 这一个开关选择。

### 怎么用

```yaml
env:
  dim: 3        # 2 = 原来的 2D 环境 | 3 = 新的 3D 环境
```

```bash
python scripts/train_adjoint.py  --config configs/example_3d.yaml
python scripts/train_tracking.py --config configs/example_3d.yaml
python scripts/train_cmaes.py    --config configs/example_3d.yaml
```

`dim` 同时决定配置里读哪些键：3D 读 `start.u_left_3d / u_right_3d` 和 `task.target_xyz`，2D 读不带后缀的那几个。`rod / energy / rhc / train / output` 两种模式共用。`configs/example_3d.yaml` 是一个只覆盖差异字段的最小 3D 配置（更细的杆、更少的节点、离面目标点）。

### 1. 3D 环境

新增 `src/SimulatorEnv_3D.py`，API 和 2D 完全一致（`reset / step / get_state / set_state / render`），配套的几块：

| 文件 | 内容 |
|---|---|
| `src/energy.py` | 新增 `der3d`：自然直、无扭转杆的 Discrete Elastic Rod 能量 |
| `src/implicit_grad_tools.py` | 新增 `boundary_node_jacobian_3d`（12×10），伴随法入口统一加 `dim` 参数 |
| `src/contact.py` | 新增 `SphereObstacle`，`get_contact_forces` 变成 dim-aware |
| `src/env_factory.py` | **新文件**。把所有和维度相关的构造集中到一处，三个训练脚本因此只写一遍、两种维度都能跑 |

**控制维度的变化**（对照 §4a）：两端都是 clamped，控制量给的是位置 + 切线方向。3D 需要两个角而不是一个 —— 方位角 `theta` 和仰角 `phi` —— 所以每端是 `(x, y, z, theta, phi)`，**控制向量 10 维、被钉死的边界 DOF 12 个**。`phi = 0` 精确还原 2D 的约定。

**为什么 `der3d` 的精确 Hessian 是廉价的**：弯曲各向同性 + 静止曲率为零，`|kb|²` 塌缩成单个标量 `c = te·tf` 的函数，所以伴随法要求的精确二阶导有闭式表达。有限差分验证到 ~1e-11。

**没建模扭转**。要加扭转需要 material frame 和 parallel transport，见 README。

### 2. Newton 求解器：线搜索 + 自适应正则

受压的杆是后屈曲状态，`J_ff` 会出现负特征值（3D 的负特征值数量是 2D 的两倍 —— 每个屈曲模态在 y 和 z 两个方向各存在一次）。在这种不定系统上走无阻尼的完整 Newton 步会跑飞：实测残差从 ~1e0 涨到 ~1e21，把迭代次数烧光，然后刷屏 "Maximum number of iterations reached."

`src/newton.py`（新文件）加了**非单调回溯线搜索**和**自适应 Tikhonov 正则**，2D/3D 共用。

**线搜索为什么必须是非单调的**：这里的残差在下降途中会合理地上跳（1.5e6 → 2.7e6 → 7.8e5 → … → 6.8e2）。要求每步都下降的话，一步要花 27 次迭代 + 94 次回溯；用 Grippo 准则（不超过最近 N 次迭代的最差残差就接受）只要 10 次干净的完整步。

**对原有结果的影响**：原来能收敛的地方，结果 bit-identical；原来发散的地方差别是决定性的 —— 一个 2D 受压算例，无阻尼时返回弧长 3.5e6 的杆，开线搜索后是正确的 0.998。`newton.line_search: false` 可精确还原原来的迭代。

### 3. BLAS 线程：3D 慢 18 倍的真正原因

每次 Newton 迭代都要解一个 (n_free × n_free) 的稠密系统：2D 是 94，3D 是 141。OpenBLAS 在 n ≈ 100 附近开始多线程，而这么小的矩阵，线程开销是分解本身的 ~2000 倍 —— 141×141 默认要 260 ms，单线程只要 0.127 ms。**2D 刚好落在阈值下面、3D 刚好落在上面**，这就是 3D 看起来远比 1.5 倍 DOF 增长该有的速度更慢的全部原因。

训练脚本现在在 import NumPy **之前**调用 `limit_blas_threads()`（`src/blas_threads.py`）。默认 3D 配置跑两个 epoch：**396 s → 22 s**。

### 已验证

- `der3d` 的梯度 / Hessian，以及完整的 `dL/du` 链条，对有限差分 ~1e-11
- 3D 降维到平面（`z = 0`，重力 `[0, -9.81, 0]`）与 2D 逐节点一致，误差 1e-16
- 重构后 2D 的轨迹、loss、伴随法辅助函数**完全没变**（差 0）
- 三个脚本 × 两种维度 × 开关 RHC，全部跑通

### 下一步

- 扭转（material frame + parallel transport）
- 3D 下的自定义 energy（`@register_energy` 注册的类需要和 `env.dim` 匹配，否则加载时会直接报错）

---

## 2026-09-22 — 控制过程 GIF 动画（已完成）

训练结束时除了原来的 `outputs/neural_control_result.png`（最终形状 + loss 曲线），现在还能把**整段控制过程逐帧合成 GIF**，存成 `outputs/neural_control_animation.gif`。

### 怎么用

`configs/default.yaml` 的 `output:` 段新增四个键，**默认关闭**（渲染要花几秒）：

```yaml
output:
  experiment_name: dlo_final_shape
  save_gif: false          # 改成 true 就会输出 GIF
  gif_fps: 20              # 播放速度
  gif_dpi: 100             # 分辨率，150 以上文件会迅速变大
  gif_max_frames: 200      # 帧数上限，更长的 rollout 会均匀抽帧（首尾帧一定保留）
```

三个训练脚本（`train_adjoint.py` / `train_tracking.py` / `train_cmaes.py`）都支持，2D / 3D 都支持。

### 画面内容

- 黑色带点的曲线 = 杆的当前形状；绿色方块 = 两端边界节点
- 红点 = 中点，红色细线 = 中点已经走过的轨迹
- 蓝色叉 = 目标。tracking 任务下额外画出整条目标曲线（虚线）+ 当前时刻的目标点
- 坐标轴范围在整段 rollout 上**固定**（3D 用等比例立方体），所以杆不会随帧抖动
- 标题显示 `step k/N`，抽帧后显示的仍是原始 rollout 的步号

### 动的是哪一段轨迹

GIF 画的是**最终提交的那次 rollout**，不是训练中间的 epoch：

| 脚本 / 模式 | 取的是哪段 |
|---|---|
| `train_adjoint` 非 RHC | 最后一个 Newton 收敛的 epoch 的整条 `q_list` |
| `train_adjoint` RHC | 每段 commit 时那次 rollout，按段拼接成完整时域 |
| `train_tracking` | `_visualize_tracking` 里那次最终 rollout（本来就会跑一遍，顺手存下完整的 `q`，不额外花时间） |
| `train_cmaes` | `all_rollouts` 逐段拼接；每段的首帧等于上一段的末帧，拼接时去重 |

### 改了哪些文件

| 文件 | 改了什么 |
|---|---|
| `src/utils.py` | 新增 `save_trajectory_gif()`（FuncAnimation + PillowWriter，2D/3D 通用）和 `maybe_save_trajectory_gif()`（读 `output.save_gif` 开关的封装） |
| `configs/default.yaml` | `output:` 段新增 `save_gif / gif_fps / gif_dpi / gif_max_frames` |
| `scripts/train_adjoint.py` | 记录 `q_traj`（非 RHC 取最后收敛的 epoch，RHC 按段拼接），末尾调用 |
| `scripts/train_tracking.py` | `_visualize_tracking` 里顺带收集完整 `q`，末尾调用 |
| `scripts/train_cmaes.py` | 末尾拼接 `all_rollouts` 后调用 |
| `src/__init__.py` | 导出两个新函数 |
| `README.md` | 目录树 + "Animating the rollout" 小节 |

### 已验证

- 3D point-to-point（`train_adjoint`，3 epoch）→ 50 帧 GIF，杆/中点/目标位置都对
- 2D tracking（`train_tracking`，circle）→ 40 帧，目标曲线 + 当前目标点正常
- 2D CMA-ES + RHC 2 段 → 40 帧，跨段拼接无重复帧
- `save_gif: false`（默认）时完全不产生 GIF，原有行为不变

### 依赖

用的是 matplotlib 自带的 `PillowWriter`，只需要 Pillow（环境里已有 12.2.0），**不需要 ffmpeg 或 imagemagick**。

---

## 2026-09-22 — 材料预设选择器（已完成）

原来杆的材料/截面是写死在 `rod:` 里的四个数，换材料要手工改四行、也没法记录用的是什么。现在改成**预设注册表 + 一个选择键**。

### 怎么用

```yaml
rod:
  material: soft_strip     # aluminum_tube | soft_strip | steel_tube | silicone_cord | custom
```

`configs/default.yaml` 里新增了一段 `materials:`，每个预设只管四个字段 —— `R_outer / r_inner / E_al / rho_al`。其余全部是推出来的：

```
I  = pi/4 * (R_outer^4 - r_inner^4)      A  = pi * (R_outer^2 - r_inner^2)
EI = E * I  （弯曲刚度）                  EA = E * A  （拉伸刚度）
```

### 四个预设（1 m 杆）

| 预设 | EI [N·m²] | 说明 |
|---|---|---|
| `steel_tube` | 2.19e+3 | 最硬的极端，求解很慢 |
| `aluminum_tube` | 7.65e+2 | **原来的默认值**，行为不变 |
| `soft_strip` | 1.27e-1 | 细软聚合物条，比铝管软约 6000 倍；**迭代最快**，3D/GIF 的测试都用它 |
| `silicone_cord` | 2.45e-3 | 橡胶绳，最接近真实的 DLO（线缆）；重力主导的大变形 |

> 注意 EI 正比于半径的**四次方**，所以截面比材料本身影响更大。硬杆在边界控制下几乎不变形，而且 Newton 要反复线搜索，所以跑得慢 —— 之前一次铝管 RHC 测试 10 分钟没跑完，换 `soft_strip` 后几十秒就完。

### 优先级规则

1. `rod:` 里显式写的字段 **覆盖** 预设 —— 想"同样的材料但壁薄一点"很方便：

```yaml
rod:
  material: aluminum_tube
  r_inner: 0.012
```

2. `material: custom`（或不写）= 完全不用预设，四个字段自己在 `rod:` 里给全。
3. 名字拼错、预设里写了不该写的字段、custom 但字段给不全 —— 都会在建环境时直接抛 `ValueError` 并列出可用名字，不会静默用错材料。

每次训练开头会打印解析结果，例如：

```
Rod material: soft_strip: R_outer=0.003 m, r_inner=0 m, E=2e+09 Pa, rho=1200 kg/m^3  ->  EI=0.127 N*m^2, EA=5.65e+04 N
```

### 改了哪些文件

| 文件 | 改了什么 |
|---|---|
| `configs/default.yaml` | 新增 `materials:` 注册表（四个预设 + 说明）；`rod:` 里的四个材料字段移走，换成 `material:` 键 |
| `src/env_factory.py` | 新增 `MATERIAL_FIELDS`、`resolve_material()`（预设 + 覆盖 + 报错）、`material_summary()`；`build_env` 改成 `**resolve_material(cfg)` |
| `configs/example_3d.yaml` | 那四行手写材料 → `material: soft_strip`（数值完全相同） |
| `scripts/train_*.py` | 开头多打印一行 `Rod material: ...` |
| `src/__init__.py` | 导出两个新函数 |
| `README.md` | 新增 "Choosing a rod material" 小节 |

### 已验证

- 四个预设 + 覆盖 + custom + 三种报错路径，逐一检查解析结果和 EI/EA 数值
- `soft_strip` 跑通 `train_adjoint` / `train_tracking`（2D），`silicone_cord` 跑通 `train_cmaes`（2D）
- **默认行为不变**：`configs/default.yaml` 默认仍是 `aluminum_tube`，解析出的四个数与改动前逐字相同
- `configs/example_3d.yaml` 解析结果与改动前相同

---

## 2026-09-28 — 3D 杆的扭转能量 + 夹爪 yaw 控制（已完成）

提交：`a538c2e`（主体）、`4414fe5`（泊松比归入材料）、`462d436`（`energy.type` 不再说谎）、`0f705fa`（动画画出截面朝向）。

补上了 09-21 那条"下一步"里留的坑：**扭转**。用 `rod.twist` 一个开关控制，关掉时全部逻辑失效，无扭转的 3D 路径与改动前**逐位相同**。

### 怎么用

```yaml
rod:
  twist: true
  material: soft_strip     # 预设自带泊松比，GJ 跟着材料走
start:
  u_left_twist:  [0, 0, 0, 0, 0, 0]      # 每端 6 个数
  u_right_twist: [1, 0, 0, 0, 0, 0]
  init_yaw: 0.0
```

```bash
python scripts/train_adjoint.py --config configs/example_3d_twist.yaml
python scripts/check_twist.py                     # 扭转模型本身的有限差分自检
```

### 1. 模型：DOF 多了一维，能量多了一项

沿用 Discrete Elastic Rods（Bergou et al. / DEFORM）的 material frame 做法。每条边带一个**参考指向子**（reference director，用无扭转的平行移动搬运）加一个**标量扭转角** `theta^i`，DOF 向量从 `3*nv` 长到 `4*nv - 1`：

```
q = [ 3*nv 个位置 | nv-1 个扭转角 ]

E = E_stretch(x) + E_bend(x) + E_twist(x, theta)
E_twist = Σ_i (GJ / 2dL) * m_i² ,   m_i = theta^i - theta^{i-1} + psi_i(x)
```

圆截面下 `J = 2I`、`G = E/(2(1+nu))`，所以 `GJ = EI / (1 + nu)`。拉伸和弯曲一个字没动，`GJ = 0` 精确还原 `der3d`（实测差 0.0）。

**`psi_i`（reference twist）是整件事的关键**，它是平行移动沿杆走一圈"合不拢"的那部分。它**依赖位置**，这正是扭转与弯曲耦合的来源（离散版的 link = twist + writhe）。少了它，`E_twist` 就只依赖 `theta`，中心线对扭转毫无反应，`dL/d(yaw)` 恒等于零 —— yaw 会变成一个死掉的控制输入。

### 2. 控制：每端多一个 yaw

每端从 `(x, y, z, theta, phi)` 变成 `(x, y, z, theta, phi, yaw)`，**控制向量 12 维，边界映射 14×12**（对照 §4a / 09-21 那条的 10 维与 12×10）。`theta/phi` 指定切线方向，`yaw` 是绕切线的自旋 —— 第三个转动自由度。

yaw 是**相对边的参考指向子**量的，而那个标架本身由无扭转平行移动搬运，所以"yaw 保持不变"就等于"不施加扭转"。结果是 yaw 直接对应到被钉死的两个扭转角上，**中间没有任何几何换算**（`boundary_node_jacobian_3d_twist` 里那两行就是两个 1）。

### 3. 两个花了真功夫才 debug 出来的结论

这两条记下来，以后再碰 material frame 不用重踩：

- **`hess(psi)` 是链式法则矩阵的对称部分。** 直接对 `kb/2L` 的梯度表达式求导，会多出一块伪反对称项，误差 **>100%**。
- **`grad(psi)` 比经典的 holonomy 多一个"重新适配项"**，来自把每个存下来的指向子搬到当前切线上。当存储切线等于当前切线时它为零 —— 所以在一步的**起点看不见**，只有 Newton 迭代走动之后才冒出来。漏掉它梯度差 **~50%**。

### 4. 还剩一个近似

标架是在一步 commit 之后才推进的，所以被扰动的控制会走一条略微不同的标架路径，而单步隐式梯度看不见这件事。这和时域伴随法本来就在做的近似是同一类。3 步 rollout 上实测相对误差 **~6.5e-04**。

### 5. 连带改的三件小事

| 提交 | 改了什么 |
|---|---|
| `4414fe5` | **泊松比归入材料预设**。`nu` 决定扭转刚度，它是"杆是什么做的"的属性，不该松散地挂在 `rod:` 下面、换材料时还留在原地。四个预设各自带值（steel 0.30 / aluminum 0.33 / soft_strip 0.40 / silicone 0.49），换材料时弯曲刚度和扭转刚度一起变。不写仍然默认 0.3，`rod.poisson` 仍可单点覆盖 |
| `462d436` | `energy.type: der` 原本读起来像在陈述事实，但 `custom: false`（默认）时 `build_energy` 根本不看这一行 —— 真正建的模型由**杆**决定（2D 是 der，3D 是 der3d，开 twist 是 der3d_twist）。填一个根本没注册的名字也毫无反应，可见这个字段多误导。现在默认是 `type: null` |
| `0f705fa` | 动画里画出**截面朝向**。纯扭转只让截面自转、形状不变，所以只画中心线的话，最有意思的那个自由度是看不见的。现在在两端和中点画出 material frame（切线 + m1/m2）。指向子是求解器状态、`q` 里没有，所以按记录的构型重放（确定性的，训练脚本不用额外存东西） |

### 改了哪些文件

| 文件 | 改了什么 |
|---|---|
| `src/rodframes.py` | **新文件**（294 行）。material frame、平行移动、`reference_twist`、`psi_grad_hess` |
| `src/energy.py` | 新增 `der3d_twist`（`DERElasticEnergy3DTwist`），复用 `der3d` 算拉伸/弯曲 |
| `src/implicit_grad_tools.py` | 新增 `boundary_node_jacobian_3d_twist`（14×12），伴随法入口统一加 `twist` 参数 |
| `src/SimulatorEnv_3D.py` | 扭转 DOF、标架状态（`get_state/set_state` 要一起存，否则读档后扭转会跳）、12 维控制 |
| `src/contact.py` / `src/newton.py` | 加 `n_pos`：只作用在位置 DOF 上，别碰扭转角 |
| `src/utils.py` | 改走 `env.positions()`，原来的 `reshape(-1, dim)` 只在无扭转时才对 |
| `src/env_factory.py` | 维度 + twist 感知的控制维数 / 初始控制 / 目标 |
| `scripts/check_twist.py` | **新文件**。9 项有限差分自检 |
| `configs/example_3d_twist.yaml` | **新文件**。最小的 twist 配置 |

### 已验证

`scripts/check_twist.py` 把伴随法依赖的每一环都对了有限差分，**9/9 通过**。提交当时记录的数值：

| 检查项 | 相对误差 |
|---|---|
| `grad(psi)` | 3.7e-09 |
| `hess(psi)` | 8.4e-07 |
| 扭转能量的力 / 雅可比 | ~7e-11 |
| `GJ = 0` 退化回 `der3d` | 精确 0.0 |
| 边界映射 14×12 | 5.3e-10 |
| 含 yaw 的 `dL/du` | 6.8e-08 |
| 标架历史的漂移（3 步，属于 §4 那个近似，**不算失败项**） | ~6.5e-04 |

09-29 重跑仍然 **9/9**，数值同量级（边界映射 5.3e-10、`dL/du` 1.1e-09、`dL/d(yaw)` 3.7e-07、标架漂移 4.2e-04）—— 具体数字随算例浮动，看的是量级和是否过容差。

另外：三个训练脚本 × 2D / 3D / 3D+twist 全部跑通；**3D 不开 twist 的结果与改动前最后一位都相同**。

### 下一步

- 各向异性截面（现在只建模了各向同性弯曲，所以弯曲不需要 material frame，标架只在扭转项里出现）
- 自然曲率 / 自然扭转（现在假设杆天然是直的）

---

## 2026-09-29 — 计算冗余审计（已完成）

跑 `configs/example_3d_twist.yaml` 时每个 epoch 要 12~25 s，怀疑内层循环在重复算 Hessian 之类的东西。Profiler 只能说"哪个函数慢"，但真正决定怎么改的问题是"**这些计算里有多少是以前已经算过的**"。新增 `src/compute_audit.py` 来回答后者。

### 怎么用

```bash
# 短跑一次拿数字，开销约 2.5%
python scripts/train_adjoint.py --config configs/example_3d_twist.yaml --audit

# 再往下一层，看每个 stencil kernel（调用量极大，会明显变慢）
python scripts/train_adjoint.py --config configs/example_3d_twist.yaml --audit --audit-kernels
```

长训练想**边跑边看**，配置里写：

```yaml
audit:
  enabled: true
  report_every: 10        # 每 10 个 epoch 重写一次报告（RHC 模式是每 10 次梯度更新）
```

然后另开终端 `watch -n 30 cat experiments/<name>/runs/<时间戳>/compute_audit.txt`。周期性写入会跳过 `.npz`（保存里最慢的那部分）。

默认 `audit.enabled: false`，此时**什么都不包**，热路径完全没动过。

### 原理

包住几个热点：`env.step`、Newton 求解、`force_and_jacobian / force / gradient / hessian`、伴随法的 `local_dxf_dxb` 和 `multistep_adjoint_gradient`。每次调用记三样东西 —— **输入**（决定结果的那个向量）、**耗时**（扣掉嵌套在内部的、同样被记录的调用，所以不重复计时）、**位置**（第几个 epoch / rollout 第几步 / 第几个 RHC segment）。

然后对每次调用去找**之前**最近的同类输入：完全相同用字节哈希精确判定；"接近"用随机投影 LSH + JL sketch，所以每次调用的查找是 **O(1)** 而不是 O(n) —— 否则审计本身会比训练还慢。

输出写进 run 目录：`compute_audit.txt`（人读）、`.json`（程序读）、`compute_audit_records.npz`（每次调用一行的原始表，字段有耗时 / 到最近历史输入的相对距离 / epoch / step / seg）。

报告有三张表：

- **调用重复率** —— `exact%` 是"加个一行缓存就能去掉、结果完全不变"的部分；`<=1e-4` 这类列是累积的相对距离，也就是冻结 Jacobian 或热启动能有的精度预算
- **时间重复率** —— 同一件事按耗时加权，这才是真能省下的
- **逐 epoch 对比** —— 把每个 rollout step 和上一个 epoch 的同一 step 配对，直接回答"是不是每个 epoch 都在重算上一个 epoch"

### 测出来什么（8 epoch，nv=21，N=100）

| kernel | 调用次数 | 输入与之前某次**逐字节相同** |
|---|---|---|
| `psi_grad_hess` | 301 785 | **64.4%** |
| `RodFrames.reference_twist` | 15 891 | **64.4%** |
| `_bend_stencil` | 301 929 | **68.8%** |
| `_stretch_stencil` | 317 820 | **71.9%** |
| `energy.gradient` | 10 378 | **45.4%** |

约三分之二的 stencil 计算是**字面意义上的重算**，来自两个叠加的原因：

1. `RodEnergy.force_and_jacobian` 是 `(-gradient(q), -hessian(q))`。对带扭转的杆，这两半走的是**同一批 stencil**：`reference_twist` 各调一次，`psi_grad_hess` 每个内部节点各调一次，而 `psi_grad_hess` 本来就同时返回 psi 的梯度**和** Hessian。所以 Hessian 那一遍把 gradient 那一遍刚算完的东西又算了一次。
2. Newton 的线搜索先算 `force(q_try)`，这一步被接受后，下一轮迭代立刻在**同一个 `q`** 上调 `force_and_jacobian(q)`。这就是 `energy.gradient` 那 45%。

**原本的猜想里不成立的那一半**：跨 epoch 的重复。逐 epoch 表显示相邻 epoch 在同一 rollout step 上的输入相距约 `1e-1`，只有 ~3% 的配对落在 `1e-2` 以内 —— **跨 epoch 缓存没有价值，冗余全部在单个 step 内部**。另外 `<=1e-2` 那一列几乎处处是 99%，这是 modified Newton（多轮迭代之间冻结 Jacobian）能用的精度预算。

### 改了哪些文件

| 文件 | 改了什么 |
|---|---|
| `src/compute_audit.py` | **新文件**。`ComputeAudit`：包热点、记录、LSH + sketch 找最近历史输入、出报告 |
| `scripts/train_adjoint.py` | `--audit` / `--audit-kernels` 两个开关；epoch / step / segment 打标；训练结束（在画图之前）出报告；`report_every` 的周期性刷新 |
| `configs/default.yaml` | 新增 `audit:` 段 |
| `README.md` | 目录树 + Performance 下的 "Where the repeated work is: the compute audit" 小节 |

### 已验证

- 近似重复的判定对着注入的已知扰动测过：1e-8 / 1e-4 / 1e-2 三档的中位数都对得上，`1e-2` 以内召回率 100%
- 关掉审计（默认）时行为与改动前一致；开 `--audit` 的额外开销约 **2.5%**
- `report_every` 的周期性刷新在 6 epoch 的短跑里按预期刷了 3 次

### 两个限制

- 近似重复的检测在相对距离 `1e-2` 以内可靠，**超出这个范围计数是下界**（LSH 不再碰撞）。精确重复永远是精确计数
- 开 `--audit-kernels` 时，wrapper 自身的开销会抬高那几个高频 kernel 在**时间**列里的占比；重复**比例**是调用计数，不受影响

### 下一步

审计只做测量，**没有动任何 kernel**。按上面的数字，可做的有两件：

1. 让 `force_and_jacobian` 一趟同时出梯度和 Hessian（`psi_grad_hess` / `_bend_stencil` 本来就两个都算），并把线搜索最后一次 `force(q)` 的结果缓存给紧接着的 `force_and_jacobian(q)`
2. 更激进的话，利用 `<=1e-2` 那 99%，在 Newton 迭代之间冻结 Jacobian（modified Newton）

另外 `--audit` 目前只接到了 `train_adjoint.py`，`train_tracking.py` 要的话是同样的三行改动。
