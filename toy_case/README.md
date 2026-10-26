# Toy DER: a minimal Discrete Elastic Rods implementation

A self-contained 3D DER written from scratch against
**Bergou et al., "Discrete Elastic Rods", ACM TOG 2008** (`reference/Discrete elastic rods.pdf`).
It has **stretching + bending + twisting** energies, no contact, no control.
It is independent of `src/` and only needs `torch` (CPU is fine), `numpy`, `matplotlib` and `pyyaml`.

> 中文：这是对照 DER 论文从零写的独立 3D 离散弹性杆，包含拉伸、弯曲、扭转三种能量，没有接触和控制部分，和 `src/` 完全独立。

```
toy_case/
├── der.py            # the model: geometry, frames, the three energies (one function per paper formula)
├── solver.py         # Newton (static equilibrium) + backward Euler (dynamics)
├── run.py            # 5 validation / demo experiments
├── profile_steps.py  # where the time goes in one experiment (timers around the real functions)
├── config.yaml       # every tunable parameter
└── outputs/          # experiment figures
```

```bash
cd toy_case
python run.py gradcheck          # force correctness + comparison with the paper's closed forms
python run.py cantilever         # gravity sag of a cantilever vs. Euler-Bernoulli
python run.py twist              # pure twist of a clamped-clamped rod vs. paper Eq. (5)(6)
python run.py buckle             # twist buckling vs. Greenhill; twist -> writhe
python run.py swing              # dynamics: horizontal cantilever swinging under gravity
python run.py all
# override any config entry from the command line:
python run.py twist --set rod.rest_curvature=3.0 rod.radius_b=0.004 phi=6.0
# time breakdown of a real run: whole-run timeline, load steps 1-3 in detail, per-stage shares (§5)
python profile_steps.py buckle --show 1 2 3
```

> 中文：每个实验都可以用 `--set 键=值` 在命令行覆盖任意参数，不用改 yaml。

## How the files fit together

Dependencies go one way only: `run.py` → `solver.py` → `der.py`. `config.yaml` is read only by `run.py`.
`der.py` knows nothing about solvers or experiments, and `solver.py` only needs a `Rod` object.

> 中文：依赖是单向的：`run.py` 调用 `solver.py`，`solver.py` 调用 `der.py`。只有 `run.py` 读 `config.yaml`。`der.py` 只负责"给定构型算能量"，不知道求解器和实验的存在；`solver.py` 只需要一个 `Rod` 对象。

```
config.yaml ──► run.py
                 │ main(): parse CLI → load_cfg(): merge base + experiments.<name> + --set
                 │ EXPERIMENTS[name](cfg)          e.g. exp_twist(cfg)
                 │
                 ├─► der.Rod(cfg)                  build rest shape, stiffness, masses, BCs   (once)
                 │
                 │   loop over load steps / time steps:
                 ├─► rod.apply_end_twist / rod.set_bc_targets   change the prescribed BC values
                 ├─► solver.static_solve(rod, z)   or   solver.dynamic_step(rod, z, v, dt)
                 │     └─► solver.newton(obj, z)
                 │           obj(z) = rod.elastic_energy(z) + rod.gravity_energy(z) [+ inertia]
                 │           g = torch.func.grad(obj),  H = torch.func.hessian(obj)
                 │           solve H dz = -g, line search, repeat until |dz| < tol
                 │     └─► rod.track_holonomy(z)   advance the end-clamp holonomy reference
                 │
                 └─► rod.energy_terms(...) for logging → matplotlib → outputs/<name>.png
```

> 中文：一次实验的流程：`run.py` 读配置，创建 `Rod`（只建一次），然后在每个加载步或时间步里先改边界条件，再调用 `solver` 求解，最后记录能量、画图。`solver.newton` 把能量函数当成黑箱，用 autograd 算梯度和 Hessian 来做牛顿迭代。每步收敛以后调用 `track_holonomy`，更新夹持端的和乐角。

What happens inside one energy evaluation `rod.elastic_energy(z)` (all in `der.py`):

```
z (free DOFs only)
 └─► Rod.unpack(z)            x = x_bc + S_x z_x,  θ = S_θ z_θ      (insert fixed BC values)
      └─► Rod.frames(x, θ)
           edges → tangents t^j
           bishop_frame(t, u^0)          parallel transport along the rod (§4.2.2)
           clamped end: θ^n = Φ − δ(x)   holonomy at the end edge (§5, Eq. 10)
           material_frame(u, v, θ)       m1, m2
      └─► Rod.energy_terms
           stretch_energy(e)                                    k_s
           curvature_binormal(e) → material_curvatures → bend_energy   Eq. (1)(2)(3), B
           twist_energy(θ)                                      β
      └─► sum  →  scalar E   (autograd differentiates back through this whole chain)
```

> 中文：一次能量计算的内部流程：先把自由度 z 展开成完整的 x 和 θ（把边界条件固定的值填进去），再算切向、Bishop 标架、材料标架（夹持端的 θⁿ 在这一步由和乐决定），最后算三种能量并求和。autograd 会沿着这条链一路反向求导，所以力和 Hessian 都不用手写。

---

## 1. Code ↔ paper correspondence

DOFs: `q = [x_0 … x_{n+1}, θ^0 … θ^n]`. There are N = n+2 vertices (lower index) and n+1 edges
(upper index). Curvature lives on the interior vertices i = 1..n. Barred quantities (x̄, ē, l̄_i, ω̄)
describe the undeformed configuration and are computed once in `Rod.__init__` (Algorithm 1, line 1).

> 中文：下标是顶点，上标是边；带横线的量是静止构型，只在初始化时算一次。

| Paper | What | Code (`der.py`) |
|---|---|---|
| §4.2 | edges e^j = x_{j+1} − x_j, tangents t^j = e^j/\|e^j\| | `edges`, `Rod.frames` |
| **Eq. (1)** | curvature binormal (κb)_i = 2 e^{i−1}×e^i / (\|e^{i−1}\|\|e^i\| + e^{i−1}·e^i) | `curvature_binormal` |
| §4.2.2 | discrete parallel transport P_i: rotation about t^{i−1}×t^i taking t^{i−1} to t^i | `parallel_transport` |
| §4.2.2 | Bishop frame u^j = P_j u^{j−1}, v^j = t^j × u^j | `bishop_frame` |
| §4.2.2 | material frame m1 = cosθ u + sinθ v, m2 = −sinθ u + cosθ v | `material_frame` |
| **Eq. (2)** | material curvatures ω_i^j = ((κb)_i·m2^j, −(κb)_i·m1^j), j∈{i−1,i} | `material_curvatures` |
| **Eq. (3)** | E_bend = Σ_i 1/(2 l̄_i) Σ_j (ω_i^j − ω̄_i^j)ᵀ B (ω_i^j − ω̄_i^j) | `bend_energy` |
| §4.2.2 | E_twist = Σ_i β (θ^i − θ^{i−1})² / l̄_i | `twist_energy` |
| §4.1 ("straightforward to add") | E_stretch = ½ Σ_j k_s (\|e^j\|/\|ē^j\| − 1)² \|ē^j\| | `stretch_energy` |
| §4.1.1 | rest material frame = Bishop frame ⇒ θ̄ = 0, m̄ = 0; ω̄ from the rest shape | `Rod.__init__` |
| §4.2.2 "Bishop frame update" | free start: u^0 is parallel-transported **in time** | `Rod.frames` |
| §5 Eq. (4) | quasistatic material frame ∂E/∂θ^j = 0 | `solver.py`: θ is massless and minimized in the same Newton solve as x |
| §5 "Boundary conditions" | clamped end prescribes the material frame | `Rod` (see below) |
| §6 Eq. (9)(10) | holonomy gradient ∇Ψ_j | **not hand-coded**: autograd through `bishop_frame` produces it; checked by `gradcheck` [3] |
| §7.1 | forces on the centerline | **not hand-coded**: `-autograd(E)` |
| §5.1 Eq. (7) | closed form of ∂E/∂θ^j | used in `gradcheck` [2] to check autograd |

**Why autograd instead of the hand-derived forces of §7?**
Each function here is a direct transcription of a formula. The Bishop frame is built by explicit
parallel transport, so it is a differentiable function of x. Differentiating E therefore produces the
holonomy terms of §6 (the twist-bend coupling forces) automatically. `gradcheck` compares autograd
with the paper's Eq. (7) and Eq. (9)(10); they agree to machine precision (~1e-16). If you later switch
to hand-derived analytic forces/Hessians (faster), this code is the reference answer.

> 中文：力和 Hessian 没有照论文 §7 手推，而是对能量做自动微分。因为 Bishop 标架是用平行移动显式算出来的，是 x 的可微函数，所以 §6 的和乐项（扭转-弯曲耦合力）会自动出现。`gradcheck` 已经用论文的解析式核对过。以后想换成更快的手推解析力，可以拿这里当参考答案。

**θ^n at a clamped end (paper §5 boundary conditions).** A clamped end fixes m1^n in the lab frame,
but u^n is parallel-transported from u^0 along the whole (deformed) rod, so θ^n is not a constant:

  θ^n = Φ_end − δ(x), where δ is the rotation of u^n from its rest value, i.e. the total holonomy Ψ_n (Eq. 10).

`Rod.frames` implements exactly this. δ is accumulated step by step (`track_holonomy`), so it can wind past ±π.

> 中文：夹持端在实验室坐标系里固定了材料方向，但 Bishop 标架会随杆的变形而转动（和乐），所以 θ^n 依赖于 x。δ 是逐步累加的，可以超过 ±π。

## 2. Parameters (`config.yaml`)

| Parameter | Meaning | Effect |
|---|---|---|
| `rod.n_nodes` | number of vertices N | accuracy / speed (O(1/N) convergence, see cantilever) |
| `rod.length` | rod length L | |
| `rod.radius_a`, `rod.radius_b` | elliptic cross-section semi-axes (along m1 / m2) | a≠b ⇒ anisotropic B = diag(E I1, E I2), Eq. (3) |
| `rod.youngs`, `rod.poisson` | E, ν | k_s = EA, B = EI, β = GJ, G = E/2(1+ν) |
| `rod.density` | ρ | lumped vertex mass ρA·(Voronoi length) |
| `rod.rest_curvature` | natural curvature κ̄ (circular arc in the x-y plane) | ω̄ ≠ 0, the barred terms in Eq. (2)(3) |
| `energy.{stretch,bend,twist}` | on/off switch per energy | debugging: turn one term off to see what it does |
| `energy.*_scale` | stiffness multipliers | debugging: scale one term's stiffness alone |
| `bc.start` | `clamp` / `free` | clamp: x0, x1 fixed, θ^0 = 0 |
| `bc.end` | `clamp` / `pin` / `free` | clamp: last edge fixed + end material frame prescribed |
| `bc.end_twist` | rotation Φ of the clamped end [rad] | |
| `bc.end_shift` | translation of the end [m] | compression / tension |
| `gravity` | gravity vector | |
| `solver.tol`, `max_iter`, `verbose` | Newton tolerance / max iterations / per-iteration printing | |
| `dynamics.dt`, `steps`, `damping` | backward-Euler step / number of steps / mass-proportional damping | |

Each experiment has its own overrides under `experiments:`, plus experiment-specific parameters
(`phi`, `mesh`, `phi_max`, `slack`, …).

> 中文：调试时最常用的是 `energy.*` 的开关和 `*_scale` 乘子，可以单独看每种能量的作用；`radius_a ≠ radius_b` 得到各向异性截面，`rest_curvature` 得到自然弯曲的杆。

## 3. Experiments and current results

| Experiment | What it checks | Result |
|---|---|---|
| `gradcheck` | [1] autograd force vs. finite differences (anisotropic, clamped + twisted, random state)<br>[2] ∂E/∂θ vs. paper Eq. (7)<br>[3] holonomy gradient vs. paper Eq. (9)(10) | 6.7e-11<br>2.4e-16<br>6.1e-16 |
| `cantilever` | gravity sag vs. δ = qL⁴/8EI | sim/EB(L_eff) = 0.993 → 0.998 (N = 11 → 81) |
| `twist` | straight clamped-clamped rod twisted by Φ = π: uniform twist (Eq. 5), E = βΦ²/2L (Eq. 6) | both exact to 1e-15 |
| `buckle` | critical twist of the straight rod vs. Greenhill M·L/EI = 2.861π = 8.988 | N = 21/41/81: 9.167 / 9.029 / 8.997 |
| `swing` | dynamics with all energies + inertia | see `outputs/swing.png` |

Things to keep in mind when reading the results:

- **Effective length in `cantilever`.** A DER clamp fixes the whole first edge e^0, which is equivalent
  to a built-in support at the midpoint of e^0: L_eff = L − \|ē^0\|/2. Comparing against the nominal L
  gives an O(1/N) gap (≈0.81 at N = 11); that is not a discretization error of the energy.
  > 中文：DER 夹持的是整条首边，等效固支点在首边中点，所以要用 L_eff 比较。
- **`buckle`.** The onset is found by bisecting the smallest Hessian eigenvalue on the exact
  "straight rod + uniform twist" equilibrium, so it does not depend on whether Newton leaves that branch.
  Phase 1 then twists on to Φ = 20. With a fixed end distance and a large k_s, the rod can only bulge
  slightly. Phase 2 brings the ends together, and E_twist converts into E_bend (twist → writhe, the
  mechanism behind the paper's Fig. 1). For slack ≳ 0.35 the coil **passes through itself** (no contact)
  and the energy jumps; handling that is what contact will be for.
  > 中文：失稳点是在精确平衡态上二分 Hessian 最小特征值得到的。没有接触时，slack ≳ 0.35 杆会自己穿过自己，所以默认 slack = 0.3。
- **Zero mode after buckling (isotropic rod).** Rotating the buckled shape rigidly about the axis
  costs no energy, so the Hessian has an eigenvalue ≈ 0 and Newton is slower there. This is a real
  physical symmetry, not a bug.
  > 中文：各向同性杆屈曲后整体绕轴旋转不改变能量，Hessian 有零特征值，这是真实的对称性。
- **Total energy drops in `swing`.** Backward Euler is numerically dissipative; a smaller `dynamics.dt` loses less.
  > 中文：后向欧拉有数值耗散，减小 `dt` 即可减少能量损失。
- **Fig. 5 effect** (natural curvature + anisotropy ⇒ non-uniform twist):
  `python run.py twist --set rod.rest_curvature=3.0 rod.radius_b=0.004 phi=6.0`
  > 中文：自然弯曲加各向异性截面时扭转不再均匀，对应论文 Fig. 5。

## 4. Solver notes (`solver.py`)

- Statics: minimize Π = E_elastic + E_gravity. Dynamics (backward Euler, incremental potential):
  Π = 1/(2h²)(x − x_n − h v_n)ᵀM(x − x_n − h v_n) + E.
- θ has no mass, so in dynamics it is simply minimized every step. That is the quasistatic material
  frame of §5, and lines 6 and 11 of Algorithm 1 collapse into a single Newton solve.
- Newton with backtracking line search. When the Hessian is not positive definite (near buckling),
  μI is added. Gradients/Hessians are dense, computed with `torch.func`; at N ≈ 40 one Hessian takes
  ≈ 0.05 s, fine at toy scale. For larger rods, switch to a sparse hand-derived Hessian.
- Each time the end twist grows by Δ, `Rod.apply_end_twist` first spreads Δ over θ in proportion to
  arc length as a predictor (the solution of Eq. 5). Without this the whole Δ lands on the last vertex
  as one large local twist, and Newton degrades into a gradient-descent-like crawl.

> 中文：静力学和动力学都写成对同一个标量目标函数的最小化，θ 无质量，所以每步直接求解，等价于论文的准静态材料标架。增加末端扭角时先把增量均匀分配到 θ 上作为预测值，否则 Newton 会收敛得很慢。

## 5. Where the time goes (`profile_steps.py`)

`python profile_steps.py buckle --show 1 2 3` wraps every function a load step goes through
(`Rod.apply_end_twist` / `set_bc_targets`, `solver.newton`, `torch.func.grad` / `hessian`, Cholesky,
`track_holonomy`, …) with `time.perf_counter` timers, then runs the experiment through `run.main()`.
`der.py`, `solver.py` and `run.py` are not edited, and the printed results are identical to `python run.py buckle`.
It writes `outputs/profile_<exp>.png`: (a) the whole run to scale, (b) the `--show` load steps with every
Newton iteration, (c) the share of each stage. Every raw span goes to `outputs/profile_<exp>.json`.

`buckle` on a 16-core CPU machine (N = 41, 149 free DOFs, torch 8 threads), one run:

| Phase | Time | Share |
|---|---|---|
| imports (torch 0.80 s, matplotlib 0.25 s) | 1.08 s | 2.1% |
| preparation: config, 2 × `Rod()`, `straight_branch_onset` (19 × ∇²Π) | 2.02 s | 4.0% |
| 49 load steps (537 Newton iterations) | 47.70 s | 93.5% |
| plotting | 0.21 s | 0.4% |

- **Inside a load step, ∇²Π dominates:** about 76–80% of the step is the dense autograd Hessian and about
  15–19% is ∇Π. The linear solve (Cholesky + back-substitution) takes about 0.5–1.4%. The BC change and the
  holonomy update take about 0.1% each.
- **A step's cost is set by its Newton iterations:** roughly (iterations) × (one ∇Π + one ∇²Π) ≈ 85 ms each.
  - Before buckling, a step needs 3 iterations and takes 0.27–0.30 s.
  - Step 17 (Φ = 13.6, just past Φ_c) needs 16 iterations and 43 Cholesky attempts (μI added), about 1.5 s.
  - The slack phase needs 13–30 iterations per step, about 1.1–2.7 s.
- `solver.newton` evaluates ∇Π once more after convergence, only to report `res`. That costs about 13 ms,
  about 5% of a 3-iteration step and 1.3% of all steps.
- The first `torch.func` call of a process pays about 0.6 s of one-time setup (even for a 5-element toy
  function). Here it falls on the first ∇²Π in `straight_branch_onset`: 725 ms vs. 69 ms afterwards.
  In `twist` it falls on the first ∇Π of step 1.

> 中文：`profile_steps.py` 不改动 der/solver/run，只给真实函数套上计时器，再原样运行实验。它画出三张图：全程时间线、展开的加载步、各阶段耗时占比。buckle 实验里，每个加载步约 76–80% 的时间花在稠密 Hessian（autograd）上，梯度约占 15–19%，线性求解只占约 1%，改边界和和乐更新各约 0.1%。每步耗时基本正比于牛顿迭代次数：屈曲前 3 次迭代约 0.27–0.30 s，屈曲那一步 16 次迭代约 1.5 s。要提速，应先换成解析或稀疏的 Hessian。
