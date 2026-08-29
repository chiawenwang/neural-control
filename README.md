# Neural Control

A minimal, self-contained implementation of **Neural Control**, a method for
controlling deformable linear objects (DLOs, e.g. elastic rods/strips) whose
state is defined *implicitly* by an equilibrium condition rather than an
explicit forward dynamics model.

Paper: **"Neural Control: Adjoint Learning Through Equilibrium Constraints"**
(<https://arxiv.org/abs/2605.03288>)

This branch (`neural-control-core`) intentionally strips out everything that
is not the core method — no RL baselines, no STO acceleration variant, no
robot-data preprocessing, no exploratory toy-case scripts. Those live on
other branches. What remains here is the equilibrium simulator, the adjoint
gradient tools, and the training/control scripts needed to reproduce the
paper's point-to-point and trajectory-tracking tasks.

## The problem

Many physical systems (elastic rods, cloth, cables, contact-rich
manipulation) don't have a convenient explicit dynamics update `x_{t+1} =
f(x_t, u_t)`. Instead, given a boundary/control input `z` (e.g. the pose of
the two ends of a rod), the internal configuration `x` is whatever satisfies
a force/energy balance:

```
G(x, z) = 0
```

This is a root-finding (equilibrium) problem, not a one-step function
evaluation. Two complications follow:

1. **The system can be multistable.** The same boundary condition `z` can
   correspond to several valid equilibria `x`; which one you end up in
   depends on the actuation *history*, not just the current `z`.
2. **Differentiating through the solver is expensive.** Backpropagating
   through an iterative equilibrium solve (Newton-Raphson, in this case)
   means unrolling every solver iteration.

## The method

**Neural Control** trains a boundary-actuation controller by differentiating
the equilibrium condition directly (implicit/adjoint differentiation)
instead of unrolling the solver, and uses receding-horizon continuation to
keep training anchored to the equilibrium branch actually realized by the
simulator.

1. **Equilibrium-governed simulation** (`src/SimulatorEnv_2D.py`)
   At every timestep, the elastic-rod configuration `x` (free/internal
   nodes) is solved via Newton-Raphson given the current boundary condition
   `z` (controlled end poses), warm-started from the previous equilibrium so
   the solver tracks a *specific* branch rather than jumping between them.

2. **Implicit adjoint gradients** (`src/implicit_grad_tools.py`)
   Rather than backpropagating through the Newton-Raphson iterations, the
   loss gradient w.r.t. the controls is obtained by differentiating the
   equilibrium residual `G(x, z) = 0` at the converged solution:

   ```
   J_ff^T v = a_free          # adjoint solve at the converged equilibrium
   g_qb     = -J_fb^T v       # gradient w.r.t. boundary DOFs
   g_u      = (dq_b/du)^T g_qb  # chain rule to the actual control parameters
   ```

   where `J = dG/dx` is the equilibrium Jacobian, `a_free = dLoss/dx_free`,
   and `q_b`/`u` are the boundary DOFs and control parameters respectively.
   This avoids storing/unrolling the solver's inner iterations — only the
   final converged Jacobian is needed ("frozen-tangent" adjoint).

3. **Neural / parametric controller**
   The controller `u(λ) = u_Θ(λ)` maps a continuation parameter `λ ∈ [0, 1]`
   (progress through the current control horizon) to boundary actuation
   rates. Two parameterizations are provided:
   - an MLP (`scripts/train_adjoint.py`, `scripts/train_tracking.py`),
     trained with the adjoint gradients above via gradient descent;
   - a low-order polynomial (`scripts/train_cmaes.py`), trained
     gradient-free with CMA-ES — simpler, more robust, no gradient
     derivation needed per task, and the recommended starting point.

4. **Receding-horizon control (RHC)**
   A long control horizon is split into `M` shorter segments. Each segment
   is optimized independently and warm-started from the end state of the
   previous one, then executed before moving to the next. This bounds error
   accumulation and prevents the optimizer from exploiting equilibrium
   branches that the warm-started simulator wouldn't actually follow at
   execution time.

## Repository layout

```
neural-control/
├── src/
│   ├── SimulatorEnv_2D.py      # Equilibrium (Newton-Raphson) rod simulator
│   ├── contact.py              # Penalty-based obstacle contact forces (used by the simulator)
│   ├── implicit_grad_tools.py  # Adjoint / implicit-gradient computation
│   ├── render_neurips.py       # DLO rendering utilities
│   └── utils.py                # Result visualization
├── scripts/
│   ├── train_cmaes.py          # Task 1: point-to-point, CMA-ES controller (recommended)
│   ├── train_adjoint.py        # Task 1: point-to-point, adjoint-gradient MLP controller
│   ├── train_tracking.py       # Task 2: trajectory tracking, adjoint-gradient MLP controller
│   ├── snapshot_overlay.py     # Multi-frame DLO configuration visualization
│   └── test_gpu.py             # MPS/GPU availability check
├── img/                        # README figures
└── outputs/                    # Generated figures (created at runtime)
```

## Setup

```bash
conda env create -f requirement.yml
conda activate neural-control
```

or install directly with pip: `numpy`, `matplotlib`, `tqdm`, `torch`,
`tensorboard`, `cma`.

## Usage

Run everything from the `neural-control/` directory.

```bash
# Task 1: point-to-point boundary control (CMA-ES, recommended entry point)
python scripts/train_cmaes.py

# Task 1: point-to-point, full adjoint-gradient training
python scripts/train_adjoint.py --epochs 200

# Task 2: trajectory tracking (circle by default)
python scripts/train_tracking.py --epochs 300 --traj circle

# Visualize a resulting trajectory as overlaid snapshots
python scripts/snapshot_overlay.py --traj circle
```

Programmatic usage (mirrors `train_cmaes.py`):

```python
import sys; sys.path.insert(0, 'src')
import numpy as np
from SimulatorEnv_2D import SimulatorEnv_2D
from scripts.train_cmaes import NeuralControlAgent

env = SimulatorEnv_2D(nv=15, dt=0.01, rod_length=1.0, total_time=1.0)
agent = NeuralControlAgent(env, segment_horizon=6, degree=3)
target_pos = np.array([0.5, 0.2])

results = agent.run_rhc(
    target_pos=target_pos,
    n_segments=4,
    n_iterations_per_segment=25,
)
```

## Example result

Task 1 (point-to-point): elastic strip with 15 nodes, midpoint driven from
`(0.50, 0.00)` to target `(0.50, 0.20)` via CMA-ES + RHC, 4 segments × 25
iterations, final tracking error ≈ 0.0227 m.

![Neural Control result](img/neural_control_result.png)

## Notes on this implementation vs. the paper

- `train_cmaes.py` uses CMA-ES rather than the paper's analytical adjoint —
  it's the simplest, most robust way to reproduce the qualitative result and
  needs no per-task gradient derivation. `train_adjoint.py` and
  `train_tracking.py` use the actual adjoint gradients described above.
- The demo simulator uses 15 nodes rather than the paper's 101, for faster
  iteration; `SimulatorEnv_2D`'s `nv` argument controls this.
- Other approaches explored for this problem (an RL baseline, an accelerated
  "Stateful Tangent Operator" variant, real-robot data preprocessing) are
  kept out of this branch to keep the core method easy to read and run; see
  other branches of this repository for those.
