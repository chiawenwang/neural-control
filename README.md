# Neural Control

A minimal, self-contained implementation of **Neural Control**, a method for
controlling deformable linear objects (DLOs, e.g. elastic rods/strips) whose
state is defined *implicitly* by an equilibrium condition (`G(x, z) = 0`)
rather than an explicit forward dynamics model.

Paper: **"Neural Control: Adjoint Learning Through Equilibrium Constraints"**
(<https://arxiv.org/abs/2605.03288>)

This branch (`neural-control-core`) contains only the core method: the
equilibrium simulator, the adjoint gradient tools, and the training scripts
needed to reproduce the paper's point-to-point and trajectory-tracking
tasks. Other approaches (RL baseline, STO acceleration variant, real-robot
data preprocessing) live on other branches.

## How it works

Instead of unrolling the Newton-Raphson equilibrium solver to get gradients
(expensive), Neural Control differentiates the equilibrium condition itself
at the converged solution ("implicit adjoint" gradients), then trains a
boundary-actuation controller with those gradients:

1. **`src/SimulatorEnv_2D.py`** — simulates the rod by solving for its
   equilibrium at each timestep, warm-started from the previous step so the
   solver stays on one equilibrium branch instead of jumping between them.
2. **`src/implicit_grad_tools.py`** — computes the adjoint gradient of the
   loss w.r.t. the controls, and chains per-step sensitivities across the
   whole horizon (`multistep_adjoint_gradient()`) so every control step gets
   credit/blame proportional to its own equilibrium Jacobian.
3. **Controller** — maps a continuation parameter `λ ∈ [0, 1]` (progress
   through the control horizon) to boundary actuation rates. Two options:
   - MLP, trained with adjoint gradients (`train_adjoint.py`, `train_tracking.py`)
   - low-order polynomial, trained gradient-free with CMA-ES
     (`train_cmaes.py`) — simpler and the recommended starting point.
4. **Receding-horizon control (RHC)** — optional (`rhc.enabled` in
   `configs/default.yaml`). Splits the horizon into shorter segments,
   optimized and committed one at a time, warm-started from the previous
   segment's end state. Bounds error accumulation over long horizons.
   `train_cmaes.py` always uses this scheme.

## Repository layout

```
neural-control/
├── configs/
│   └── default.yaml            # Global config: rod/task/start pose/RHC/training params
├── src/
│   ├── SimulatorEnv_2D.py      # Equilibrium (Newton-Raphson) rod simulator
│   ├── contact.py              # Penalty-based obstacle contact forces (used by the simulator)
│   ├── implicit_grad_tools.py  # Adjoint / implicit-gradient computation
│   ├── config.py               # YAML config loader (deep-merges onto configs/default.yaml)
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
`tensorboard`, `cma`, `pyyaml`.

## Configuration

All three training scripts are driven by a single YAML config,
[`configs/default.yaml`](configs/default.yaml). Edit it directly, or pass
`--config path/to/your.yaml` with a file that only overrides the fields you
care about (deep-merged onto the defaults). Key sections:

- `rod` — physical/geometric parameters (node count, timestep, horizon, material)
- `start` — initial boundary pose for both rod ends
- `task` — task type (`point_to_point` / `tracking`) and target/trajectory
- `rhc` — whether to use receding-horizon control, segment count, re-init policy
- `train` — epochs, learning rate, network width, seed, device

## Usage

Run everything from the `neural-control/` directory.

```bash
# Task 1: point-to-point boundary control (CMA-ES, recommended entry point)
python scripts/train_cmaes.py

# Task 1: point-to-point, full adjoint-gradient training
python scripts/train_adjoint.py

# Task 2: trajectory tracking (circle/figure8/line, set in configs/default.yaml)
python scripts/train_tracking.py

# Any of the above with a custom config
python scripts/train_adjoint.py --config configs/my_experiment.yaml

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

## Notes vs. the paper

- `train_cmaes.py` uses CMA-ES rather than the paper's analytical adjoint —
  simplest way to reproduce the qualitative result with no per-task gradient
  derivation. `train_adjoint.py` and `train_tracking.py` use the actual
  adjoint gradients.
- The demo simulator uses `nv` nodes (`configs/default.yaml`, `rod.nv`)
  rather than the paper's 101, for faster iteration.
