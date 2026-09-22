# Neural Control

A minimal, self-contained implementation of **Neural Control**, a method for
controlling deformable linear objects (DLOs, e.g. elastic rods/strips) whose
state is defined *implicitly* by an equilibrium condition (`G(x, z) = 0`)
rather than an explicit forward dynamics model.

This branch (`neural-control-core`) contains only the core method: the
equilibrium simulator, the adjoint gradient tools, and the training scripts
needed to reproduce the paper's point-to-point and trajectory-tracking
tasks.

## How it works

Instead of unrolling the Newton-Raphson equilibrium solver to get gradients
(expensive), Neural Control differentiates the equilibrium condition itself
at the converged solution ("implicit adjoint" gradients), then trains a
boundary-actuation controller with those gradients:

1. **`src/SimulatorEnv_2D.py` / `src/SimulatorEnv_3D.py`** — simulate the rod
   by solving for its equilibrium at each timestep, warm-started from the
   previous step so the solver stays on one equilibrium branch instead of
   jumping between them. Pick one with `env.dim` in the config (see
   [2D vs. 3D](#2d-vs-3d)).
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
│   ├── SimulatorEnv_2D.py      # Equilibrium (Newton-Raphson) rod simulator, 2D
│   ├── SimulatorEnv_3D.py      # Equilibrium (Newton-Raphson) rod simulator, 3D
│   ├── energy.py               # Pluggable elastic-energy models (der, der3d, custom)
│   ├── contact.py              # Penalty-based obstacle contact forces (Circle/Sphere)
│   ├── implicit_grad_tools.py  # Adjoint / implicit-gradient computation (2D and 3D)
│   ├── env_factory.py          # dim-aware env/control/target construction
│   ├── config.py               # YAML config loader (deep-merges onto configs/default.yaml)
│   ├── render_neurips.py       # DLO rendering utilities (2D)
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

- `env` — `dim: 2` or `dim: 3`, i.e. which simulator to use
- `rod` — physical/geometric parameters (node count, timestep, horizon, material,
  and `gravity` in 3D)
- `energy` — elastic energy model (built-in DER, or your own; see `src/energy.py`)
- `start` — initial boundary pose for both rod ends
- `task` — task type (`point_to_point` / `tracking`) and target/trajectory
- `newton` — equilibrium-solver safeguards (line search, regularization); see
  [Newton convergence](#newton-convergence)
- `rhc` — whether to use receding-horizon control, segment count, re-init policy
- `train` — epochs, learning rate, network width, seed, device

## 2D vs. 3D

Set `env.dim` in the config; all three training scripts, the adjoint
gradients, contact, and the plots follow it.

|                      | `dim: 2`                   | `dim: 3`                                 |
| -------------------- | -------------------------- | ---------------------------------------- |
| simulator            | `SimulatorEnv_2D`          | `SimulatorEnv_3D`                        |
| DOFs per node        | `(x, y)`                   | `(x, y, z)`                              |
| control per rod end  | `(x, y, θ)`                | `(x, y, z, θ, φ)`                        |
| full control vector  | 6D                         | 10D                                      |
| elastic energy       | `der`                      | `der3d`                                  |
| obstacles            | `CircleObstacle`           | `SphereObstacle`                         |
| target / trajectory  | `task.target_xy`           | `task.target_xyz`                        |
| initial end poses    | `start.u_left`/`u_right`   | `start.u_left_3d`/`u_right_3d`           |

Both rod ends are **clamped**: the control prescribes each end's position and
its tangent direction. In 3D the tangent needs two angles instead of one —
azimuth `θ` in the x-y plane and elevation `φ` out of it:

```
d(θ, φ) = ( cos φ · cos θ,  cos φ · sin θ,  sin φ )
```

so `φ = 0` reproduces the 2D convention exactly. The two clamped nodes at each
end are `node_0 = p_l`, `node_1 = p_l + ΔL·d(θ_l, φ_l)` and symmetrically at the
right end.

The 3D elastic energy (`der3d` in `src/energy.py`) is the Discrete Elastic Rod
model for a naturally straight rod: per-edge stretching springs plus isotropic
bending `E_b = Σ (EI / 2ΔL)·|κb|²`. Because the rod is naturally straight and
bending is isotropic, `|κb|²` reduces to a function of the single scalar
`c = t_e · t_f`, which is what makes the *exact* Hessian — required by the
adjoint gradient — cheap to write in closed form. A per-edge **twist** DOF
would need material frames and parallel transport, and is deliberately not
modelled; the 3D environment captures out-of-plane bending only.

Gravity in 3D is the config vector `rod.gravity`, defaulting to
`[0, -9.81, 0]` so that y stays "down" as in 2D. With that default, a 3D run
confined to `z = 0` and `φ = 0` reproduces the 2D run node for node (verified
to machine precision). Use `[0, 0, -9.81]` for the usual z-up convention.

Tracking trajectories: `circle`, `figure8` and `line` work in both dims (the
planar curves sit at `z = 0` in 3D); `helix` and `figure8_3d` are 3D-only.

`scripts/snapshot_overlay.py` and `src/render_neurips.py` remain 2D-only.

## Performance

The dominant per-step cost is the dense solve `J_ff \ f_free`, where
`n_free = dim*nv - 4*dim` — **94** for the 2D default (nv=51) and **141** for
the 3D one. That 1.5x growth in DOFs is not what makes 3D slow. OpenBLAS
switches from its serial kernel to a threaded one at about n = 100, and for
matrices this small the thread spawn/sync overhead dwarfs the factorization.
Measured on a 16-core machine:

| solve size | default threads | `OPENBLAS_NUM_THREADS=1` |
| ---------- | --------------- | ------------------------ |
| 94 x 94 (2D default)  | 0.057 ms   | 0.057 ms |
| 100 x 100             | **96.4 ms**| 0.063 ms |
| 141 x 141 (3D default)| **260.8 ms**| 0.127 ms |
| 256 x 256             | 604.8 ms   | 0.615 ms |

2D lands just under the threshold and 3D just over it, so 3D paid a ~2000x
penalty on its single hottest operation. The training scripts therefore call
`limit_blas_threads()` ([`src/blas_threads.py`](src/blas_threads.py)) before
importing NumPy. Real effect, 2 epochs of the default 3D config
(nv=51, N=200): **396 s -> 22 s, ~18x**.

The variables are set with `setdefault`, so the shell always wins if you want
something else:

```bash
OPENBLAS_NUM_THREADS=4 python scripts/train_adjoint.py --config configs/example_3d.yaml
```

If you drive the simulator from your own script rather than `scripts/train_*.py`,
do the same thing before NumPy loads:

```python
import sys; sys.path.insert(0, 'src')
from blas_threads import limit_blas_threads
limit_blas_threads()
import numpy as np          # only now
```

After that fix the bottleneck is the **Python-loop assembly** of the elastic
force and Hessian (~8.6 ms per call in 3D, ~18.3 ms in 2D), at ~10 Newton
iterations per step. Vectorizing those stencil loops — not a faster linear
solver — is the next worthwhile optimization; the Hessian is banded (only
2.8% nonzero in 3D), so a banded solver would help only for much larger `nv`.

## Newton convergence

If training prints a stream of `Maximum number of iterations reached.`, the
per-step equilibrium solve is diverging. The cause is almost always the same:
a **compressed rod is post-buckled**, so the equilibrium Jacobian `J_ff` has
negative eigenvalues (one pair per buckling mode — and 3D has twice as many as
2D, because every mode exists in both the y and the z direction). An undamped
full Newton step on an indefinite system can increase the residual without
bound; in practice it runs away from ~1e0 to ~1e18 in a few iterations, burns
all `rod.max_newton_iter` iterations, and prints that message.

The solver in [`src/newton.py`](src/newton.py) is globalized against this:

1. **backtracking line search** — the full Newton step is always tried first
   and only halved if it fails to reduce `||residual||`;
2. **adaptive Tikhonov regularization** — if no step length helps, `J_ff` is
   shifted by `λ·scale·I` with λ escalating, bending the step toward gradient
   descent; λ relaxes again on success.

Neither changes the answer when plain Newton was already converging (verified
bit-identical, ~1e-16). Where plain Newton diverged, the difference is
decisive — one 2D compression case returns a rod of arc length 3.5×10⁶ (i.e.
garbage) undamped, and the correct arc length 0.998 with the line search on.

Knobs (`newton:` in the config):

```yaml
newton:
  line_search: true    # false -> exactly the original undamped iteration
  max_backtrack: 12
  reg: 0.0             # raise to ~1e-8 for a persistently unstable setup
  verbose: false       # one-line diagnosis (residual, iterations, reason) per failed step
```

`env.n_newton_failures` counts non-converged steps, and `info["newton"]` from
`env.step()` carries the per-step residual, iteration count and failure reason.

If failures persist after this, the physics — not the solver — is the problem.
Check in this order: the control is compressing the rod far past buckling
(shorten the travel or raise `rod.total_time` so each step moves less); the
rod is too stiff for the step size (`rod.E_al`, `rod.R_outer`); or the target
is simply unreachable, so the policy drives the ends to extremes.

## Usage

Run everything from the `neural-control/` directory.

```bash
# Task 1: point-to-point boundary control (CMA-ES, recommended entry point)
python scripts/train_cmaes.py

# Task 1: point-to-point, full adjoint-gradient training
python scripts/train_adjoint.py

# Task 2: trajectory tracking (circle/figure8/line, set in configs/default.yaml)
python scripts/train_tracking.py

# Any of the above in 3D: set `env: {dim: 3}` in your config
python scripts/train_adjoint.py --config configs/my_3d_experiment.yaml

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

The same in 3D — only the env class, the control width and the target change:

```python
from SimulatorEnv_3D import SimulatorEnv_3D

env = SimulatorEnv_3D(nv=15, dt=0.01, rod_length=1.0, total_time=1.0)
agent = NeuralControlAgent(env, segment_horizon=6, degree=3)   # picks up env.dim
results = agent.run_rhc(target_pos=np.array([0.5, 0.2, 0.1]), n_segments=4,
                        n_iterations_per_segment=25)
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

## Still missing in 3D

The 3D environment models out-of-plane bending, not torsion. Adding twist
would mean:

- a per-edge material frame with parallel transport, and a twisting energy
  (`getFt`) alongside the existing stretching/bending terms in `der3d`;
- state grows from `3*nv` positions to `3*nv + (nv-1)` twist angles, with
  `free_index`/`fixed_index` extended to match;
- the end-effector control grows from `(x, y, z, θ, φ)` to a full 6-DOF pose
  per end — at which point the tangent-angle parameterization should be
  replaced by axis-angle or quaternions (Euler angles gimbal-lock);
- `implicit_grad_tools.py` still needs **no change** beyond the `d(q_b)/d(u)`
  map: the adjoint solve itself is dimension-agnostic linear algebra.
