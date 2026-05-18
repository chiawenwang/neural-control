# neural-control
Neural control in DLO manipulation

<!-- img -->
![Neural Control for DLO Manipulation](img/neural_control_result.png)

# Neural Control for Deformable Linear Objects (DLO)

Implementation of the paper: **"Neural Control: Adjoint Learning Through Equilibrium Constraints"**

## Task 1: Point-to-Point Manipulation

**Objective**: Drive the midpoint of an elastic strip to a target position using boundary control.

## Method Overview

This implementation follows the Neural Control framework from the paper:

### 1. **Equilibrium-Governed System**
- The elastic strip state is defined implicitly by equilibrium conditions
- At each step, we solve: `G(x, z) = 0` where:
  - `x`: free degrees of freedom (internal nodes)
  - `z`: controllable boundary conditions

### 2. **Neural Controller**
- Parameterized controller: `u(λ) = u_Θ(λ)`
- `λ ∈ [0,1]`: continuation parameter
- Controller outputs boundary actuation rates

### 3. **Receding Horizon Control (RHC)**
- Split long horizon into M short segments
- For each segment:
  - Initialize controller parameters
  - Optimize over local horizon
  - Execute and shift to next segment
- This mitigates error accumulation and basin switching in multi-stable systems

### 4. **Optimization**
Two implementations provided:

#### a) **Full Adjoint-based Gradients** (`neural_control_numpy.py`)
- Computes trajectory-dependent gradients via implicit differentiation
- Frozen-tangent approximation: treats sensitivity `S(λ) = ∂x/∂z` as locally constant
- Memory-efficient: avoids unrolling equilibrium solver iterations
- **Note**: Currently uses numerical gradients (finite differences) for simplicity
  - In production, should use analytical Jacobians for speed

#### b) **Simplified CMA-ES** (`neural_control_simple.py` - RECOMMENDED)
- Uses evolution strategy (CMA-ES) for optimization
- Simpler and more robust for demonstration
- Polynomial controller: `u(λ) = Σ aᵢ λⁱ`
- Successfully demonstrated on the task

## Results

### Task 1: Point-to-Point Manipulation

**Setup**:
- Elastic strip with 15 nodes
- Initial midpoint: (0.50, 0.00)
- Target position: (0.50, 0.20)

**Performance**:
- **Method**: Neural Control + RHC (CMA-ES)
- **Segments**: 4
- **Iterations per segment**: 25
- **Total time**: 141.78s
- **Final error**: 0.0227 m
- **Segment losses**: [0.0268, 0.0143, 0.0084, 0.0008]

The controller successfully drives the midpoint from (0.50, 0.00) to near the target (0.50, 0.20) through coordinated boundary actuation.

## Key Features Implemented

✅ **Equilibrium-based simulation**: Solves implicit elastic rod equilibrium at each step
✅ **Continuation control**: Controller parameterized by λ ∈ [0,1]
✅ **Receding Horizon Control**: Segment-wise optimization with parameter reinitialization
✅ **Multi-stability handling**: Warm-started solver follows realized equilibrium branch
✅ **Smooth actuation**: Incremental boundary control for numerical stability

## Project Structure

```
neural-control/
├── src/                        # Core library
│   ├── SimulatorEnv_2D.py      # Elastic rod physics simulator
│   ├── implicit_grad_tools.py  # Adjoint/implicit gradient tools
│   ├── sto_simulation.py       # Stateful Tangent Operator (STO)
│   ├── render_neurips.py       # NeurIPS-style DLO renderer
│   └── utils.py                # Shared visualization utilities
│
├── scripts/                    # Entry-point scripts (run from neural-control/)
│   ├── train_adjoint.py        # Task 1: adjoint-based gradient training
│   ├── train_cmaes.py          # Task 1: CMA-ES evolution strategy (recommended)
│   ├── train_tracking.py       # Task 2: trajectory tracking
│   ├── train_sto.py            # Task 1: STO-accelerated adjoint training
│   ├── snapshot_overlay.py     # DLO snapshot visualization
│   └── test_gpu.py             # MPS/GPU availability check
│
├── experiments/                # TensorBoard logs (auto-created at runtime)
├── outputs/                    # Generated figures and PDFs
├── toy_case/                   # Toy-case analysis and figures
├── preprocessing/              # Robot data preprocessing (Sawyer ee/force data)
├── rl_based/                   # RL baseline (MuJoCo DLO environment)
└── img/                        # README images
```

## Usage

```bash
# Run from the neural-control/ directory

# Task 1: point-to-point (CMA-ES, recommended)
python scripts/train_cmaes.py

# Task 1: adjoint-based gradient
python scripts/train_adjoint.py --epochs 200

# Task 1: STO-accelerated adjoint
python scripts/train_sto.py --epochs 200 --kappa_abs 1e6

# Task 2: trajectory tracking
python scripts/train_tracking.py --epochs 300 --traj circle

# DLO snapshot visualization
python scripts/snapshot_overlay.py --traj angle_oscillation
```

```python
# Programmatic usage
import sys; sys.path.insert(0, 'src')
from SimulatorEnv_2D import SimulatorEnv_2D
from scripts.train_cmaes import NeuralControlAgent

env = SimulatorEnv_2D(nv=15, dt=0.01, rod_length=1.0, total_time=1.0)
agent = NeuralControlAgent(env, segment_horizon=6, degree=3)
target_pos = np.array([0.5, 0.2])

results = agent.run_rhc(
    target_pos=target_pos,
    n_segments=4,
    n_iterations_per_segment=25
)
```

## Dependencies

- NumPy
- Matplotlib
- SimulatorEnv_2D (provided elastic strip simulator)

## Implementation Notes

### Differences from Paper

1. **Gradient Computation**: 
   - Paper uses analytical adjoint sensitivities
   - This demo uses CMA-ES (evolution strategy) for robustness and simplicity
   - Numerical adjoint version provided but slower

2. **Controller Architecture**:
   - Paper uses MLP with multiple hidden layers
   - This demo uses polynomial controller for fewer parameters

3. **Scale**:
   - Paper uses N=101 nodes
   - This demo uses N=15 nodes for faster demonstration

### Numerical Stability

The implementation includes several stability measures:
- Small incremental control steps (δ-scaling)
- Warm-started equilibrium solver
- Failure detection and penalty
- Regularization on control magnitudes

## Future Improvements

1. **Analytical Adjoint Gradients**: Implement full implicit differentiation with analytical Jacobians
2. **GPU Acceleration**: Port equilibrium solver to GPU for parallel rollouts
3. **Advanced Tasks**: 
   - Task 2: Trajectory tracking
   - Task 3: Multi-stable shape formation
4. **Real Robot Transfer**: Sim-to-real with domain randomization
