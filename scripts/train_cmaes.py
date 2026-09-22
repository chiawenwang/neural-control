# -*- coding: utf-8 -*-
"""
Neural Control: Simplified Implementation using Evolution Strategy
Task 1: Point-to-point manipulation (drive midpoint to target position)

This version uses a simpler optimization method for demonstration
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

# Must run before numpy is imported: OpenBLAS goes multithreaded at about
# n = 100, and for the small dense solves here (94x94 in 2D, 141x141 in 3D)
# the threading overhead costs ~2000x. See src/blas_threads.py.
from blas_threads import limit_blas_threads
limit_blas_threads()

import argparse
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import time

from utils import visualize_results
from config import load_config
from env_factory import (
    build_env,
    reset_env,
    env_dim,
    control_dim,
    initial_control,
    task_target,
    node_position,
)


class SimpleController:
    """Simple polynomial controller u(λ) = sum_i a_i * λ^i"""
    def __init__(self, output_dim: int = 6, degree: int = 3, u_max: float = 0.1):
        self.output_dim = output_dim
        self.degree = degree
        self.u_max = u_max
        
        # Parameters: coefficients for each output dimension
        self.params = np.zeros((output_dim, degree + 1))
        
    def forward(self, lamda: float) -> np.ndarray:
        """Evaluate polynomial at λ"""
        # Compute powers of λ
        powers = np.array([lamda ** i for i in range(self.degree + 1)])
        
        # Evaluate polynomial for each output
        u = self.params @ powers
        
        # Apply bounds
        u = np.tanh(u) * self.u_max
        
        return u
    
    def get_params(self):
        return self.params.flatten()
    
    def set_params(self, params_flat):
        self.params = params_flat.reshape(self.output_dim, self.degree + 1)


class CMAESOptimizer:
    """Simple CMA-ES optimizer"""
    def __init__(self, n_params: int, pop_size: int = 20, sigma: float = 0.3):
        self.n_params = n_params
        self.pop_size = pop_size
        self.sigma = sigma
        
        # Mean
        self.mean = np.zeros(n_params)
        
        # Covariance (diagonal for simplicity)
        self.C = np.eye(n_params)
        
    def ask(self) -> List[np.ndarray]:
        """Sample population"""
        population = []
        for _ in range(self.pop_size):
            z = np.random.randn(self.n_params)
            x = self.mean + self.sigma * (self.C @ z)
            population.append(x)
        return population
    
    def tell(self, population: List[np.ndarray], fitness: List[float]):
        """Update distribution based on fitness"""
        # Sort by fitness (lower is better)
        idx = np.argsort(fitness)
        
        # Select top half as elites
        n_elite = self.pop_size // 2
        elites = [population[i] for i in idx[:n_elite]]
        
        # Update mean
        self.mean = np.mean(elites, axis=0)
        
        # Update covariance (simple version: use empirical covariance of elites)
        if len(elites) > 1:
            elite_array = np.array(elites)
            self.C = np.cov(elite_array.T) + 1e-6 * np.eye(self.n_params)
            
            # Shrink sigma slightly
            self.sigma *= 0.99


class NeuralControlAgent:
    """Agent using evolution strategy for optimization.

    Works with either environment: ``env.dim`` selects the control width
    (6D in 2D, 10D in 3D) and the rollout's boundary-pose bookkeeping.
    """
    def __init__(self, env, segment_horizon: int = 8, degree: int = 3, 
                 u_max: float = 0.1, pop_size: int = 15):
        self.env = env
        self.dim = getattr(env, "dim", 2)
        self.n_u = control_dim(self.dim)
        self.segment_horizon = segment_horizon
        self.degree = degree
        self.u_max = u_max
        self.pop_size = pop_size
        
        # Controller
        self.controller = SimpleController(output_dim=self.n_u, degree=degree, u_max=u_max)
        
        # Optimizer
        n_params = self.controller.get_params().size
        self.optimizer = CMAESOptimizer(n_params=n_params, pop_size=pop_size, sigma=0.2)
        
    def forward_rollout(self, initial_q: np.ndarray, lamda_start: float,
                       steps: int, safe: bool = True) -> Dict:
        """Execute forward rollout"""
        if self.dim == 2:
            from SimulatorEnv_2D import objfun
        else:
            from SimulatorEnv_3D import objfun_3d as objfun
        
        q_traj = [initial_q.copy()]
        u_traj = []
        q_current = initial_q.copy()
        failed = False
        
        for k in range(steps):
            lamda = lamda_start + (k + 1) / steps
            
            # Get control
            u = self.controller.forward(lamda)
            u_traj.append(u)
            
            # Build control vector incrementally from current state.
            # Per end the layout is (position..., angles...):
            #   2D -> (x, y, theta)            3D -> (x, y, z, theta, phi)
            d = self.dim
            n_ang = d - 1                       # 1 angle in 2D, 2 in 3D
            per_end = d + n_ang
            left_pos = q_current[:d]            # node_0
            right_pos = q_current[-d:]          # node_{n-1}

            # Smooth incremental control
            delta_scale = 0.05  # Small steps
            control_vec = np.empty(self.n_u)
            control_vec[:d] = left_pos + u[:d] * delta_scale
            control_vec[d:per_end] = u[d:per_end] * 0.1
            control_vec[per_end:per_end + d] = right_pos + u[per_end:per_end + d] * delta_scale
            control_vec[per_end + d:] = u[per_end + d:] * 0.1

            control_node = self.env._control_node_from_xytheta(control_vec)
            
            # Solve equilibrium
            try:
                q_new, flag, J, _ninfo = objfun(
                    q_old=q_current,
                    u_old=np.zeros_like(q_current),
                    dt=self.env.dt,
                    tol=self.env.tol * 10,  # Looser tolerance for speed
                    maximum_iter=50,  # Fewer iterations
                    m=self.env.m,
                    mMat=self.env.mMat,
                    energy=self.env.energy,
                    W=self.env.W,
                    deltaL=self.env.deltaL,
                    free_index=self.env.free_index,
                    control_node=control_node,
                    use_inertia=False
                )
                
                if flag < 0:
                    failed = True
                    break
                
                q_traj.append(q_new.copy())
                q_current = q_new
                
            except:
                failed = True
                break
        
        return {'q_traj': q_traj, 'u_traj': u_traj, 'failed': failed}
    
    def compute_loss(self, rollout_data: Dict, target_pos: np.ndarray) -> float:
        """Compute loss with penalties"""
        if rollout_data.get('failed', False) or len(rollout_data['q_traj']) < 2:
            return 1e6
        
        q_final = rollout_data['q_traj'][-1]
        p_mid = node_position(q_final, self.dim, self.env.mid_node)
        
        # Distance to target
        dist = np.linalg.norm(p_mid - target_pos)
        
        # Regularization: penalize large controls
        u_penalty = 0.0
        for u in rollout_data['u_traj']:
            u_penalty += np.sum(u ** 2) * 0.001
        
        return dist ** 2 + u_penalty
    
    def train_segment(self, initial_q: np.ndarray, target_pos: np.ndarray,
                     lamda_start: float, n_iterations: int = 30, verbose: bool = True) -> Dict:
        """Train using CMA-ES"""
        best_loss = float('inf')
        best_params = None
        loss_history = []
        
        for iter in range(n_iterations):
            # Ask for population
            population = self.optimizer.ask()
            
            # Evaluate population
            fitness = []
            for params in population:
                self.controller.set_params(params)
                rollout = self.forward_rollout(initial_q, lamda_start, self.segment_horizon)
                loss = self.compute_loss(rollout, target_pos)
                fitness.append(loss)
            
            # Update optimizer
            self.optimizer.tell(population, fitness)
            
            # Track best
            min_loss = min(fitness)
            if min_loss < best_loss:
                best_loss = min_loss
                best_params = population[np.argmin(fitness)].copy()
            
            loss_history.append(min_loss)
            
            if verbose and iter % 5 == 4:
                print(f"  Iteration {iter+1}/{n_iterations}: loss={min_loss:.6f}, best={best_loss:.6f}")
            
            # Early stopping
            if best_loss < 1e-5:
                if verbose:
                    print(f"  Early stop at iteration {iter+1}")
                break
        
        # Set best parameters
        if best_params is not None:
            self.controller.set_params(best_params)
        
        # Final rollout
        final_rollout = self.forward_rollout(initial_q, lamda_start, self.segment_horizon)
        
        return {
            'losses': loss_history,
            'best_loss': best_loss,
            'final_rollout': final_rollout
        }
    
    def run_rhc(self, target_pos: np.ndarray, n_segments: int = 5,
               n_iterations_per_segment: int = 30, verbose: bool = True) -> Dict:
        """Run RHC"""
        obs = self.env.reset()
        initial_q = self.env.q.copy()
        
        all_losses = []
        segment_losses = []
        all_rollouts = []
        
        for m in range(n_segments):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Segment {m+1}/{n_segments}")
                print(f"{'='*60}")
            
            # Reinitialize controller
            self.controller = SimpleController(output_dim=self.n_u, degree=self.degree, u_max=self.u_max)
            n_params = self.controller.get_params().size
            self.optimizer = CMAESOptimizer(n_params=n_params, pop_size=self.pop_size, sigma=0.15)
            
            # Train
            lamda_start = m / n_segments
            result = self.train_segment(initial_q, target_pos, lamda_start, 
                                       n_iterations_per_segment, verbose)
            
            all_losses.extend(result['losses'])
            segment_losses.append(result['best_loss'])
            all_rollouts.append(result['final_rollout'])
            
            # Update for next segment
            initial_q = result['final_rollout']['q_traj'][-1]
            
            # Check error
            p_mid = node_position(initial_q, self.dim, self.env.mid_node)
            error = np.linalg.norm(p_mid - target_pos)
            
            if verbose:
                print(f"Segment {m+1} error: {error:.6f}")
            
            if error < 5e-3:
                print(f"\nTarget reached after {m+1} segments!")
                break
        
        return {
            'all_losses': all_losses,
            'segment_losses': segment_losses,
            'all_rollouts': all_rollouts,
            'final_error': error,
            'final_q': initial_q
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config overriding configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    print("="*60)
    print("Neural Control: Task 1 - Point-to-Point Manipulation")
    print("Simplified Implementation (CMA-ES)")
    print("="*60)

    # Create environment
    print("\nInitializing environment...")
    dim = env_dim(cfg)
    env = build_env(cfg)
    reset_env(env, cfg)

    # Target
    target_pos = task_target(cfg)

    print(f"Environment: {type(env).__name__}  (dim={dim}, control={control_dim(dim)}D)")
    print("\nTarget: (" + ", ".join(f"{v:.2f}" for v in target_pos) + ")")
    print("Initial midpoint: ("
          + ", ".join(f"{v:.2f}" for v in env.nodes[env.mid_node]) + ")")

    # RHC toggle: disabled -> a single "segment" spanning the whole horizon.
    n_segments = cfg.rhc.num_segments if cfg.rhc.enabled else 1
    print(f"RHC: {'enabled (' + str(n_segments) + ' segments)' if cfg.rhc.enabled else 'disabled (single horizon)'}")

    # Create agent
    agent = NeuralControlAgent(
        env=env,
        segment_horizon=max(1, env.Nsteps // n_segments),
        degree=3,
        u_max=0.15,
        pop_size=12
    )

    # Run
    print("\nStarting Neural Control...")
    start_time = time.time()

    results = agent.run_rhc(
        target_pos=target_pos,
        n_segments=n_segments,
        n_iterations_per_segment=cfg.train.iterations_per_segment,
        verbose=True
    )

    elapsed = time.time() - start_time

    # Results
    print(f"\n{'='*60}")
    print("Completed!")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Final error: {results['final_error']:.6f}")
    print(f"Segment losses: {[f'{l:.4f}' for l in results['segment_losses']]}")

    # Visualize
    visualize_results(results, env, target_pos, save_path=str(_ROOT / 'outputs'))

    return results


if __name__ == "__main__":
    results = main()
