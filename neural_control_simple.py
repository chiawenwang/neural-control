# -*- coding: utf-8 -*-
"""
Neural Control: Simplified Implementation using Evolution Strategy
Task 1: Point-to-point manipulation (drive midpoint to target position)

This version uses a simpler optimization method for demonstration
"""

import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import time


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
    """Agent using evolution strategy for optimization"""
    def __init__(self, env, segment_horizon: int = 8, degree: int = 3, 
                 u_max: float = 0.1, pop_size: int = 15):
        self.env = env
        self.segment_horizon = segment_horizon
        self.degree = degree
        self.u_max = u_max
        self.pop_size = pop_size
        
        # Controller
        self.controller = SimpleController(output_dim=6, degree=degree, u_max=u_max)
        
        # Optimizer
        n_params = self.controller.get_params().size
        self.optimizer = CMAESOptimizer(n_params=n_params, pop_size=pop_size, sigma=0.2)
        
    def forward_rollout(self, initial_q: np.ndarray, lamda_start: float, 
                       steps: int, safe: bool = True) -> Dict:
        """Execute forward rollout"""
        from SimulatorEnv_2D import objfun
        
        q_traj = [initial_q.copy()]
        u_traj = []
        q_current = initial_q.copy()
        failed = False
        
        for k in range(steps):
            lamda = lamda_start + (k + 1) / steps
            
            # Get control
            u = self.controller.forward(lamda)
            u_traj.append(u)
            
            # Build control vector incrementally from current state
            current_left_x = q_current[0]
            current_left_y = q_current[1]
            current_right_x = q_current[-2]
            current_right_y = q_current[-1]
            
            # Smooth incremental control
            delta_scale = 0.05  # Small steps
            new_left_x = current_left_x + u[0] * delta_scale
            new_left_y = current_left_y + u[1] * delta_scale
            new_left_theta = u[2] * 0.1
            new_right_x = current_right_x + u[3] * delta_scale
            new_right_y = current_right_y + u[4] * delta_scale
            new_right_theta = u[5] * 0.1
            
            control_vec = np.array([new_left_x, new_left_y, new_left_theta,
                                   new_right_x, new_right_y, new_right_theta])
            control_node = self.env._control_node_from_xytheta(control_vec)
            
            # Solve equilibrium
            try:
                q_new, flag, J = objfun(
                    q_old=q_current,
                    u_old=np.zeros_like(q_current),
                    dt=self.env.dt,
                    tol=self.env.tol * 10,  # Looser tolerance for speed
                    maximum_iter=50,  # Fewer iterations
                    m=self.env.m,
                    mMat=self.env.mMat,
                    EI=self.env.EI,
                    EA=self.env.EA,
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
        mid_idx = self.env.mid_node
        p_mid = np.array([q_final[2*mid_idx], q_final[2*mid_idx+1]])
        
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
            self.controller = SimpleController(output_dim=6, degree=self.degree, u_max=self.u_max)
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
            mid_idx = self.env.mid_node
            p_mid = np.array([initial_q[2*mid_idx], initial_q[2*mid_idx+1]])
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


def visualize_results(results, env, target_pos, save_path='/mnt/user-data/outputs'):
    """Visualize results"""
    final_q = results['final_q']
    x_arr = final_q[::2]
    y_arr = final_q[1::2]
    mid_idx = env.mid_node
    midx, midy = x_arr[mid_idx], y_arr[mid_idx]
    
    # Plot configuration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Configuration plot
    ax1.plot(x_arr, y_arr, 'ko-', linewidth=2, markersize=6, label='Final config', alpha=0.7)
    ax1.plot(x_arr[0], y_arr[0], 'gs', markersize=10, label='Left boundary')
    ax1.plot(x_arr[-1], y_arr[-1], 'gs', markersize=10)
    ax1.plot(midx, midy, 'ro', markersize=12, label='Midpoint', zorder=10)
    ax1.plot(target_pos[0], target_pos[1], 'bx', markersize=15, markeredgewidth=3, 
            label='Target', zorder=10)
    ax1.plot([midx, target_pos[0]], [midy, target_pos[1]], 'r--', alpha=0.5, linewidth=1,
            label=f'Error: {results["final_error"]:.4f}')
    ax1.set_xlabel('x (m)', fontsize=11)
    ax1.set_ylabel('y (m)', fontsize=11)
    ax1.set_title('Final Configuration', fontsize=12, fontweight='bold')
    ax1.axis('equal')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Loss curve
    ax2.semilogy(results['all_losses'], linewidth=2, color='steelblue')
    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Loss (log scale)', fontsize=11)
    ax2.set_title('Training Progress', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/neural_control_result.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to {save_path}/neural_control_result.png")
    plt.close()


def main():
    print("="*60)
    print("Neural Control: Task 1 - Point-to-Point Manipulation")
    print("Simplified Implementation (CMA-ES)")
    print("="*60)
    
    import sys
    sys.path.append('/mnt/user-data/uploads')
    from SimulatorEnv_2D import SimulatorEnv_2D
    
    # Create environment
    print("\nInitializing environment...")
    env = SimulatorEnv_2D(
        nv=15,  # Fewer nodes for speed
        dt=0.01,
        rod_length=1.0,
        total_time=1.0,
        save_history=False
    )
    
    # Target
    target_x = 0.4
    target_y = 0.25
    target_pos = np.array([target_x, target_y])
    
    print(f"\nTarget: ({target_x:.2f}, {target_y:.2f})")
    print(f"Initial midpoint: ({env.nodes[env.mid_node, 0]:.2f}, {env.nodes[env.mid_node, 1]:.2f})")
    
    # Create agent
    agent = NeuralControlAgent(
        env=env,
        segment_horizon=6,
        degree=3,
        u_max=0.15,
        pop_size=12
    )
    
    # Run
    print("\nStarting Neural Control with RHC...")
    start_time = time.time()
    
    results = agent.run_rhc(
        target_pos=target_pos,
        n_segments=4,
        n_iterations_per_segment=25,
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
    visualize_results(results, env, target_pos)
    
    return results


if __name__ == "__main__":
    results = main()
