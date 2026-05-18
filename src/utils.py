import matplotlib.pyplot as plt

def visualize_results(results, env, target_pos, save_path='img'):
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