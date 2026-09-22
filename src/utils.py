import matplotlib.pyplot as plt


def visualize_results(results, env, target_pos, save_path='img'):
    """Visualize results (works for both SimulatorEnv_2D and SimulatorEnv_3D)."""
    final_q = results['final_q']
    dim = getattr(env, 'dim', 2)
    pts = final_q.reshape(-1, dim)
    mid_idx = env.mid_node
    mid = pts[mid_idx]

    fig = plt.figure(figsize=(14, 5))

    # ---- Configuration plot ----
    if dim == 2:
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(pts[:, 0], pts[:, 1], 'ko-', linewidth=2, markersize=6,
                 label='Final config', alpha=0.7)
        ax1.plot(pts[0, 0], pts[0, 1], 'gs', markersize=10, label='Left boundary')
        ax1.plot(pts[-1, 0], pts[-1, 1], 'gs', markersize=10)
        ax1.plot(mid[0], mid[1], 'ro', markersize=12, label='Midpoint', zorder=10)
        ax1.plot(target_pos[0], target_pos[1], 'bx', markersize=15, markeredgewidth=3,
                 label='Target', zorder=10)
        ax1.plot([mid[0], target_pos[0]], [mid[1], target_pos[1]], 'r--',
                 alpha=0.5, linewidth=1,
                 label=f'Error: {results["final_error"]:.4f}')
        ax1.axis('equal')
    else:
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'ko-', linewidth=2, markersize=4,
                 label='Final config', alpha=0.7)
        ax1.plot([pts[0, 0]], [pts[0, 1]], [pts[0, 2]], 'gs', markersize=9,
                 label='Left boundary')
        ax1.plot([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]], 'gs', markersize=9)
        ax1.plot([mid[0]], [mid[1]], [mid[2]], 'ro', markersize=10,
                 label='Midpoint', zorder=10)
        ax1.plot([target_pos[0]], [target_pos[1]], [target_pos[2]], 'bx',
                 markersize=13, markeredgewidth=3, label='Target', zorder=10)
        ax1.plot([mid[0], target_pos[0]], [mid[1], target_pos[1]],
                 [mid[2], target_pos[2]], 'r--', alpha=0.5, linewidth=1,
                 label=f'Error: {results["final_error"]:.4f}')
        ax1.set_zlabel('z (m)', fontsize=11)
        try:
            ax1.set_box_aspect((1, 1, 1))
        except AttributeError:      # matplotlib < 3.3
            pass

    ax1.set_xlabel('x (m)', fontsize=11)
    ax1.set_ylabel('y (m)', fontsize=11)
    ax1.set_title('Final Configuration', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ---- Loss curve ----
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.semilogy(results['all_losses'], linewidth=2, color='steelblue')
    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Loss (log scale)', fontsize=11)
    ax2.set_title('Training Progress', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}/neural_control_result.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to {save_path}/neural_control_result.png")
    plt.close()
