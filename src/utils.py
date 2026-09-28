"""Result plotting helpers shared by the training scripts.

* ``visualize_results``  -> outputs/neural_control_result.png (final shape + loss curve)
* ``save_trajectory_gif`` -> outputs/<name>.gif (the whole control rollout, frame by frame)
* ``maybe_save_trajectory_gif`` -> the same, but only if ``output.save_gif`` is on
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def visualize_results(results, env, target_pos, save_path='img'):
    """Visualize results (works for both SimulatorEnv_2D and SimulatorEnv_3D)."""
    final_q = results['final_q']
    dim = getattr(env, 'dim', 2)
    pts = _positions(final_q, env, dim)
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


# ---------------------------------------------------------------------------
# Trajectory animation
# ---------------------------------------------------------------------------

def _positions(q, env, dim):
    """Node coordinates out of a DOF vector, ``(nv, dim)``.

    With torsion the DOF vector carries per-edge twist angles after the
    positions, so a plain ``reshape(-1, dim)`` would be wrong; the environment
    knows the split.
    """
    get = getattr(env, 'positions', None)
    if get is not None:
        return get(q)
    return np.asarray(q).reshape(-1, dim)


def _as_frames(q_traj, dim, max_frames, env=None):
    """Stack a list of flat q vectors into (n_frames, nv, dim), subsampled."""
    frames = np.asarray([_positions(q, env, dim) for q in q_traj])
    if max_frames and len(frames) > max_frames:
        # Keep the first and last frame, evenly subsample in between.
        idx = np.unique(np.linspace(0, len(frames) - 1, max_frames).round().astype(int))
        frames = frames[idx]
    return frames


def save_trajectory_gif(q_traj, env, target_pos, save_path='outputs',
                        filename='neural_control_animation.gif',
                        fps=20, dpi=100, max_frames=200, title='DLO control rollout'):
    """Animate the full control rollout of the rod and write it out as a GIF.

    Parameters
    ----------
    q_traj : sequence of flat configuration vectors (length ``dim * nv`` each),
        one per control step -- i.e. the committed rollout, not a training epoch.
    target_pos : the point-to-point target ``(dim,)``, or a tracking target
        trajectory ``(n_steps, dim)``; a trajectory is drawn as a dashed curve
        with the current target marked on it.

    Returns the path written, or ``None`` if there was nothing to animate.
    """
    if q_traj is None or len(q_traj) == 0:
        print("No trajectory recorded - skipping GIF.")
        return None

    dim = getattr(env, 'dim', 2)
    frames = _as_frames(q_traj, dim, max_frames, env)
    n_frames = len(frames)
    mid_idx = env.mid_node

    target = np.asarray(target_pos, dtype=float)
    is_traj_target = target.ndim == 2
    # Frame k of the animation corresponds to this index in the original rollout,
    # which is what the per-step tracking target has to be looked up with.
    src_idx = np.linspace(0, len(q_traj) - 1, n_frames).round().astype(int)

    # Fixed axis limits over the whole rollout, so the rod does not jitter.
    pts_all = frames.reshape(-1, dim)
    ref = np.vstack([pts_all, target.reshape(-1, dim)])
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    span = max((hi - lo).max(), 1e-3)
    pad = 0.08 * span
    ctr = 0.5 * (lo + hi)
    lims = np.stack([ctr - 0.5 * span - pad, ctr + 0.5 * span + pad], axis=1)

    fig = plt.figure(figsize=(6, 6))
    if dim == 2:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel('z (m)', fontsize=11)
        ax.set_zlim(*lims[2])
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:      # matplotlib < 3.3
            pass
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    if dim == 2:
        ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.grid(True, alpha=0.3)

    coords = lambda p: [p[..., c] for c in range(dim)]       # noqa: E731

    if is_traj_target:
        ax.plot(*coords(target), 'b--', linewidth=1.5, alpha=0.6,
                label='Target trajectory')
    else:
        ax.plot(*[[v] for v in target], 'bx', markersize=13, markeredgewidth=3,
                label='Target', zorder=10)

    rod, = ax.plot([], [], *([[]] if dim == 3 else []), 'ko-', linewidth=2,
                   markersize=4 if dim == 3 else 5, alpha=0.75, label='Rod')
    ends, = ax.plot([], [], *([[]] if dim == 3 else []), 'gs', markersize=9,
                    linestyle='None', label='Boundary')
    mid, = ax.plot([], [], *([[]] if dim == 3 else []), 'ro', markersize=9,
                   linestyle='None', label='Midpoint', zorder=10)
    trail, = ax.plot([], [], *([[]] if dim == 3 else []), 'r-', linewidth=1,
                     alpha=0.5, label='Midpoint path')
    tgt_now = None
    if is_traj_target:
        tgt_now, = ax.plot([], [], *([[]] if dim == 3 else []), 'bx', markersize=11,
                           markeredgewidth=2.5, linestyle='None', label='Target now',
                           zorder=10)
    ax.legend(fontsize=9, loc='upper right')

    def _set(artist, pts):
        pts = np.atleast_2d(pts)
        artist.set_data(pts[:, 0], pts[:, 1])
        if dim == 3:
            artist.set_3d_properties(pts[:, 2])

    def update(k):
        pts = frames[k]
        _set(rod, pts)
        _set(ends, pts[[0, -1]])
        _set(mid, pts[mid_idx])
        _set(trail, frames[:k + 1, mid_idx])
        artists = [rod, ends, mid, trail]
        if is_traj_target:
            j = min(src_idx[k], len(target) - 1)
            _set(tgt_now, target[j])
            artists.append(tgt_now)
        ax.set_title(f'{title}   step {src_idx[k] + 1}/{len(q_traj)}',
                     fontsize=12, fontweight='bold')
        return artists

    os.makedirs(save_path, exist_ok=True)
    out = os.path.join(save_path, filename)
    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Saved {n_frames}-frame animation to {out}")
    return out


def maybe_save_trajectory_gif(cfg, q_traj, env, target_pos, save_path='outputs',
                              filename='neural_control_animation.gif',
                              title='DLO control rollout'):
    """``save_trajectory_gif`` gated on the ``output.save_gif`` config switch."""
    out_cfg = cfg.get('output', {}) or {}
    if not out_cfg.get('save_gif', False):
        return None
    return save_trajectory_gif(
        q_traj, env, target_pos, save_path=save_path, filename=filename,
        fps=out_cfg.get('gif_fps', 20),
        dpi=out_cfg.get('gif_dpi', 100),
        max_frames=out_cfg.get('gif_max_frames', 200),
        title=title,
    )
