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
    """Stack a list of flat q vectors into (n_frames, nv, dim), subsampled.

    Also returns the source step index of every kept frame, so anything else
    derived per step (the tracking target, the material frames) can be
    subsampled to match instead of guessing.
    """
    frames = np.asarray([_positions(q, env, dim) for q in q_traj])
    idx = np.arange(len(frames))
    if max_frames and len(frames) > max_frames:
        # Keep the first and last frame, evenly subsample in between.
        idx = np.unique(np.linspace(0, len(frames) - 1, max_frames).round().astype(int))
        frames = frames[idx]
    return frames, idx


# ---------------------------------------------------------------------------
# Material frames (torsion only)
# ---------------------------------------------------------------------------

#: Where the cross-section triads are drawn, and how each axis is coloured.
_TRIAD_COLORS = ("#7f7f7f", "#d55e00", "#009e73")      # tangent, m1, m2
_TRIAD_LABELS = ("tangent", "m1 (cross-section)", "m2")


def _material_triads(q_traj, env, idx):
    """Material frames at the two ends and the midpoint, per kept animation frame.

    Returns ``(anchors, triads)`` of shapes ``(n_frames, 3, 3)`` and
    ``(n_frames, 3, 3, 3)`` -- three locations, and for each a ``(tangent, m1,
    m2)`` triad -- or ``None`` when the rod has no torsion.

    The reference directors are simulator *state*, not something ``q`` carries,
    so they are replayed here: seeded exactly as ``env.reset()`` does and then
    parallel transported along the recorded configurations.  Replaying is
    deterministic given the positions, so this reproduces the directors the
    solver actually used, without the training scripts having to record them.
    """
    if not getattr(env, "twist", False):
        return None
    from rodframes import RodFrames, tangents

    nv = env.nv
    # Edges whose cross-section we show, anchored at the node we draw them from.
    edges = (0, min(max(env.mid_node, 0), nv - 2), nv - 2)
    nodes = (0, env.mid_node, nv - 1)

    frames_state = RodFrames(nv)
    frames_state.reset(env.nodes)

    keep = set(int(k) for k in idx)
    anchors, triads = [], []
    for k, q in enumerate(q_traj):
        pos = env.positions(q)
        frames_state.advance(pos)
        if k not in keep:
            continue
        theta = env.twist_angles(q)
        t = tangents(pos)
        u = frames_state.directors
        tri = np.empty((3, 3, 3))
        for j, e in enumerate(edges):
            v = np.cross(t[e], u[e])
            c, sn = np.cos(theta[e]), np.sin(theta[e])
            tri[j, 0] = t[e]
            tri[j, 1] = c * u[e] + sn * v          # material axis m1
            tri[j, 2] = -sn * u[e] + c * v         # material axis m2
        anchors.append(pos[list(nodes)])
        triads.append(tri)
    return np.asarray(anchors), np.asarray(triads)


def save_trajectory_gif(q_traj, env, target_pos, save_path='outputs',
                        filename='neural_control_animation.gif',
                        fps=20, dpi=100, max_frames=200, title='DLO control rollout',
                        track_node=None, track_label='Midpoint', gripper_node=None):
    """Animate the full control rollout of the rod and write it out as a GIF.

    Parameters
    ----------
    q_traj : sequence of flat configuration vectors (length ``dim * nv`` each),
        one per control step -- i.e. the committed rollout, not a training epoch.
    target_pos : the point-to-point target ``(dim,)``, or a tracking target
        trajectory ``(n_steps, dim)``; a trajectory is drawn as a dashed curve
        with the current target marked on it.
    track_node : the node whose position and path are highlighted (default:
        the mid node); ``track_label`` names it in the legend.
    gripper_node : if given, that node's path is drawn as well (the gripper).

    Rigid geometry the environment carries (``env.pipe``) is drawn underneath.

    Returns the path written, or ``None`` if there was nothing to animate.
    """
    if q_traj is None or len(q_traj) == 0:
        print("No trajectory recorded - skipping GIF.")
        return None

    dim = getattr(env, 'dim', 2)
    frames, src_idx = _as_frames(q_traj, dim, max_frames, env)
    n_frames = len(frames)
    mid_idx = env.mid_node if track_node is None else int(track_node)
    triad_data = _material_triads(q_traj, env, src_idx) if dim == 3 else None

    target = np.asarray(target_pos, dtype=float)
    is_traj_target = target.ndim == 2
    # `src_idx` (from _as_frames) says which step each animation frame came
    # from, which is what the per-step tracking target is looked up with.

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
    pipe = getattr(env, 'pipe', None)
    if pipe is not None and dim == 2:
        pipe.draw(ax)

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
                   linestyle='None', label=track_label, zorder=10)
    trail, = ax.plot([], [], *([[]] if dim == 3 else []), 'r-', linewidth=1,
                     alpha=0.5, label=f'{track_label} path')
    grip_trail = None
    if gripper_node is not None:
        grip_trail, = ax.plot([], [], *([[]] if dim == 3 else []), 'g-', linewidth=1.5,
                              alpha=0.7, label='Gripper path')
    # Cross-section orientation at the two ends and the midpoint. Only a
    # twisting rod has one: without torsion the material frame is not a degree
    # of freedom and there would be nothing to show.
    triad_artists = []
    if triad_data is not None:
        arrow_len = 0.12 * span
        from matplotlib.lines import Line2D
        for color, label in zip(_TRIAD_COLORS, _TRIAD_LABELS):
            ax.add_line(Line2D([], [], color=color, lw=2, label=label))

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
        if grip_trail is not None:
            _set(grip_trail, frames[:k + 1, gripper_node])
            artists.append(grip_trail)
        if triad_data is not None:
            # quiver has no in-place update, so the arrows are rebuilt each frame
            while triad_artists:
                triad_artists.pop().remove()
            anchors, triads = triad_data
            for j in range(anchors.shape[1]):
                o = anchors[k, j]
                for a, color in enumerate(_TRIAD_COLORS):
                    d = triads[k, j, a] * arrow_len
                    triad_artists.append(ax.quiver(
                        o[0], o[1], o[2], d[0], d[1], d[2],
                        color=color, linewidth=1.6, arrow_length_ratio=0.25))
            artists.extend(triad_artists)
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
                              title='DLO control rollout', **kwargs):
    """``save_trajectory_gif`` gated on the ``output.save_gif`` config switch."""
    out_cfg = cfg.get('output', {}) or {}
    if not out_cfg.get('save_gif', False):
        return None
    return save_trajectory_gif(
        q_traj, env, target_pos, save_path=save_path, filename=filename,
        fps=out_cfg.get('gif_fps', 20),
        dpi=out_cfg.get('gif_dpi', 100),
        max_frames=out_cfg.get('gif_max_frames', 200),
        title=title, **kwargs,
    )


def visualize_pipe_task(q_traj, env, target_pos, losses, tip_node=0, q_init=None,
                        reference=None, metrics=None, save_path='outputs',
                        filename='pipe_task_result.png', n_snapshots=6):
    """Pipe-insertion summary figure (2D).

    Left: the pipe, the initial rod, a few intermediate shapes, the final
    shape, the gripper path, the tip trajectory, the reference path and the
    target.  Right: the training loss.  ``metrics`` (a dict) is printed into
    the title of the left panel.
    """
    pts = np.asarray([_positions(q, env, 2) for q in q_traj])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                   gridspec_kw={'width_ratios': [1.3, 1]})
    if getattr(env, 'pipe', None) is not None:
        env.pipe.draw(ax1)
    if q_init is not None:
        p0 = _positions(q_init, env, 2)
        ax1.plot(p0[:, 0], p0[:, 1], '-', color='0.6', lw=2, label='Initial rod')
    snap = np.unique(np.linspace(0, len(pts) - 1, n_snapshots + 2).round().astype(int))[1:-1]
    for j, k in enumerate(snap):
        ax1.plot(pts[k, :, 0], pts[k, :, 1], '-', color='tab:purple', lw=1,
                 alpha=0.25 + 0.5 * j / max(len(snap), 1),
                 label='Intermediate rods' if j == 0 else None)
    ax1.plot(pts[-1, :, 0], pts[-1, :, 1], 'k.-', lw=2, ms=4, label='Final rod')
    grip = pts[:, -1]
    ax1.plot(grip[:, 0], grip[:, 1], 'g-', lw=2.5, label='Gripper path')
    ax1.plot(grip[-1, 0], grip[-1, 1], 'gs', ms=9)
    tip = pts[:, tip_node]
    ax1.plot(tip[:, 0], tip[:, 1], 'r-', lw=1.5, label='Tip trajectory')
    ax1.plot(tip[-1, 0], tip[-1, 1], 'ro', ms=7)
    if reference is not None:
        ref = np.asarray(reference)
        ax1.plot(ref[:, 0], ref[:, 1], 'b:', lw=1.2, label='Reference path')
    ax1.plot(target_pos[0], target_pos[1], 'bx', ms=14, mew=3, label='Target', zorder=10)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='lower right')
    if metrics:
        ax1.set_title('  '.join(f'{k}={v}' for k, v in metrics.items()), fontsize=9)

    if losses:
        ax2.semilogy(losses, lw=2, color='steelblue')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training Progress', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    out = os.path.join(save_path, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved to {out}")
    return out
