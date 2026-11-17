"""Result plotting helpers shared by the training scripts.

* ``visualize_results``  -> neural_control_result.png in the run directory (final shape + loss curve)
* ``save_trajectory_gif`` -> <name>.gif there too (the whole control rollout, frame by frame)
* ``maybe_save_trajectory_gif`` -> the same, but only if ``output.save_gif`` is on

Both draw the environment's obstacles translucently (``draw_obstacles``) and
the gripper frames at the clamped ends (``draw_gripper_frames``: x along the
tangent, y the yaw director, z = x cross y), so what the robot holds and what
the rod must avoid are visible in every output.
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

    draw_obstacles(ax1, env)
    draw_workspace(ax1, env)
    if getattr(env, 'pipe', None) is not None:
        env.pipe.draw(ax1)
    draw_gripper_frames(ax1, env, final_q, 0.1 * env.rod_length)
    gripper_frame_legend(ax1, dim)

    ax1.set_xlabel('x (m)', fontsize=11)
    ax1.set_ylabel('y (m)', fontsize=11)
    ax1.set_title('Final Configuration', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
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

    With torsion the DOF vector carries the two gripper yaws after the
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
# Obstacles and gripper frames (drawn in every figure and animation)
# ---------------------------------------------------------------------------

#: The gripper frame's axes: x along the clamped tangent, y the yaw director
#: (the cross-section direction the gripper holds), z = x cross y.
_GRIPPER_AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")
_GRIPPER_AXIS_LABELS = ("gripper x (tangent)", "gripper y (yaw director)", "gripper z")


def draw_obstacles(ax, env, alpha=0.35, color="steelblue"):
    """Draw the environment's rigid obstacles translucently.

    Circles (2D) and spheres (3D) from ``env.obstacles``; whole-rod geometry
    with its own ``draw`` (the pipe) is drawn through ``env.pipe`` by the
    callers, so it is skipped here.  Returns the artists.
    """
    dim = getattr(env, "dim", 2)
    artists = []
    for obs in getattr(env, "obstacles", []) or []:
        if hasattr(obs, "draw_surface"):                   # the peg (a cylinder)
            artists.extend(obs.draw_surface(ax, alpha=alpha, color=color))
            continue
        if hasattr(obs, "draw") or not hasattr(obs, "radius"):
            continue
        if dim == 2:
            from matplotlib.patches import Circle
            patch = Circle((obs.cx, obs.cy), obs.radius, color=color, alpha=alpha, zorder=2,
                           label="obstacle" if not artists else None)
            ax.add_patch(patch)
            artists.append(patch)
        else:
            u = np.linspace(0, 2 * np.pi, 24)
            v = np.linspace(0, np.pi, 16)
            xs = obs.cx + obs.radius * np.outer(np.cos(u), np.sin(v))
            ys = obs.cy + obs.radius * np.outer(np.sin(u), np.sin(v))
            zs = obs.cz + obs.radius * np.outer(np.ones_like(u), np.cos(v))
            artists.append(ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0,
                                           shade=True))
    return artists


def draw_workspace(ax, env, color="darkorange"):
    """Draw the gripper workspace (``env.workspace_box``, set by ``build_env``).

    A dashed rectangle (2D) or wireframe box (3D) over the allowed gripper
    positions; nothing when the workspace is off.  Returns the artists.
    """
    box = getattr(env, "workspace_box", None)
    if box is None:
        return []
    lo, hi = box
    label = "gripper workspace"
    if getattr(env, "dim", 2) == 2:
        from matplotlib.patches import Rectangle
        patch = Rectangle(lo[:2], *(hi[:2] - lo[:2]), fill=False, ec=color, lw=1.4, ls="--",
                          zorder=3, label=label)
        ax.add_patch(patch)
        return [patch]
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    artists = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(corners[i] != corners[j]) == 1:      # an edge of the box
                seg = corners[[i, j]]
                artists.extend(ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, lw=1.0,
                                       ls="--", alpha=0.8, label=label if not artists else None))
    return artists


def gripper_frames(env, q):
    """``[(origin, axes), ...]`` of every gripped end, from the configuration.

    ``axes`` is ``(dim, dim)`` with the rows x, y(, z): x is the clamped
    edge's tangent, y the yaw director -- in 3D ``rodframes.gripper_director``
    rotated about the tangent by that end's yaw (zero without torsion), in
    2D the in-plane normal -- and z = x cross y.  Derived from ``q`` alone, so
    recorded rollouts can be drawn without their commands.
    """
    dim = getattr(env, "dim", 2)
    pos = _positions(q, env, dim)
    nv = pos.shape[0]
    ends = []
    if getattr(env, "grasp_mode", "both") in ("both", "left"):
        ends.append((0, pos[1] - pos[0]))
    if getattr(env, "grasp_mode", "both") in ("both", "right"):
        ends.append((nv - 1, pos[nv - 1] - pos[nv - 2]))
    yaws = env.twist_angles(q) if getattr(env, "twist", False) else (0.0, 0.0)
    out = []
    for (node, t), yaw in zip(ends, yaws if len(ends) == 2 else yaws[-1:]):
        t = t / max(np.linalg.norm(t), 1e-12)
        if dim == 2:
            out.append((pos[node], np.array([t, [-t[1], t[0]]])))
            continue
        from rodframes import gripper_director
        try:
            d = gripper_director(t)
        except ValueError:                     # vertical tangent: any perpendicular
            d = np.cross(t, [1.0, 0.0, 0.0])
            d /= np.linalg.norm(d)
        e = np.cross(t, d)
        y = np.cos(yaw) * d + np.sin(yaw) * e
        out.append((pos[node], np.array([t, y, np.cross(t, y)])))
    return out


def draw_gripper_frames(ax, env, q, length, linewidth=2.0):
    """Draw the gripper frames of ``q`` as arrows of ``length``; returns the artists."""
    dim = getattr(env, "dim", 2)
    artists = []
    for origin, axes in gripper_frames(env, q):
        for a, color in zip(axes, _GRIPPER_AXIS_COLORS):
            d = a * length
            if dim == 2:
                artists.append(ax.quiver(origin[0], origin[1], d[0], d[1], color=color,
                                         angles="xy", scale_units="xy", scale=1, width=0.006,
                                         zorder=11))
            else:
                artists.append(ax.quiver(origin[0], origin[1], origin[2], d[0], d[1], d[2],
                                         color=color, linewidth=linewidth,
                                         arrow_length_ratio=0.25, zorder=11))
    return artists


def gripper_frame_legend(ax, dim):
    """Legend entries for the gripper axes (a quiver has none of its own)."""
    from matplotlib.lines import Line2D
    for color, label in zip(_GRIPPER_AXIS_COLORS[:dim], _GRIPPER_AXIS_LABELS[:dim]):
        ax.add_line(Line2D([], [], color=color, lw=2, label=label))


# ---------------------------------------------------------------------------
# Material frames (torsion only)
# ---------------------------------------------------------------------------

#: Where the cross-section triads are drawn, and how each axis is coloured.
_TRIAD_COLORS = ("#7f7f7f", "#d55e00", "#009e73")      # t, m1, m2 of the middle edge
_TRIAD_LABELS = ("mid edge t", "mid edge m1", "mid edge m2")


def _material_triads(q_traj, env, idx):
    """Material frame at the midpoint, per kept animation frame.

    Returns ``(anchors, triads)`` of shapes ``(n_frames, 1, 3)`` and
    ``(n_frames, 1, 3, 3)`` -- the location and its ``(tangent, m1, m2)``
    triad -- or ``None`` when the rod has no torsion.

    The frames are a function of the recorded configuration alone (the Bishop
    frame is rebuilt from the positions and the twist is uniform; see
    :mod:`rodframes`), so nothing has to be replayed and nothing the training
    scripts did not record is missing.
    """
    if not getattr(env, "twist", False):
        return None
    from rodframes import material_frames, tangents

    nv = env.nv
    # Frames live on edges (Bergou et al. 2008, Sec. 4.2: "we naturally assign
    # frames to edges, rather than to vertices"), so the triad is the middle
    # edge's and is drawn at that edge's midpoint; a vertex carries curvature
    # and twist, not an orientation.  Only the middle edge is shown: at the
    # clamped ends the material frame is the gripper frame, which
    # draw_gripper_frames shows.
    edges = (min(max(env.mid_node, 0), nv - 2),)

    anchors, triads = [], []
    for k in idx:
        q = q_traj[int(k)]
        pos = env.positions(q)
        yaw_l, yaw_r = env.twist_angles(q)
        m1, m2 = material_frames(pos, yaw_l, yaw_r)
        t = tangents(pos)
        tri = np.empty((len(edges), 3, 3))
        for j, e in enumerate(edges):
            tri[j, 0] = t[e]
            tri[j, 1] = m1[e]
            tri[j, 2] = m2[e]
        anchors.append(np.array([0.5 * (pos[e] + pos[e + 1]) for e in edges]))
        triads.append(tri)
    return np.asarray(anchors), np.asarray(triads)


def save_trajectory_gif(q_traj, env, target_pos, save_path='outputs',
                        filename='neural_control_animation.gif',
                        fps=20, dpi=100, max_frames=200, title='DLO control rollout',
                        track_node=None, track_label='Midpoint', gripper_node=None,
                        target_shape=None):
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
    target_shape : a whole target rod shape ``(nv, dim)`` (shape formation),
        drawn as a static curve; ``target_pos`` may then be None.

    Rigid geometry the environment carries (``env.pipe``, ``env.obstacles``)
    is drawn underneath, translucently, and the gripper frames (x along the
    clamped tangent, y the yaw director, z their cross product) follow the
    grippers frame by frame.

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

    target = None if target_pos is None else np.asarray(target_pos, dtype=float)
    is_traj_target = target is not None and target.ndim == 2
    # `src_idx` (from _as_frames) says which step each animation frame came
    # from, which is what the per-step tracking target is looked up with.

    # Fixed axis limits over the whole rollout, so the rod does not jitter.
    pts_all = frames.reshape(-1, dim)
    extra = [target.reshape(-1, dim)] if target is not None else []
    if target_shape is not None:
        extra.append(np.asarray(target_shape, dtype=float).reshape(-1, dim))
    for obs in getattr(env, 'obstacles', []) or []:
        if hasattr(obs, 'bounds'):
            extra.append(np.stack(obs.bounds()))
        elif hasattr(obs, 'radius') and not hasattr(obs, 'draw'):
            c = np.array([obs.cx, obs.cy] + ([obs.cz] if dim == 3 else []), dtype=float)
            extra.append(np.stack([c - obs.radius, c + obs.radius]))
    ws_box = getattr(env, 'workspace_box', None)
    if ws_box is not None:
        extra.append(np.stack(ws_box))
    ref = np.vstack([pts_all] + extra)
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    span = max((hi - lo).max(), 1e-3)
    pad = 0.08 * span
    ctr = 0.5 * (lo + hi)
    lims = np.stack([ctr - 0.5 * span - pad, ctr + 0.5 * span + pad], axis=1)

    fig = plt.figure(figsize=(8, 6))
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
    if pipe is not None:
        pipe.draw(ax)
    draw_obstacles(ax, env)
    draw_workspace(ax, env)

    coords = lambda p: [p[..., c] for c in range(dim)]       # noqa: E731

    if target_shape is not None:
        ax.plot(*coords(np.asarray(target_shape, dtype=float)), color='k', linewidth=2.2,
                alpha=0.8, label='Target shape')
    if is_traj_target:
        ax.plot(*coords(target), color='tab:blue', ls='--', linewidth=2.2, alpha=0.95,
                label='Target path', zorder=6)
    elif target is not None:
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
    # Cross-section orientation at the midpoint (the ends are the gripper
    # frames). Only a twisting rod has one: without torsion the material frame
    # is not a degree of freedom and there would be nothing to show.
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
    # The gripper frames, rebuilt every frame (quiver has no in-place update).
    frame_len = 0.1 * span
    frame_artists = []
    gripper_frame_legend(ax, dim)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
    fig.subplots_adjust(left=0.02, right=0.76)

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
        while frame_artists:
            frame_artists.pop().remove()
        frame_artists.extend(draw_gripper_frames(ax, env, q_traj[int(src_idx[k])], frame_len))
        artists.extend(frame_artists)
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


def visualize_pipe_task_3d(q_traj, env, target_pos, losses, tip_node=0, q_init=None,
                           reference=None, metrics=None, save_path='outputs',
                           filename='pipe_task_result.png', n_snapshots=6):
    """The 3D counterpart of :func:`visualize_pipe_task`: the pipe's bores,
    the initial rod, intermediate shapes, the final rod with its gripper
    frame, the tip path and the target, plus the loss curve."""
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    env.pipe.draw(ax)
    draw_obstacles(ax, env)
    draw_workspace(ax, env)
    if q_init is not None:
        P = _positions(q_init, env, 3)
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color='0.6', lw=1.2, label='initial rod')
    idx = np.unique(np.linspace(0, len(q_traj) - 1, n_snapshots).round().astype(int))
    for i in idx[:-1]:
        P = _positions(q_traj[i], env, 3)
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color='0.45', lw=0.8, alpha=0.5)
    P = _positions(q_traj[-1], env, 3)
    ax.plot(P[:, 0], P[:, 1], P[:, 2], 'ko-', lw=2, ms=3, label='final rod')
    tips = np.array([_positions(q, env, 3)[tip_node] for q in q_traj])
    ax.plot(tips[:, 0], tips[:, 1], tips[:, 2], 'r-', lw=1.2, label='tip path')
    if reference is not None:
        R = np.asarray(reference)
        ax.plot(R[:, 0], R[:, 1], R[:, 2], 'b--', lw=1, alpha=0.6, label='reference (centerline)')
    ax.plot([target_pos[0]], [target_pos[1]], [target_pos[2]], 'bx', ms=12, mew=3, label='target')
    draw_gripper_frames(ax, env, q_traj[-1], 0.1 * env.rod_length)
    gripper_frame_legend(ax, 3)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass
    pts = np.vstack([P, tips, np.asarray(target_pos)[None, :], env.pipe.P0[None, :], env.pipe.P3[None, :]]
                     + ([np.stack(env.workspace_box)] if getattr(env, 'workspace_box', None) is not None else []))
    lo, hi = pts.min(0) - 0.15, pts.max(0) + 0.15
    span = (hi - lo).max()
    ctr = 0.5 * (lo + hi)
    ax.set_xlim(ctr[0] - span / 2, ctr[0] + span / 2)
    ax.set_ylim(ctr[1] - span / 2, ctr[1] + span / 2)
    ax.set_zlim(ctr[2] - span / 2, ctr[2] + span / 2)
    title = 'Pipe insertion (3D)'
    if metrics:
        title += '  ' + ', '.join(f'{k}={v}' for k, v in metrics.items())
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax2 = fig.add_subplot(1, 2, 2)
    if losses:
        ax2.semilogy(losses, color='steelblue', lw=2)
    ax2.set_xlabel('epoch'); ax2.set_ylabel('loss'); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    out = os.path.join(save_path, filename)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved to {out}")
    return out


def visualize_pipe_task(q_traj, env, target_pos, losses, tip_node=0, q_init=None,
                        reference=None, metrics=None, save_path='outputs',
                        filename='pipe_task_result.png', n_snapshots=6):
    if getattr(env, 'dim', 2) == 3:
        return visualize_pipe_task_3d(q_traj, env, target_pos, losses, tip_node=tip_node,
                                      q_init=q_init, reference=reference, metrics=metrics,
                                      save_path=save_path, filename=filename,
                                      n_snapshots=n_snapshots)
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
    draw_workspace(ax1, env)
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
