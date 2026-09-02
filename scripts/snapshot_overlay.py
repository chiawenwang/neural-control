# -*- coding: utf-8 -*-
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
_OUT = _ROOT / 'outputs'
_OUT.mkdir(exist_ok=True)

"""
DLO live simulation + snapshot overlay

Whenever the simulation progress reaches 0%, 10%, ..., 100%, the current rod
shape is frozen and overlaid on the figure.

Usage:
    python snapshot_overlay.py                         # default: angle_oscillation
    python snapshot_overlay.py --traj moving_forward
    python snapshot_overlay.py --traj angle_oscillation
    python snapshot_overlay.py -h                      # see all options
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl

from SimulatorEnv_2D import SimulatorEnv_2D
from render_neurips import render_dlo_neurips, DLO_CMAP, _node_curvatures

# ===== Matplotlib global styling for NeurIPS-ready figures =====
# Use Times-family serif fonts so the PDF is self-contained in Illustrator
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Nimbus Roman No9 L', 'Liberation Serif']
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
mpl.rcParams['pdf.use14corefonts'] = False
mpl.rcParams['ps.useafm'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['savefig.dpi'] = 300
# =================================================================


# =========================================================
# Global physical parameters
# =========================================================
L = 1.0    # rod length (m)
T = 10.0   # main simulation duration (s)

# Material & mesh
ROD_NV      = 31
ROD_DT      = 0.01
ROD_R_OUTER = 0.006
ROD_E       = 2e9
ROD_RHO     = 1200

# Snapshot & render config
PAPER_MODE            = True   # True -> output paper size (2.75"); False -> screen preview size
_PAPER_FIG_W          = 2.75   # paper output width (inch)
_ORIG_FIG_W           = 7.0    # original width in render_neurips.py
_PAPER_S              = (_PAPER_FIG_W / _ORIG_FIG_W) if PAPER_MODE else 1.0

SNAP_PERCENTS         = list(range(0, 101, 25)) + [93,94]# 0%, 20%, 40%, 60%, 80%, 100%
SHOW_SNAP_LABELS      = False
ARROW_SCALE           = 0.5
ARROW_ENDPOINT_COLOR  = '#1a1a1a'
ARROW_MIDPOINT_COLOR  = '#d62728'
ARROW_QUARTER_COLOR_1 = '#0088dd'
ARROW_QUARTER_COLOR_3 = '#9b2dca'
SHOW_QUARTER_ARROWS   = True
SHOW_TRAILS           = True       # whether to show trails of the three middle points (mid, 1/4, 3/4)
HIGHLIGHT_SNAP        = [93,94]   # a single percent, a list of percents, or None (no highlight)
DIMMED_ALPHA          = 0.1


# =========================================================
# Trajectory base class
# =========================================================

class Trajectory:
    """
    Subclasses must implement:
      - name  : str, trajectory identifier (used for filenames, CLI choices)
      - xlim / ylim : display range
      - control(t)  : returns the 6D control vector at time t
    Optional overrides:
      - warmup(env) : warmup phase (not counted toward the main simulation
        duration/snapshots)
    """
    name: str = ""
    xlim = (-0.5 * L, 1.5 * L)
    ylim = (-0.5 * L, 0.5 * L)

    def warmup(self, env) -> dict:
        """No warmup by default; just returns the current observation."""
        return env._get_observation()

    def control(self, t: float) -> np.ndarray:  # noqa: ARG002
        raise NotImplementedError

    def _reset_env_counters(self, env):
        """Resets the time counters and history log after warmup ends."""
        env.current_step = 0
        env.ctime        = 0.0
        env.Nsteps       = int(round(T / env.dt))
        for k in env.history:
            env.history[k] = []

    @staticmethod
    def _smoothstep(t, t_total):
        s = min(t / t_total, 1.0)
        return s * s * (3 - 2 * s)


# =========================================================
# Trajectory 1: horizontal translation (both ends move in parallel, fixed angle)
# =========================================================

class MovingForward(Trajectory):
    """
    Left end moves left by 0.35L, right end moves left by 1.3L, angle stays
    at pi/2 throughout.
    Warmup: smoothly transitions from the default straight rod to the start
    configuration.
    """
    name = "moving_forward"
    xlim = (-1.13 * L, 0.7 * L)
    ylim = (-0.1  * L, 0.7 * L)

    # [x_l, y_l, th_l, x_r, y_r, th_r]
    _start = np.array([0.0, 0.0, np.pi/2,
                       0.3*L, 0.65*L, np.pi/2])

    def warmup(self, env) -> dict:
        """Smoothly moves the right end from (L, 0, 0) to the start configuration over 2 seconds."""
        print("Warmup: transitioning to start position ...")
        T_warmup = 2.0
        N = int(round(T_warmup / env.dt))
        init = np.array([0.0, 0.0, 0.0, L, 0.0, 0.0])   # straight-rod state after reset()
        obs = None
        for i in range(N):
            s    = self._smoothstep((i + 1) * env.dt, T_warmup)
            ctrl = init + s * (self._start - init)
            obs, _, _ = env.step(ctrl)
        self._reset_env_counters(env)
        print("Warmup done.")
        return obs

    def control(self, t: float) -> np.ndarray:
        s = self._start
        return np.array([
            s[0] - 0.35*L*(t/T), s[1], s[2],
            s[3] - 1.30*L*(t/T), s[4], s[5],
        ])


# =========================================================
# Trajectory 2: angle oscillation (endpoints fixed, angle down->up->down->up)
# =========================================================

class AngleOscillation(Trajectory):
    """
    Warmup: right end horizontally compresses from x=L to x=0.6L (angle held at +/- pi/3).
    Main phase: both ends' xy fixed, angle oscillates cosine-wise down->up->down->up (1.5 cycles).
    """
    name = "angle_oscillation"
    xlim = (-0.2 * L, 0.8 * L)
    ylim = (-0.5 * L, 0.5 * L)

    _X_END    = 0.6 * L
    _TH_INIT  = 5.5 * np.pi / 12  # fixed angle amplitude during warmup
    _TH_AMP   = 5.5 * np.pi / 12  # oscillation angle amplitude
    _T_WARMUP = 2.0

    def warmup(self, env) -> dict:
        """Smoothly compresses the right end from x=L to x=0.6L, keeping the angle at +/- pi/3."""
        print("Warmup: compressing rod ...")
        N = int(round(self._T_WARMUP / env.dt))
        obs = None
        for i in range(N):
            s   = self._smoothstep((i + 1) * env.dt, self._T_WARMUP)
            x_r = L + s * (self._X_END - L)      # L -> 0.6L
            ctrl = np.array([0.0, 0.0,  self._TH_INIT,
                             x_r, 0.0, -self._TH_INIT])
            obs, _, _ = env.step(ctrl)
        self._reset_env_counters(env)
        print("Warmup done.")
        return obs

    def control(self, t: float) -> np.ndarray:
        th = -self._TH_AMP * np.cos(2 * np.pi * t / T)
        return np.array([0.0, 0.0, -th, self._X_END, 0.0, th])


# =========================================================
# Trajectory 3: angle oscillation + both ends drifting right at constant speed
# =========================================================

class AngleOscillationDrift(AngleOscillation):
    """
    Inherits AngleOscillation's warmup (compress to 0.6L).
    Main phase: while oscillating the angle, both ends also drift right at a
    constant speed _DRIFT (L/s).
    """
    name  = "angle_oscillation_drift"
    xlim  = (-0.2 * L, 1.2 * L)   # widened on the right to accommodate the drift
    ylim  = (-0.5 * L, 0.5 * L)

    _DRIFT = 0.0 * L   # drift speed (m/s); over 10s each end moves right by 0.5L

    def control(self, t: float) -> np.ndarray:
        th  = -self._TH_AMP * np.cos(3 * np.pi * t / T)
        x_l = 0.0          + self._DRIFT * t   # left end drifts right
        x_r = self._X_END  + self._DRIFT * t   # right end drifts right (spacing stays 0.6L)
        return np.array([x_l, 0.0, -th, x_r, 0.0, th])


# =========================================================
# Trajectory registry (to add a new trajectory: subclass Trajectory and
# register it here)
# =========================================================

TRAJ_REGISTRY: dict[str, type[Trajectory]] = {
    "moving_forward":          MovingForward,
    "angle_oscillation":       AngleOscillation,
    "angle_oscillation_drift": AngleOscillationDrift,
}


# =========================================================
# Rendering helper functions
# =========================================================

def _get_key_velocities(env):
    mid = env.mid_node
    q1  = env.nv // 4
    q3  = 3 * env.nv // 4
    return [
        (env.q[0],      env.q[1],      env.u[0],      env.u[1],      ARROW_ENDPOINT_COLOR),
        (env.q[2*q1],   env.q[2*q1+1], env.u[2*q1],   env.u[2*q1+1], ARROW_QUARTER_COLOR_1),
        (env.q[2*mid],  env.q[2*mid+1],env.u[2*mid],  env.u[2*mid+1],ARROW_MIDPOINT_COLOR),
        (env.q[2*q3],   env.q[2*q3+1], env.u[2*q3],   env.u[2*q3+1], ARROW_QUARTER_COLOR_3),
        (env.q[-2],     env.q[-1],     env.u[-2],     env.u[-1],     ARROW_ENDPOINT_COLOR),
    ]


def _draw_velocity_arrows(ax, key_vels):
    for x, y, vx, vy, color in key_vels:
        if color in (ARROW_QUARTER_COLOR_1, ARROW_QUARTER_COLOR_3) and not SHOW_QUARTER_ARROWS:
            continue
        if np.hypot(vx, vy) < 1e-6:
            continue
        ax.annotate('',
                    xy=(x + vx*ARROW_SCALE, y + vy*ARROW_SCALE),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle='-|>', color=color,
                                   lw=round(1.2 * _PAPER_S, 2),
                                   mutation_scale=round(9 * _PAPER_S, 1),
                                   shrinkA=0, shrinkB=0),
                    zorder=25)


def _draw_frozen_snaps(ax, frozen_snaps, xlim, ylim):
    ax.autoscale(False)

    # choose a perceptually-uniform colormap for curvature
    cmap_name = 'viridis'

    # supports a single value or a list: find, for each target percent, the
    # nearest frame (>= target and not 100%)
    highlight_fracs: set[float] = set()
    if HIGHLIGHT_SNAP is not None and frozen_snaps:
        targets = HIGHLIGHT_SNAP if isinstance(HIGHLIGHT_SNAP, (list, tuple)) else [HIGHLIGHT_SNAP]
        for target in targets:
            cands = [f for f, *_ in frozen_snaps if f*100 >= target and f < 1.0]
            if cands:
                highlight_fracs.add(min(cands))

    # compute a common normalization across all snapshots so the colorbar is meaningful
    global_vmax = 1.0
    if frozen_snaps:
        global_vmax = max(item[4] for item in frozen_snaps)
    norm = plt.Normalize(vmin=0.0, vmax=max(global_vmax, 1e-6))

    for frac, x_arr, y_arr, seg_kappa, vmax, key_vels in frozen_snaps:
        is_highlight = frac in highlight_fracs
        # make highlighted snapshots visually bolder
        if highlight_fracs and not is_highlight:
            alpha, linewidth = DIMMED_ALPHA, round(1.0 * _PAPER_S, 2)
        elif is_highlight:
            alpha = 0.95
            linewidth = round(3.6 * _PAPER_S, 2)
        else:
            alpha, linewidth = 0.45 + 0.50*frac, round(2.5 * _PAPER_S, 2)

        pts  = np.column_stack([x_arr, y_arr])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap=cmap_name,
                      norm=norm,
                      linewidths=linewidth, capstyle='round',
                      alpha=alpha,
                      zorder=20 if not is_highlight else 22)
        lc.set_array(seg_kappa)
        ax.add_collection(lc)
        _draw_velocity_arrows(ax, key_vels)

        if SHOW_SNAP_LABELS:
            mid = len(x_arr) // 2
            ax.text(x_arr[mid], y_arr[mid], f'{int(round(frac*100))}%',
                    fontsize=6.5, color='#1b4332',
                    alpha=min(alpha+0.20, 1.0),
                    ha='center', va='bottom', zorder=21)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # NOTE: colorbar is created once after the simulation finishes
    # to avoid adding a new colorbar every frame during live updates.


# =========================================================
# theta-H phase plot
# =========================================================

def _plot_theta_H_curve(theta_hist, H_hist, traj_name):
    theta  = np.array(theta_hist)
    H      = np.array(H_hist)
    t_norm = np.linspace(0, 1, len(theta))
    font_name = 'Times New Roman'

    # Scale factor relative to original 6" wide figure
    _S = 1.375 / 6.0   # ~= 0.229

    # use a compact figure size suitable for two-column layout
    fig, ax = plt.subplots(figsize=(2.75 * _PAPER_S, 2.2 * _PAPER_S))

    # Time-colored curve (use perceptually-uniform colormap)
    pts  = np.column_stack([theta, H]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap='viridis',
                          norm=plt.Normalize(0, 1),
                          linewidth=1.6, zorder=2)
    lc.set_array(t_norm[:-1])
    ax.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Progress (0 → 1)', fontsize=10, fontfamily=font_name)
    cbar.ax.tick_params(labelsize=8)

    # Direction arrows only at theta = 0 crossings
    half_win = max(3, len(theta) // 100)
    for i in range(1, len(theta)):
        if theta[i - 1] * theta[i] > 0:   # no sign change
            continue
        i0 = max(0,          i - half_win)
        i1 = min(len(theta), i + half_win)
        dth = theta[i1 - 1] - theta[i0]
        dH  = H[i1 - 1]     - H[i0]
        norm = np.hypot(dth, dH)
        if norm < 1e-10:
            continue
        th_range = theta.max() - theta.min() + 1e-6
        H_range  = H.max()     - H.min()     + 1e-6
        scale    = 0.03
        dth_draw = dth / norm * th_range * scale
        dH_draw  = dH  / norm * H_range  * scale
        th_c, H_c = theta[i], H[i]
        ax.annotate('',
                xy=(th_c + dth_draw, H_c + dH_draw),
                xytext=(th_c, H_c),
                arrowprops=dict(arrowstyle='->', color=ARROW_ENDPOINT_COLOR,
                       lw=1.2,
                       mutation_scale=12,
                       shrinkA=0, shrinkB=0),
                zorder=5)

    margin_th = (theta.max() - theta.min()) * 0.05 + 1e-4
    margin_H  = (H.max()     - H.min())     * 0.08 + 1e-4
    ax.set_xlim(theta.min() - margin_th, theta.max() + margin_th)
    ax.set_ylim(H.min()     - margin_H,  H.max()     + margin_H)
    ax.set_xlabel('$\\theta$ (rad)', fontsize=10, fontfamily=font_name)
    ax.set_ylabel('$H$ (m)',          fontsize=10, fontfamily=font_name)
    ax.set_title('$\\theta$ vs midpoint height $H$', fontsize=10, fontfamily=font_name)
    ax.tick_params(labelsize=8, length=3.5, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(font_name)
    ax.xaxis.label.set_fontfamily(font_name)
    ax.yaxis.label.set_fontfamily(font_name)
    ax.title.set_fontfamily(font_name)
    for spine in ax.spines.values():
        spine.set_linewidth(round(0.8 * _S, 2))
    cbar.outline.set_visible(False)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(labelsize=8, length=3.5, width=0.8)
    cbar.ax.yaxis.label.set_fontfamily(font_name)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily(font_name)
    ax.grid(False)

    # subfigure label inside the axes
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes,
            ha='left', va='top', fontsize=10, fontweight='bold',
            fontfamily=font_name)

    fig.tight_layout()

    out = str(_OUT / f'theta_H_{traj_name}')
    fig.savefig(f'{out}.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{out}.pdf', bbox_inches='tight')
    print(f'Saved: {out}.png / .pdf')
    plt.show()


# =========================================================
# Main simulation loop
# =========================================================

def run_simulation(traj: Trajectory, show_snaps: bool = True, show_trails: bool = True):
    env = SimulatorEnv_2D(nv=ROD_NV, dt=ROD_DT, rod_length=L,
                          R_outer=ROD_R_OUTER, r_inner=0.0,
                          E_al=ROD_E, rho_al=ROD_RHO,
                          total_time=T, save_history=True)

    # reset() always initializes the rod as a straight horizontal segment x=0->L
    obs = env.reset()

    # Warmup phase (not counted toward progress, no snapshots recorded)
    obs = traj.warmup(env)

    frozen_snaps = []
    snap_targets = sorted(SNAP_PERCENTS) if show_snaps else []
    theta_hist, H_hist = [], []   # for the theta-H phase plot

    _fig_resized = False
    done = False
    while not done:
        t_osc   = obs["time"] + env.dt
        control = traj.control(t_osc)
        obs, done, _ = env.step(control)

        # Record the right-end angle theta_r and midpoint height H (skip the
        # first frame to avoid the warmup->main-phase jump)
        if env.current_step > 1:
            theta_hist.append(control[5])
            H_hist.append(env.q[2 * env.mid_node + 1])

        pct = env.current_step / env.Nsteps * 100.0

        # Freeze the current shape when a snapshot target is hit (only if show_snaps)
        if snap_targets and pct >= snap_targets[0] and \
                (snap_targets[0] > 0 or env.current_step >= 2):
            frac      = snap_targets[0] / 100.0
            x_arr     = env.q[::2].copy()
            y_arr     = env.q[1::2].copy()
            kappa     = _node_curvatures(x_arr, y_arr)
            seg_kappa = 0.5 * (kappa[:-1] + kappa[1:])
            vmax      = max(np.percentile(kappa, 95), 1e-3)
            frozen_snaps.append(
                (frac, x_arr, y_arr, seg_kappa, vmax, _get_key_velocities(env))
            )
            snap_targets.pop(0)

        # Draw the current frame
        render_dlo_neurips(env, control=control, show=False,
                           xlim=traj.xlim, ylim=traj.ylim,
                           show_gradient=True, show_trail=show_trails,
                           show_quarter_trails=show_trails, show_colorbar=False,
                           show_legend=False, show_midpoint=False,
                           lw_scale=_PAPER_S)

        if PAPER_MODE and not _fig_resized:
            _w, _h = env._nr_fig.get_size_inches()
            env._nr_fig.set_size_inches(_PAPER_FIG_W, _h * _PAPER_FIG_W / _w)
            _fig_resized = True

        ax = env._nr_ax
        ax.set_facecolor('white')
        if HIGHLIGHT_SNAP is not None:
            for artist in ax.collections + ax.lines:
                artist.set_alpha(DIMMED_ALPHA)
        _draw_velocity_arrows(ax, _get_key_velocities(env))
        ax.grid(False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        if show_snaps:
            _draw_frozen_snaps(ax, frozen_snaps, traj.xlim, traj.ylim)

        env._nr_fig.canvas.draw()
        plt.pause(0.01)

    print("Simulation finished.")

    plt.ioff()
    out_base = str(_OUT / f'snapshot_overlay_{traj.name}')
    # add a single colorbar for curvature (left plot) after simulation
    if show_snaps and frozen_snaps:
        try:
            global_vmax = max(item[4] for item in frozen_snaps)
            norm = mpl.colors.Normalize(vmin=0.0, vmax=max(global_vmax, 1e-6))
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap='viridis')
            mappable.set_array([])
            cbar = env._nr_fig.colorbar(mappable, ax=env._nr_ax, pad=0.015, fraction=0.030)
            cbar.set_label('Curvature', fontsize=10)
            cbar.ax.tick_params(labelsize=8)
            cbar.outline.set_visible(False)
            for spine in cbar.ax.spines.values():
                spine.set_visible(False)
        except Exception:
            pass

    # add subfigure label inside the snapshot figure
    try:
        env._nr_ax.text(0.02, 0.98, '(a)', transform=env._nr_ax.transAxes,
                        ha='left', va='top', fontsize=10, fontweight='bold')
    except Exception:
        pass

    env._nr_fig.savefig(f'{out_base}.png', dpi=200, bbox_inches='tight')
    env._nr_fig.savefig(f'{out_base}.pdf', bbox_inches='tight')
    print(f'Saved: {out_base}.png / .pdf')
    plt.show()

    # theta-H phase plot
    _plot_theta_H_curve(theta_hist, H_hist, traj.name)


# =========================================================
# CLI entry point
# =========================================================

def parse_args():
    traj_list = '\n'.join(f'  {k}' for k in TRAJ_REGISTRY)
    parser = argparse.ArgumentParser(
        description='DLO snapshot overlay simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'available trajectories:\n{traj_list}',
    )
    parser.add_argument(
        '--traj', '-t',
        default='angle_oscillation',
        choices=list(TRAJ_REGISTRY.keys()),
        metavar='NAME',
        help='trajectory to run (default: angle_oscillation)',
    )
    parser.add_argument(
        '--no-snaps',
        action='store_true',
        help='disable frozen snapshot overlay (show live rod only)',
    )
    parser.add_argument(
        '--no-trails',
        action='store_true',
        help='hide trails of midpoint, 1/4-point, and 3/4-point',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    traj        = TRAJ_REGISTRY[args.traj]()
    show_trails = SHOW_TRAILS and not args.no_trails
    print(f"Trajectory  : {traj.name}")
    print(f"Show snaps  : {not args.no_snaps}")
    print(f"Show trails : {show_trails}")
    run_simulation(traj, show_snaps=not args.no_snaps, show_trails=show_trails)
