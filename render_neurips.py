# -*- coding: utf-8 -*-
"""
NeurIPS-style rendering for the DLO (Discrete Elastic Rod) 2D simulator.

Usage:
  from render_neurips import render_dlo_neurips

  env.reset()
  done = False
  while not done:
      obs, done, info = env.step(control)
      render_dlo_neurips(env, control=control, target=(x_t, y_t), show=True)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba


# -------------------------------------------------------------------------
# Colormap A: dark-green → orange-yellow gradient (show_gradient=True)
# -------------------------------------------------------------------------
_DLO_CMAP_COLORS = [
    (0.00, '#1b4332'),   # deep forest green  (low curvature)
    (0.25, '#2d6a4f'),   # dark green
    (0.50, '#74a72e'),   # olive-green
    (0.72, '#e07b00'),   # dark orange
    (0.88, '#f5a623'),   # amber
    (1.00, '#f7d155'),   # orange-yellow      (high curvature)
]
DLO_CMAP = LinearSegmentedColormap.from_list(
    'dlo_curvature',
    [(p, c) for p, c in _DLO_CMAP_COLORS],
    N=256,
)

# Flat color used when show_gradient=False
_ROD_FLAT_COLOR = '#1b4332'   # deep forest green


# -------------------------------------------------------------------------
# Helper: discrete curvature at each node
# -------------------------------------------------------------------------
def _node_curvatures(x_arr, y_arr):
    n = len(x_arr)
    kappa = np.zeros(n)
    for k in range(1, n - 1):
        e0 = np.array([x_arr[k] - x_arr[k - 1], y_arr[k] - y_arr[k - 1]])
        e1 = np.array([x_arr[k + 1] - x_arr[k], y_arr[k + 1] - y_arr[k]])
        n0, n1 = np.linalg.norm(e0), np.linalg.norm(e1)
        if n0 < 1e-12 or n1 < 1e-12:
            continue
        t0, t1 = e0 / n0, e1 / n1
        cross = t0[0] * t1[1] - t0[1] * t1[0]
        dot   = t0[0] * t1[0] + t0[1] * t1[1]
        kappa[k] = 2.0 * abs(cross) / max(1.0 + dot, 1e-12)
    kappa[0]  = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa


# -------------------------------------------------------------------------
# Main render function
# -------------------------------------------------------------------------
def render_dlo_neurips(env,
                       control=None,
                       target=None,
                       # --- per-element visibility switches ---
                       show_rod=True,
                       show_halo=True,
                       show_clamps=True,
                       show_clamp_arrows=False,
                       show_midpoint=True,
                       show_target=False,
                       show_trail=True,
                       show_quarter_trails=False,
                       show_gravity=False,
                       show_colorbar=True,
                       show_legend=True,
                       show_gradient=True,
                       # --- layout / style ---
                       title_prefix='',
                       saveimg=False,
                       save_dir='imgs',
                       show=True,
                       ax=None,
                       xlim=None,
                       ylim=None,
                       curvature_cmap=DLO_CMAP,
                       curvature_vmax=None,
                       dpi=140,
                       lw_scale=1.0):
    """
    Render the current DLO configuration in a NeurIPS-friendly style.

    Parameters
    ----------
    env : SimulatorEnv_2D
    control : array-like (6,), optional
        [x_l, y_l, th_l, x_r, y_r, th_r].  Only used when
        show_clamp_arrows=True.
    target : tuple (x, y), optional
    show_rod : bool            draw the rod body
    show_halo : bool           draw the soft halo underneath the rod
    show_clamps : bool         draw the boundary clamp squares
    show_clamp_arrows : bool   draw orientation arrows at clamps
    show_midpoint : bool       draw the midpoint ring marker
    show_target : bool         draw the target marker (requires target != None)
    show_trail : bool          draw past midpoint trajectory
    show_gravity : bool        draw gravity arrow in corner
    show_colorbar : bool       draw the curvature colorbar
    show_legend : bool         draw the legend
    show_gradient : bool       True → dark-green-to-orange-yellow curvature
                               gradient; False → solid deep forest green
    title_prefix : str
    saveimg : bool
    save_dir : str
    show : bool
    ax : matplotlib Axes, optional
    xlim, ylim : tuple, optional
    curvature_cmap : matplotlib colormap
    curvature_vmax : float, optional
    dpi : int
    """
    # ---- Extract state ----
    x_arr = env.q[::2]
    y_arr = env.q[1::2]
    midx  = x_arr[env.mid_node]
    midy  = y_arr[env.mid_node]

    # ---- Initialize figure/axes on first call ----
    if ax is None:
        if not getattr(env, '_nr_initialized', False):
            plt.rcParams.update({
                'font.family':      'serif',
                'font.serif':       ['Times New Roman', 'Times', 'Nimbus Roman No9 L', 'Liberation Serif'],
                'mathtext.fontset': 'custom',
                'mathtext.rm':      'Times New Roman',
                'mathtext.it':      'Times New Roman:italic',
                'mathtext.bf':      'Times New Roman:bold',
                'pdf.use14corefonts': False,
                'ps.useafm':         False,
                'pdf.fonttype':      42,
                'ps.fonttype':       42,
                'axes.edgecolor':   '#2d2d2d',
                'axes.labelcolor':  '#1a1a1a',
                'xtick.color':      '#1a1a1a',
                'ytick.color':      '#1a1a1a',
                'axes.linewidth':   0.9,
            })
            env._nr_fig, env._nr_ax = plt.subplots(figsize=(7.0, 4.2), dpi=dpi)
            env._nr_initialized  = True
            env._nr_cbar_added   = False
            plt.ion()
        ax = env._nr_ax
    ax.cla()

    # ---- Background ----
    ax.set_facecolor('#fafbfd')
    ax.grid(True, which='major', linestyle='-', linewidth=round(0.5 * lw_scale, 2),
            color='#d9dde3', alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    # ---- Axis limits ----
    if xlim is None:
        pad  = 0.25 * env.rod_length
        xlim = (-pad, env.rod_length + pad)
    if ylim is None:
        ylim = (-0.6 * env.rod_length, 0.6 * env.rod_length)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')

    # ---- Curvature ----
    kappa = _node_curvatures(x_arr, y_arr)
    if curvature_vmax is None:
        v = np.percentile(kappa, 95)
        curvature_vmax = max(v, 1e-3)

    pts      = np.column_stack([x_arr, y_arr])
    segs     = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_kappa = 0.5 * (kappa[:-1] + kappa[1:])

    rod_collection = None   # needed for colorbar

    # ---- Halo ----
    if show_halo:
        halo = LineCollection(
            segs,
            colors=[to_rgba('#9dd8d8', alpha=0.30)] * len(segs),
            linewidths=round(6 * lw_scale, 2),
            capstyle='round',
            zorder=3,
        )
        ax.add_collection(halo)

    # ---- Rod ----
    if show_rod:
        if show_gradient:
            lc = LineCollection(
                segs,
                cmap=curvature_cmap,
                norm=plt.Normalize(vmin=0.0, vmax=curvature_vmax),
                linewidths=round(3.0 * lw_scale, 2),
                capstyle='round',
                zorder=4,
            )
            lc.set_array(seg_kappa)
            rod_collection = ax.add_collection(lc)
        else:
            lc = LineCollection(
                segs,
                colors=[to_rgba(_ROD_FLAT_COLOR, alpha=1.0)] * len(segs),
                linewidths=round(3.0 * lw_scale, 2),
                capstyle='round',
                zorder=4,
            )
            ax.add_collection(lc)

        # Thin white core for crispness
        ax.plot(x_arr, y_arr, color='white', linewidth=round(0.6 * lw_scale, 2),
                alpha=0.50, solid_capstyle='round', zorder=5)

    # ---- Past midpoint trail ----
    if show_trail and getattr(env, 'save_history', False):
        hist_x = np.asarray(env.history.get('midx', []), dtype=float)
        hist_y = np.asarray(env.history.get('midy', []), dtype=float)
        if len(hist_x) > 1:
            ax.plot(hist_x, hist_y, '-', color='#d62728', lw=0.9,
                    alpha=0.30, zorder=2, label='midpoint trail')

    # ---- 1/4 and 3/4 point trails ----
    if show_quarter_trails and getattr(env, 'save_history', False):
        for key_x, key_y, color in [('q1x', 'q1y', '#0088dd'),
                                     ('q3x', 'q3y', '#9b2dca')]:
            hx = np.asarray(env.history.get(key_x, []), dtype=float)
            hy = np.asarray(env.history.get(key_y, []), dtype=float)
            if len(hx) > 1:
                ax.plot(hx, hy, '-', color=color, lw=0.9,
                        alpha=0.30, zorder=2)

    # ---- Boundary clamps ----
    clamp_color = '#1a4f6e'
    if show_clamps:
        for cx, cy in [(x_arr[0], y_arr[0]), (x_arr[-1], y_arr[-1])]:
            # Outer filled block (mechanical grip appearance)
            ax.plot(cx, cy, 's',
                    markersize=round(10 * lw_scale, 1),
                    markerfacecolor=clamp_color,
                    markeredgecolor='white',
                    markeredgewidth=round(1.8 * lw_scale, 2),
                    zorder=9)
            # Inner highlight dot for depth
            ax.plot(cx, cy, 's',
                    markersize=round(4 * lw_scale, 1),
                    markerfacecolor='white',
                    markeredgecolor='white',
                    alpha=0.40,
                    zorder=10)

        # ---- Optional clamp orientation arrows ----
        if show_clamp_arrows:
            arrow_len = 0.08 * env.rod_length
            if control is not None and len(control) == 6:
                th_l, th_r = control[2], control[5]
            else:
                dl = np.array([x_arr[1]  - x_arr[0],  y_arr[1]  - y_arr[0]])
                dr = np.array([x_arr[-1] - x_arr[-2], y_arr[-1] - y_arr[-2]])
                th_l = np.arctan2(dl[1], dl[0])
                th_r = np.arctan2(dr[1], dr[0])

            def _arrow(x0, y0, dx, dy):
                ax.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle='-|>', color=clamp_color,
                                            lw=round(2.0 * lw_scale, 2),
                                            mutation_scale=round(12 * lw_scale, 1),
                                            alpha=0.90, shrinkA=0, shrinkB=0),
                            zorder=11)

            _arrow(x_arr[0],  y_arr[0],
                    arrow_len * np.cos(th_l),  arrow_len * np.sin(th_l))
            _arrow(x_arr[-1], y_arr[-1],
                   -arrow_len * np.cos(th_r), -arrow_len * np.sin(th_r))

    # ---- Midpoint marker ----
    if show_midpoint:
        ax.plot(midx, midy, 'o', markersize=round(5 * lw_scale, 1),
                markerfacecolor='white', markeredgecolor='#d62728',
                markeredgewidth=round(1.6 * lw_scale, 2), zorder=11)
        ax.plot(midx, midy, 'o', markersize=round(2 * lw_scale, 1),
                markerfacecolor='#d62728', markeredgecolor='#d62728',
                zorder=12)

    # ---- Target marker ----
    if show_target and target is not None:
        tx, ty = target
        ax.plot(tx, ty, marker='P', markersize=11,
                markerfacecolor='#ffd60a', markeredgecolor='#1a1a1a',
                markeredgewidth=1.2, zorder=13, label='target')
        if show_midpoint:
            ax.plot([midx, tx], [midy, ty], ':', color='#8a8f98',
                    lw=1.0, alpha=0.70, zorder=6)

    # ---- Gravity arrow ----
    if show_gravity:
        gx    = xlim[0] + 0.07 * (xlim[1] - xlim[0])
        gy    = ylim[1] - 0.08 * (ylim[1] - ylim[0])
        g_len = 0.08 * (ylim[1] - ylim[0])
        ax.annotate('', xy=(gx, gy - g_len), xytext=(gx, gy),
                    arrowprops=dict(arrowstyle='-|>', color='#2d2d2d',
                                    lw=1.4, mutation_scale=10),
                    zorder=20)
        ax.text(gx + 0.015, gy - g_len * 0.5, r'$\mathbf{g}$',
                fontsize=11, color='#2d2d2d', va='center', zorder=20)

    # ---- Colorbar ----
    if show_colorbar and rod_collection is not None:
        if not getattr(env, '_nr_cbar_added', False):
            cbar = plt.colorbar(rod_collection, ax=ax,
                                shrink=0.78, pad=0.02, aspect=24)
            cbar.set_label(r'curvature  $|\kappa|$ (1/m)', fontsize=10)
            cbar.ax.tick_params(labelsize=9)
            env._nr_cbar       = cbar
            env._nr_cbar_added = True
        else:
            env._nr_cbar.mappable.set_clim(0.0, curvature_vmax)

    # ---- Axes labels / title ----
    ax.set_xlabel(r'$x$ (m)', fontsize=11)
    ax.set_ylabel(r'$y$ (m)', fontsize=11)
    ax.set_title(title_prefix + f't = {env.ctime:6.3f} s',
                 fontsize=12, pad=6, loc='left', color='#1a1a1a')

    for spine in ax.spines.values():
        spine.set_linewidth(round(0.9 * lw_scale, 2))
        spine.set_color('#2d2d2d')

    # ---- Legend ----
    if show_legend:
        from matplotlib.lines import Line2D
        handles = []
        if show_rod:
            rod_legend_color = '#2d6a4f' if not show_gradient else '#1aab99'
            rod_legend_label = 'rod' if not show_gradient else 'rod (color = $|\\kappa|$)'
            handles.append(Line2D([0], [0], color=rod_legend_color, lw=3.5,
                                   label=rod_legend_label))
        if show_clamps:
            handles.append(Line2D([0], [0], marker='s', color='w',
                                   markerfacecolor=clamp_color,
                                   markeredgecolor='white',
                                   markeredgewidth=1.5, markersize=9, lw=0,
                                   label='clamp'))
        if show_midpoint:
            handles.append(Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor='white',
                                   markeredgecolor='#d62728',
                                   markeredgewidth=2.0, markersize=9, lw=0,
                                   label='midpoint (obs.)'))
        if show_target and target is not None:
            handles.append(Line2D([0], [0], marker='P', color='w',
                                   markerfacecolor='#ffd60a',
                                   markeredgecolor='#1a1a1a',
                                   markeredgewidth=1.0, markersize=10, lw=0,
                                   label='target'))
        if show_trail and getattr(env, 'save_history', False):
            handles.append(Line2D([0], [0], color='#d62728', lw=1.2,
                                   alpha=0.45, label='midpoint trail'))
        if handles:
            ax.legend(handles=handles, loc='lower center', ncol=3,
                      fontsize=8.5, frameon=True, framealpha=0.95,
                      edgecolor='#2d2d2d',
                      bbox_to_anchor=(0.5, -0.28))

    # ---- Save frame ----
    if saveimg:
        os.makedirs(save_dir, exist_ok=True)
        base = f'{save_dir}/render_{env.ctime:06.3f}'
        plt.savefig(base + '.png', dpi=dpi, bbox_inches='tight')

    if show:
        plt.pause(0.01)


# -------------------------------------------------------------------------
# Stand-alone demo
# -------------------------------------------------------------------------
if __name__ == '__main__':
    from SimulatorEnv_2D import SimulatorEnv_2D

    env = SimulatorEnv_2D(nv=31, dt=0.01, rod_length=1.0,
                          total_time=3.0, save_history=True)
    obs = env.reset()

    snapshot_times = [0.5, 1.0, 1.5, 2.0]
    saved_snaps    = {}

    done = False
    while not done:
        t        = obs['time']
        x_target = env.rod_length
        y_target = 0.22 * np.sin(2 * np.pi * 0.5 * t)
        control  = np.array([0.0, 0.0, 0.0, x_target, y_target, 0.0])

        obs, done, info = env.step(control)

        render_dlo_neurips(
            env,
            control=control,
            target=(x_target, y_target),
            show_rod=True,
            show_halo=True,
            show_clamps=True,
            show_clamp_arrows=False,
            show_midpoint=True,
            show_target=True,
            show_trail=True,
            show_gravity=False,
            show_colorbar=True,
            show_legend=True,
            title_prefix='DLO simulation  |  ',
            saveimg=False,
            show=False,
        )

        for st in list(snapshot_times):
            if abs(env.ctime - st) < 0.5 * env.dt:
                fname = f'dlo_snap_t{st:.2f}.png'
                plt.savefig(fname, dpi=200, bbox_inches='tight')
                saved_snaps[st] = fname
                print(f'  saved {fname}')
                snapshot_times.remove(st)
                break

    plt.savefig('dlo_final_snapshot.png', dpi=200, bbox_inches='tight')
    plt.savefig('dlo_final_snapshot.pdf', bbox_inches='tight')
    print('Saved final snapshot: dlo_final_snapshot.png / .pdf')
