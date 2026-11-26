"""IEEE T-RO figure style for every plot of the manuscript (2D and 3D).

Implements the project rules in ``TRO_figure_style.md``:

* final print size: ``COL_W`` = 3.5 in, ``DBL_W`` = 7.16 in;
* Times New Roman 8 pt (labels, titles) / 7 pt (ticks, legend), stix math;
* thin lines (data 1.0 pt, reference 0.8 pt dashed, axes 0.5 pt), inward ticks;
* Okabe-Ito colours, one look per quantity (``ROLE``) in every figure;
* PDF + SVG with editable text (``pdf.fonttype 42``, ``svg.fonttype none``).

and adds what 3D scenes (rod, paths, obstacles, workspace) need on top of it:
white panes, a faint grid, an equal-scale box fitted to the data instead of a
cube around everything, clipped workspace edges and a quiet restyle of the
obstacle surfaces that ``utils.draw_obstacles`` / ``pipe.draw`` produce.

Usage::

    import tro_style as ts
    with ts.style():                                  # rcParams only inside the block
        fig = plt.figure(figsize=(ts.COL_W, 2.8))
        ax = fig.add_subplot(projection="3d")
        ts.style_3d(ax, view=(22, -58))
        ax.plot(*ref.T, **ts.ROLE["reference"], label="Reference")
        ...
        ts.fit_box(ax, [ref, mids, rod])              # after everything is drawn
        ts.legend_top(fig, ax)
        ts.save(fig, "fig5_tracking", width=ts.COL_W) # .pdf, .svg (+ .png preview)

``ts.apply()`` sets the same rcParams globally, for a script that only makes
paper figures.  Everything (creating *and* saving the figure) must happen while
the style is active, because fonts are resolved when the file is written.

Self-test (fonts, export, a synthetic rod scene)::

    python src/tro_style.py [out_dir]
"""
from __future__ import annotations

import contextlib
import warnings
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------- constants

COL_W = 3.5         # in, single column (88.9 mm)
DBL_W = 7.16        # in, double column (181.9 mm)
MAX_H = 8.5         # in, leave room for the caption
FS_LABEL = 8        # pt: axis labels, panel titles, panel letters
FS_TICK = 7         # pt: tick labels, legend, annotations

# Okabe-Ito (colour-blind safe)
BLUE, VERMILION, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
ORANGE, SKY, YELLOW = "#E69F00", "#56B4E9", "#F0E442"
BLACK, GRAY, MID_GRAY, LIGHT_GRAY = "#000000", "#555555", "#8C8C8C", "#BDBDBD"
SHADE = "#DCE8F5"                  # light region shade (e.g. penetration d < 0)
OBSTACLE = "#9FB6CD"               # base tone of obstacle / pipe surfaces
PANE_EDGE, GRID_3D = "#B5B5B5", "#E3E3E3"

SERIF = ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman",
         "Nimbus Roman No9 L", "TeX Gyre Termes", "STIXGeneral", "DejaVu Serif"]

AXIS_LABELS_3D = (r"$x$ (m)", r"$y$ (m)", r"$z$ (m)")

#: One look per quantity, identical in every figure: ``ax.plot(..., **ROLE[k])``.
#: Colour is never the only cue -- width / dash / marker shape differ as well,
#: so the figures stay readable in grayscale print.
ROLE = {
    # desired path or target shape
    "reference": dict(color=GRAY, linestyle=(0, (4, 2)), linewidth=0.8),
    # what the tracked node (mid node, tip) actually did
    "achieved": dict(color=VERMILION, linestyle="-", linewidth=1.0),
    # rod configuration at the end of the rollout / at the start
    "rod": dict(color=BLUE, linestyle="-", linewidth=1.5, solid_capstyle="round"),
    "rod_init": dict(color="#C2C2C2", linestyle="-", linewidth=1.0, solid_capstyle="round"),
    # where the robot moved the gripper (the commanded end of the rod)
    "gripper_path": dict(color=GREEN, linestyle=(0, (1, 1.2)), linewidth=1.0),
    # constraint boxes / limits
    "workspace": dict(color=MID_GRAY, linestyle=(0, (3, 2)), linewidth=0.5),
    # markers
    "start": dict(marker="o", markersize=4, markerfacecolor="white", markeredgecolor=BLACK,
                  markeredgewidth=0.7, linestyle="none"),
    "goal": dict(marker="x", markersize=5, color=BLACK, markeredgewidth=1.0, linestyle="none"),
    "gripper": dict(marker="s", markersize=3.5, markerfacecolor=GRAY, markeredgecolor=BLACK,
                    markeredgewidth=0.5, linestyle="none"),
}

#: zorder bands used when ``computed_zorder`` is off (see ``style_3d``)
Z_SCENE, Z_WS, Z_ROD, Z_PATH, Z_MARK = 1, 1.5, 3, 4, 5


# ---------------------------------------------------------------- rcParams

def rc(font: str | None = None) -> dict:
    """The rcParams of the style (``font`` puts another serif family first)."""
    serif = ([font] if font else []) + [f for f in SERIF if f != font]
    r = {
        # text
        "font.family": "serif",
        "font.serif": serif,
        "mathtext.fontset": "stix",
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_LABEL,
        "figure.titlesize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,
        "legend.title_fontsize": FS_TICK,
        "axes.titleweight": "normal",
        "axes.titlepad": 3.0,
        "axes.labelpad": 2.0,
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": True,
        "text.usetex": False,                       # usetex outlines all text
        # lines and axes
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        "lines.solid_capstyle": "butt",
        "lines.dash_capstyle": "butt",
        "patch.linewidth": 0.5,
        "hatch.linewidth": 0.4,
        "axes.linewidth": 0.5,
        "axes.edgecolor": BLACK,
        "axes.labelcolor": BLACK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=[BLUE, VERMILION, GREEN, PURPLE, ORANGE, SKY]),
        "grid.color": GRID_3D,
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        # legend: frameless, compact
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.25,
        "legend.borderaxespad": 0.2,
        "legend.borderpad": 0.2,
        # export: editable text, reproducible files
        "figure.dpi": 150,
        "savefig.dpi": 600,                         # only matters for rasterized parts
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "tro-figure",
        "image.cmap": "cividis",
    }
    for key in ("axes3d.xaxis.panecolor", "axes3d.yaxis.panecolor", "axes3d.zaxis.panecolor"):
        if key in mpl.rcParams:                      # matplotlib >= 3.8
            r[key] = (1.0, 1.0, 1.0, 0.0)
    return r


_FONT_CHECKED = False


def resolved_font() -> tuple[str, str]:
    """(family name, file) of the serif font matplotlib will actually use."""
    from matplotlib.font_manager import FontProperties, findfont, get_font
    path = str(findfont(FontProperties(family="serif"), fallback_to_default=True))
    return get_font(path).family_name, path


def _check_font():
    global _FONT_CHECKED
    if _FONT_CHECKED:
        return
    _FONT_CHECKED = True
    name, path = resolved_font()
    if name != mpl.rcParams["font.serif"][0]:
        warnings.warn(
            f"tro_style: '{mpl.rcParams['font.serif'][0]}' is not installed, using '{name}' "
            f"({path}). Liberation Serif / TeX Gyre Termes / Nimbus Roman have the same metrics, "
            "but the PDF embeds the fallback. Linux: install ttf-mscorefonts-installer, then "
            "delete ~/.cache/matplotlib.", stacklevel=3)


@contextlib.contextmanager
def style(font: str | None = None):
    """Context manager: the T-RO rcParams inside the block, the old ones after."""
    with mpl.rc_context(rc(font)):
        _check_font()
        yield


def apply(font: str | None = None):
    """Set the T-RO rcParams globally (affects every later figure)."""
    mpl.rcParams.update(rc(font))
    _check_font()


# ---------------------------------------------------------------- 2D helpers

def panel_title(ax, label: str, text: str | None = None, **kw):
    """``(a)`` (optionally followed by ``text``), regular weight, left-aligned."""
    s = label if not text else f"{label} {text}"
    kw.setdefault("fontsize", FS_LABEL)
    kw.setdefault("fontweight", "normal")
    return ax.set_title(s, loc="left", **kw)


def style_2d(ax, grid: bool = False, nticks: int | None = None):
    """Spines, ticks and log formatting of a 2D axes (rcParams do most of it)."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(grid)
    if nticks:
        ax.xaxis.set_major_locator(MaxNLocator(nticks))
        if ax.get_yscale() != "log":
            ax.yaxis.set_major_locator(MaxNLocator(nticks))
    if ax.get_yscale() == "log":
        ax.yaxis.set_minor_formatter(NullFormatter())
    return ax


def loss_curve(values, stem, xlabel="Epoch", ylabel="Loss", log=None, title=None,
               height=1.55, color=BLUE, preview=True):
    """Single-column training curve -> ``stem``.pdf/.svg(/.png).

    ``log=None`` picks a log axis when the finite values span > 1.5 decades.
    """
    y = np.asarray(values, float)
    with style():
        fig, ax = plt.subplots(figsize=(COL_W, height))
        ax.plot(np.arange(len(y)), y, color=color, linewidth=1.0)
        fin = y[np.isfinite(y) & (y > 0)]
        if log is None:
            log = fin.size > 1 and fin.max() / fin.min() > 10 ** 1.5
        if log:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, max(len(y) - 1, 1))
        if title:
            ax.set_title(title)
        style_2d(ax)
        paths = save(fig, stem, width=COL_W, preview=preview)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------- 3D helpers

def style_3d(ax, view=(22, -58), proj="ortho", grid=True, nticks=4,
             labels=AXIS_LABELS_3D, labelpad=-3.0, tickpad=-2.5, manual_zorder=True):
    """White panes, faint grid, thin axis lines, short ticks, tight label pads.

    ``proj="ortho"`` removes perspective distortion, so equal lengths look
    equal; pass ``"persp"`` for a more pictorial view.  With ``manual_zorder``
    the draw order follows the ``Z_*`` bands (scene < rod < paths < markers)
    instead of matplotlib's per-artist depth sort, which is unreliable for a
    rod passing a cylinder; obstacles are then drawn translucent (see
    ``restyle_scene``) so nothing important disappears behind them.
    """
    if proj:
        ax.set_proj_type(proj)
    if view is not None:
        ax.view_init(*view)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        except AttributeError:                                   # very old matplotlib
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor(PANE_EDGE)
        axis.pane.set_linewidth(0.4)
        axis.line.set_linewidth(0.5)
        axis.line.set_color(BLACK)
        info = getattr(axis, "_axinfo", None)                    # private, but stable since 3.1
        if info is not None:
            info.setdefault("grid", {}).update(color=GRID_3D, linewidth=0.35, linestyle="-")
            info.setdefault("tick", {}).update(inward_factor=0.0, outward_factor=0.12)
            if isinstance(info["tick"].get("linewidth"), dict):
                info["tick"]["linewidth"] = {True: 0.5, False: 0.4}
            info.setdefault("axisline", {}).update(linewidth=0.5, color=BLACK)
        axis.set_major_locator(MaxNLocator(nticks))
    ax.grid(grid)
    ax.tick_params(axis="both", which="major", labelsize=FS_TICK, pad=tickpad)
    ax.zaxis.set_tick_params(labelsize=FS_TICK, pad=tickpad)
    if labels:
        ax.set_xlabel(labels[0], labelpad=labelpad)
        ax.set_ylabel(labels[1], labelpad=labelpad)
        ax.set_zlabel(labels[2], labelpad=labelpad)
    if manual_zorder and hasattr(ax, "computed_zorder"):
        ax.computed_zorder = False
    ax.patch.set_visible(False)
    ax._tro_nticks = nticks
    return ax


def fit_box(ax, arrays, pad=0.04, min_frac=0.15, zoom=1.0, limits=None):
    """Equal-scale limits fitted to ``arrays`` (point sets of the axes' dimension).

    3D: the box takes the proportions of the data (``set_box_aspect``), so a
    scene that is long in x and flat in y fills the panel instead of sitting
    in a corner of a cube; ``min_frac`` keeps a flat axis at least that
    fraction of the longest one (a planar path would otherwise give a
    zero-depth box), and short axes get fewer ticks.
    2D: limits with ``pad`` and ``aspect='equal'`` (the axes shrink to fit).
    ``pad`` is the margin around the data as a fraction of its largest side.
    ``limits=(lo, hi)`` sets the view exactly instead (still equal scale).
    Returns ``(lo, hi)`` of the new limits.
    """
    d = 3 if getattr(ax, "name", "") == "3d" else 2
    if limits is not None:
        lo, hi = (np.asarray(v, float).reshape(d) for v in limits)
        if np.any(hi <= lo):
            raise ValueError(f"view limits need lo < hi on every axis, got {lo} / {hi}")
    else:
        pts = [np.asarray(a, float).reshape(-1, d) for a in arrays
               if a is not None and np.size(a)]
        P = np.vstack(pts)
        P = P[np.isfinite(P).all(axis=1)]
        lo, hi = P.min(0), P.max(0)
        L = float((hi - lo).max()) or 1.0
        if d == 2:
            lo, hi = lo - pad * L, hi + pad * L
        else:
            half = 0.5 * np.maximum(hi - lo, min_frac * L) + pad * L
            ctr = 0.5 * (lo + hi)
            lo, hi = ctr - half, ctr + half
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    if d == 2:
        ax.set_aspect("equal", adjustable="box")
        return lo, hi
    ax.set_zlim(lo[2], hi[2])
    span = hi - lo
    try:
        ax.set_box_aspect(tuple(span), zoom=zoom)
    except TypeError:                                            # matplotlib < 3.6: no zoom
        ax.set_box_aspect(tuple(span))
    # a short (foreshortened) axis gets fewer ticks, so its labels do not collide
    nt = getattr(ax, "_tro_nticks", 4)
    for axis, frac in zip((ax.xaxis, ax.yaxis, ax.zaxis), span / span.max()):
        axis.set_major_locator(MaxNLocator(max(2, int(round(nt * frac))), min_n_ticks=2))
    return lo, hi


def draw_box(ax, lo, hi, clip=True, label=None, zorder=Z_WS, cut_alpha=0.4, **kw):
    """The axis-aligned box [lo, hi] (2D rectangle or 3D box) as dashed edges.

    3D with ``clip`` (call it after ``fit_box``): the box is intersected with
    the view.  Its own edges inside the view are drawn in the workspace
    style; where one of its faces runs out of the view, the line along which
    the view boundary cuts that face is drawn fainter (``cut_alpha``), so a
    partly visible face reads as a plane instead of a stray segment.  Faces
    outside the view are not drawn.  2D: matplotlib clips to the axes;
    ``clip`` only keeps the box out of the autoscaling.
    Returns a legend handle, or None if nothing is visible.
    """
    import itertools
    style_kw = dict(ROLE["workspace"])
    style_kw.update(kw)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    d = len(lo)
    is3d = getattr(ax, "name", "") == "3d"
    handle = Line2D([], [], label=label, **style_kw)
    if not (clip and is3d):
        extra = {} if is3d or not clip else dict(scalex=False, scaley=False)
        for k in range(d):
            others = [j for j in range(d) if j != k]
            for fixed in itertools.product(*[(lo[j], hi[j]) for j in others]):
                p0, p1 = np.zeros(d), np.zeros(d)
                for j, v in zip(others, fixed):
                    p0[j] = p1[j] = v
                p0[k], p1[k] = lo[k], hi[k]
                ax.plot(*np.stack([p0, p1]).T, zorder=zorder, **style_kw, **extra)
        return handle
    lim = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    ilo, ihi = np.maximum(lo, lim[:, 0]), np.minimum(hi, lim[:, 1])     # box ∩ view
    if np.any(ihi <= ilo):
        return None
    tol = 1e-9 * max(float(np.ptp(lim)), 1.0)
    face_lo = np.abs(ilo - lo) <= tol           # that side of box ∩ view is a box face
    face_hi = np.abs(ihi - hi) <= tol           # (False: it is the view boundary)
    drawn = False
    for k in range(3):                          # edges of box ∩ view parallel to axis k
        i, j = [m for m in range(3) if m != k]
        for a, fa in ((ilo[i], face_lo[i]), (ihi[i], face_hi[i])):
            for b, fb in ((ilo[j], face_lo[j]), (ihi[j], face_hi[j])):
                n_faces = int(fa) + int(fb)
                if n_faces == 0:                # only the view's own edge
                    continue
                p0, p1 = np.zeros(3), np.zeros(3)
                p0[i] = p1[i] = a
                p0[j] = p1[j] = b
                p0[k], p1[k] = ilo[k], ihi[k]
                kw_e = dict(style_kw)
                if n_faces == 1:                # a face cut by the view boundary
                    kw_e["alpha"] = cut_alpha
                ax.plot(*np.stack([p0, p1]).T, zorder=zorder, **kw_e)
                drawn = True
    return handle if drawn else None


def workspace_mode(mode, lo, hi, extent, ratio=1.6):
    """Resolve ``"auto"``: ``"fit"`` if the box [lo, hi] is at most ``ratio``
    times the size of the scene ``extent`` (largest side), else ``"off"`` --
    a workspace much larger than the scene would shrink it to a corner."""
    if mode != "auto" or lo is None:
        return mode
    d = len(lo)
    P = np.vstack([np.asarray(e, float).reshape(-1, d) for e in extent if np.size(e)])
    ws = float(np.max(np.asarray(hi, float) - np.asarray(lo, float)))
    return "fit" if ws <= ratio * float(np.ptp(P, axis=0).max()) else "off"


def data_extent(ax):
    """(lo, hi) of what has been drawn on ``ax`` so far (data limits), or (None, None).

    Call right after drawing a piece of scene geometry (e.g. ``pipe.draw``) to
    get its extent without knowing how it was drawn.
    """
    if getattr(ax, "name", "") == "3d":
        xy, zz = ax.xy_dataLim, ax.zz_dataLim
        lo = np.array([xy.x0, xy.y0, zz.x0])
        hi = np.array([xy.x1, xy.y1, zz.x1])
    else:
        dl = ax.dataLim
        lo, hi = np.array([dl.x0, dl.y0]), np.array([dl.x1, dl.y1])
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()) or np.any(hi < lo):
        return None, None
    return lo, hi


def snapshot(ax) -> set:
    """Ids of the artists already on ``ax`` (pass to ``restyle_scene``)."""
    return {id(a) for a in list(ax.collections) + list(ax.lines) + list(ax.patches)}


#: light direction for surfaces drawn without shading (upper left front)
_LIGHT = np.array([-0.45, -0.55, 0.70]) / np.linalg.norm([-0.45, -0.55, 0.70])


def _face_shading(coll):
    """Per-face intensity 0..1 of a Poly3DCollection from its facet normals
    (two-sided Lambert), or None if the 3D vertices are not accessible."""
    faces = getattr(coll, "_faces", None)                        # matplotlib >= 3.10
    if faces is None:
        vec, sl = getattr(coll, "_vec", None), getattr(coll, "_segslices", None)
        if vec is None or sl is None:
            return None
        faces = [np.asarray(vec)[:3, k].T for k in sl]
    t = []
    for f in faces:
        f = np.asarray(np.ma.getdata(f), float)
        n = np.cross(f, np.roll(f, -1, axis=0)).sum(axis=0)     # Newell's normal
        nn = np.linalg.norm(n)
        t.append(abs(n @ _LIGHT) / nn if nn > 1e-15 else 0.5)
    return np.asarray(t) if t else None


def _shaded(colors, base, lo=0.72, hi=1.12, t=None):
    """The tone of ``base`` with the light/dark pattern of ``colors`` (or the
    intensities ``t``, 0..1, one per face)."""
    c = np.asarray(colors, float).reshape(-1, 4)
    if t is None:
        lum = c[:, :3] @ np.array([0.299, 0.587, 0.114])
        span = np.ptp(lum)
        t = (lum - lum.min()) / span if span > 1e-6 else np.full(len(c), 0.6)
    t = np.asarray(t, float)
    rgb = np.clip(np.outer(lo + (hi - lo) * t, to_rgba(base)[:3]), 0.0, 1.0)
    return np.column_stack([rgb, np.ones(len(t))])


def restyle_scene(ax, before, color=OBSTACLE, alpha=0.55, line_color=GRAY,
                  line_width=0.5, zorder=Z_SCENE, rasterize=False):
    """Quiet, uniform look for what was drawn since ``before = snapshot(ax)``.

    Meant for geometry from project code (``utils.draw_obstacles``,
    ``env.pipe.draw``): surfaces get one muted tone (their own shading is
    kept), no facet edges (no seams in the PDF) and ``alpha``; lines get
    ``line_color``.  Their own legend labels are removed -- add one
    ``scene_handle`` instead.  Returns the number of restyled artists.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    hollow = isinstance(color, str) and color.lower() == "none"
    n = 0
    for c in list(ax.collections):
        if id(c) in before:
            continue
        n += 1
        c.set_zorder(zorder)
        c.set_label("_nolegend_")
        if isinstance(c, Poly3DCollection) and hollow:
            c.set_facecolor("none")
            c.set_edgecolor(line_color)
            c.set_linewidth(line_width)
        elif isinstance(c, Poly3DCollection):
            fc = getattr(c, "_facecolor3d", None)               # 3D (unsorted) face colours
            if fc is None or np.size(fc) == 0:
                fc = np.array([to_rgba(color)])
            fc = np.asarray(fc, float).reshape(-1, 4)
            lum = fc[:, :3] @ np.array([0.299, 0.587, 0.114])
            # drawn without shading (e.g. shade=False): shade from the facet normals
            t = _face_shading(c) if np.ptp(lum) <= 1e-6 else None
            new = _shaded(fc, color, t=t)
            c.set_facecolor(new)
            c.set_alpha(alpha)
            c.set_edgecolor((0, 0, 0, 0))
            c.set_linewidth(0.0)
            c.set_rasterized(rasterize)
        elif isinstance(c, Line3DCollection):
            c.set_color(line_color)
            c.set_linewidth(line_width)
        else:
            c.set_alpha(alpha)
    for a in list(ax.lines) + list(ax.patches):
        if id(a) in before:
            continue
        n += 1
        a.set_zorder(zorder)
        a.set_label("_nolegend_")
        if isinstance(a, Line2D):
            a.set_color(line_color)
            a.set_linewidth(line_width)
        else:
            a.set_facecolor("none" if hollow else to_rgba(color, alpha))
            a.set_edgecolor(line_color)
            a.set_linewidth(line_width)
    return n


def scene_handle(label="Obstacle", color=OBSTACLE, alpha=0.55, edgecolor="none", linewidth=0.0):
    """Legend patch matching ``restyle_scene``."""
    return Patch(facecolor=to_rgba(color, alpha), edgecolor=edgecolor, linewidth=linewidth,
                 label=label)


class _Group(tuple):
    """Several legend handles drawn side by side under one label."""

    def get_label(self):
        return self._label


def group_handle(handles, label):
    """One legend entry showing ``handles`` side by side (e.g. an x/y/z triad)."""
    g = _Group(handles)
    g._label = label
    return g


def triad_handle(colors, label="Gripper frame", linewidth=1.0):
    """Legend entry for a coordinate triad: one short segment per axis colour."""
    return group_handle([Line2D([], [], color=c, linewidth=linewidth) for c in colors], label)


def snapshot_colors(n, light="#ADADAD", dark="#6E6E6E"):
    """Gray ramp, light (early) -> dark (late), for intermediate rod configurations.

    Starts darker than ``ROLE["rod_init"]`` and ends lighter than the blue
    final rod in grayscale, so time reads light -> dark.
    """
    a, b = np.array(to_rgba(light)), np.array(to_rgba(dark))
    return [tuple(a + (b - a) * t) for t in np.linspace(0, 1, max(n, 1))]


def _row_major(handles, ncol):
    """Order handles so that a column-filled legend reads row by row."""
    rows = int(np.ceil(len(handles) / ncol))
    grid = [handles[r * ncol:(r + 1) * ncol] for r in range(rows)]
    return [g[c] for c in range(ncol) for g in grid if c < len(g)]


def _finite(bb):
    """A usable bbox: finite and not a point (empty legend-proxy lines report a
    zero-size box at the display origin, which would drag the crop there)."""
    return (bb is not None and np.isfinite(bb.bounds).all() and bb.width >= 0
            and bb.height >= 0 and (bb.width > 0 or bb.height > 0))


def content_bbox(ax, renderer, with_title=True):
    """Display bbox of what is actually drawn in ``ax``.

    For a 3D axes this is the box, its ticks/labels and the artists -- not the
    square axes rectangle matplotlib reserves around any 3D view.
    """
    from matplotlib.transforms import Bbox
    if getattr(ax, "name", "") != "3d":
        return ax.get_tightbbox(renderer)
    parts = []
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if axis.get_visible():
            parts += [axis.get_tightbbox(renderer), axis.pane.get_window_extent(renderer)]
    for a in list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.texts):
        if a.get_visible():
            try:
                parts.append(a.get_window_extent(renderer))
            except Exception:                                    # noqa: BLE001
                pass
    if with_title:
        for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
            if t is not None and t.get_text():
                parts.append(t.get_window_extent(renderer))
        if ax.get_legend() is not None:
            parts.append(ax.get_legend().get_window_extent(renderer))
    return Bbox.union([b for b in parts if _finite(b)])


def figure_bbox(fig, renderer):
    """Display bbox of all drawn content of ``fig`` (3D-aware ``tight``)."""
    from matplotlib.transforms import Bbox
    parts = [content_bbox(ax, renderer) for ax in fig.axes if ax.get_visible()]
    for a in list(fig.legends) + list(fig.texts):
        if a.get_visible():
            parts.append(a.get_window_extent(renderer))
    st = getattr(fig, "_suptitle", None)
    if st is not None and st.get_text():
        parts.append(st.get_window_extent(renderer))
    return Bbox.union([b for b in parts if _finite(b)])


def _replace(fig):
    for f in getattr(fig, "_tro_place", []):
        f()


def legend_top(fig, ax=None, handles=None, ncol=3, title=None, gap=3.0, row_major=True, **kw):
    """One frameless legend centred just above the drawn content of ``ax``.

    ``title`` (optional, 8 pt) goes above the legend -- keep it for drafts,
    the caption replaces it in the paper.  The position is recomputed whenever
    :func:`save` resizes the figure.
    """
    axes = [ax] if ax is not None else list(fig.axes)
    if handles is None:
        handles = []
        for a in axes:
            hs, _ = a.get_legend_handles_labels()
            handles += [h for h in hs if h not in handles]
    handles = [h for h in handles if h is not None]
    ncol = max(1, min(ncol, len(handles)))
    if row_major:
        handles = _row_major(handles, ncol)
    kw.setdefault("ncol", ncol)
    from matplotlib.legend_handler import HandlerTuple
    handler_map = {_Group: HandlerTuple(ndivide=None, pad=0.0)}
    handler_map.update(kw.pop("handler_map", {}) or {})
    labels = [h.get_label() for h in handles]
    leg = fig.legend(handles=handles, labels=labels, loc="lower center",
                     handler_map=handler_map, **kw)
    ttl = fig.text(0.5, 1.0, title, ha="center", va="bottom", fontsize=FS_LABEL) if title else None

    def place():
        from matplotlib.transforms import Bbox
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        bb = Bbox.union([content_bbox(a, r, with_title=False) for a in axes]).transformed(inv)
        g = gap / 72.0 / fig.get_size_inches()[1]
        leg.set_bbox_to_anchor((bb.x0, bb.y1 + g, bb.width, 0.0), transform=fig.transFigure)
        if ttl is not None:
            lh = leg.get_window_extent(r).transformed(inv).height
            ttl.set_position((bb.x0 + 0.5 * bb.width, bb.y1 + 2 * g + lh))

    fig._tro_place = getattr(fig, "_tro_place", []) + [place]
    place()
    return leg


# ---------------------------------------------------------------- export

def declutter_3d(fig, step=1.5, max_pad=40.0):
    """Push 3D axis labels off the tick labels they overlap.

    matplotlib offsets 3D labels in proportion to the box depth along the
    other axes while text keeps its size, so in a long, flat box (planar
    motion, a pipe scene) a label lands on its tick labels whatever the
    padding.  This measures instead: while an axis label overlaps any tick
    label it gets ``step`` pt more padding.  Returns the number of changes.
    """
    axes3d = [a for a in fig.axes if getattr(a, "name", "") == "3d"]
    n = 0
    for ax in axes3d:
        axes = (ax.xaxis, ax.yaxis, ax.zaxis)
        for axis in axes:
            if not axis.get_visible() or not axis.label.get_text():
                continue
            while axis.labelpad < max_pad:
                fig.canvas.draw()
                r = fig.canvas.get_renderer()
                lb = axis.label.get_window_extent(r)
                ticks = [t.get_window_extent(r) for a in axes for t in a.get_majorticklabels()
                         if t.get_visible() and t.get_text()]
                if not any(lb.overlaps(tb) for tb in ticks):
                    break
                axis.labelpad += step
                n += 1
    return n


def _crop_bbox(fig, pad):
    """Content bbox in inches, padded (what ``save`` writes)."""
    _replace(fig)
    fig.canvas.draw()
    bb = figure_bbox(fig, fig.canvas.get_renderer())
    return bb.transformed(fig.dpi_scale_trans.inverted()).padded(pad)


def _crop_width(fig, pad):
    bb = _crop_bbox(fig, pad)
    return bb.width, bb.height


def fit_width(fig, width, pad=None, tol=0.004, iters=6):
    """Rescale the canvas so that the *cropped* figure is exactly ``width`` in.

    ``bbox_inches='tight'`` removes the empty margins (large around 3D axes)
    but would make the file narrower than the column; LaTeX would then scale
    it up and the 7/8 pt text would grow.  Text keeps its size while the axes
    grow with the canvas, so a few secant steps converge.
    """
    pad = mpl.rcParams["savefig.pad_inches"] if pad is None else pad
    W, H = fig.get_size_inches()
    w, _ = _crop_width(fig, pad)
    prev = None
    for _ in range(iters):
        if abs(w - width) < tol:
            break
        if prev is None:
            s = width / w
        else:
            (W0, w0) = prev
            slope = (w - w0) / (W - W0) if abs(W - W0) > 1e-9 else 1.0
            s = (W + (width - w) / (slope if abs(slope) > 0.05 else 1.0)) / W
        prev = (W, w)
        W, H = W * s, H * s
        fig.set_size_inches(W, H)
        w, _ = _crop_width(fig, pad)
    return _crop_width(fig, pad)


def save(fig, stem, formats=("pdf", "svg"), preview=True, width=None, gray=False,
         tight=True, pad=None, verbose=True):
    """Write ``stem``.pdf / .svg (editable text) and a 300 dpi ``stem``.png preview.

    ``stem`` may carry an extension (``x.png`` -> ``x.pdf``, ``x.svg``, ``x.png``).
    ``width`` (e.g. ``COL_W``) makes the cropped output exactly that wide, see
    :func:`fit_width`.  ``gray=True`` also writes ``stem_gray.png`` to check
    grayscale readability.  Returns the written paths.
    """
    stem = Path(stem)
    if stem.suffix.lower() in (".png", ".pdf", ".svg", ".eps", ".jpg"):
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    pad = mpl.rcParams["savefig.pad_inches"] if pad is None else pad
    declutter_3d(fig)
    size = fit_width(fig, width, pad) if width else None
    if width and declutter_3d(fig):                 # resizing moved the labels: once more
        size = fit_width(fig, width, pad)
    kw = dict(bbox_inches=_crop_bbox(fig, pad) if tight else None)
    meta = {"pdf": {"Creator": None, "Producer": None, "CreationDate": None},
            "svg": {"Date": None, "Creator": None}}
    out = []
    for ext in formats:
        p = stem.parent / f"{stem.name}.{ext}"       # append: "run_v1.2" keeps its ".2"
        fig.savefig(p, metadata=meta.get(ext), **kw)
        out.append(p)
    if preview:
        p = stem.parent / f"{stem.name}.png"
        fig.savefig(p, dpi=300, **kw)
        out.append(p)
        if gray:
            from PIL import Image
            g = stem.parent / f"{stem.name}_gray.png"
            Image.open(p).convert("L").save(g)
            out.append(g)
    if verbose:
        w, h = size if size else _crop_width(fig, pad)
        print(f"[tro_style] {stem}.{{{','.join(formats)}}}  {w:.2f} x {h:.2f} in "
              f"({25.4 * w:.1f} x {25.4 * h:.1f} mm), font: {resolved_font()[0]}")
    return out


# ---------------------------------------------------------------- self-test

def _demo_scene():
    """A synthetic DLO tracking scene (no simulator needed)."""
    s = np.linspace(0, 1, 41)[:, None]
    A, B0, B1 = np.array([0.0, 0.0, 0.42]), np.array([0.62, 0.0, 0.42]), np.array([0.40, 0.04, 0.48])
    sag = lambda d: np.column_stack([0 * s, 0 * s, -d * np.sin(np.pi * s)])  # noqa: E731
    rod0 = A + s * (B0 - A) + sag(0.12)
    rod1 = A + s * (B1 - A) + sag(0.30) + np.column_stack([0 * s, 0.03 * np.sin(2 * np.pi * s), 0 * s])
    t = np.linspace(0, 1, 101)[:, None]
    m0, m1 = rod0[20], rod1[20] + np.array([0.0, 0.0, 0.0])
    ref = m0 + t * (m1 - m0) + 0.03 * np.sin(2 * np.pi * t) * np.array([0, 0, 1.0])
    rng = np.random.default_rng(0)
    mids = ref + np.cumsum(rng.normal(0, 4e-4, ref.shape), 0) * np.sin(np.pi * t)
    snaps = [m * rod1 + (1 - m) * rod0 + sag(0.06 * np.sin(np.pi * m)) for m in (0.25, 0.5, 0.75)]
    return rod0, rod1, ref, mids, snaps


def _demo_cylinder(ax, c=(0.40, -0.02, 0.32), r=0.02, h=0.12):
    """What a typical ``draw_obstacles`` does: a shaded steel-blue surface."""
    th, y = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(-h / 2, h / 2, 2))
    ax.plot_surface(c[0] + r * np.cos(th), c[1] + y, c[2] + r * np.sin(th),
                    color="steelblue", alpha=0.4, shade=True)
    return np.array([[c[0] - r, c[1] - h / 2, c[2] - r], [c[0] + r, c[1] + h / 2, c[2] + r]])


def _self_test(out_dir="tro_style_selftest"):
    out = Path(out_dir)
    rod0, rod1, ref, mids, snaps = _demo_scene()
    with style():
        print(f"serif font in use: {resolved_font()}")
        fig = plt.figure(figsize=(COL_W, 3.0))
        ax = fig.add_subplot(projection="3d")
        style_3d(ax, view=(20, -62))
        before = snapshot(ax)
        bounds = _demo_cylinder(ax)
        restyle_scene(ax, before)
        handles = [scene_handle("Obstacle")]
        for P, c in zip(snaps, snapshot_colors(len(snaps))):
            ax.plot(*P.T, color=c, linewidth=0.7, zorder=Z_ROD - 0.5)
        ax.plot(*rod0.T, **ROLE["rod_init"], zorder=Z_ROD, label="Initial rod")
        ax.plot(*rod1.T, **ROLE["rod"], zorder=Z_ROD, label="Final rod")
        ax.plot(*mids.T, **ROLE["achieved"], zorder=Z_PATH, label="Mid node")
        ax.plot(*ref.T, **ROLE["reference"], zorder=Z_PATH + 0.1, label="Reference")
        ax.plot(*ref[0], **ROLE["start"], zorder=Z_MARK, label="Start")
        ax.plot(*ref[-1], **ROLE["goal"], zorder=Z_MARK, label="End")
        fit_box(ax, [rod0, rod1, ref, mids, bounds])
        h_ws = draw_box(ax, (0.1, -0.3, 0.05), (0.9, 0.3, 0.75), label="Gripper workspace")
        hs, _ = ax.get_legend_handles_labels()
        legend_top(fig, ax, handles=hs + handles + [h_ws], ncol=3,
                   title="Mid-node tracking (synthetic)")
        paths = save(fig, out / "selftest_3d", width=COL_W, gray=True)
        plt.close(fig)
    paths += loss_curve(np.exp(-np.linspace(0, 6, 200)) + 1e-3, out / "selftest_loss",
                        ylabel="Tracking loss")
    return paths


if __name__ == "__main__":
    import sys
    for p in _self_test(sys.argv[1] if len(sys.argv) > 1 else "tro_style_selftest"):
        print("  wrote", p)
