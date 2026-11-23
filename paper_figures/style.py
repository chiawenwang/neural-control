"""One style for every paper figure: fonts, sizes, line widths and colors.

Everything is sized for the *final* canvas (IEEE single column, 3.5 in), so a
component is never scaled in Illustrator: scaling would change stroke widths
and font sizes.

    import paper_figures.style as style
    style.apply()
    color = style.lambda_color(0.4)
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

PT = 1.0 / 72.0                 # inches per point

# --- text -------------------------------------------------------------------
FONT_SERIF = ["Times New Roman", "Times", "STIXGeneral"]
FONT_SIZE = 7.0                 # tick labels, captions inside components
LABEL_SIZE = 8.0                # axis labels, panel labels
MIN_FONT_SIZE = 7.0             # QA threshold (checks.py)

# --- strokes [pt] -------------------------------------------------------------
LW_DATA = 1.0                   # curves, rods
LW_DATA_THIN = 0.8              # snapshots, secondary data
LW_DATA_THICK = 1.2             # final shape, realized path
LW_AUX = 0.5                    # outlines, axes, dashed guides
LW_ARROW = 0.6                  # process arrows (one style for all of them)

# --- colors -----------------------------------------------------------------
INK = "#1A1A1A"                 # text and axes
OBSTACLE_FILL = "#D9D9D9"
OBSTACLE_EDGE = "#4D4D4D"
CLAMP = "#404040"               # fixed clamp and gripper glyphs (dark gray)
ACCENT = "#D55E00"              # Okabe-Ito vermillion: contacting edges
BALL_RING = "#D7191C"            # red outline of the balls on the surface: stands out on viridis blues/greens
BALL_RING_LW = 0.9              # [pt]
AUX_GRAY = "#8C8C8C"            # dashed guides (local maxima)
PLACEHOLDER = "#BFBFBF"         # placeholder tile border and text

# --- lambda (homotopy progress) ------------------------------------------------
#: viridis truncated to [0.05, 0.92]: perceptually uniform, colorblind safe,
#: monotone in grayscale, and without the near-black and near-white ends that
#: vanish against the page or the gray obstacles.
LAMBDA_RANGE = (0.05, 0.92)
LAMBDA_CMAP = LinearSegmentedColormap.from_list(
    "lambda_viridis",
    mpl.colormaps["viridis"](np.linspace(*LAMBDA_RANGE, 256)),
    N=256,
)


def lambda_color(lam):
    """RGBA of homotopy progress ``lam`` in [0, 1] (tuple, or (n, 4) for arrays)."""
    lam = np.clip(np.asarray(lam, dtype=float), 0.0, 1.0)
    rgba = LAMBDA_CMAP(lam)
    return tuple(float(c) for c in rgba) if lam.ndim == 0 else rgba


def apply() -> None:
    """Install the shared rcParams (idempotent)."""
    if "lambda_viridis" not in mpl.colormaps:
        mpl.colormaps.register(LAMBDA_CMAP)
    mpl.rcParams.update({
        # fonts: Times-like serif, STIX math; TrueType embedded, SVG text kept as text
        "font.family": "serif",
        "font.serif": FONT_SERIF,
        "mathtext.fontset": "stix",
        "font.size": FONT_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": LABEL_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "paper_figures",      # stable SVG ids between runs
        # exact canvases, transparent backgrounds
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "savefig.dpi": 600,
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
        "path.simplify": False,               # keep every data vertex
        # strokes
        "lines.linewidth": LW_DATA,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "butt",
        "patch.linewidth": LW_AUX,
        "axes.linewidth": LW_AUX,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.labelpad": 1.5,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": LW_AUX,
        "ytick.major.width": LW_AUX,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,
        "legend.frameon": False,
        # no default matplotlib blue anywhere, no rainbow maps
        "axes.prop_cycle": cycler(color=[INK]),
        "image.cmap": "lambda_viridis",
    })
