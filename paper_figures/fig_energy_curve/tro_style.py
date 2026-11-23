"""
tro_style.py -- unified matplotlib style for IEEE T-RO figures.

Usage
-----
    import sys; sys.path.insert(0, "source")
    import tro_style as ts
    ts.apply()                                   # once, before creating figures
    fig, axs = plt.subplots(2, 2, figsize=ts.figsize(ts.COL_W, 2.6))
    ...
    ts.save(fig, "fig7_contact_energy")          # -> .pdf + .svg (editable text)

See TRO_figure_style.md for the rules this module implements.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- page geometry
COL_W = 3.50          # in, IEEE single column (88.9 mm)
DBL_W = 7.16          # in, IEEE double column / text width (181.9 mm)
MAX_H = 8.50          # in, maximum figure height incl. caption space

# ---------------------------------------------------------------- typography
FONT_SERIF = ["Times New Roman", "Times", "Liberation Serif",
              "Nimbus Roman", "TeX Gyre Termes", "STIXGeneral", "DejaVu Serif"]
FS_LABEL = 8          # axis labels, panel titles
FS_TICK = 7           # tick labels
FS_LEGEND = 7         # legend
FS_ANNOT = 7          # in-plot annotations
FS_PANEL = 8          # (a), (b), ... labels

# ---------------------------------------------------------------- lines & colors
LW_DATA = 1.0         # data curves
LW_REF = 0.8          # reference / limit curves (dashed)
LW_AXES = 0.5         # spines, ticks, grid
# colorblind-safe palette (Okabe-Ito); first two match the paper's blue/orange
C = dict(blue="#0072B2", orange="#D55E00", green="#009E73", purple="#CC79A7",
         sky="#56B4E9", yellow="#E69F00", black="#000000", gray="#555555",
         shade="#DCE8F5")


def apply():
    """Set global rcParams for T-RO figures."""
    mpl.rcParams.update({
        # fonts
        "font.family": "serif",
        "font.serif": FONT_SERIF,
        "mathtext.fontset": "stix",            # Times-compatible math
        "mathtext.rm": "serif",
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_LABEL,
        "axes.titleweight": "normal",
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        # editable text in vector output
        "pdf.fonttype": 42,                    # embed TrueType -> text stays text
        "ps.fonttype": 42,
        "svg.fonttype": "none",                # <text> elements, not paths
        "text.usetex": False,                  # usetex turns text into paths
        # lines / axes
        "lines.linewidth": LW_DATA,
        "axes.linewidth": LW_AXES,
        "xtick.major.width": LW_AXES, "ytick.major.width": LW_AXES,
        "xtick.minor.width": LW_AXES * 0.8, "ytick.minor.width": LW_AXES * 0.8,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.2,
        # layout / export
        "figure.dpi": 150,
        "savefig.dpi": 600,                    # only affects embedded rasters
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
    })


def figsize(width=COL_W, height=None, aspect=0.75):
    """(w, h) in inches; height defaults to width*aspect."""
    return (width, height if height is not None else width * aspect)


def panel_title(ax, letter, text, **kw):
    """Left-aligned panel title '(a) text' above the axes."""
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=FS_PANEL, pad=3, **kw)


def save(fig, stem, formats=("pdf", "svg")):
    """Save vector outputs with editable text. Returns list of paths."""
    paths = []
    for ext in formats:
        p = f"{stem}.{ext}"
        fig.savefig(p, format=ext)
        paths.append(p)
    return paths
