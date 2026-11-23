"""
Fig. 2: Credit assignment of the state-based gradient (Sec. IV-C).

Cell (k, j): contribution w_k * pi_k^T of the loss measured at step k to the
rate applied at step j <= k (zero for j > k), Eqs. (22)-(25).
  (a) Terminal objective   : only row M is nonzero; every column receives pi_M.
  (b) Trajectory objective : full lower triangle; column sums are the
                             costates g_j = Delta_lambda * sum_{k>=j} w_k pi_k.

Style: project rules in TRO_figure_style.md (uses source/tro_style.py when it
is importable, otherwise an equivalent local fallback).
Output (next to this file): fig2_credit_assignment.pdf / .svg
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ----------------------------------------------------------------- style
try:
    import tro_style as ts
    ts.apply()
    COL_W = ts.COL_W
    _save = ts.save
except ImportError:
    COL_W = 3.5
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif",
                       "Nimbus Roman", "TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.5, "lines.linewidth": 1.0,
        "pdf.fonttype": 42, "svg.fonttype": "none", "text.usetex": False,
    })

    def _save(fig, stem):
        for ext in ("pdf", "svg"):
            fig.savefig(f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)
        # SVG text must request "Times New Roman" even if a metric-compatible
        # fallback was used for layout on this machine
        svg = Path(f"{stem}.svg")
        txt = svg.read_text()
        for alt in ("Liberation Serif", "Nimbus Roman", "TeX Gyre Termes", "Times"):
            txt = txt.replace(f"'{alt}'", "'Times New Roman'")
        svg.write_text(txt)

# Okabe-Ito palette (TRO_figure_style.md, Sec. 3)
BLUE, VERMILION, GRAY = "#0072B2", "#D55E00", "#555555"
EMPTY_EC = "#C8C8C8"

FS_TITLE, FS_LABEL, FS_ANNOT = 8, 8, 7   # titles / axis labels / everything else

# ---------------------------------------------------------------- geometry
M = 6
CELL, GAP = 1.0, 0.18
PITCH = CELL + GAP
W = M * PITCH - GAP


def cell_xy(k, j):
    """Lower-left corner of cell (row k, column j); row 1 at the top."""
    return (j - 1) * PITCH, (M - k) * PITCH


def draw_cell(ax, k, j, active, ec=None, label=None):
    """Active cells: white fill, solid colored outline, black symbol.
    Empty cells: no fill, thin light-gray outline. The symbol is never
    printed on a tinted background, and active/empty differ by line weight,
    not by color alone (grayscale-safe)."""
    x, y = cell_xy(k, j)
    if active:
        ax.add_patch(FancyBboxPatch((x, y), CELL, CELL,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    fc="white", ec=ec, lw=1.0))
        ax.text(x + CELL / 2, y + CELL / 2, label, ha="center", va="center",
                fontsize=FS_ANNOT, color="black")
    else:
        ax.add_patch(FancyBboxPatch((x, y), CELL, CELL,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    fc="none", ec=EMPTY_EC, lw=0.5))


def draw_grid(ax, objective):
    color = VERMILION if objective == "terminal" else BLUE
    for k in range(1, M + 1):
        for j in range(1, M + 1):
            if objective == "terminal":
                active = (k == M)
            else:
                active = (j <= k)
            draw_cell(ax, k, j, active, color, rf"$\pi_{k}$")

    # step indices
    for i in range(1, M + 1):
        _, y = cell_xy(i, 1)
        ax.text(-0.3, y + CELL / 2, f"{i}", ha="right", va="center",
                fontsize=FS_ANNOT, color=GRAY)
        x, _ = cell_xy(1, i)
        ax.text(x + CELL / 2, W + 0.25, f"{i}", ha="center", va="bottom",
                fontsize=FS_ANNOT, color=GRAY)

    # axis labels
    ax.text(W / 2, W + 1.2, r"rate applied at step $j$ $\rightarrow$",
            ha="center", va="bottom", fontsize=FS_LABEL)
    ax.text(-1.3, W / 2, r"$\leftarrow$ loss measured at step $k$",
            ha="center", va="center", rotation=90, fontsize=FS_LABEL)

    # column sums under a rule
    ax.plot([0.0, W], [-0.3, -0.3], color=GRAY, lw=0.5)
    for j in range(1, M + 1):
        x, _ = cell_xy(M, j)
        s = rf"$\pi_{M}$" if objective == "terminal" else rf"$g_{j}$"
        ax.text(x + CELL / 2, -0.55, s, ha="center", va="top", fontsize=FS_LABEL)

    # (b): mark one column sum, g_3 = Delta_lambda (w_3 pi_3 + ... + w_6 pi_6)
    if objective == "trajectory":
        jh = 3
        x, y_bot = cell_xy(M, jh)
        _, y_top = cell_xy(jh, jh)
        ax.add_patch(FancyBboxPatch((x - 0.11, y_bot - 0.11), CELL + 0.22,
                                    y_top - y_bot + CELL + 0.22,
                                    boxstyle="round,pad=0,rounding_size=0.18",
                                    fc="none", ec=GRAY, lw=0.8, ls=(0, (3, 2))))

    ax.set_xlim(-1.65, W + 0.1)
    ax.set_ylim(-1.6, W + 2.15)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.1))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.9, wspace=0.05)

    draw_grid(axes[0], "terminal")
    draw_grid(axes[1], "trajectory")
    axes[0].set_title("(a) Terminal objective", loc="left", fontsize=FS_TITLE, pad=1)
    axes[1].set_title("(b) Trajectory objective", loc="left", fontsize=FS_TITLE, pad=1)

    stem = Path(__file__).resolve().parent / "fig2_credit_assignment"
    _save(fig, str(stem))


if __name__ == "__main__":
    main()
