"""
Fig. 2: Credit assignment of the state-based gradient (Sec. IV-C).

Cell (k, j): contribution of the rate applied at step j to the loss measured
at step k,  w_k * pi_k^T  for j <= k, zero for j > k   (Eqs. (22)-(25)).
  (a) Terminal objective   : only row M is filled; every column receives pi_M.
  (b) Trajectory objective : full lower triangle; column sums are the
                             costates g_j = Delta_lambda * sum_{k>=j} pi_k.

IEEE single column (3.5 in), Arial. Outputs: fig_credit_assignment.pdf / .svg
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

M = 6  # homotopy steps shown

# ---------------------------------------------------------------- style
_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Arial" if "Arial" in _avail else "Liberation Sans"  # metric-compatible fallback
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT, "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
    "font.size": 8,
    "pdf.fonttype": 42,        # embed TrueType (IEEE PDF eXpress)
    "svg.fonttype": "none",    # keep text editable in Illustrator
})

FS_TITLE, FS_AXIS, FS_IDX, FS_CELL, FS_SUM = 8, 7, 7, 6.5, 7

C_EMPTY_FC, C_EMPTY_EC = "#F4F5F7", "#D5D8DD"
C_TERM_FC, C_TERM_EC, C_TERM_TX = "#FBE3D6", "#D9774B", "#A2451C"   # terminal
C_TRAJ_FC, C_TRAJ_EC, C_TRAJ_TX = "#DCE9F6", "#4A7FB5", "#1F4E86"   # trajectory
C_TEXT, C_IDX = "#222222", "#555555"

CELL, GAP = 1.0, 0.16          # grid units
PITCH = CELL + GAP


def cell_xy(k, j):
    """Lower-left corner of cell (row k, column j); k = 1 at the top."""
    return (j - 1) * PITCH, (M - k) * PITCH


def draw_cell(ax, k, j, fc, ec, label=None, tc=C_TEXT, lw=0.6):
    x, y = cell_xy(k, j)
    ax.add_patch(FancyBboxPatch((x, y), CELL, CELL,
                                boxstyle="round,pad=0,rounding_size=0.14",
                                fc=fc, ec=ec, lw=lw))
    if label:
        ax.text(x + CELL / 2, y + CELL / 2, label, ha="center", va="center",
                fontsize=FS_CELL, color=tc)


def draw_grid(ax, objective):
    W = M * PITCH - GAP
    for k in range(1, M + 1):
        for j in range(1, M + 1):
            if objective == "terminal" and k == M:
                draw_cell(ax, k, j, C_TERM_FC, C_TERM_EC, rf"$\pi_{M}$", C_TERM_TX)
            elif objective == "trajectory" and j <= k:
                draw_cell(ax, k, j, C_TRAJ_FC, C_TRAJ_EC, rf"$\pi_{k}$", C_TRAJ_TX)
            else:
                draw_cell(ax, k, j, C_EMPTY_FC, C_EMPTY_EC)

    # indices
    for i in range(1, M + 1):
        x, y = cell_xy(i, 1)
        ax.text(-0.35, y + CELL / 2, f"{i}", ha="right", va="center",
                fontsize=FS_IDX, color=C_IDX)
        x, _ = cell_xy(1, i)
        ax.text(x + CELL / 2, W + 0.3, f"{i}", ha="center", va="bottom",
                fontsize=FS_IDX, color=C_IDX)

    # axis labels
    ax.text(W / 2, W + 1.15, r"rate applied at step $j$ $\rightarrow$",
            ha="center", va="bottom", fontsize=FS_AXIS, color=C_TEXT)
    ax.text(-1.25, W / 2, r"$\leftarrow$ loss measured at step $k$",
            ha="center", va="center", rotation=90, fontsize=FS_AXIS, color=C_TEXT)

    # column sums
    ax.plot([-0.05, W + 0.05], [-0.28, -0.28], color="#888888", lw=0.5)
    for j in range(1, M + 1):
        x, _ = cell_xy(M, j)
        if objective == "terminal":
            s, c = rf"$\pi_{M}$", C_TERM_TX
        else:
            s, c = rf"$g_{j}$", C_TRAJ_TX
        ax.text(x + CELL / 2, -0.55, s, ha="center", va="top",
                fontsize=FS_SUM, color=c, fontweight="bold")

    # highlight one column sum in (b): g_3 = Δλ (π3 + π4 + π5 + π6)
    if objective == "trajectory":
        jh = 3
        x, y_bot = cell_xy(M, jh)
        _, y_top = cell_xy(jh, jh)
        ax.add_patch(FancyBboxPatch((x - 0.1, y_bot - 0.1), CELL + 0.2,
                                    y_top - y_bot + CELL + 0.2,
                                    boxstyle="round,pad=0,rounding_size=0.2",
                                    fc="none", ec=C_TRAJ_TX, lw=0.9, ls=(0, (2.5, 1.5))))

    ax.set_xlim(-1.6, W + 0.15)
    ax.set_ylim(-1.55, W + 1.75)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.9, wspace=0.04)

    draw_grid(axes[0], "terminal")
    draw_grid(axes[1], "trajectory")
    axes[0].set_title("(a) Terminal objective", fontsize=FS_TITLE, pad=2)
    axes[1].set_title("(b) Trajectory objective", fontsize=FS_TITLE, pad=2)

    for ext in ("pdf", "svg"):
        fig.savefig(f"fig_credit_assignment.{ext}", bbox_inches="tight",
                    pad_inches=0.02)
    if FONT != "Arial":  # make the editable SVG text request Arial explicitly
        s = open("fig_credit_assignment.svg").read().replace(f"'{FONT}'", "'Arial'")
        open("fig_credit_assignment.svg", "w").write(s)
    fig.savefig("fig_credit_assignment_preview.png", dpi=400,
                bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
