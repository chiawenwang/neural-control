"""Fig. 2 of the T-RO manuscript: framework of the state-based method.

Double-column schematic (7.16 in wide, ``figure*``), symbols first::

              xi  <------------------- xi <- xi + tau d -------------------+
              |                                                            |
   u_{k-1} -> u_k --Gamma--> z_k --G(x,z_k)=0--> x_k --l--> l_k        [ L-BFGS ]
              | d/dxi                            | Hessian    | d/dx     [ Wolfe +  ]
   J_{k-1} -> J_k                              G_{x,k} > 0  |          [ snap rule]
              |                                  |          v              ^
 zeta_{k-1}-> zeta_k <-+w pi^T J-- pi_k <-B^T G_z^T-- eta_k <-G^-1-- a_k   |
              |                                                            |
              +------------------ zeta_M = dL/dxi -------------------------+

* forward (black, solid, white nodes) along the top;
* adjoint (vermilion, dashed, light fill) along the bottom, right to left,
  reusing the Cholesky factor of the stability check;
* outer loop (blue, thick): L-BFGS with the line search of Algorithm 1.

The figure is a schematic and reads no data.  Symbols follow
``paper-TRO/preamble/notation.tex``; the style follows the project rules
("IEEE T-RO figure style", ``src/tro_style.py``): drawn at its final size
(data units are inches), Times / stix, 8 pt symbols in the nodes and titles,
7 pt labels, Okabe-Ito colours plus line style (readable in grayscale),
editable text in PDF (TrueType) and SVG (``<text>``), ``gid`` on every element;
framed legend; no symbol or label on a line (``check_layout`` tests every text
against every line, frame edge and node, and fails the run otherwise).

Run from anywhere::

    python paper_figures/fig02_method_overview/fig02_method_overview.py

Writes ``fig02_method_overview.{pdf,svg,png}`` next to this file.  Include with
``\\includegraphics[width=\\textwidth]{...}`` in a ``figure*``; do not rescale.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tro_style as ts  # noqa: E402

STEM = HERE / "fig02_method_overview"

# ---------------------------------------------------------------- canvas (inches)
W, H = ts.DBL_W, 1.96

# ---------------------------------------------------------------- look
FS_NODE = ts.FS_LABEL           # 8 pt: symbols in the nodes, titles
FS_EDGE = ts.FS_TICK            # 7 pt: operators on the edges, legend
LW_NODE, LW_FRAME, LW_ARROW, LW_LOOP = 0.6, 0.5, 0.7, 1.1
DASH = (0, (3, 1.5))
ADJ_FILL = "#FBE5D6"
CARRY_FILL = "#F4F4F4"
NODE_W, NODE_H = 0.40, 0.23

KIND = {   # edge colour, line style, fill, line width
    "forward": dict(color=ts.BLACK, ls="-", fill="white", lw=LW_ARROW),
    "adjoint": dict(color=ts.VERMILION, ls=DASH, fill=ADJ_FILL, lw=LW_ARROW),
    "loop": dict(color=ts.BLUE, ls="-", fill=ts.SHADE, lw=LW_LOOP),
    "carry": dict(color=ts.GRAY, ls="-", fill=CARRY_FILL, lw=LW_ARROW),
}

# ---------------------------------------------------------------- symbols (notation.tex)
x, z, u, ud = r"\mathbf{x}", r"\mathbf{z}", r"\mathbf{u}", r"\dot{\mathbf{u}}"
J, zeta, xi = r"\mathbf{J}", r"\boldsymbol{\zeta}", r"\boldsymbol{\xi}"
eta, pi, a = r"\boldsymbol{\eta}", r"\boldsymbol{\pi}", r"\mathbf{a}"
Gam, B, d = r"\boldsymbol{\Gamma}", r"\mathbf{B}", r"\mathbf{d}"
Gx, Gz, G = r"\mathbf{G}_{x,k}", r"\mathbf{G}_{z,k}", r"\mathbf{G}"
T, L = r"^{\mathrm{T}}", r"\mathcal{L}"


def m(s):
    return f"${s}$"


# ---------------------------------------------------------------- layout
CARRY = (0.06, 0.54)                 # carry-in column x0, x1
FRAME = (0.68, 5.36)                 # step frame x0, x1
FRAME_Y = (0.15, 1.28)
LSBOX = (5.80, 7.04)                 # line search box x0, x1
COL = (1.16, 2.40, 3.70, 4.98)       # node columns
ROW = (1.075, 0.715, 0.355)          # top (forward), middle, bottom (adjoint)
Y_XI = 1.575                         # xi node (its edge label sits above the frame)
Y_TOP = 1.77                         # outer loop, top
Y_BOT = 0.05                         # outer loop, bottom

# ---------------------------------------------------------------- helpers
_TEXTS, _NODES, _SEGS = [], [], []   # texts, node boxes, straight line pieces


def text(ax, gid, x0, y0, s, size=FS_EDGE, ha="center", va="center", color=ts.BLACK, **kw):
    t = ax.text(x0, y0, s, ha=ha, va=va, fontsize=size, color=color, zorder=5, **kw)
    t.set_gid(gid)
    _TEXTS.append(t)
    return t


def node(ax, gid, cx, cy, s, kind="forward", w=NODE_W, h=NODE_H):
    k = KIND[kind]
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                       facecolor=k["fill"], edgecolor=k["color"], linewidth=LW_NODE,
                       linestyle=k["ls"], zorder=2)
    p.set_gid(gid)
    ax.add_patch(p)
    _NODES.append((gid, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    text(ax, f"{gid}_sym", cx, cy, s, size=FS_NODE)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def path(ax, gid, pts, kind, arrow=True):
    """Polyline through ``pts``; arrow head on the last segment."""
    k = KIND[kind]
    _SEGS.extend((gid, p0, p1) for p0, p1 in zip(pts[:-1], pts[1:]))
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ln = Line2D(xs, ys, color=k["color"], linewidth=k["lw"], linestyle=k["ls"],
                    solid_joinstyle="miter", zorder=3)
        ln.set_gid(f"{gid}_line")
        ax.add_line(ln)
    style = "-|>,head_length=3.2,head_width=1.6" if arrow else "-"
    fa = FancyArrowPatch(pts[-2], pts[-1], arrowstyle=style, mutation_scale=1.0,
                         color=k["color"], linewidth=k["lw"], linestyle=k["ls"],
                         shrinkA=0, shrinkB=0, zorder=3)
    fa.set_gid(gid)
    ax.add_patch(fa)


def hedge(ax, gid, b0, b1, y, kind, lab=None, below=None):
    """Horizontal edge from node box ``b0`` to node box ``b1`` at height ``y``."""
    g = 0.015
    if b1[0] > b0[2]:
        p0, p1 = (b0[2] + g, y), (b1[0] - g, y)
    else:
        p0, p1 = (b0[0] - g, y), (b1[2] + g, y)
    path(ax, gid, [p0, p1], kind)
    xm = 0.5 * (p0[0] + p1[0])
    if lab:
        text(ax, f"{gid}_lab", xm, y + 0.035, lab, va="bottom")
    if below:
        text(ax, f"{gid}_lab2", xm, y - 0.04, below, va="top", color=ts.GRAY)


def vedge(ax, gid, b0, b1, x0, kind, lab=None, side="right", lab_y=None):
    """Vertical edge from node box ``b0`` down to node box ``b1``; the label
    beside it at mid-height, or at ``lab_y`` (e.g. to keep it off a frame)."""
    g = 0.015
    p0, p1 = (x0, b0[1] - g), (x0, b1[3] + g)
    path(ax, gid, [p0, p1], kind)
    if lab:
        dx = 0.045 if side == "right" else -0.045
        text(ax, f"{gid}_lab", x0 + dx, 0.5 * (p0[1] + p1[1]) if lab_y is None else lab_y, lab,
             ha="left" if side == "right" else "right")


def draw():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------------- frames
    for gid, (x0, x1), fill, edge in (("step_frame", FRAME, "none", ts.LIGHT_GRAY),
                                      ("carry_in", CARRY, CARRY_FILL, ts.LIGHT_GRAY)):
        p = FancyBboxPatch((x0, FRAME_Y[0]), x1 - x0, FRAME_Y[1] - FRAME_Y[0],
                           boxstyle="round,pad=0,rounding_size=0.05", facecolor=fill,
                           edgecolor=edge, linewidth=LW_FRAME, zorder=1)
        p.set_gid(gid)
        ax.add_patch(p)
        (fy0, fy1) = FRAME_Y
        _SEGS.extend([(gid, (x0, fy0), (x1, fy0)), (gid, (x0, fy1), (x1, fy1)),
                      (gid, (x0, fy0), (x0, fy1)), (gid, (x1, fy0), (x1, fy1))])
    text(ax, "step_title", FRAME[1], FRAME_Y[1] + 0.035, r"homotopy step $k=1,\dots,M$",
         size=FS_NODE, ha="right", va="bottom")

    # ---------------- nodes
    c1, c2, c3, c4 = COL
    rt, rm, rb = ROW
    n_u = node(ax, "n_u", c1, rt, m(rf"{u}_k"))
    n_z = node(ax, "n_z", c2, rt, m(rf"{z}_k"))
    n_x = node(ax, "n_x", c3, rt, m(rf"{x}_k"))
    n_l = node(ax, "n_l", c4, rt, m(r"\ell_k"))
    n_J = node(ax, "n_J", c1, rm, m(rf"{J}_k"))
    n_G = node(ax, "n_G", c3, rm, m(rf"{Gx}\succ0"), w=0.62)
    n_zeta = node(ax, "n_zeta", c1, rb, m(rf"{zeta}_k"), kind="adjoint")
    n_pi = node(ax, "n_pi", c2, rb, m(rf"{pi}_k"), kind="adjoint")
    n_eta = node(ax, "n_eta", c3, rb, m(rf"{eta}_k"), kind="adjoint")
    n_a = node(ax, "n_a", c4, rb, m(rf"{a}_k"), kind="adjoint")
    n_xi = node(ax, "n_xi", c1, Y_XI, m(xi), kind="loop")

    # ---------------- forward (top row, left to right)
    hedge(ax, "e_uz", n_u, n_z, rt, "forward", m(Gam))
    hedge(ax, "e_zx", n_z, n_x, rt, "forward", m(rf"{G}({x},{z}_k)=\mathbf{{0}}"),
          below="Newton from " + m(rf"{x}_{{k-1}}"))
    hedge(ax, "e_xl", n_x, n_l, rt, "forward", m(r"\ell"))
    vedge(ax, "e_xi_u", n_xi, n_u, c1, "forward", m(rf"\Delta\lambda\,{ud}(\lambda_{{k-1}};{xi})"),
          lab_y=0.5 * (FRAME_Y[1] + n_xi[1]))     # between frame top and xi: off the frame
    vedge(ax, "e_uJ", n_u, n_J, c1, "forward", m(rf"\mathrm{{d}}/\mathrm{{d}}{xi}"))
    vedge(ax, "e_xG", n_x, n_G, c3, "forward", m(rf"\nabla^2_{{{x}{x}}}E"))

    # ---------------- adjoint (bottom row, right to left)
    vedge(ax, "e_la", n_l, n_a, c4, "adjoint", m(rf"w_k\neq0:\ \partial/\partial{x}_k"), side="left")
    hedge(ax, "e_aeta", n_a, n_eta, rb, "adjoint", m(rf"{Gx}^{{-1}}"))
    hedge(ax, "e_etapi", n_eta, n_pi, rb, "adjoint", m(rf"-{B}_k{T}{Gz}{T}"))
    hedge(ax, "e_pizeta", n_pi, n_zeta, rb, "adjoint", m(rf"+\,w_k{pi}_k{T}{J}_k"))
    vedge(ax, "e_Geta", n_G, n_eta, c3, "adjoint", "same factor")
    vedge(ax, "e_Jzeta", n_J, n_zeta, c1, "forward")

    # ---------------- carried from step k-1
    cx = 0.5 * sum(CARRY)
    for gid, row, s, b, kind in (("c_u", rt, rf"{u}_{{k-1}}", n_u, "carry"),
                                 ("c_J", rm, rf"{J}_{{k-1}}", n_J, "carry"),
                                 ("c_zeta", rb, rf"{zeta}_{{k-1}}", n_zeta, "carry")):
        text(ax, gid, cx, row, m(s), size=FS_NODE)
        path(ax, f"e_{gid}", [(CARRY[1] - 0.02, row), (b[0] - 0.015, row)], kind)

    # ---------------- outer loop
    ls = node(ax, "n_ls", 0.5 * sum(LSBOX), 0.5 * (ROW[0] + ROW[2]), "", kind="loop",
              w=LSBOX[1] - LSBOX[0], h=0.62)
    xls = 0.5 * sum(LSBOX)
    text(ax, "ls_1", xls, 0.5 * (ROW[0] + ROW[2]) + 0.12, "L-BFGS", size=FS_NODE)
    text(ax, "ls_2", xls, 0.5 * (ROW[0] + ROW[2]) - 0.02, "line search:")
    text(ax, "ls_3", xls, 0.5 * (ROW[0] + ROW[2]) - 0.15, "Wolfe + snap rule")
    # zeta_M: from the accumulator, under the step, into the line search
    path(ax, "e_zetaM", [(c1 - 0.08, n_zeta[1] - 0.015), (c1 - 0.08, Y_BOT), (xls, Y_BOT),
                         (xls, ls[1] - 0.015)], "loop")
    text(ax, "e_zetaM_lab", 0.5 * (FRAME[1] + xls) - 0.05, Y_BOT + 0.035,
         m(rf"{zeta}_M=\mathrm{{d}}{L}/\mathrm{{d}}{xi}"), va="bottom")
    # next rollout: back to xi over the step
    path(ax, "e_next", [(xls, ls[3] + 0.015), (xls, Y_TOP), (c1, Y_TOP),
                        (c1, n_xi[3] + 0.015)], "loop")
    text(ax, "e_next_lab", 0.5 * (c1 + xls), Y_TOP + 0.03,
         m(rf"{xi}\leftarrow{xi}+\tau{d}") + "  (next rollout)", va="bottom")

    # ---------------- legend (colour + line style + width)
    lx, ly, dy = 0.13, 1.80, 0.115
    fr = FancyBboxPatch((0.06, ly - 2 * dy - 0.085), 0.80, 2 * dy + 0.17, boxstyle="square,pad=0",
                        facecolor="white", edgecolor=ts.BLACK, linewidth=0.5, zorder=1)
    fr.set_gid("legend_frame")
    ax.add_patch(fr)
    for i, (k, s) in enumerate((("forward", "forward"), ("adjoint", "adjoint"),
                                ("loop", "outer loop"))):
        yy = ly - i * dy
        st = KIND[k]
        ln = Line2D([lx, lx + 0.20], [yy, yy], color=st["color"], linestyle=st["ls"],
                    linewidth=st["lw"] * 1.3)
        ln.set_gid(f"legend_{k}_line")
        ax.add_line(ln)
        text(ax, f"legend_{k}", lx + 0.26, yy, s, ha="left")
    return fig, ax


def check_layout(fig):
    """No text outside the canvas, no two texts overlapping, no edge label on a node."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.axes[0].transData.inverted()
    boxes = {t.get_gid(): t.get_window_extent(r).transformed(inv) for t in _TEXTS
             if t.get_text()}
    bad = []
    for g, bb in boxes.items():
        if bb.x0 < 0 or bb.x1 > W or bb.y0 < 0 or bb.y1 > H:
            bad.append(f"{g}: outside the canvas")
    for (g0, b0), (g1, b1) in itertools.combinations(boxes.items(), 2):
        if b0.overlaps(b1):
            bad.append(f"{g0} overlaps {g1}")
    e = 0.01                                    # in: clearance between a text and a line
    for g, bb in boxes.items():
        for gs, (xa, ya), (xb, yb) in _SEGS:
            if (min(xa, xb) < bb.x1 + e and max(xa, xb) > bb.x0 - e
                    and min(ya, yb) < bb.y1 + e and max(ya, yb) > bb.y0 - e):
                bad.append(f"{g} on line {gs}")
    for g, bb in boxes.items():
        for gn, (x0, y0, x1, y1) in _NODES:
            inside = g.startswith(gn + "_") or (gn == "n_ls" and g.startswith("ls_"))
            hit = bb.x1 > x0 and bb.x0 < x1 and bb.y1 > y0 and bb.y0 < y1
            if inside and not (bb.x0 >= x0 + 0.02 and bb.x1 <= x1 - 0.02):
                bad.append(f"{g} does not fit in {gn}")
            elif hit and not inside:
                bad.append(f"{g} touches node {gn}")
    return bad


def main():
    with ts.style():
        fig, _ = draw()
        bad = check_layout(fig)
        for b in bad:
            print("[layout]", b)
        ts.save(fig, STEM, formats=("pdf", "svg"), preview=False, tight=False)
        fig.savefig(STEM.with_suffix(".png"), dpi=600)
        plt.close(fig)
    if bad:
        raise SystemExit(f"{len(bad)} layout problem(s)")


if __name__ == "__main__":
    main()
