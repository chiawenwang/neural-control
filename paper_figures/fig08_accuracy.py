"""Fig. 8 (fig:accuracy) -- gradient accuracy and cost, double column (7.16 in).

(a) e_FD: relative error of the directional derivatives of the loss along
    four random unit directions in xi against central finite differences,
    state-based gradient vs. frozen tangent, one column of points per scene;
    filled triangles: the frozen tangent has the wrong sign.
(b1)-(b4) e_rate: per-rate relative difference ||g_hat_j - g_j|| / ||g_j|| of
    the frozen tangent from the state-based gradient along lambda, one panel
    per scene (S1, S2, P1, P2), with the pegs (solid) and without them
    (dashed, S only); dotted: first peg contact of the initial rollout.
(c) t_grad: wall time of one gradient (after the rollout), state-based vs.
    frozen tangent; numbers above the bars: adjoint solves.
Axis labels are symbols with the unit in parentheses (defined in Sec. VII-C of the paper);
legends are per panel, framed; reference lines are labelled in the plot.

Data: the gradient checks of the code repository, seed 0, at the common
initialization of each scene (written by ``scripts/train_shape.py check`` and
``scripts/train_path.py check``):

    S1 = ../runs/shape_p1/checks/check_P1_net_s0.json   (target P1, shape task)
    S2 = ../runs/shape_w1/checks/check_W1_net_s0.json   (target W1, shape task)
    P1 = ../runs/path_p1/checks/check_P1_net_s0.json    (path task)
    P2 = ../runs/path_p2/checks/check_P2_net_s0.json    (path task; added
         automatically as soon as the file exists)

Scenes whose check file is missing are skipped, so the figure grows from
three to four scenes without editing this script.

Style: ``src/tro_style.py`` + project rules "IEEE T-RO figure style": panel
letters 10 pt, labels 8 pt, ticks/legend/annotations 7 pt Times; symbols as
axis labels, no words or formulas; closed axes boxes, ticks inward on all four
sides; framed legend per panel; Okabe-Ito plus marker/line style (grayscale);
editable text in PDF and SVG; nothing outside the page. Colours used here and to be kept in the
other result figures: state-based (ours) = blue ``#0072B2``, frozen tangent =
vermilion ``#D55E00``.

Run from anywhere:  python paper_figures/fig08_accuracy/fig08_accuracy.py [out_dir]
Writes fig08_accuracy.{pdf,svg,png} next to this script (or into out_dir).
LaTeX: \\includegraphics[width=\\textwidth]{figures/fig08_accuracy.pdf} in figure*.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# repository root = first ancestor holding runs/ and src/ (works whether this
# script sits in paper_figures/ or in paper_figures/fig08_accuracy/)
REPO = next(p for p in HERE.parents if (p / "runs").is_dir() and (p / "src").is_dir())
sys.path.insert(0, str(REPO / "src"))

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402
from matplotlib.patches import Patch                       # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator  # noqa: E402

import tro_style as ts                                     # noqa: E402

# ------------------------------------------------------------------ inputs
#: (paper name, check file relative to the repository root)
SCENES = [
    ("S1", "runs/shape_p1/checks/check_P1_net_s0.json"),
    ("S2", "runs/shape_w1/checks/check_W1_net_s0.json"),
    ("P1", "runs/path_p1/checks/check_P1_net_s0.json"),
    ("P2", "runs/path_p2/checks/check_P2_net_s0.json"),
]

# ------------------------------------------------------------------ look
C_STATE, C_PROXY = ts.BLUE, ts.VERMILION
C_NOPEG = ts.GRAY
LW_DATA, LW_REF = 1.0, 0.8
DASH_NOPEG = (0, (3.0, 1.5))
DOT_CONTACT = (0, (1.0, 1.5))
MS = 3.6                       # marker size (pt)

FIG_H = 1.62                   # in, before fit_width (which keeps the aspect)

# layout (inches). The horizontal positions are not fixed: tight_row() measures
# the labels of every panel and fills the 7.16 in so that the gap between the
# drawn content of neighbouring panels is GAP and the outer margin is MARGIN.
Y0, H_AX = 0.30, 1.30          # bottom and height of every axes
MARGIN = 0.02                  # in, left/right of the outermost text
GAP = 0.10                     # in, between the content of (a), (b) and (c)
GAP_B = 0.13                   # in, between the small panels of (b)
WEIGHT_A, WEIGHT_B_EACH, WEIGHT_C = 0.95, 0.80, 0.85   # share of the axes width


def load():
    out = []
    for name, rel in SCENES:
        f = REPO / rel
        if not f.exists():
            print(f"[fig08] {name}: {rel} not found -- skipped")
            continue
        d = json.load(open(f))
        if "per_rate_pegs" in d:            # shape task (S): with and without pegs
            pr, npr = d["per_rate_pegs"], d.get("per_rate_no_pegs")
            onset = pr["onset"] if pr.get("onset") is not None else pr["split_at"]
        else:                               # path task (P): with pegs only
            pr, npr = d["per_rate"], None
            onset = pr["split_at"]
        rel_p = np.asarray(pr["rel"], float)
        out.append(dict(
            name=name, M=rel_p.size, onset=int(onset),
            fd=np.array([r["fd"] for r in d["dirs"]]),
            state=np.array([r["state"] for r in d["dirs"]]),
            proxy=np.array([r["proxy"] for r in d["dirs"]]),
            e_state=np.array([r["state_relerr"] for r in d["dirs"]]),
            e_proxy=np.array([r["proxy_relerr"] for r in d["dirs"]]),
            rel=rel_p, rel_nopeg=None if npr is None else np.asarray(npr["rel"], float),
            t_state=1e3 * d["state_grad_time"], t_proxy=1e3 * d["proxy_grad_time"],
            n_state=int(d["state_adjoint_solves"]), n_proxy=int(d["proxy_adjoint_solves"]),
        ))
        s = out[-1]
        print(f"[fig08] {name}: M={s['M']}, onset {s['onset']}, state rel err median "
              f"{np.median(s['e_state']):.2e}, proxy {np.median(s['e_proxy']):.3f}, "
              f"wrong sign {int(np.sum(s['fd'] * s['proxy'] < 0))}/{s['fd'].size}, "
              f"grad {s['t_state']:.2f} / {s['t_proxy']:.2f} ms")
    if not out:
        raise SystemExit("no check file found")
    return out


FS_LETTER = 10                 # pt, panel letters (a), (b), (c)
LEGEND_GAP = 3.0               # pt, between the x-axis labels and the legend


def panel_letter(ax, letter):
    """``(a)`` alone, 10 pt, regular, top left of the panel's content: its left
    edge follows the y-axis label (also after resizing), its baseline sits just
    above the axes. No panel title -- the caption says what the panel shows."""
    return ax.annotate(letter, xy=(0, 1), xycoords=(ax.yaxis.label, "axes fraction"),
                       xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
                       fontsize=FS_LETTER, fontweight="normal", annotation_clip=False)


def letter_at_left(fig, ax, letter):
    """Panel letter above an axes without a y label: left edge = leftmost text
    of the panel (tick labels) or the axes; re-placed when ts.save resizes."""
    t = fig.text(0, 0, letter, ha="left", va="bottom", fontsize=FS_LETTER)

    def place():
        fig.canvas.draw()
        inv = fig.transFigure.inverted()
        b = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(inv)
        g = 3.0 / 72.0 / fig.get_size_inches()[1]
        t.set_position((b.x0, ax.get_position().y1 + g))

    fig._tro_place = getattr(fig, "_tro_place", []) + [place]
    place()
    return t


def box(ax):
    """Closed axes box: all four spines, ticks inward on all four sides."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def leader_label(ax, text, xy_text, xy, color, ha="left", va="center"):
    """In-plot label with a thin leader line (0.5 pt, colour of the element,
    no arrowhead) from the edge of the text to the point xy on the element."""
    return ax.annotate(text, xy=xy, xytext=xy_text, fontsize=ts.FS_TICK, color=color,
                       ha=ha, va=va, linespacing=0.95, annotation_clip=False,
                       arrowprops=dict(arrowstyle="-", lw=0.5, color=color,
                                       shrinkA=1.0, shrinkB=0.0))


def framed_legend(ax, handles, **kw):
    """Legend of one panel inside its axes, in a thin black frame."""
    leg = ax.legend(handles=handles, frameon=True, fancybox=False, framealpha=1.0,
                    edgecolor=ts.BLACK, borderpad=0.35, handletextpad=0.4,
                    labelspacing=0.25, handlelength=1.4, borderaxespad=0.4,
                    fontsize=ts.FS_TICK, **kw)
    leg.get_frame().set_linewidth(0.5)
    return leg


def legend_bottom(fig, axes, handles, ncol):
    """One frameless legend centred below the drawn content of ``axes``; its
    position is recomputed whenever ts.save resizes the figure."""
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.transforms import Bbox
    leg = fig.legend(handles=handles, labels=[h.get_label() for h in handles],
                     loc="upper center", ncol=ncol,
                     handler_map={ts._Group: HandlerTuple(ndivide=None, pad=0.0)})

    def place():
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        bb = Bbox.union([a.get_tightbbox(r) for a in axes]).transformed(inv)
        g = LEGEND_GAP / 72.0 / fig.get_size_inches()[1]
        leg.set_bbox_to_anchor((bb.x0, bb.y0 - g, bb.width, 0.0), transform=fig.transFigure)

    fig._tro_place = getattr(fig, "_tro_place", []) + [place]
    place()
    return leg


def log_axis(ax, lo, hi, every=1):
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    dec = np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1, every)
    ax.yaxis.set_major_locator(FixedLocator(10.0 ** dec))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(), numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())


def panel_a(ax, S):
    x = np.arange(len(S))
    off = 0.17
    jit = np.array([-0.06, -0.02, 0.02, 0.06])
    for i, s in enumerate(S):
        n = s["e_state"].size
        j = jit[:n] if n <= jit.size else np.linspace(-0.06, 0.06, n)
        ax.plot(x[i] - off + j, s["e_state"], "o", ms=MS, mfc=C_STATE, mec=C_STATE,
                mew=0.5, ls="none", zorder=3)
        wrong = s["fd"] * s["proxy"] < 0
        for k in range(n):
            ax.plot(x[i] + off + j[k], s["e_proxy"][k], "^", ms=MS + 0.4, mec=C_PROXY, mew=0.7,
                    mfc=C_PROXY if wrong[k] else "white", ls="none", zorder=3)
        for xc, e, c in ((x[i] - off, s["e_state"], C_STATE), (x[i] + off, s["e_proxy"], C_PROXY)):
            ax.plot([xc - 0.11, xc + 0.11], [np.median(e)] * 2, color=c, lw=LW_DATA, zorder=2)
    log_axis(ax, 1e-12, 1e1, every=3)
    ax.set_xticks(x, [s["name"] for s in S])
    ax.set_xlim(-0.55, len(S) - 0.45)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(r"$e_{\mathrm{FD}}$")
    ts.style_2d(ax)
    box(ax)
    ax.tick_params(axis="x", which="both", length=0)
    framed_legend(ax, legend_a(), loc="center", bbox_to_anchor=(0.5, 0.5))
    panel_letter(ax, "(a)")


def panel_b(fig, S, h):
    axes = []
    for i, s in enumerate(S):
        ax = fig.add_axes([0.3 + 0.1 * i, Y0 / h, 0.08, H_AX / h],
                          sharey=axes[0] if axes else None)   # x set by tight_row
        M = s["M"]
        lam = np.arange(1, M + 1) / M                     # rate j is applied in step j
        lc = s["onset"] / M
        ax.axvline(lc, color=ts.BLACK, lw=0.6, ls=DOT_CONTACT, zorder=1)
        if s["rel_nopeg"] is not None:
            r = np.where(s["rel_nopeg"] > 0, s["rel_nopeg"], np.nan)
            ax.plot(lam, r, color=C_NOPEG, lw=LW_REF, ls=DASH_NOPEG, zorder=2)
        r = np.where(s["rel"] > 0, s["rel"], np.nan)      # last rate: exactly 0
        ax.plot(lam, r, color=C_PROXY, lw=LW_DATA, zorder=3)
        if i == 0:
            log_axis(ax, 1e-3, 1e1)
            ax.set_ylim(1e-3, 3e1)
            ax.set_ylabel(r"$e_{\mathrm{rate}}$")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_xlabel(r"$\lambda$", labelpad=1.0)
        ts.style_2d(ax)
        box(ax)
        # direct labels (once): with / without pegs in the S1 panel,
        # first contact in the P1 panel
        def at(curve, x):                             # value of a curve at lambda = x
            return float(np.interp(x, lam, np.nan_to_num(curve, nan=1e-3)))
        if s["name"] == "S1":
            leader_label(ax, "with pegs", (0.06, 9.0), (0.32, at(s["rel"], 0.32)), C_PROXY)
            leader_label(ax, "without\npegs", (0.06, 3.0e-3),
                         (0.30, at(s["rel_nopeg"], 0.30)), C_NOPEG)
        if s["name"] == "P1":
            leader_label(ax, "first\ncontact", (lc - 0.12, 4.0e-3), (lc, 1.5e-2), ts.BLACK,
                         ha="right")
        axes.append(ax)
    return axes


def panel_c(ax, S):
    x = np.arange(len(S))
    bw = 0.40
    y_lo, y_hi = 0.1, 2000
    ts_ = np.array([s["t_state"] for s in S])
    tp_ = np.array([s["t_proxy"] for s in S])
    # Bars start at the lower axis limit, not at 0: on a log axis a bar from 0
    # is drawn down to log(0) -- tens of thousands of points below the page,
    # clipped on screen but enlarging the artwork box in Illustrator/Inkscape.
    ax.bar(x - bw / 2, ts_ - y_lo, bw, bottom=y_lo, color=C_STATE, edgecolor=C_STATE,
           linewidth=0.5, zorder=2)
    ax.bar(x + bw / 2, tp_ - y_lo, bw, bottom=y_lo, color="white", edgecolor=C_PROXY,
           linewidth=0.7, hatch="//////", zorder=2)
    for i, s in enumerate(S):
        for xc, t, nn in ((x[i] - bw / 2, s["t_state"], s["n_state"]),
                          (x[i] + bw / 2, s["t_proxy"], s["n_proxy"])):
            ax.text(xc, t * 1.25, f"{nn}", ha="center", va="bottom", rotation=90,
                    fontsize=ts.FS_TICK - 1)
    log_axis(ax, y_lo, 1e3)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(x, [s["name"] for s in S])
    ax.set_xlim(-0.55, len(S) - 0.45)
    ax.set_ylabel(r"$t_{\mathrm{grad}}$ (ms)")
    ts.style_2d(ax)
    box(ax)
    ax.tick_params(axis="x", which="both", length=0)
    framed_legend(ax, legend_c(), loc="upper left")
    panel_letter(ax, "(c)")


def tight_row(fig, groups, weights, h, passes=3):
    """Place the axes of each group side by side across the full width.

    ``groups``: lists of axes; the axes of one group are equally wide and
    GAP_B apart. Each pass measures how far the labels of a group reach out
    of its axes (left: y label and tick labels; right: tick labels, titles),
    then sizes the axes so that the drawn content of neighbouring groups is
    exactly GAP apart and the row spans MARGIN..DBL_W-MARGIN.
    """
    W = ts.DBL_W
    for _ in range(passes):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        dpi = fig.dpi
        over = []
        for g in groups:
            bbs = [a.get_tightbbox(r) for a in g]
            left = min(b.x0 for b in bbs) / dpi
            right = max(b.x1 for b in bbs) / dpi
            a0 = g[0].get_position().x0 * W
            a1 = g[-1].get_position().x1 * W
            over.append((a0 - left, right - a1))
        n_inner = sum(len(g) - 1 for g in groups)
        free = (W - 2 * MARGIN - sum(lo + ro for lo, ro in over)
                - (len(groups) - 1) * GAP - n_inner * GAP_B)
        x = MARGIN
        for g, wt, (lo, ro) in zip(groups, weights, over):
            wg = free * wt / sum(weights)                 # axes width of the group
            wa = (wg - (len(g) - 1) * GAP_B) / len(g)     # width of one axes
            x += lo
            for k, a in enumerate(g):
                a.set_position([(x + k * (wa + GAP_B)) / W, Y0 / h, wa / W, H_AX / h])
            x += wg + ro + GAP
    return over


def legend_a():
    return [Line2D([], [], marker="o", ms=MS, mfc=C_STATE, mec=C_STATE, ls="none",
                   label="State-based"),
            Line2D([], [], marker="^", ms=MS + 0.4, mfc="white", mec=C_PROXY, mew=0.7,
                   ls="none", label="Frozen tangent"),
            Line2D([], [], marker="^", ms=MS + 0.4, mfc=C_PROXY, mec=C_PROXY, ls="none",
                   label="Wrong sign")]


def legend_c():
    return [Patch(facecolor=C_STATE, edgecolor=C_STATE, linewidth=0.5, label="State-based"),
            Patch(facecolor="white", edgecolor=C_PROXY, linewidth=0.7, hatch="//////",
                  label="Frozen tangent")]


def main(out_dir=None):
    S = load()
    out = Path(out_dir) if out_dir else HERE
    with ts.style():
        fig = plt.figure(figsize=(ts.DBL_W, FIG_H))
        h = FIG_H
        ax_a = fig.add_axes([0.05, Y0 / h, 0.2, H_AX / h])
        panel_a(ax_a, S)
        axes_b = panel_b(fig, S, h)
        ax_c = fig.add_axes([0.8, Y0 / h, 0.15, H_AX / h])
        panel_c(ax_c, S)
        tight_row(fig, [[ax_a], axes_b, [ax_c]],
                  [WEIGHT_A, WEIGHT_B_EACH * len(axes_b), WEIGHT_C], h)
        for i, ax in enumerate(axes_b):
            letter_at_left(fig, ax, f"(b{i + 1})")
        ts.save(fig, out / "fig08_accuracy", width=ts.DBL_W, gray=True)
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
