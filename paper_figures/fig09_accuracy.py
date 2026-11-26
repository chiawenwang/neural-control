"""Fig. 9 (fig:accuracy) -- gradient accuracy and cost, double column (7.16 in).

(a) Directional derivatives of the loss along four random unit directions in
    xi, relative error against central finite differences: state-based
    gradient vs. frozen tangent, one column of points per scene. Filled
    triangles: the frozen tangent has the wrong sign.
(b) Per-rate relative difference of the frozen tangent from the state-based
    gradient along lambda, ||G_hat_j - G_j|| / ||G_j||, one small panel per
    scene, with the pegs (solid) and without them (dashed, S only); dotted:
    first peg contact of the initial rollout.
(c) Wall time of one gradient (after the rollout), state-based vs. frozen
    tangent; the number of adjoint solves is written (vertically) above each bar.

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

Style: ``src/tro_style.py`` (IEEE T-RO rules: Times 8/7 pt, Okabe-Ito, 0.5 pt
axes, editable text in PDF and SVG). Colours used here and to be kept in the
other result figures: state-based (ours) = blue ``#0072B2``, frozen tangent =
vermilion ``#D55E00``.

Run from anywhere:  python paper_figures/fig09_accuracy/fig09_accuracy.py [out_dir]
Writes fig09_accuracy.{pdf,svg,png} next to this script (or into out_dir).
LaTeX: \\includegraphics[width=\\textwidth]{figures/fig09_accuracy.pdf} in figure*.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# repository root = first ancestor holding runs/ and src/ (works whether this
# script sits in paper_figures/ or in paper_figures/fig09_accuracy/)
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

FIG_H = 1.95                   # in, before fit_width (which keeps the aspect)

# layout (inches). The horizontal positions are not fixed: tight_row() measures
# the labels of every panel and fills the 7.16 in so that the gap between the
# drawn content of neighbouring panels is GAP and the outer margin is MARGIN.
Y0, H_AX = 0.30, 1.30          # bottom and height of every axes
MARGIN = 0.02                  # in, left/right of the outermost text
GAP = 0.10                     # in, between the content of (a), (b) and (c)
GAP_B = 0.08                   # in, between the small panels of (b)
WEIGHT_A, WEIGHT_B_EACH, WEIGHT_C = 0.95, 0.80, 0.85   # share of the axes width


def load():
    out = []
    for name, rel in SCENES:
        f = REPO / rel
        if not f.exists():
            print(f"[fig09] {name}: {rel} not found -- skipped")
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
        print(f"[fig09] {name}: M={s['M']}, onset {s['onset']}, state rel err median "
              f"{np.median(s['e_state']):.2e}, proxy {np.median(s['e_proxy']):.3f}, "
              f"wrong sign {int(np.sum(s['fd'] * s['proxy'] < 0))}/{s['fd'].size}, "
              f"grad {s['t_state']:.2f} / {s['t_proxy']:.2f} ms")
    if not out:
        raise SystemExit("no check file found")
    return out


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
    ax.set_ylabel("Relative error vs. FD")
    ts.style_2d(ax)
    ts.panel_title(ax, "(a)", "Directional derivatives")


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
            ax.set_ylim(1e-3, 3e1)                        # headroom for the scene label
            ax.set_ylabel(r"$\|\hat{\mathbf{g}}_j-\mathbf{g}_j\|\,/\,\|\mathbf{g}_j\|$")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_xlabel(r"$\lambda$", labelpad=1.0)
        ts.style_2d(ax)
        ax.text(0.97, 0.97, s["name"], transform=ax.transAxes, ha="right", va="top",
                fontsize=ts.FS_TICK)
        axes.append(ax)
    ts.panel_title(axes[0], "(b)", "Per-rate difference of the frozen tangent")
    return axes


def panel_c(ax, S):
    x = np.arange(len(S))
    bw = 0.40
    y_lo, y_hi = 0.1, 250
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
    log_axis(ax, y_lo, 1e2)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(x, [s["name"] for s in S])
    ax.set_xlim(-0.55, len(S) - 0.45)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Time per gradient (ms)")
    ts.style_2d(ax)
    ts.panel_title(ax, "(c)", "Cost")


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


def legend_handles():
    state = ts.group_handle(
        [Line2D([], [], marker="o", ms=MS, mfc=C_STATE, mec=C_STATE, ls="none"),
         Patch(facecolor=C_STATE, edgecolor=C_STATE, linewidth=0.5)], "State-based (ours)")
    proxy = ts.group_handle(
        [Line2D([], [], marker="^", ms=MS + 0.4, mfc="white", mec=C_PROXY, mew=0.7, ls="none"),
         Line2D([], [], color=C_PROXY, lw=LW_DATA),
         Patch(facecolor="white", edgecolor=C_PROXY, linewidth=0.7, hatch="//////")],
        "Frozen tangent")
    wrong = Line2D([], [], marker="^", ms=MS + 0.4, mfc=C_PROXY, mec=C_PROXY, ls="none",
                   label="Wrong sign")
    nopeg = Line2D([], [], color=C_NOPEG, lw=LW_REF, ls=DASH_NOPEG, label="Without pegs")
    contact = Line2D([], [], color=ts.BLACK, lw=0.6, ls=DOT_CONTACT, label="First contact")
    return [state, proxy, wrong, nopeg, contact]


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
        ts.legend_top(fig, None, handles=legend_handles(), ncol=5, gap=2.0)
        ts.save(fig, out / "fig09_accuracy", width=ts.DBL_W, gray=True)
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
