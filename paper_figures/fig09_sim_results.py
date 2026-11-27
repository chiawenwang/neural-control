"""Fig. 9 (fig:sim-results) -- contact-rich tasks in simulation, double column.

One panel per scene: (a) S1, (b) S2, (c) P1, (d) P2 (a scene is skipped if its
runs are missing). The scenes themselves are shown in Fig. 7 (fig:simtask),
so this figure shows only the task errors.

Each method is one curve, the geometric mean over the 5 seeds (the errors span
decades, so the mean is taken in log space), with a band from the smallest to
the largest seed (lighter for the Adam runs).
  S (terminal): e_task (mm) of the current iterate versus n_roll, the number of
     rollouts (iter_rms at the rollout of each accepted iterate, ending with the
     final rollout of the run, final_rms of its JSON); a run that has stopped
     keeps its final value up to the budget (500), so every curve ends at the
     final errors of Table results. Adam is non-monotone, so its curves rise
     where the runs drift away from the target. Methods: state-based gradient
     (ours) and frozen tangent, each with L-BFGS and with Adam. The threshold
     (5 mm) is labelled in (a), with a leader line.
  P (trajectory): e_mid (mm), the midpoint error of the final rollout, along
     lambda. Methods: initial trajectory, state-based and frozen tangent with
     L-BFGS and with Adam, end pose (60 steps), greedy (deterministic, one run,
     no band). Gray, labelled "contact" (with a leader line): steps in which the rod of the
     reference motion touches a peg.
Grayscale: the gradient is told by the marker shape (state-based: circle,
frozen tangent: triangle), the optimizer by the marker fill and the line width
(L-BFGS: filled marker, 1.0 pt; Adam: open marker, 0.7 pt; both solid);
greedy: diamonds; end pose: dash-dot; initial: thin gray. One framed legend per task (below (a)-(b) and below (c)-(d)).

Style: project rules "IEEE T-RO figure style" (src/tro_style.py): panel letters
(a), (b), ... 10 pt, axis labels 8 pt (symbols, unit in parentheses), ticks/legend
7 pt Times; no panel titles; closed axes boxes; methods: ours blue #0072B2, frozen tangent
vermilion #D55E00 (L-BFGS 1.0 pt with band, Adam 0.7 pt with a lighter band, both solid; legend: line + marker only, no band swatch), greedy
purple #CC79A7, end pose orange #E69F00 (dash-dot), initial gray; bands in the
colour of their curve; contact steps light gray #EDEDED; editable text in PDF
and SVG; nothing outside the page.

Run:   python paper_figures/fig09_sim_results/fig09_sim_results.py [out_dir]
Writes fig09_sim_results.{pdf,svg,png} (+ _gray.png) next to this script.
LaTeX: \\includegraphics[width=\\textwidth]{figures/fig09_sim_results.pdf} in figure*.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# repository root = first ancestor holding runs/ and src/
REPO = next(p for p in HERE.parents if (p / "runs").is_dir() and (p / "src").is_dir())
sys.path.insert(0, str(REPO / "src"))

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter  # noqa: E402

import tro_style as ts                                     # noqa: E402

# ------------------------------------------------------------------ inputs
#: one panel per scene; skipped if its runs are missing
SCENES = [
    dict(name="S1", task="shape", run="runs/shape_p1", tag="P1"),
    dict(name="S2", task="shape", run="runs/shape_w1", tag="W1"),
    dict(name="P1", task="path", run="runs/path_p1", tag="P1"),
    dict(name="P2", task="path", run="runs/path_p2", tag="P2"),
]
SEEDS = range(5)
BUDGET = 500                  # rollouts
THRESHOLD_MM = 5.0            # S

# ------------------------------------------------------------------ look
C_OURS, C_PROXY = ts.BLUE, ts.VERMILION
C_GREEDY, C_ENDPOSE, C_INIT = ts.PURPLE, ts.ORANGE, "#9A9A9A"
DASH_ADAM = "-"               # Adam: solid like L-BFGS; told apart by open markers and LW_ADAM
DASH_END = (0, (4.0, 1.2, 1.0, 1.2))   # dash-dot: end pose
MS = 3.0                      # marker size (pt)
MARK = {"state": "o", "proxy": "^", "greedy": "D", "endpose": "s"}
N_MARK = 8                    # markers per curve
DOT_THR = (0, (1.0, 1.5))
LW = 1.0                      # L-BFGS curves (and all others)
LW_ADAM = 0.7                 # Adam curves: thinner, so that the two curves of one colour separate
BAND_ALPHA = 0.18
BAND_ALPHA_ADAM = 0.10        # Adam bands: lighter, so that four bands stay readable
C_CONTACT = "#EDEDED"         # steps in contact (not the blue SHADE: ours is blue)
FS_LETTER = 10

FIG_H = 2.15                  # in, before fit_width (which keeps the aspect)
Y0, H_AX = 0.62, 1.30         # bottom and height of every axes (in); legends below
MARGIN = 0.02                 # in, left/right of the outermost text
GAP = 0.12                    # in, between the drawn content of neighbouring panels
LEGEND_GAP = 3.0              # pt, x labels -> legend

YLIM_S, XLIM_S = (0.03, 300.0), (1.0, BUDGET)
YLIM_P = (1e-3, 200.0)


def _npz(path):
    return np.load(path) if Path(path).exists() else None


def iterate_curve(z, j):
    """Task error (mm) of the current iterate after r = 1..BUDGET rollouts.

    The accepted iterates are ``iter_rms`` at the rollouts ``eval_rollouts[iter_eval]``
    (for L-BFGS one per accepted line-search step, for Adam one per update, so
    the non-monotone course of Adam is kept). The curve ends with the final
    rollout of the run (``final_rms`` of the run JSON at its ``rollouts``): when a
    run stops because a line-search trial reaches the stopping tolerance, that
    trial is not an iterate. A run that has stopped keeps its final value up to
    the budget, so the end of every curve is the final error of Table results.
    """
    ro = np.r_[z["eval_rollouts"][z["iter_eval"]].astype(float), float(j["rollouts"])]
    e = np.r_[z["iter_rms"], j["final_rms"]] * 1e3
    r = np.arange(1, BUDGET + 1)
    idx = np.searchsorted(ro, r, side="right") - 1          # last iterate with ro <= r
    return np.where(idx >= 0, e[np.clip(idx, 0, None)], np.nan)


def load():
    out = []
    for sc in SCENES:
        run = REPO / sc["run"]
        if not (run / f"{sc['tag']}_net_state_lbfgs_s0.npz").exists():
            print(f"[fig09] {sc['name']}: no runs in {sc['run']} -- skipped")
            continue
        d = dict(sc)
        if sc["task"] == "shape":
            for key in ("state_lbfgs", "state_adam", "proxy_lbfgs", "proxy_adam"):
                curves = []
                for s in SEEDS:
                    z = _npz(run / f"{sc['tag']}_net_{key}_s{s}.npz")
                    if z is not None:
                        j = json.load(open(run / f"{sc['tag']}_net_{key}_s{s}.json"))
                        curves.append(iterate_curve(z, j))
                d[key] = np.array(curves)
        else:
            for key, f in (("state_lbfgs", "net_state_lbfgs"), ("state_adam", "net_state_adam"),
                           ("proxy_lbfgs", "net_proxy_lbfgs"), ("proxy_adam", "net_proxy_adam"),
                           ("endpose", "endpose60_state_lbfgs")):
                runs = [_npz(run / f"{sc['tag']}_{f}_s{s}.npz") for s in SEEDS]
                d[key] = np.array([z["final_err"] * 1e3 for z in runs if z is not None])
                if key == "state_lbfgs":
                    d["init"] = np.array([z["init_err"] * 1e3 for z in runs if z is not None])
            z = _npz(run / f"{sc['tag']}_greedy_state_greedy_s0.npz")
            d["greedy"] = None if z is None else z["final_err"] * 1e3
            tg = _npz(run / "targets" / f"{sc['tag']}.npz")
            d["contact"] = (None if tg is None else
                            np.asarray(tg["touching"]).reshape(len(d["init"][0]), -1).any(axis=1))
        out.append(d)
        print(f"[fig09] {sc['name']}: " + ", ".join(
            f"{k} {len(v)}" for k, v in d.items()
            if isinstance(v, np.ndarray) and v.ndim == 2))
    if not out:
        raise SystemExit("no scene found")
    return out


# ------------------------------------------------------------------ drawing
def band_style(color, adam=False):
    """Min-max band: translucent fill in the colour of its curve (lighter for Adam)."""
    return dict(facecolor=color, edgecolor="none", lw=0,
                alpha=BAND_ALPHA_ADAM if adam else BAND_ALPHA)


def mark_idx(x, k, logx):
    """Indices of N_MARK marker positions along x, shifted by k/4 of a spacing
    so that the markers of different curves do not sit on top of each other."""
    u = np.log(x) if logx else np.asarray(x, float)
    lo, hi = u[0], u[-1]
    step = (hi - lo) / N_MARK
    t = lo + step * (np.arange(N_MARK) + 0.25 + 0.17 * k)
    return np.unique(np.clip(np.searchsorted(u, t), 0, len(u) - 1))


def mean_band(ax, x, Y, lo_clip, color, ls="-", z=3, band=True, marker=None,
              open_marker=False, k=0, logx=False, adam=False):
    """Geometric mean over the rows of Y (one row per seed), a min-max band,
    and N_MARK markers on the mean (shape = gradient, open = Adam)."""
    Y = np.maximum(np.asarray(Y, float), lo_clip)            # nothing below the axis floor
    ok = np.all(np.isfinite(Y), axis=0)
    gm = np.exp(np.mean(np.log(Y[:, ok]), axis=0))
    if band:
        ax.fill_between(x[ok], Y[:, ok].min(0), Y[:, ok].max(0), zorder=z - 0.5,
                        **band_style(color, adam))
    ax.plot(x[ok], gm, color=color, ls=ls, lw=LW_ADAM if adam else LW, zorder=z)
    if marker:
        i = mark_idx(x[ok], k, logx)
        ax.plot(x[ok][i], gm[i], ls="none", marker=marker, ms=MS, mec=color, mew=0.7,
                mfc="white" if open_marker else color, zorder=z + 0.1)


def leader_label(ax, text, xy_text, xy, color, ha="left", va="center"):
    """In-plot label with a thin leader line (0.5 pt, colour of the element,
    no arrowhead) from the edge of the text to the point xy on the element."""
    return ax.annotate(text, xy=xy, xytext=xy_text, fontsize=ts.FS_TICK, color=color,
                       ha=ha, va=va, linespacing=0.95, annotation_clip=False,
                       arrowprops=dict(arrowstyle="-", lw=0.5, color=color,
                                       shrinkA=1.0, shrinkB=0.0))


def box(ax):
    """Closed axes box: all four spines, ticks inward on all four sides."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def log_y(ax, lo, hi):
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    dec = np.arange(np.ceil(np.log10(lo)), np.floor(np.log10(hi)) + 1)
    ax.yaxis.set_major_locator(FixedLocator(10.0 ** dec))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(), numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())


def panel_letters(fig, axes):
    """(a), (b), ... alone, 10 pt, just above each axes, left-aligned with the
    leftmost text of the panel (y label or tick labels); re-placed when
    ts.save resizes the figure."""
    texts = [fig.text(0, 0, f"({chr(97 + i)})", ha="left", va="bottom",
                      fontsize=FS_LETTER, fontweight="normal") for i in range(len(axes))]

    def place():
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        g = 3.0 / 72.0 / fig.get_size_inches()[1]
        for t, a in zip(texts, axes):
            b = a.get_tightbbox(r).transformed(inv)
            t.set_position((b.x0, a.get_position().y1 + g))

    fig._tro_place = getattr(fig, "_tro_place", []) + [place]
    place()
    return texts


def panel_shape(ax, d, ylabel, label_ref=False):
    x = np.arange(1, BUDGET + 1, dtype=float)
    lo = YLIM_S[0]
    kw = dict(logx=True)
    mean_band(ax, x, d["proxy_adam"], lo, C_PROXY, DASH_ADAM, z=2, adam=True,
              marker=MARK["proxy"], open_marker=True, k=3, **kw)
    mean_band(ax, x, d["proxy_lbfgs"], lo, C_PROXY, "-", z=3, marker=MARK["proxy"], k=2, **kw)
    mean_band(ax, x, d["state_adam"], lo, C_OURS, DASH_ADAM, z=4, adam=True,
              marker=MARK["state"], open_marker=True, k=1, **kw)
    mean_band(ax, x, d["state_lbfgs"], lo, C_OURS, "-", z=5, marker=MARK["state"], k=0, **kw)
    ax.axhline(THRESHOLD_MM, color=ts.GRAY, lw=0.8, ls=DOT_THR, zorder=1)
    if label_ref:                                   # reference line, labelled with a leader
        leader_label(ax, "threshold", (1.15, 0.45), (2.4, THRESHOLD_MM), ts.GRAY)
    ax.set_xscale("log")
    ax.set_xlim(*XLIM_S)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100]))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())
    log_y(ax, *YLIM_S)
    ax.set_xlabel(r"$n_{\mathrm{roll}}$", labelpad=1.0)
    ax.set_ylabel(r"$e_{\mathrm{task}}$ (mm)" if ylabel else "")
    ts.style_2d(ax)
    box(ax)
    ax.yaxis.set_minor_formatter(NullFormatter())


def panel_path(ax, d, label_ref=False):
    M = d["state_lbfgs"].shape[1]
    lam = np.arange(1, M + 1) / M
    if d["contact"] is not None:
        c = np.r_[False, d["contact"], False].astype(int)
        for a, b in zip(np.where(np.diff(c) == 1)[0], np.where(np.diff(c) == -1)[0]):
            ax.axvspan(a / M, b / M, color=C_CONTACT, lw=0, zorder=0)
            if label_ref:                           # region labelled with a leader
                leader_label(ax, "contact", (0.04, 90.0), (a / M + 0.08, 90.0), ts.GRAY)
    lo = YLIM_P[0]
    mean_band(ax, lam, d["init"], lo, C_INIT, "-", z=1)
    mean_band(ax, lam, d["endpose"], lo, C_ENDPOSE, DASH_END, z=2)
    mean_band(ax, lam, d["proxy_adam"], lo, C_PROXY, DASH_ADAM, z=2.5, adam=True,
              marker=MARK["proxy"], open_marker=True, k=3)
    mean_band(ax, lam, d["proxy_lbfgs"], lo, C_PROXY, "-", z=3, marker=MARK["proxy"], k=2)
    mean_band(ax, lam, d["state_adam"], lo, C_OURS, DASH_ADAM, z=4.5, adam=True,
              marker=MARK["state"], open_marker=True, k=1)
    if d["greedy"] is not None:
        g = np.maximum(d["greedy"], lo)
        ax.plot(lam, g, color=C_GREEDY, lw=LW, zorder=4)
        i = mark_idx(lam, 1.5, False)
        ax.plot(lam[i], g[i], ls="none", marker=MARK["greedy"], ms=MS - 0.4, mec=C_GREEDY,
                mfc=C_GREEDY, zorder=4.1)
    mean_band(ax, lam, d["state_lbfgs"], lo, C_OURS, "-", z=5, marker=MARK["state"], k=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    log_y(ax, *YLIM_P)
    ax.set_xlabel(r"$\lambda$", labelpad=1.0)
    ax.set_ylabel(r"$e_{\mathrm{mid}}$ (mm)")
    ts.style_2d(ax)
    box(ax)


def _line(c, ls, marker, open_marker, label, adam=False):
    """Legend entry: line + marker only (the bands are explained in the caption)."""
    return Line2D([], [], color=c, lw=LW_ADAM if adam else LW, ls=ls, marker=marker, ms=MS,
                  mec=c, mew=0.7, mfc="white" if open_marker else c, label=label)


def legend_S():
    return [_line(C_OURS, "-", MARK["state"], False, "State-based, L-BFGS (ours)"),
            _line(C_OURS, DASH_ADAM, MARK["state"], True, "State-based, Adam", True),
            _line(C_PROXY, "-", MARK["proxy"], False, "Frozen tangent, L-BFGS"),
            _line(C_PROXY, DASH_ADAM, MARK["proxy"], True, "Frozen tangent, Adam", True)]


def legend_P():
    return legend_S() + [
        _line(C_GREEDY, "-", MARK["greedy"], False, "Greedy"),
        _line(C_ENDPOSE, DASH_END, None, False, "End pose"),
        _line(C_INIT, "-", None, False, "Initial")]


# ------------------------------------------------------------------ layout
def tight_row(fig, axes, h, passes=3):
    """Equal-width axes side by side over the full width; the drawn content of
    neighbouring panels is GAP apart and MARGIN from the page edges."""
    W = ts.DBL_W
    for _ in range(passes):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        over = []
        for a in axes:
            b = a.get_tightbbox(r)
            p = a.get_position()
            over.append((p.x0 * W - b.x0 / fig.dpi, b.x1 / fig.dpi - p.x1 * W))
        wa = (W - 2 * MARGIN - sum(lo + ro for lo, ro in over) - (len(axes) - 1) * GAP) / len(axes)
        x = MARGIN
        for a, (lo, ro) in zip(axes, over):
            x += lo
            a.set_position([x / W, Y0 / h, wa / W, H_AX / h])
            x += wa + ro + GAP


def legend_below(fig, axes, handles, ncol):
    """One framed legend centred below a group of axes (one task); re-placed
    when ts.save resizes the figure."""
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.transforms import Bbox
    leg = fig.legend(handles=handles, labels=[x.get_label() for x in handles],
                     loc="upper center", ncol=ncol, frameon=True, fancybox=False,
                     framealpha=1.0, edgecolor=ts.BLACK, borderpad=0.35,
                     handletextpad=0.4, labelspacing=0.25, columnspacing=0.9,
                     handlelength=2.2, fontsize=ts.FS_TICK,
                     handler_map={ts._Group: HandlerTuple(ndivide=None, pad=0.0)})
    leg.get_frame().set_linewidth(0.5)

    def place():
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        bb = Bbox.union([a.get_tightbbox(r) for a in axes]).transformed(fig.transFigure.inverted())
        g = LEGEND_GAP / 72.0 / fig.get_size_inches()[1]
        leg.set_bbox_to_anchor((bb.x0, bb.y0 - g, bb.width, 0.0), transform=fig.transFigure)

    fig._tro_place = getattr(fig, "_tro_place", []) + [place]
    place()
    return leg


def main(out_dir=None):
    S = load()
    out = Path(out_dir) if out_dir else HERE
    with ts.style():
        fig = plt.figure(figsize=(ts.DBL_W, FIG_H))
        axes = []
        first = {"shape": True, "path": True}
        for i, d in enumerate(S):
            ax = fig.add_axes([0.1 + 0.25 * i, Y0 / FIG_H, 0.18, H_AX / FIG_H])
            if d["task"] == "shape":
                panel_shape(ax, d, ylabel=first["shape"], label_ref=first["shape"])
            else:
                panel_path(ax, d, label_ref=first["path"])
            first[d["task"]] = False
            axes.append(ax)
        tight_row(fig, axes, FIG_H)
        panel_letters(fig, axes)
        ax_s = [a for a, d in zip(axes, S) if d["task"] == "shape"]
        ax_p = [a for a, d in zip(axes, S) if d["task"] == "path"]
        if ax_s:
            legend_below(fig, ax_s, legend_S(), ncol=2)
        if ax_p:
            legend_below(fig, ax_p, legend_P(), ncol=3)
        ts.save(fig, out / "fig09_sim_results", width=ts.DBL_W, gray=True)
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
