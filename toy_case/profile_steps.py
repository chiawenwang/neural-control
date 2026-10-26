# -*- coding: utf-8 -*-
"""Where the time goes in one toy-DER experiment, measured on the real code.

    python profile_steps.py [buckle|twist] [--show 1 2 3] [--set key=value ...]

der.py / solver.py / run.py are not modified.  Their functions are wrapped with
wall-clock timers (time.perf_counter), then run.main() runs the experiment
exactly as `python run.py <experiment>` does - same config, same printed lines.
Every timed call is stored as a span (name, start, end, parent); the table and
the figure are computed from those spans alone.

    outputs/profile_<exp>.png   (a) whole run  (b) the --show load steps  (c) time shares
    outputs/profile_<exp>.json  every raw span, in s since the first line of this script
"""
import time

T0 = time.perf_counter()

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import unicodedata  # noqa: E402

T_STDLIB = time.perf_counter()
import torch  # noqa: E402

T_TORCH = time.perf_counter()
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

T_MPL = time.perf_counter()
import der  # noqa: E402
import run  # noqa: E402
import solver  # noqa: E402

T_MODS = time.perf_counter()


############################
# Recording
############################

class Recorder:
    """Nested wall-clock spans.  A span's parent is the innermost span open when it starts."""

    def __init__(self):
        self.spans, self.stack = [], []

    def add(self, name, t0, t1, parent=None, **meta):
        self.spans.append(dict(id=len(self.spans), name=name, parent=parent, t0=t0, t1=t1, **meta))
        return self.spans[-1]["id"]

    def wrap(self, name, fn, meta=None):
        @functools.wraps(fn)
        def timed(*args, **kwargs):
            m = meta(*args, **kwargs) if meta else {}
            sid = self.add(name, time.perf_counter(), None, self.stack[-1] if self.stack else None, **m)
            self.stack.append(sid)
            try:
                return fn(*args, **kwargs)
            finally:
                self.stack.pop()
                self.spans[sid]["t1"] = time.perf_counter()
        return timed


def bc_meta(self, end_twist=None, end_shift=None):
    """Prescribed values of a BC call (same signature as Rod.set_bc_targets), for labelling the load steps."""
    m = {}
    if end_twist is not None:
        m["phi"] = float(end_twist)
    if end_shift is not None:
        m["shift"] = [float(v) for v in end_shift]
    return m


def instrument(rec):
    """Wrap every function one load step goes through (README, 'How the files fit together')."""
    R = der.Rod
    R.__init__ = rec.wrap("rod_init", R.__init__, meta=lambda self, cfg: dict(n_nodes=int(cfg["rod"]["n_nodes"])))
    R.apply_end_twist = rec.wrap("bc", R.apply_end_twist, meta=lambda self, z, end_twist: bc_meta(self, end_twist))
    R.set_bc_targets = rec.wrap("bc", R.set_bc_targets, meta=bc_meta)
    R.track_holonomy = rec.wrap("holonomy", R.track_holonomy)
    run.load_cfg = rec.wrap("config", run.load_cfg)
    run.straight_branch_onset = rec.wrap("onset", run.straight_branch_onset)
    run.solve = rec.wrap("solve", run.solve)
    run.EXPERIMENTS = {k: rec.wrap("experiment", f) for k, f in run.EXPERIMENTS.items()}

    # inside solver.newton: Π(z) for the initial value and the line search, ∇Π, ∇²Π, the linear solve
    newton = rec.wrap("newton", solver.newton, meta=lambda obj, z0, *a, **k: dict(n_dof=len(z0)))
    solver.newton = lambda obj, *a, **k: newton(rec.wrap("energy", obj), *a, **k)
    grad, hessian = torch.func.grad, torch.func.hessian
    solver.grad = lambda f: rec.wrap("grad", grad(f))
    solver.hessian = lambda f: rec.wrap("hessian", hessian(f))
    torch.func.hessian = lambda f, *a, **k: rec.wrap("hessian", hessian(f, *a, **k))   # straight_branch_onset
    torch.linalg.cholesky_ex = rec.wrap("cholesky", torch.linalg.cholesky_ex)
    torch.cholesky_solve = rec.wrap("backsolve", torch.cholesky_solve)
    torch.linalg.eigvalsh = rec.wrap("eigvalsh", torch.linalg.eigvalsh)
    # first figure of the experiment = end of its last load step
    plt.figure = rec.wrap("plot", plt.figure)
    plt.subplots = rec.wrap("plot", plt.subplots)


############################
# Analysis (pure function of the spans)
############################

# category -> (short label, legend label, colour), in the order a load step runs through them
CATS = {
    "bc": ("改边界", "改边界条件 apply_end_twist / set_bc_targets", "#008300"),
    "energy": ("能量 Π", "能量 Π（牛顿初值 + 线搜索）", "#1baf7a"),
    "grad": ("梯度 ∇Π", "梯度 ∇Π（每次迭代 + 收敛后残差）", "#eb6834"),
    "hessian": ("Hessian ∇²Π", "Hessian ∇²Π", "#2a78d6"),
    "linalg": ("线性求解", "线性求解 Cholesky / 回代 / 特征值", "#eda100"),
    "holonomy": ("和乐更新", "和乐更新 track_holonomy", "#6250d6"),
    "rod_init": ("建杆", "建杆 Rod()", "#e87ba4"),
    "startup": ("导入/配置", "导入库 / 读配置", "#898781"),
    "other": ("其他", "其他：记录 / 打印 / 画图 / Python 开销", "#c3c2b7"),
}
# a leaf span's whole duration is charged to its category (Π evaluated inside ∇Π / ∇²Π counts as ∇Π / ∇²Π)
LEAF = {
    "import": "startup", "config": "startup", "rod_init": "rod_init", "bc": "bc", "holonomy": "holonomy",
    "energy": "energy", "grad": "grad", "hessian": "hessian",
    "cholesky": "linalg", "backsolve": "linalg", "eigvalsh": "linalg",
}


def pieces(span, kids):
    """Cover [t0, t1] of a span with (t0, t1, category): leaves whole, uncovered time -> 'other'."""
    if span["name"] in LEAF:
        return [(span["t0"], span["t1"], LEAF[span["name"]])]
    out, t = [], span["t0"]
    for c in kids[span["id"]]:
        if c["t0"] > t:
            out.append((t, c["t0"], "other"))
        out += pieces(c, kids)
        t = c["t1"]
    if span["t1"] > t:
        out.append((t, span["t1"], "other"))
    return out


def totals(flat, t0, t1):
    """Seconds per category inside the window [t0, t1]."""
    tot = dict.fromkeys(CATS, 0.0)
    for a, b, c in flat:
        lo, hi = max(a, t0), min(b, t1)
        if hi > lo:
            tot[c] += hi - lo
    return tot


def analyse(spans):
    kids = {s["id"]: [] for s in spans}
    kids[-1] = []
    for s in spans:
        kids[s["parent"] if s["parent"] is not None else -1].append(s)
    for v in kids.values():
        v.sort(key=lambda s: s["t0"])
    main = next(s for s in spans if s["name"] == "main")
    exp = next(s for s in spans if s["name"] == "experiment")
    flat = pieces(dict(id=-1, name="run", t0=0.0, t1=main["t1"]), kids)

    # load step = its BC calls + the solve after them + whatever the experiment does until the next
    # step's BC call (logging).  The last step ends at the experiment's first figure.
    xk = kids[exp["id"]]
    plot_t0 = next((c["t0"] for c in xk if c["name"] == "plot"), exp["t1"])
    steps, bcs = [], []
    for c in xk:
        if c["t0"] >= plot_t0:
            break
        if c["name"] == "bc":
            bcs.append(c)
        elif c["name"] == "solve":
            meta = {k: v for b in bcs for k, v in b.items() if k in ("phi", "shift")}
            steps.append(dict(k=len(steps) + 1, t0=(bcs[0] if bcs else c)["t0"], solve=c, **meta))
            bcs = []
    for s, nxt in zip(steps, steps[1:] + [None]):
        s["t1"] = nxt["t0"] if nxt else plot_t0
        nt = next(c for c in kids[s.pop("solve")["id"]] if c["name"] == "newton")
        nk = kids[nt["id"]]
        g = [c for c in nk if c["name"] == "grad"]          # one per iteration + the residual after the loop
        s["iters"] = [(a["t0"], b["t0"]) for a, b in zip(g[:-1], g[1:])]
        s["residual"] = (g[-1]["t0"], g[-1]["t1"])
        s["n_ls"] = sum(c["name"] == "energy" for c in nk) - 1   # minus the initial Π(z)
        s["n_chol"] = sum(c["name"] == "cholesky" for c in nk)
        s["n_dof"] = nt["n_dof"]
        s["tot"] = totals(flat, s["t0"], s["t1"])

    t_step0 = steps[0]["t0"]
    t_imp = max(s["t1"] for s in spans if s["name"] == "import")
    prep = [w for n, w in (("config", "读配置"), ("rod_init", "建杆"), ("onset", "稳定性分析"))
            if any(s["name"] == n and s["t0"] < t_step0 for s in spans)]
    phases = [("导入库", 0.0, t_imp), (f"准备（{' / '.join(prep)}）", t_imp, t_step0),
              (f"加载步 1–{len(steps)}", t_step0, plot_t0), ("画图保存", plot_t0, main["t1"])]
    n_hess_prep = sum(s["name"] == "hessian" and s["t0"] < t_step0 for s in spans)
    n_nodes = next(s["n_nodes"] for s in spans if s["name"] == "rod_init")

    # the first call of a torch.func transform pays a one-time setup cost
    warm = []
    for name in ("grad", "hessian"):
        d = [s for s in spans if s["name"] == name]
        if len(d) > 2:
            typical = sorted(s["t1"] - s["t0"] for s in d[1:])[(len(d) - 1) // 2]
            if d[0]["t1"] - d[0]["t0"] > 3 * typical:
                warm.append(dict(name=name, t0=d[0]["t0"], t1=d[0]["t1"], typical=typical))
    return dict(flat=flat, steps=steps, phases=phases, T=main["t1"], n_nodes=n_nodes, warm=warm,
                n_hess_prep=n_hess_prep, startup=totals(flat, 0.0, t_step0), all_steps=totals(flat, t_step0, plot_t0))


############################
# Report
############################

def fmt_t(s):
    if s < 5e-5:
        return "<0.1 ms" if s > 0 else "0"
    return f"{s * 1e3:.1f} ms" if s < 1 else f"{s:.2f} s"


def cols(s):
    """Terminal width of s: CJK characters take two columns."""
    return sum(1 + (unicodedata.east_asian_width(ch) in "WF") for ch in s)


def rjust(s, w):
    return " " * (w - cols(s)) + s


def step_label(s):
    out = f"第{s['k']}步"
    if "phi" in s:
        out += f"  Φ={s['phi']:.2f}"
    if any(s.get("shift", [0])):
        out += f"  shift={s['shift'][0]:+.3f}"
    return out


def print_report(prof, show):
    keys = list(CATS)
    print("\n=== 耗时分解（各类别占该行总时长的百分比）===")
    print("  " + rjust("", 26) + rjust("总时长", 10) + "".join(rjust(CATS[k][0], 12) for k in keys)
          + rjust("迭代", 6) + rjust("线搜索", 8) + rjust("Cholesky", 9))

    def row(label, tot, extra=""):
        T = sum(tot.values())
        print("  " + label + " " * (26 - cols(label)) + rjust(fmt_t(T), 10)
              + "".join(rjust(f"{100 * tot[k] / T:.2f}%", 12) for k in keys) + extra)

    row("启动·准备", prof["startup"])
    for s in prof["steps"]:
        mark = " *" if s["k"] in show else ""
        row(step_label(s) + mark, s["tot"], f"{len(s['iters']):6d}{s['n_ls']:8d}{s['n_chol']:9d}")
    row(f"全部 {len(prof['steps'])} 步", prof["all_steps"])
    print(f"  总计 {prof['T']:.2f} s;  * = 图 (b) 中展开的步")


############################
# Figure
############################

INK, INK2, MUTED, SURFACE, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9", "#c3c2b7"
CJK = ("Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC", "WenQuanYi Zen Hei",
       "Microsoft YaHei", "PingFang SC", "SimHei")
STYLE = {
    "font.family": [n for n in CJK if n in {f.name for f in font_manager.fontManager.ttflist}] + ["DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 8.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK2, "xtick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
}


def bars(ax, segs, y, h, gap):
    """segs: (x0, x1, cat).  gap > 0 draws a surface-coloured edge between touching pieces."""
    for c in CATS:
        xs = [(a, b - a) for a, b, cc in segs if cc == c]
        if xs:
            ax.broken_barh(xs, (y - h / 2, h), facecolors=CATS[c][2],
                           edgecolors=SURFACE if gap else "none", linewidth=gap)


def merged(segs):
    out = []
    for a, b, c in segs:
        if out and out[-1][2] == c and abs(out[-1][1] - a) < 1e-12:
            out[-1] = (out[-1][0], b, c)
        else:
            out.append((a, b, c))
    return out


def clip(flat, t0, t1):
    """Pieces inside [t0, t1], in ms since t0."""
    return [((max(a, t0) - t0) * 1e3, (min(b, t1) - t0) * 1e3, c) for a, b, c in flat if b > t0 and a < t1]


def bracket(ax, x0, x1, y, dy):
    """A bracket over [x0, x1] at height y whose ends point dy down (dy < 0: points up)."""
    ax.plot([x0, x0, x1, x1], [y - dy, y, y, y - dy], color=MUTED, lw=0.8, clip_on=False)


def label_free(ax, x0, x1, y, dy, text, placed, va="bottom", **kw):
    """Text over the middle of [x0, x1] at the first height y + k·dy that overlaps nothing in `placed`.
    Kept inside the axes horizontally.  Returns the height used."""
    r = ax.figure.canvas.get_renderer()
    box = ax.get_window_extent(r)
    for k in range(6):
        t = ax.text((x0 + x1) / 2, y + k * dy, text, ha="center", va=va, **kw)
        bb = t.get_window_extent(r)
        if bb.x0 < box.x0:
            t.set_x(x0), t.set_ha("left")
        elif bb.x1 > box.x1:
            t.set_x(x1), t.set_ha("right")
        bb = t.get_window_extent(r)
        if not any(bb.overlaps(o) for o in placed) or k == 5:
            placed.append(bb)
            return y + k * dy
        t.remove()


def warm_text(w):
    sym = "∇Π" if w["name"] == "grad" else "∇²Π"
    return f"首次调用 {sym}：{fmt_t(w['t1'] - w['t0'])}，含一次性初始化（之后每次约 {fmt_t(w['typical'])}）"


def fits(ax, t, x0, x1, pad=4):
    """Does text t fit horizontally between data x0 and x1?"""
    span = ax.transData.transform((x1, 0))[0] - ax.transData.transform((x0, 0))[0]
    return t.get_window_extent(ax.figure.canvas.get_renderer()).width < span - pad


def name_stages(ax, s, segs, y):
    """Name the stages of one load step on its row: inside the wide pieces of iteration 1,
    with leader lines below the bar for the thin ones."""
    t0 = s["t0"]
    ms = lambda t: (t - t0) * 1e3
    it0, it1 = (ms(t) for t in s["iters"][0])
    res0, res1 = (ms(t) for t in s["residual"])
    end = ms(s["t1"])

    def dur(a, b, cat):
        return fmt_t(sum(min(x1, b) - max(x0, a) for x0, x1, c in segs if c == cat and x1 > a and x0 < b) / 1e3)

    first = lambda x0: it0 - 1e-6 <= x0 < it1 - 1e-6          # piece starts inside iteration 1
    for x0, x1, c in segs:
        if c in ("grad", "hessian") and first(x0) or c == "grad" and x0 >= res0 - 1e-6:
            text = {"grad": "∇Π", "hessian": "∇²Π"}[c] if first(x0) else "残差 ∇Π"
            t = ax.text((x0 + x1) / 2, y, text, ha="center", va="center", fontsize=7.5, color="#ffffff")
            if not fits(ax, t, x0, x1):
                t.remove()
    h1 = max(x1 for x0, x1, c in segs if c == "hessian" and first(x0))
    notes = [(0.0, it0, f"改边界 {dur(0, it0, 'bc')} → Π 初值 {dur(0, it0, 'energy')}", "left"),
             (h1, it1, f"线性求解 {dur(h1, it1, 'linalg')} → 线搜索 Π {dur(h1, it1, 'energy')}", "center"),
             (res1, end, f"和乐更新 {dur(res1, end, 'holonomy')} → 记录等 {dur(res1, end, 'other')}", "right")]
    r, placed = ax.figure.canvas.get_renderer(), []
    for a, b, text, ha in notes:
        t = ax.text({"left": a, "center": (a + b) / 2, "right": b}[ha], y - 0.42, text, ha=ha, va="top",
                    fontsize=7.5, color=INK2)
        while hit := [o for o in placed if t.get_window_extent(r).overlaps(o)]:   # short row: slide right
            t.set(x=ax.transData.inverted().transform((max(o.x1 for o in hit) + 12, 0))[0], ha="left")
        placed.append(t.get_window_extent(r))
        ax.plot([(a + b) / 2, t.get_position()[0]], [y - 0.18, y - 0.4], color=MUTED, lw=0.6)


def leader(ax, a, b, y0, y1, others):
    """Vertical line from a bracket over [a, b] (height y0) to its label (height y1), at the first
    of centre / left end / right end where it crosses none of the `others` label boxes."""
    T = ax.transData
    for x in ((a + b) / 2, a, b):
        px = T.transform((x, 0))[0]
        lo, hi = sorted(T.transform([(x, y0), (x, y1)])[:, 1])
        if not any(o.x0 <= px <= o.x1 and o.y0 < hi and o.y1 > lo for o in others):
            break
    ax.plot([x, x], [y0, y1], color=MUTED, lw=0.6)


def runs(ks):
    """[1, 2, 3, 7] -> [(1, 3), (7, 7)]"""
    out = []
    for k in sorted(ks):
        if out and k == out[-1][1] + 1:
            out[-1] = (out[-1][0], k)
        else:
            out.append((k, k))
    return out


def plot(prof, show, path, meta):
    steps, T = prof["steps"], prof["T"]
    sel = [steps[k - 1] for k in show]
    keys = list(CATS)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14, 7.8 + 0.75 * len(sel)))
        gs = fig.add_gridspec(3, len(sel) + 2, height_ratios=[1.6, 0.75 * len(sel) + 0.5, 2.6],
                              left=0.075, right=0.985, top=0.855, bottom=0.055, hspace=0.62, wspace=0.12)

        # (a) whole run, to scale
        ax = fig.add_subplot(gs[0, :])
        ax.set_xlim(0, T)
        ax.set_ylim(-1.0, 1.75)
        bars(ax, merged(prof["flat"]), 0.0, 0.5, 0)
        for s in steps[1:]:
            ax.plot([s["t0"]] * 2, [-0.25, 0.25], color=SURFACE, lw=0.6)
        # labels right to left, so a raised label's leader passes left of the labels already placed
        placed, top, bottom = [], 0.46, -0.46
        for lab, a, b in reversed(prof["phases"]):
            bracket(ax, a, b, 0.38, 0.08)
            y = label_free(ax, a, b, 0.46, 0.3, f"{lab}  {fmt_t(b - a)} · {100 * (b - a) / T:.1f}%", placed,
                           fontsize=8, color=INK2)
            if y > 0.46:
                leader(ax, a, b, 0.38, y, placed[:-1])
            top = max(top, y)
        marks = [(steps[k0 - 1]["t0"], steps[k1 - 1]["t1"], f"第 {k0}–{k1} 步" if k1 > k0 else f"第 {k0} 步")
                 for k0, k1 in runs(show)]
        marks += [(w["t0"], w["t1"], warm_text(w)) for w in prof["warm"]       # a warm-up inside a shown
                  if not any(s["t0"] <= w["t0"] < s["t1"] for s in sel)]       # step is marked in (b) instead
        for a, b, text in marks:
            bracket(ax, a, b, -0.38, -0.08)
            y = label_free(ax, a, b, -0.46, -0.3, text, placed, va="top", fontsize=8, color=INK2)
            if y < -0.46:
                leader(ax, a, b, -0.38, y, placed[:-1])
            bottom = min(bottom, y)
        ax.set_ylim(bottom - 0.42, top + 0.45)
        ax.set_yticks([])
        ax.set_xlabel("运行时间 [s]（从脚本第一行起）")
        ax.set_title("(a) 全程时间线（按真实比例；白线 = 加载步分界，下方括号 = (b) 中展开的步）", loc="left",
                     color=INK, fontsize=10)

        # (b) the selected load steps, one row each, time since the step started
        ax = fig.add_subplot(gs[1, :])
        width = max(s["t1"] - s["t0"] for s in sel) * 1e3
        pitch = 1.3                                      # row spacing; the first row carries the stage names
        ax.set_xlim(0, width * 1.1)
        ax.set_ylim(-pitch * (len(sel) - 1) - 0.4, 0.62)
        named = False
        for row, s in enumerate(sel):
            y = -pitch * row
            segs = merged(clip(prof["flat"], s["t0"], s["t1"]))
            bars(ax, segs, y, 0.36, 0.8)
            for i, (a, b) in enumerate(s["iters"]):
                x0, x1 = (a - s["t0"]) * 1e3, (b - s["t0"]) * 1e3
                bracket(ax, x0, x1, y + 0.25, 0.05)
                for text in (f"迭代 {i + 1}", f"{i + 1}"):
                    t = ax.text((x0 + x1) / 2, y + 0.27, text, ha="center", va="bottom", fontsize=7.5, color=INK2)
                    if fits(ax, t, x0, x1):
                        break
                    t.remove()
            ax.text(width * 1.012, y, f"{fmt_t(s['t1'] - s['t0'])}\n{len(s['iters'])} 次牛顿迭代",
                    va="center", ha="left", fontsize=8, color=INK2)
            warm = [w for w in prof["warm"] if s["t0"] <= w["t0"] < s["t1"]]
            for w in warm:
                x0, x1 = (w["t0"] - s["t0"]) * 1e3, (w["t1"] - s["t0"]) * 1e3
                t = ax.text((x0 + x1) / 2, y, warm_text(w), ha="center", va="center", fontsize=7.5, color="#ffffff")
                if not fits(ax, t, x0, x1, pad=8):
                    t.set(x=x0, y=y - 0.21, ha="left", va="top", color=INK2)
            if not warm and not named:
                name_stages(ax, s, segs, y)
                named = True
        ax.set_yticks([-pitch * i for i in range(len(sel))], [step_label(s).replace("  ", "\n", 1) for s in sel])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("该步开始后的时间 [ms]")
        ax.set_title("(b) 加载步内部：改边界 → [∇Π → ∇²Π → 线性求解 → 线搜索] × 迭代 → 残差 ∇Π → 和乐更新 → 记录",
                     loc="left", color=INK, fontsize=10)

        # (c) share of each category, one panel per window
        n = prof["n_hess_prep"]
        wins = [("启动·准备", prof["startup"], f" · 含 {n} 次 ∇²Π" if n else "")] + \
               [(step_label(s), s["tot"], f" · {len(s['iters'])} 次迭代") for s in sel] + \
               [(f"全部 {len(steps)} 个加载步", prof["all_steps"], f" · {sum(len(s['iters']) for s in steps)} 次迭代")]
        for j, (lab, tot, extra) in enumerate(wins):
            ax = fig.add_subplot(gs[2, j])
            if j == 0:
                pos = ax.get_position()
                fig.text(pos.x0, pos.y1 + 0.5 / fig.get_figheight(), "(c) 各类计算的耗时占比（占该栏总时长）",
                         fontsize=10, color=INK, va="bottom")
            Tw = sum(tot.values())
            pct = [100 * tot[k] / Tw for k in keys]
            ax.barh(range(len(keys)), pct, height=0.6, color=[CATS[k][2] for k in keys])
            for i, (k, p) in enumerate(zip(keys, pct)):
                if tot[k] == 0:
                    continue
                txt = f"{p:.1f}%  {fmt_t(tot[k])}" if p >= 0.05 else f"<0.1%  {fmt_t(tot[k])}"
                if p > 55:
                    ax.text(p - 2, i, txt, va="center", ha="right", fontsize=7.5, color="#ffffff")
                else:
                    ax.text(p + 2, i, txt, va="center", ha="left", fontsize=7.5, color=INK2)
            ax.set_xlim(0, 100)
            ax.set_ylim(len(keys) - 0.5, -0.5)
            ax.set_xticks([0, 50, 100], ["0", "50%", "100%"])
            ax.grid(axis="x", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            ax.set_yticks(range(len(keys)), [CATS[k][0] for k in keys] if j == 0 else [])
            ax.tick_params(axis="y", length=0)
            ax.set_title(f"{lab}\n{fmt_t(Tw)}{extra}", fontsize=8.5, color=INK, loc="left")

        fig.text(0.012, 0.985, f"toy DER · {meta['experiment']} 耗时分解（真实运行）", fontsize=13, color=INK,
                 va="top", weight="bold")
        fig.text(0.012, 0.955, f"总 {T:.1f} s · N={prof['n_nodes']} 节点 · {steps[0]['n_dof']} 个自由度 · "
                 f"{len(steps)} 个加载步 · torch {meta['torch']} · {meta['threads']} 线程 · "
                 "计时 = time.perf_counter 包装真实函数，der/solver/run 未改动", fontsize=8.5, color=INK2, va="top")
        fig.legend(handles=[Patch(color=CATS[k][2], label=CATS[k][1]) for k in keys], loc="upper left",
                   bbox_to_anchor=(0.008, 0.932), ncol=5, frameon=False, fontsize=8.5, handlelength=1.1,
                   columnspacing=1.6)
        fig.savefig(path, dpi=130)
        plt.close(fig)


############################
# Main
############################

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?", default="buckle", choices=["buckle", "twist"])
    ap.add_argument("--show", nargs="+", type=int, default=[1, 2, 3], metavar="K",
                    help="load steps drawn in detail in (b) and (c), 1-based")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="passed on to run.py")
    args = ap.parse_args()

    rec = Recorder()
    for what, a, b in [("stdlib", T0, T_STDLIB), ("torch", T_STDLIB, T_TORCH),
                       ("matplotlib", T_TORCH, T_MPL), ("der/run/solver", T_MPL, T_MODS)]:
        rec.add("import", a, b, what=what)
    instrument(rec)
    sys.argv = ["run.py", args.experiment] + (["--set", *args.set] if args.set else [])
    out = run.OUT
    with tempfile.TemporaryDirectory() as tmp:
        run.OUT = tmp                       # the experiment's own figure is drawn (and timed), not kept
        rec.wrap("main", run.main)()
    run.OUT = out
    spans = [dict(s, t0=s["t0"] - T0, t1=s["t1"] - T0) for s in rec.spans]

    prof = analyse(spans)
    steps = prof["steps"]
    bad = [k for k in args.show if not 1 <= k <= len(steps)]
    if bad:
        sys.exit(f"--show {bad}: this run has load steps 1..{len(steps)}")
    print_report(prof, args.show)

    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f"profile_{args.experiment}")
    meta = dict(experiment=args.experiment, argv=sys.argv, torch=torch.__version__, threads=torch.get_num_threads())
    with open(stem + ".json", "w") as f:                  # one span per line
        f.write(json.dumps(meta)[:-1] + ', "spans": [\n' + ",\n".join(map(json.dumps, spans)) + "\n]}\n")
    plot(prof, args.show, stem + ".png", meta)
    print(f"  -> {stem}.png, {stem}.json")


if __name__ == "__main__":
    main()
