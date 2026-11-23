"""Panel (a): the energy landscape the robot steers.

    python -m paper_figures.fig01.landscape [--force] [--workers N]

A clamped strip compressed to ``(1 - eps) L`` is a bistable arch. Rotating the
right clamp (``theta_r = lambda theta_max``) tilts its energy landscape
``E(s; lambda)`` -- the energy minimized with the midpoint height pinned at
``s``. The realized equilibrium follows the floor of the valley it started in
until that valley ends at a fold, then snaps into the other one.
The physics is in :mod:`landscape_compute`, cached by a hash of the settings
(``--force`` recomputes).

Components (``out/fig01/components``), every plot as ``*_bare`` (graphics only)
and ``*_axes`` (axis lines and ticks, no text where marked below); text lives in
its own layer on the same canvas (``a_surface_text``, ``a_rods_column_label``):

a_surface_*              E(s, lambda) as a rasterized surface (main style), the
                         realized path on top: balls joined by dashed links, a thin
                         arrow beside each branch, a curved arrow for the snap, the
                         saddle line (dashed on a white halo); labels start / snap /
                         end / saddle
a_rods_column_*          the process as a stack of rods (flipbook): start, snap,
                         end solid, the frames between translucent
a_ridgeline_*            curves E(s; lambda_i) stacked by lambda, balls at the
                         realized equilibria, the valley-floor path colored by
                         lambda, dashed gray line through the local maxima
a_ridgeline_path_solid   the valley-floor path as single-color polylines
a_rods, a_rod_lam_<v>    the same rods as a strip and one per file
colorbar_lambda_*        the lambda colorbar, horizontal and vertical
... and the same with ``_obstacle`` for the scenario with a circle above the arch.

Data (``out/fig01/data``): curves, realized path and surface as CSV/NPZ,
``landscape_summary.json`` (theta_c, snap, validation) and ``annotations.json``.
"""
from __future__ import annotations

import paper_figures  # noqa: F401  (src/ on sys.path, BLAS threads)

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import yaml                                                     # noqa: E402
from matplotlib.patches import Rectangle                        # noqa: E402

from paper_figures import style                                 # noqa: E402
from paper_figures.common import draw                           # noqa: E402
from paper_figures.common.export import Exporter, data_to_inches  # noqa: E402
from paper_figures.common.paths import CACHE_DIR, rel           # noqa: E402
from paper_figures.fig01 import landscape_compute as lcm        # noqa: E402

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_cfg(path=CONFIG_PATH):
    return yaml.safe_load(Path(path).read_text())


def load_or_compute(cfg, force=False, workers=None):
    spec = lcm.physics_spec(cfg)
    h = lcm.spec_hash(spec)
    path = CACHE_DIR / f"fig01_landscape_{h}.pkl"
    if path.exists() and not force:
        print(f"cached computation: {rel(path)}")
        with open(path, "rb") as f:
            return pickle.load(f)
    res = lcm.compute(cfg, workers)
    res["_hash"], res["_spec"] = h, spec
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    print(f"cached to {rel(path)}")
    return res


def lam_tags(lams):
    """Labels for lambda values on the step grid: the shortest exact decimal,
    at least two places (0.00, 0.20, 0.765, 0.85)."""
    tags = []
    for v in lams:
        nd = next((n for n in (2, 3, 4) if abs(v - round(v, n)) < 1e-9), 6)
        tags.append(f"{v:.{nd}f}")
    return tags


# ------------------------------------------------------------------ summary

def summarize(sc, cfg):
    """Numbers for the report and landscape_summary.json."""
    p, k = sc["path"], sc["k_snap"]
    lam = sc["lam"]
    tc = sc["theta_c"]
    out = {
        "theta_c_rad": tc["theta_c"], "theta_c_deg": float(np.degrees(tc["theta_c"])),
        "theta_c_bracket_rad": tc["theta_c_bracket"],
        "rotation_direction": "counterclockwise (+)" if tc["direction"] > 0 else "clockwise (-)",
        "directions_without_snap": tc["no_snap"],
        "theta_max_rad": sc["theta_max"],
        "lambda_fold_from_theta_c": tc["theta_c"] / sc["theta_max"],
        "k_snap": k, "lambda_snap": sc["lam_snap"], "lambda_before_snap": float(lam[k - 1]),
        "s_before_after": [float(p["s"][k - 1]), float(p["s"][k])],
        "E_before_after": [float(p["E"][k - 1]), float(p["E"][k])],
        "newton_iterations_at_snap": int(p["iterations"][k]),
        "newton_first_attempt_at_snap": int(p["first_attempt"][k]),
        "snap_attempts": p["attempts"][k],
        "newton_iterations_median": float(np.median(p["iterations"][1:])),
        "newton_iterations_max_elsewhere": int(np.max(np.delete(p["iterations"][1:], k - 1))),
        "min_eig_before_snap": float(p["min_eig"][k - 1]),
        "min_eig_at_lambda0": float(p["min_eig"][0]),
        "min_eig_after_snap": float(p["min_eig"][k]),
        "min_eig_at_theta_c": tc["min_eig_at_fold"],
        "kappa_before_snap": float(p["kappa"][k - 1]),
        "kappa_at_lambda0": float(p["kappa"][0]),
        "snap_detection": sc["snap_detection"],
        "lower_branch_jumps": sc["lower_jumps"],
        "post_snap_vs_lower_branch_m": sc["post_snap_vs_lower"],
        "branches": {n: {kk: v for kk, v in b.items() if kk != "q"}
                     for n, b in sc["branches"].items()},
    }
    # fold from the landscape curvature at the upper well: kappa ~ sqrt(lambda_f - lambda)
    sel = np.arange(max(1, k - 12), k)
    kap = p["kappa"][sel]
    if np.all(kap > 0):
        a, b = np.polyfit(lam[sel], kap ** 2, 1)
        out["lambda_fold_from_kappa_fit"] = float(-b / a)
    if "contact_fd_error" in sc:
        out["contact_energy_fd_rel_error"] = sc["contact_fd_error"]
    # validation of the constrained landscapes
    val = []
    for kk in sorted(sc["landscapes"]):
        o = sc["landscapes"][kk]
        mins = [e for e in o["extrema"] if e["kind"] == "min"]
        maxs = [e for e in o["extrema"] if e["kind"] == "max"]
        val.append({
            "k": kk, "lambda": o["lam"], "n_points": o["n_points"], "n_failed": o["n_fail"],
            "wells_s": [w["s"] for w in o["wells"]],
            "minima": [{kx: e.get(kx) for kx in ("s", "E", "s_eq", "E_eq", "ds", "dE",
                                                 "min_eig_eq", "match", "is_well_point")}
                       for e in mins],
            "maxima": [{kx: e.get(kx) for kx in ("s", "E", "s_eq", "E_eq", "ds", "dE",
                                                 "min_eig_eq", "match", "smooth")}
                       for e in maxs],
            "all_minima_match": all(e["match"] for e in mins),
        })
    out["validation"] = val
    n_min = np.array([len(v["minima"]) for v in val])
    lams = np.array([v["lambda"] for v in val])
    two = lams[n_min >= 2]
    one = lams[n_min == 1]
    out["upper_well_last_lambda"] = float(two.max()) if len(two) else None
    out["upper_well_gone_lambda"] = float(one[one > (two.max() if len(two) else -1)].min()) \
        if len(one) else None
    out["all_minima_match"] = bool(all(v["all_minima_match"] for v in val))
    out["n_failed_points"] = int(sum(v["n_failed"] for v in val))
    out["n_points"] = int(sum(v["n_points"] for v in val))
    return out


# --------------------------------------------------------------- data files

def write_data(sc, tag, ddir):
    p = sc["path"]
    with open(ddir / f"realized_path{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "lambda", "theta_r_rad", "s_over_L", "E_over_EI_L", "E_elastic_over_EI_L",
                    "newton_iterations", "min_hessian_eig", "landscape_curvature", "attempts"])
        for k in range(len(sc["lam"])):
            w.writerow([k, f"{sc['lam'][k]:.6f}", f"{sc['theta'][k]:.8f}", f"{p['s'][k]:.8f}",
                        f"{p['E'][k]:.8f}", f"{p['E_el'][k]:.8f}", p["iterations"][k],
                        f"{p['min_eig'][k]:.6e}", f"{p['kappa'][k]:.6e}", p["attempts"][k]])
    with open(ddir / f"landscape_curves{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "lambda", "theta_r_rad", "s_over_L", "E_over_EI_L", "dE_ds", "is_well",
                    "branch"])
        for kk in sc["k_curves"]:
            o = sc["landscapes"][kk]
            for i in range(len(o["curve_s"])):
                w.writerow([kk, f"{o['lam']:.6f}", f"{o['theta']:.8f}", f"{o['curve_s'][i]:.8f}",
                            f"{o['curve_E'][i]:.8f}", f"{o['curve_dEds'][i]:.6e}",
                            int(o["curve_is_well"][i]), int(o["curve_src"][i])])
    ks = sc["k_surface"]
    np.savez_compressed(ddir / f"surface{tag}.npz", s_over_L=sc["s_grid"],
                        lam=np.array([sc["lam"][kk] for kk in ks]),
                        theta=np.array([sc["theta"][kk] for kk in ks]),
                        E_over_EI_L=np.array([sc["landscapes"][kk]["grid_E"] for kk in ks]))
    np.savez_compressed(ddir / f"realized_path{tag}.npz", lam=sc["lam"], theta=sc["theta"],
                        s=p["s"], E=p["E"], nodes=p["nodes"], lower_nodes=sc["lower"]["nodes"],
                        k_snap=sc["k_snap"])


# ------------------------------------------------------------- ridgeline

class Ridge:
    """Geometry shared by the ridgeline variants: curves, offsets, limits."""

    def __init__(self, sc, cfg):
        pa = cfg["panel_a"]
        self.sc = sc
        self.ks = sc["k_curves"]
        lam = sc["lam"]
        self.lams = [float(lam[k]) for k in self.ks]
        self.tags = lam_tags(self.lams)
        L0 = sc["landscapes"][0]
        E_well0 = min(w["E"] for w in L0["wells"])
        saddles0 = [e["E_eq"] for e in L0["extrema"]
                    if e["kind"] == "max" and e.get("match") and e.get("E_eq") is not None]
        barrier0 = (max(saddles0) if saddles0 else np.nanmax(L0["curve_E"])) - E_well0
        self.E_ref = E_well0
        self.clip = E_well0 + pa["energy_clip"] * barrier0
        self.curves = []
        for k in self.ks:
            o = sc["landscapes"][k]
            s, E = o["curve_s"].copy(), o["curve_E"].copy()
            E[E > self.clip] = np.nan
            self.curves.append((s, E))
        lo = min(np.nanmin(E) for _, E in self.curves)
        height = self.clip - lo
        dl = np.diff(self.lams)
        c = pa["offset_per_lambda"]
        if c is None:
            c = (1.0 - pa["overlap"]) * height / float(np.median(dl))
        self.c = float(c)
        self.lo = lo
        smax = float(sc["s_grid"][-1])
        self.xlim = (-smax, smax)
        tops = [np.nanmax(E) + self.c * l for (_, E), l in zip(self.curves, self.lams)]
        bots = [np.nanmin(E) + self.c * l for (_, E), l in zip(self.curves, self.lams)]
        pad = 0.02 * (max(tops) - min(bots))
        self.ylim = (min(bots) - pad, max(tops) + pad)
        # realized path up to the last curve, split at the snap
        p, ksn = sc["path"], sc["k_snap"]
        kmax = self.ks[-1]
        y = p["E"] + self.c * lam
        self.pre = (np.column_stack([p["s"][:ksn], y[:ksn]]), lam[:ksn])
        self.post = (np.column_stack([p["s"][ksn:kmax + 1], y[ksn:kmax + 1]]), lam[ksn:kmax + 1])
        self.balls = [(float(p["s"][k]), float(y[k]), float(lam[k])) for k in self.ks]
        # local maxima: the barrier between the two wells at every lambda before the
        # snap (validated saddles of the full system), ending at the fold
        sad = []
        for kk in sorted(sc["landscapes"]):
            o = sc["landscapes"][kk]
            if kk >= ksn or len(o["wells"]) < 2:
                continue
            ws = sorted(w["s"] for w in o["wells"])
            cand = [e for e in o["extrema"] if e["kind"] == "max" and e.get("match")
                    and ws[0] < e["s_eq"] < ws[-1] and e["min_eig_eq"] < 0]
            if cand:
                e = max(cand, key=lambda e: e["E_eq"])
                sad.append((e["s_eq"], e["E_eq"] + self.c * lam[kk], lam[kk]))
        tc = sc["theta_c"]
        lf = tc["theta_c"] / sc["theta_max"]
        if "E_at_fold" in tc and 0.0 < lf <= 1.0:
            sad.append((tc["s_at_fold"], tc["E_at_fold"] + self.c * lf, lf))
        self.saddle = np.array(sorted(sad, key=lambda t: t[2])) if sad else None
        self.jump = (self.pre[0][-1], self.post[0][0])

    def axes(self, fig, cfg):
        W, H = fig.get_size_inches()
        l, b, w, h = cfg["panel_a"]["axes_rect_in"]
        ax = fig.add_axes([l / W, b / H, w / W, h / H])
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        return ax


def draw_ridgeline(fig, ridge, cfg, variant):
    ax = ridge.axes(fig, cfg)
    pa = cfg["panel_a"]
    if variant == "solid":
        for name, (xy, _) in (("pre", ridge.pre), ("post", ridge.post)):
            draw.polyline(ax, xy, color=style.INK, lw=style.LW_DATA_THICK,
                          gid=f"valley_path_{name}_solid", zorder=4)
        ax.set_axis_off()
        return ax
    for (s, E), lam, tag in (zip(ridge.curves, ridge.lams, ridge.tags) if variant != "axes" else ()):
        line = draw.polyline(ax, np.column_stack([s, E + ridge.c * lam]),
                             color=style.lambda_color(lam), lw=style.LW_DATA, gid=f"curve_lam_{tag}",
                             zorder=2)
        line.set_solid_capstyle("butt")
    if variant != "axes" and ridge.saddle is not None and len(ridge.saddle) > 1:
        draw.polyline(ax, ridge.saddle[:, :2], color=style.AUX_GRAY, lw=style.LW_AUX,
                      ls=(0, (2.5, 1.5)), gid="local_maxima", zorder=3)
    for name, (xy, lam) in ((("pre", ridge.pre), ("post", ridge.post)) if variant != "axes" else ()):
        if len(xy) > 1:
            draw.lambda_path(ax, xy, lam, lw=style.LW_DATA_THICK, gid=f"valley_path_{name}",
                             zorder=4)
    for (s, y, lam), tag in (zip(ridge.balls, ridge.tags) if variant != "axes" else ()):
        draw.ball(ax, s, y, color=style.lambda_color(lam), gid=f"ball_lam_{tag}",
                  size_pt=pa["ball_size_pt"])
    if variant == "bare":
        ax.set_axis_off()
    else:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_bounds(*ridge.ylim)
        ax.set_yticks([])
        ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
        ax.set_xlabel(r"$s/L$", labelpad=1.0)
        ax.set_ylabel("energy", labelpad=2.0)
    return ax


# --------------------------------------------------------------- surface

def _interval_below(s, E, clip, s_hi):
    """``[lo, hi]`` around ``s = 0`` where ``E <= clip`` (ends interpolated), ``hi <= s_hi``."""
    ok = np.isfinite(E) & (E <= clip)
    i0 = int(np.argmin(np.abs(s)))

    def end(step):
        i = i0
        while 0 <= i + step < len(s) and ok[i + step]:
            i += step
        j = i + step
        if 0 <= j < len(s) and np.isfinite(E[j]):
            return s[i] + (clip - E[i]) / (E[j] - E[i]) * (s[j] - s[i])
        return s[i]

    return end(-1), min(end(+1), s_hi)


def surface_mesh(sc, ridge, sp):
    """``(S, Lm, Z)`` of the surface, one row per lambda over the part of
    ``E(s; lambda)`` below the clip. The left edge is common to all rows; on the
    right each row stops at the clip or just past the right well of the
    realized path, whichever comes first (a taller right wall would hide the
    path)."""
    ks = sc["k_surface"]
    s_all = sc["s_grid"]
    p, ksn = sc["path"], sc["k_snap"]
    s_hi = float(np.max(p["s"][:ksn])) + sp["s_margin"]
    n = sp["n_s_plot"]
    ends = [_interval_below(s_all, sc["landscapes"][k]["grid_E"], ridge.clip, s_hi) for k in ks]
    lo = max(e[0] for e in ends)
    S = np.empty((len(ks), n))
    Z = np.empty_like(S)
    for i, (k, (_, hi)) in enumerate(zip(ks, ends)):
        E = sc["landscapes"][k]["grid_E"]
        S[i] = np.linspace(lo, hi, n)
        fin = np.isfinite(E)
        Z[i] = np.interp(S[i], s_all[fin], E[fin])
    Lm = np.repeat(np.array([sc["lam"][k] for k in ks])[:, None], n, axis=1)
    return S, Lm, Z


def project(fig, ax, xyz):
    """``(n, 3)`` data points of a 3D axes -> ``(n, 2)`` inches on the figure
    (after a draw)."""
    from mpl_toolkits.mplot3d import proj3d
    x, y, _ = proj3d.proj_transform(*np.asarray(xyz, dtype=float).T, ax.get_proj())
    return ax.transData.transform(np.column_stack([x, y])) / fig.dpi


def surface_balls(sc):
    """Balls on the surface, in order along the realized path: the curve lambdas
    before the snap, the first state after it and the last state."""
    ksn = sc["k_snap"]
    ks = [k for k in sc["k_curves"] if k < ksn] + [ksn, sc["k_curves"][-1]]
    return sorted(set(ks))


def _arclen(P):
    """Cumulative arc length along the polyline ``P``, starting at 0."""
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def _beside(P, offset, trim, smooth, step=0.002):
    """The polyline ``P`` (inches) resampled, smoothed (Gaussian of ``smooth``
    inches, so a corner of the path does not kink it), shifted ``offset`` to the
    right of the direction of travel and cut by the fractions ``trim = (tail,
    head)`` of its length."""
    c = _arclen(P)
    t = np.linspace(0.0, c[-1], max(int(c[-1] / step), 2) + 1)
    Q = np.column_stack([np.interp(t, c, P[:, i]) for i in range(2)])
    if smooth > 0.0:
        sig = smooth / (t[1] - t[0])
        m = min(int(3 * sig), len(Q) - 1)
        w = np.exp(-0.5 * (np.arange(-m, m + 1) / sig) ** 2)
        # odd reflection at both ends keeps the end points and directions
        ext = np.vstack([2 * Q[0] - Q[m:0:-1], Q, 2 * Q[-1] - Q[-2:-m - 2:-1]])
        Q = np.column_stack([np.convolve(ext[:, i], w / w.sum(), mode="valid") for i in range(2)])
    T = np.gradient(Q, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    Q = Q + offset * np.column_stack([T[:, 1], -T[:, 0]])
    c = _arclen(Q)
    return Q[(c >= trim[0] * c[-1]) & (c <= (1.0 - trim[1]) * c[-1])]


def _snap_curve(a, b, arc, trim, n=400):
    """Cubic from ``a`` to ``b`` leaving ``a`` at ``depart_deg`` and arriving at
    ``b`` from ``arrive_deg`` (handles ``height_in`` long), trimmed to start and
    end ``trim`` away from the two points."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    h = arc["height_in"]
    a0, a1 = np.radians([arc["depart_deg"], arc["arrive_deg"]])
    p1 = a + h * np.array([np.cos(a0), np.sin(a0)])
    p2 = b + h * np.array([np.cos(a1), np.sin(a1)])
    t = np.linspace(0.0, 1.0, n)[:, None]
    B = (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * b
    keep = (np.linalg.norm(B - a, axis=1) >= trim) & (np.linalg.norm(B - b, axis=1) >= trim)
    return B[keep]


def surface_overlay(fig, ax, sc, cfg, variant, box, saddle=None):
    """2D layer over the 3D axes, in inches: dashed links between consecutive
    balls, an arrow beside each branch of the path, the snap arrow, and (``axes``
    variant) hand-placed axes with the energy axis on the left, the start / snap /
    end labels and the saddle label."""
    pa = cfg["panel_a"]
    sp = pa["surface"]
    ann = sp["process"]
    W, H = fig.get_size_inches()
    ov = fig.add_axes([0, 0, 1, 1])
    ov.set_axis_off()
    ov.set_xlim(0, W)
    ov.set_ylim(0, H)
    ov.set_zorder(ax.get_zorder() + 1)
    ov.set_gid("overlay")
    p, lam, ksn = sc["path"], sc["lam"], sc["k_snap"]
    graphics = variant != "text"             # bare / axes: vectors only
    labels = variant == "text"               # text: every string and nothing else
    xyz = np.column_stack([p["s"], lam, p["E"]])
    balls = surface_balls(sc)
    C = {k: project(fig, ax, xyz[[k]])[0] for k in balls}
    r = 0.5 * pa["ball_size_pt"] * style.PT + 0.5 * style.BALL_RING_LW * style.PT
    # dashed links between consecutive balls along the projected valley floor; the
    # dash pattern is fitted to each link so it starts and ends with a whole dash
    lk = ann["links"]
    d0, g0 = lk["dash_pt"]
    for ka, kb in (zip(balls[:-1], balls[1:]) if graphics else ()):
        if ka < ksn <= kb:
            continue                         # the snap: its own arrow
        P = project(fig, ax, xyz[ka:kb + 1])
        c = _arclen(P)
        a, b = r + lk["clear_in"], c[-1] - r - lk["clear_in"]
        if b <= a:
            continue
        t = np.concatenate([[a], c[(c > a) & (c < b)], [b]])
        n = max(lk["min_dashes"], round(((b - a) / style.PT + g0) / (d0 + g0)))
        f = (b - a) / style.PT / (n * d0 + (n - 1) * g0) / lk["lw_pt"]
        draw.polyline(ov, np.column_stack([np.interp(t, c, P[:, i]) for i in range(2)]),
                      color=style.INK, lw=lk["lw_pt"], ls=(0, (f * d0, f * g0)),
                      gid=f"link_k{ka}_k{kb}")
    # one thin arrow beside each branch, on the right of the direction of travel
    # (outside the right valley, below the left one: clear of the snap arrow)
    bs = ann["beside"]
    for name, ka, kb in ((("pre", balls[0], ksn - 1), ("post", ksn, balls[-1])) if graphics else ()):
        Q = _beside(project(fig, ax, xyz[ka:kb + 1]), bs["offset_in"], bs["trim"], bs["smooth_in"])
        draw.arrow(ov, Q, head_in=ann["head_in"], gid=f"arrow_{name}")
    # the snap: a curved arrow over the landscape, ball to ball, coming down steeply
    # onto the landing ball (clear of the path that leaves it)
    B = _snap_curve(C[ksn - 1], C[ksn], ann["snap_arc"], r + ann["gap_in"])
    if graphics:
        draw.arrow(ov, B, head_in=ann["head_in"], gid="snap_arrow")
    if variant == "bare":
        return ov
    # hand-placed axes: s along the front edge, lambda along the right edge,
    # energy up the front-left edge (away from the rod column)
    s_lo, s_hi, zlo, zhi = box
    fs = style.FONT_SIZE
    tick = ann["tick_in"]

    def unit(v):
        v = np.asarray(v, dtype=float)
        return v / np.linalg.norm(v)

    o_s = project(fig, ax, [[s_lo, 0.0, zlo], [s_hi, 0.0, zlo]])
    o_l = project(fig, ax, [[s_hi, 0.0, zlo], [s_hi, 1.0, zlo]])
    o_e = project(fig, ax, [[s_lo, 0.0, zlo], [s_lo, 0.0, zhi]])
    out_s = unit(-(o_l[1] - o_l[0]))          # ticks of s point toward the viewer
    out_l = unit(o_s[1] - o_s[0])             # ticks of lambda point along +s
    for name, P in ((("s", o_s), ("lambda", o_l), ("energy", o_e)) if graphics else ()):
        draw.polyline(ov, P, color=style.INK, lw=style.LW_AUX, gid=f"axis_{name}")
    gap = ann["ticklabel_gap_in"]
    # tick labels sit below (s) or right of (lambda) the tick ends, so a tick never
    # touches its label
    ticks = {"s": ([-0.1, 0.0, 0.1], out_s, lambda v: [v, 0.0, zlo], ([0.0, -gap], "center", "top")),
             "lambda": ([0.0, 0.5, 1.0], out_l, lambda v: [s_hi, v, zlo],
                        ([gap, 0.0], "left", "center"))}
    for name, (vals, out, pt, (off, ha, va)) in ticks.items():
        for v in vals:
            q = project(fig, ax, [pt(v)])[0]
            end = q + tick * out
            if graphics:
                draw.polyline(ov, [q, end], color=style.INK, lw=style.LW_AUX,
                              gid=f"tick_{name}_{v:g}")
            if labels:
                lab = f"{v:g}".replace("-", "−")
                t = ov.text(*(end + np.array(off)), lab, fontsize=fs, ha=ha, va=va)
                t.set_gid(f"ticklabel_{name}_{v:g}")
    if not labels:
        return ov
    mid_s = 0.5 * (o_s[0] + o_s[1]) + np.array(ann["s_label_in"])
    t = ov.text(*mid_s, r"$s/L$", fontsize=style.LABEL_SIZE, ha="center", va="center")
    t.set_gid("label_s")
    mid_l = 0.5 * (o_l[0] + o_l[1]) + np.array(ann["lambda_label_in"])
    t = ov.text(*mid_l, r"$\lambda$", fontsize=style.LABEL_SIZE, ha="center", va="center")
    t.set_gid("label_lambda")
    mid_e = 0.5 * (o_e[0] + o_e[1]) + np.array([-ann["energy_label_in"], 0.0])
    t = ov.text(*mid_e, "energy", fontsize=fs, ha="center", va="center", rotation=90)
    t.set_gid("label_energy")
    # the process: start at the first ball, snap over the arc, end at the last ball
    anchors = {"start": C[balls[0]], "snap": B[int(np.argmax(B[:, 1]))], "end": C[balls[-1]]}
    for name, at in anchors.items():
        spec = ann["labels"][name]
        t = ov.text(*(at + np.array(spec["offset_in"])), name, fontsize=fs, ha=spec["ha"],
                    va="center")
        t.set_gid(f"label_{name}")
    if saddle is not None:
        # the saddle line's name, above it and along it
        sa = sp["saddle"]
        S = project(fig, ax, saddle)
        c = _arclen(S)
        i = min(int(np.searchsorted(c, sa["label_at"] * c[-1])), len(S) - 2)
        u = (S[i + 1] - S[i]) * (1.0 if S[-1, 0] >= S[0, 0] else -1.0)
        u /= np.linalg.norm(u)
        at = S[i] + sa["label_offset_in"] * np.array([-u[1], u[0]])
        t = ov.text(*at, sa["label"], fontsize=fs, ha="center", va="center",
                    rotation=float(np.degrees(np.arctan2(u[1], u[0]))), rotation_mode="anchor")
        t.set_gid("label_saddle")
    return ov


def draw_surface(fig, sc, ridge, cfg, variant):
    pa = cfg["panel_a"]
    sp = pa["surface"]
    S, Lm, Z = surface_mesh(sc, ridge, sp)
    W, H = fig.get_size_inches()
    l, b, w, h = sp["axes_rect_in"]          # the right column stays free for the rods
    ax = fig.add_axes([l / W, b / H, w / W, h / H], projection="3d")
    ax.set_proj_type("ortho")
    ax.computed_zorder = False
    ax.view_init(elev=sp["elev"], azim=sp["azim"])
    ax.set_box_aspect(sp["box_aspect"], zoom=sp["zoom"])
    box = (float(S.min()), float(S.max()), float(Z.min()), float(Z.max()))
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(0.0, float(Lm[-1, 0]))
    ax.set_zlim(box[2], box[3])
    fc = style.lambda_color(0.5 * (Lm[:-1, :-1] + Lm[1:, 1:]))
    # light shading by the slope keeps the shape readable under the lambda colors
    dz = np.abs(np.array([np.gradient(z, s) for z, s in zip(Z, S)]))
    shade = 1.0 - 0.3 * np.clip(dz[:-1, :-1] / (np.percentile(dz, 90) + 1e-12), 0, 1)
    fc[..., :3] *= shade[..., None]
    surf = ax.plot_surface(S, Lm, Z, facecolors=fc, rstride=1, cstride=1,
                           linewidth=0, antialiased=False, shade=False, rasterized=True, zorder=1)
    surf.set_gid("surface")
    p = sc["path"]                           # the path between the balls: dashed links
    balls = surface_balls(sc)                # on the overlay
    for kk, tag in zip(balls, lam_tags([sc["lam"][k] for k in balls])):
        (b,) = ax.plot([p["s"][kk]], [sc["lam"][kk]], [p["E"][kk]], ls="none", marker="o",
                       ms=pa["ball_size_pt"], mfc=style.lambda_color(sc["lam"][kk]), mec=style.BALL_RING,
                       mew=style.BALL_RING_LW, zorder=5)
        b.set_gid(f"ball_lam_{tag}")
    saddle = None
    if ridge.saddle is not None and len(ridge.saddle) > 1:
        # saddle branch in (s, lambda, E) (undo the ridgeline offset): the barrier
        # between the two valleys, up to the fold. Dark dashes on a soft white halo
        # read on the dark front of the surface and on the light back alike
        sd = ridge.saddle
        saddle = np.column_stack([sd[:, 0], sd[:, 2], sd[:, 1] - ridge.c * sd[:, 2]])
        sa = sp["saddle"]
        (halo,) = ax.plot(*saddle.T, color="white", lw=sa["halo_pt"], alpha=sa["halo_alpha"],
                          zorder=2.9, solid_capstyle="round")
        halo.set_gid("saddle_halo")
        (line,) = ax.plot(*saddle.T, color=style.INK, lw=sa["lw_pt"], dashes=sa["dashes"],
                          zorder=3)
        line.set_gid("saddle_line")
    draw.unclip(ax)
    ax.set_axis_off()                        # axes are drawn by hand on the overlay
    fig.canvas.draw()
    surface_overlay(fig, ax, sc, cfg, variant, box, saddle)
    if variant == "text":
        ax.set_visible(False)                # the 3D layer only served the projection
    return ax


# ------------------------------------------------------------------- rods

def rod_states(sc, cfg):
    p, k, lam = sc["path"], sc["k_snap"], sc["lam"]
    out = []
    for key in cfg["panel_a"]["rods"]["lambdas"]:
        if key == "first":
            kk = 0
        elif key == "pre":
            kk = k - 1
        elif key == "post":
            kk = sc["k_curves"][-1]
        else:
            kk = int(round(float(key) * (len(lam) - 1)))
        out.append((kk, float(lam[kk]), p["nodes"][kk]))
    return out


def draw_rod(ax, nodes, lam, theta_r, rcfg, scale, tag, obstacle=None, y_max=None, *,
             alpha=1.0, lw=style.LW_DATA_THICK, zorder=3):
    """One rod sketch in meters on an axes whose scale is ``scale`` in/m; an
    obstacle reaching past ``y_max`` (default: the canvas top) is cut there
    (no clipping mask)."""
    in2m = 1.0 / scale
    if obstacle is not None:
        y_top = (ax.get_ylim()[1] if y_max is None else y_max) - 0.5 * style.LW_AUX * style.PT * in2m
        draw.circle_cap(ax, obstacle["center"], obstacle["radius"], y_top,
                        gid=f"obstacle_1_lam_{tag}")
    draw.polyline(ax, nodes, color=style.lambda_color(lam), lw=lw, alpha=alpha,
                  gid=f"rod_lam_{tag}", zorder=zorder)
    cl, cw = rcfg["clamp_in"]
    t_left = nodes[1] - nodes[0]
    t_right = nodes[-2] - nodes[-1]
    draw.clamp_glyph(ax, nodes[0], t_left, length=cl * in2m, width=cw * in2m,
                     gid=f"clamp_left_lam_{tag}", alpha=alpha, zorder=zorder + 2)
    draw.clamp_glyph(ax, nodes[-1], t_right, length=cl * in2m, width=cw * in2m,
                     gid=f"clamp_right_lam_{tag}", alpha=alpha, zorder=zorder + 2)


def export_rods(ex, sc, cfg, tag_suffix, placement_strip, placements_single):
    rcfg = cfg["panel_a"]["rods"]
    scale = rcfg["scale_in_per_m"]
    states = rod_states(sc, cfg)
    tags = lam_tags([s[1] for s in states])
    obstacle = sc.get("obstacle")
    W, H = rcfg["size_in"]
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ell = float(states[0][2][-1, 0] - states[0][2][0, 0])
    n = len(states)
    gap = (W / scale - n * ell) / n
    ax.set_xlim(-0.5 * gap, W / scale - 0.5 * gap)
    ax.set_ylim(-0.5 * H / scale, 0.5 * H / scale)
    pts = {}
    for j, ((kk, lam, nodes), tag) in enumerate(zip(states, tags)):
        shift = np.array([j * (ell + gap), 0.0])
        ob = None
        if obstacle is not None:
            ob = {"center": np.asarray(obstacle["center"]) + shift, "radius": obstacle["radius"]}
        draw_rod(ax, nodes + shift, lam, sc["theta"][kk], rcfg, scale, tag, ob)
        fig.canvas.draw()
        mid = nodes[len(nodes) // 2] + shift
        pts[f"rod_lam_{tag}_midpoint"] = (*data_to_inches(fig, ax, *mid), {"lambda": lam})
        pts[f"rod_lam_{tag}_right_clamp"] = (*data_to_inches(fig, ax, *(nodes[-1] + shift)),
                                              {"lambda": lam, "theta_r_rad": float(sc["theta"][kk])})
    stem = f"a_rods{tag_suffix}"
    ex.save(fig, stem, variant="graphic", placement=placement_strip,
            data_scale={"scale_in_per_m": scale, "rod_states_k": [s[0] for s in states],
                        "lambda": [s[1] for s in states]},
            notes="rod shapes before the snap, at the fold (last state before it) and after the "
                  "snap, one shared physical scale; clamps aligned with the end tangents")
    ex.annotate(stem, pts)
    sw, sh = rcfg["single_size_in"]
    for (kk, lam, nodes), tag, plc in zip(states, tags, placements_single):
        fig = plt.figure(figsize=(sw, sh))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        m = 0.5 * (sw / scale - ell)
        ax.set_xlim(-m, ell + m)
        ax.set_ylim(-0.5 * sh / scale, 0.5 * sh / scale)
        draw_rod(ax, nodes, lam, sc["theta"][kk], rcfg, scale, tag, obstacle)
        fig.canvas.draw()
        mid = nodes[len(nodes) // 2]
        p = {"midpoint": (*data_to_inches(fig, ax, *mid), {"lambda": lam}),
             "left_clamp": (*data_to_inches(fig, ax, *nodes[0]), {}),
             "right_clamp": (*data_to_inches(fig, ax, *nodes[-1]),
                             {"theta_r_rad": float(sc["theta"][kk])})}
        stem = f"a_rod{tag_suffix}_lam_{tag}"
        ex.save(fig, stem, variant="graphic", placement=plc,
                data_scale={"scale_in_per_m": scale, "k": kk, "lambda": lam,
                            "clamp_line_y_in": sh / 2,
                            "left_clamp_x_in": round(m * scale, 4)},
                notes="single rod sketch, same scale as a_rods")
        ex.annotate(stem, p)


def _rod_outline(nodes, rcfg, scale):
    """Points [in] outlining one rod sketch, relative to its left end: the rod
    (densified) and the corners of its two clamps."""
    P = (np.asarray(nodes) - nodes[0]) * scale
    f = np.linspace(0.0, 1.0, 6)[1:, None]
    pts = [P[:1]] + [P[i] + f * (P[i + 1] - P[i]) for i in range(len(P) - 1)]
    cl, cw = rcfg["clamp_in"]
    for q, t in ((P[0], P[1] - P[0]), (P[-1], P[-2] - P[-1])):
        t = t / np.linalg.norm(t)
        n = np.array([-t[1], t[0]])
        b = q - cl * t
        pts.append(np.array([q + 0.5 * cw * n, b + 0.5 * cw * n, b - 0.5 * cw * n, q - 0.5 * cw * n]))
    return np.vstack(pts)


def _overlap(lower, upper, bin_in=0.01):
    """How far the outline ``lower`` reaches above ``upper`` (same clamp line),
    over the x range they share."""
    x0 = min(lower[:, 0].min(), upper[:, 0].min())

    def env(P, pick):
        b = np.floor((P[:, 0] - x0) / bin_in).astype(int)
        out = {}
        for i, y in zip(b, P[:, 1]):
            out[i] = pick(out[i], y) if i in out else y
        return out

    top, bot = env(lower, max), env(upper, min)
    return max((top[i] - bot[i] for i in top.keys() & bot.keys()), default=0.0)


def export_rod_column(ex, sc, cfg, tag_suffix, alt):
    """The process as a stack of rod sketches (a flipbook), in the order of the
    surface balls: start, the states between, snap (the last state before the
    jump), the first state after it, end. The named states are solid and
    labeled, joined by arrows; the frames between them are thin and translucent.
    Frames sit ``pitch_in`` apart, or more where the shape flips, so that
    neighbors never touch. Same scale as the ``a_rod_lam_*`` singles."""
    rcfg = cfg["panel_a"]["rods"]
    col = rcfg["column"]
    scale = rcfg["scale_in_per_m"]
    p, lam, ksn = sc["path"], sc["lam"], sc["k_snap"]
    ks = surface_balls(sc)
    names = {ks[0]: "start", ksn - 1: "snap", ks[-1]: "end"}
    tags = lam_tags([lam[k] for k in ks])
    W, H = col["size_in"]
    up = col["direction"] == "up"
    outlines = [_rod_outline(p["nodes"][k], rcfg, scale) for k in ks]
    ys = [col["base_in"] if up else H - col["base_in"]]
    for a, b in zip(outlines[:-1], outlines[1:]):
        need = _overlap(a, b) if up else _overlap(b, a)
        ys.append(ys[-1] + (1 if up else -1) * max(col["pitch_in"], need + col["clearance_in"]))
    ys = np.array(ys)
    if col["fill_margin_in"] is not None:
        # stretch the spacing so the stack spans the canvas height
        if up:
            room = H - col["fill_margin_in"] - outlines[-1][:, 1].max() - ys[0]
        else:
            room = ys[0] - col["fill_margin_in"] + outlines[-1][:, 1].min()
        f = room / abs(ys[-1] - ys[0])
        if f > 1.0:
            ys = ys[0] + f * (ys - ys[0])
    lo = min(y + o[:, 1].min() for y, o in zip(ys, outlines))
    hi = max(y + o[:, 1].max() for y, o in zip(ys, outlines))
    if lo < 0.0 or hi > H:
        raise ValueError(f"rod column: frames span {lo:.3f}..{hi:.3f} in, canvas height {H} in "
                         "(lower pitch_in / clearance_in or base_in)")
    obstacle = sc.get("obstacle")
    key = [j for j, k in enumerate(ks) if k in names]
    for variant in ("bare", "label"):
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, W / scale)
        ax.set_ylim(0, H / scale)
        pts = {}
        for j, (k, tag) in enumerate(zip(ks, tags)):
            nodes = p["nodes"][k]
            shift = np.array([col["rod_left_in"], ys[j]]) / scale - nodes[0]
            named = k in names
            ob, y_max = None, None
            if obstacle is not None and named:
                # cut just below the frame above, over the obstacle's width
                ob = {"center": np.asarray(obstacle["center"]) + shift, "radius": obstacle["radius"]}
                cx = (ob["center"][0] * scale) - col["rod_left_in"]
                r_in = obstacle["radius"] * scale
                above = [i for i in range(len(ks)) if ys[i] > ys[j]]
                y_max = H
                if above:
                    i = min(above, key=lambda i: ys[i])
                    o = outlines[i]
                    near = np.abs(o[:, 0] - cx) <= r_in
                    y_max = ys[i] + o[near, 1].min() - col["clearance_in"]
                y_max /= scale
            if variant == "bare":
                draw_rod(ax, nodes + shift, lam[k], sc["theta"][k], rcfg, scale, tag, ob,
                         y_max=y_max, alpha=1.0 if named else col["ghost_alpha"],
                         lw=style.LW_DATA_THICK if named else col["ghost_lw_pt"],
                         zorder=3 if named else 2)
            pts[f"rod_lam_{tag}_left_end"] = (col["rod_left_in"], ys[j],
                                              {"lambda": float(lam[k]), "k": int(k),
                                               "name": names.get(k)})
            if variant == "label" and named:
                t = ax.text(col["label_x_in"] / scale, ys[j] / scale, names[k], ha="center",
                            va="center", fontsize=style.FONT_SIZE)
                t.set_gid(f"label_{names[k]}")
        for ja, jb in (zip(key[:-1], key[1:]) if variant == "bare" else ()):
            d = np.sign(ys[jb] - ys[ja])            # up or down the column
            a = np.array([col["label_x_in"], ys[ja] + d * col["arrow_gap_in"]]) / scale
            b = np.array([col["label_x_in"], ys[jb] - d * col["arrow_gap_in"]]) / scale
            draw.arrow(ax, [a, b], head_in=col["head_in"],
                       gid=f"arrow_{names[ks[ja]]}_{names[ks[jb]]}")
        stem = f"a_rods{tag_suffix}_column_{variant}"
        px, py = col["placement_in"]
        ex.save(fig, stem, variant=variant,
                placement={"panel": "a", "x_in": px, "y_in": py, "anchor": "lower-left",
                           "alternative": alt},
                data_scale={"scale_in_per_m": scale, "frames_k": [int(k) for k in ks],
                            "lambda": [float(lam[k]) for k in ks],
                            "names": {tags[j]: names[ks[j]] for j in key},
                            "clamp_line_y_in": [round(float(y), 4) for y in ys],
                            "left_rod_end_x_in": col["rod_left_in"]},
                notes="the process as a stack of rods (" + ("bottom to top" if up else "top down") +
                      ") at the lambdas of the surface balls; start, snap, end solid, the frames "
                      "between translucent" + ("; the names only, on the same canvas as the bare "
                                               "variant" if variant == "label"
                                               else "; with the arrows"))
        ex.annotate(stem, pts)


# --------------------------------------------------------------- colorbar

def _bar_patches(ax, x0, y0, bw, bh, n, horizontal):
    """The gradient as ``n`` vector rectangles (overlapping slightly: no seams)."""
    edges = np.linspace(0.0, 1.0, n + 1)
    for i in range(n):
        lam = 0.5 * (edges[i] + edges[i + 1])
        d = edges[i + 1] - edges[i] + (0.25 / n if i < n - 1 else 0.0)
        if horizontal:
            r = Rectangle((x0 + edges[i] * bw, y0), d * bw, bh, fc=style.lambda_color(lam),
                          ec="none", clip_on=False)
        else:
            r = Rectangle((x0, y0 + edges[i] * bh), bw, d * bh, fc=style.lambda_color(lam),
                          ec="none", clip_on=False)
        r.set_gid(f"colorbar_patch_{i:03d}")
        ax.add_patch(r)


def export_colorbars(ex, cfg):
    """Horizontal bar with inline end labels ("0 [bar] 1 lambda", fits the gap
    between the panels) and a vertical bar with side labels."""
    cb = cfg["panel_a"]["colorbar"]
    n = cb["n_patches"]
    specs = {"horizontal": dict(bar=cb["horizontal_bar_in"], canvas=tuple(cb["horizontal_canvas_in"]),
                                origin=(0.12, 0.04)),
             "vertical": dict(bar=cb["vertical_bar_in"], canvas=tuple(cb["vertical_canvas_in"]),
                              origin=(0.06, 0.08))}
    for orient, sp in specs.items():
        bw, bh = sp["bar"]
        W, H = sp["canvas"]
        x0, y0 = sp["origin"]
        for variant in ("bare", "axes"):
            fig = plt.figure(figsize=(W, H))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            if variant == "bare":
                _bar_patches(ax, x0, y0, bw, bh, n, orient == "horizontal")
                if orient == "vertical":
                    for v, lab in ((0.0, "0"), (1.0, "1")):
                        y = y0 + v * bh
                        draw.polyline(ax, [[x0 + bw, y], [x0 + bw + 0.03, y]], color=style.INK,
                                      lw=style.LW_AUX, gid=f"colorbar_tick_{lab}")
            if variant == "axes":
                if orient == "horizontal":
                    ym = y0 + 0.5 * bh
                    t0 = ax.text(x0 - 0.03, ym, "0", ha="right", va="center", fontsize=style.FONT_SIZE)
                    t1 = ax.text(x0 + bw + 0.03, ym, "1", ha="left", va="center",
                                 fontsize=style.FONT_SIZE)
                    tl = ax.text(x0 + bw + 0.13, ym, r"$\lambda$", ha="left", va="center",
                                 fontsize=style.LABEL_SIZE)
                else:
                    t0 = ax.text(x0 + bw + 0.065, y0, "0", ha="left", va="center",
                                 fontsize=style.FONT_SIZE)
                    t1 = ax.text(x0 + bw + 0.065, y0 + bh, "1", ha="left", va="center",
                                 fontsize=style.FONT_SIZE)
                    tl = ax.text(x0 + 0.5 * bw, y0 + bh + 0.04, r"$\lambda$", ha="center",
                                 va="bottom", fontsize=style.LABEL_SIZE)
                t0.set_gid("colorbar_ticklabel_0")
                t1.set_gid("colorbar_ticklabel_1")
                tl.set_gid("colorbar_label")
            stem = f"colorbar_lambda_{orient}" + ("_bare" if variant == "bare" else "")
            plc = None
            if orient == "horizontal":
                px, py = cb["horizontal_placement_in"]
                plc = {"panel": "a/b", "x_in": px, "y_in": py, "anchor": "lower-left"}
            ex.save(fig, stem, variant="bare" if variant == "bare" else "axes", placement=plc,
                    data_scale={"bar_in": [bw, bh], "bar_origin_in": [x0, y0],
                                "lambda_range": [0, 1], "n_patches": n},
                    notes=f"{orient} lambda colorbar; the bar itself is {bw} x {bh} in at "
                          f"{[x0, y0]} in the canvas")


# ------------------------------------------------------------------ driver

def ridge_scale(fig, ax, ridge):
    """Data scale of a ridgeline component (inches per unit)."""
    pos = ax.get_position()
    W, H = fig.get_size_inches()
    return {"x": "s/L", "y": "E/(EI/L) + offset_per_lambda * lambda",
            "in_per_s_over_L": round(pos.width * W / (ridge.xlim[1] - ridge.xlim[0]), 5),
            "in_per_EI_over_L": round(pos.height * H / (ridge.ylim[1] - ridge.ylim[0]), 5),
            "offset_per_lambda_EI_over_L": round(ridge.c, 5),
            "data_area_in": [round(pos.x0 * W, 4), round(pos.y0 * H, 4),
                             round(pos.width * W, 4), round(pos.height * H, 4)],
            "xlim": list(ridge.xlim), "ylim": [round(v, 5) for v in ridge.ylim],
            "energy_clip_EI_over_L": round(ridge.clip, 5)}


def ridge_annotations(fig, ax, ridge, sc):
    lam = sc["lam"]
    pts = {}
    for (s, y, l), tag in zip(ridge.balls, ridge.tags):
        pts[f"ball_lam_{tag}"] = (*data_to_inches(fig, ax, s, y), {"lambda": l, "s_over_L": s})
    (s0, y0), (s1, y1) = ridge.jump
    k = sc["k_snap"]
    pts["snap_jump_start"] = (*data_to_inches(fig, ax, s0, y0),
                              {"lambda": float(lam[k - 1]), "s_over_L": float(s0)})
    pts["snap_jump_end"] = (*data_to_inches(fig, ax, s1, y1),
                            {"lambda": float(lam[k]), "s_over_L": float(s1)})
    tc = sc["theta_c"]
    if ridge.saddle is not None:
        sd = ridge.saddle
        pts["fold"] = (*data_to_inches(fig, ax, sd[-1, 0], sd[-1, 1]),
                       {"lambda": float(sd[-1, 2]), "s_over_L": float(sd[-1, 0]),
                        "theta_c_rad": tc["theta_c"],
                        "note": "upper well and saddle merge here (theta = theta_c)"})
        pts["barrier_lambda_0"] = (*data_to_inches(fig, ax, sd[0, 0], sd[0, 1]),
                                   {"lambda": float(sd[0, 2]), "s_over_L": float(sd[0, 0])})
    return pts


def surface_annotations(fig, ax, sc, ridge):
    p, lam, k = sc["path"], sc["lam"], sc["k_snap"]
    pts = {}
    balls = surface_balls(sc)
    for kk, tag in zip(balls, lam_tags([lam[b] for b in balls])):
        pts[f"ball_lam_{tag}"] = (*data_to_inches(fig, ax, p["s"][kk], lam[kk], p["E"][kk]),
                                  {"lambda": float(lam[kk])})
    pts["snap_jump_start"] = (*data_to_inches(fig, ax, p["s"][k - 1], lam[k - 1], p["E"][k - 1]),
                              {"lambda": float(lam[k - 1])})
    pts["snap_jump_end"] = (*data_to_inches(fig, ax, p["s"][k], lam[k], p["E"][k]),
                            {"lambda": float(lam[k])})
    tc = sc["theta_c"]
    if "E_at_fold" in tc:
        lf = tc["theta_c"] / sc["theta_max"]
        pts["fold"] = (*data_to_inches(fig, ax, tc["s_at_fold"], lf, tc["E_at_fold"]),
                       {"lambda": float(lf), "theta_c_rad": tc["theta_c"]})
    return pts


def print_report(summary):
    for name in ("free", "obstacle"):
        if name not in summary:
            continue
        s = summary[name]
        print(f"\n[{name}]")
        print(f"  rotation {s['rotation_direction']}; theta_c = {s['theta_c_rad']:.6f} rad "
              f"({s['theta_c_deg']:.3f} deg); theta_max = {s['theta_max_rad']:.6f} rad")
        print(f"  lambda_snap = {s['lambda_snap']:.3f} (step {s['k_snap']}; fold at "
              f"lambda = theta_c/theta_max = {s['lambda_fold_from_theta_c']:.4f}"
              + (f", kappa fit {s['lambda_fold_from_kappa_fit']:.4f}" if 'lambda_fold_from_kappa_fit' in s else "")
              + ")")
        print(f"  snap: s/L {s['s_before_after'][0]:+.4f} -> {s['s_before_after'][1]:+.4f}, "
              f"E {s['E_before_after'][0]:.3f} -> {s['E_before_after'][1]:.3f} EI/L")
        print(f"  Newton iterations at the snap {s['newton_iterations_at_snap']} "
              f"({s['snap_attempts']}; first attempt {s['newton_first_attempt_at_snap']}) vs median "
              f"{s['newton_iterations_median']:.0f}, max elsewhere {s['newton_iterations_max_elsewhere']}")
        print(f"  lowest Hessian eigenvalue: {s['min_eig_at_lambda0']:.3f} at lambda 0, "
              f"{s['min_eig_before_snap']:.4f} just before the snap, {s['min_eig_at_theta_c']:.2e} at theta_c-")
        print(f"  landscapes: {s['n_points']} constrained states, {s['n_failed_points']} failed (dropped); "
              f"all minima matched to equilibria: {s['all_minima_match']}; upper well last seen at "
              f"lambda {s['upper_well_last_lambda']}, gone at {s['upper_well_gone_lambda']}")
        for v in s["validation"]:
            mins = ", ".join(f"{m['s']:+.3f}{'*' if m['is_well_point'] else ''}"
                             f"({'ok' if m['match'] else 'NO'})" for m in v["minima"])
            maxs = ", ".join(f"{m['s']:+.3f}({'saddle' if m['match'] else 'kink' if not m.get('smooth', True) else 'NO'})"
                             for m in v["maxima"])
            print(f"    lambda {v['lambda']:.3f}: minima [{mins}]  maxima [{maxs}]  failed {v['n_failed']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="recompute instead of using the cache")
    ap.add_argument("--workers", type=int, default=None, help="processes for the landscape sweeps")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    style.apply()
    res = load_or_compute(cfg, args.force, args.workers)
    ex = Exporter(cfg["figure"]["name"], cfg["figure"]["size_in"])
    for prefix in ("a_", "colorbar_"):          # this step's outputs, written afresh
        ex.remove(prefix)
    ddir = ex.dirs["data"]
    pa = cfg["panel_a"]
    W, H = pa["size_in"]
    x0, y0 = pa["placement_in"]
    summary = {"cache_hash": res["_hash"], "settings": res["_spec"],
               "units": {"s": "L", "E": "EI/L", "theta": "rad"}}
    rc = pa["rods"]
    for name, suffix in (("free", ""), ("obstacle", "_obstacle")):
        if name not in res:
            continue
        sc = res[name]
        print(f"\npanel (a), scenario '{name}'")
        summary[name] = summarize(sc, cfg)
        write_data(sc, suffix, ddir)
        ridge = Ridge(sc, cfg)
        plc = {"panel": "a", "x_in": x0, "y_in": y0, "anchor": "lower-left"}
        alt = name != "free"
        for variant in ("bare", "axes"):
            fig = plt.figure(figsize=(W, H))
            ax = draw_ridgeline(fig, ridge, cfg, variant)
            fig.canvas.draw()
            pts = ridge_annotations(fig, ax, ridge, sc)
            stem = f"a_ridgeline{suffix}_{variant}"
            ex.save(fig, stem, variant=variant, placement=dict(plc, alternative=True),
                    data_scale=ridge_scale(fig, ax, ridge),
                    notes="E(s; lambda) stacked by lambda; balls = realized equilibria; colored "
                          "path = valley floor (split at the snap); dashed = local maxima (saddles)")
            ex.annotate(stem, pts, note="balls, fold, snap jump start/end (draw the jump arrow here)")
        fig = plt.figure(figsize=(W, H))
        ax = draw_ridgeline(fig, ridge, cfg, "solid")
        ex.save(fig, f"a_ridgeline{suffix}_path_solid", variant="graphic",
                placement=dict(plc, alternative=True), data_scale=ridge_scale(fig, ax, ridge),
                notes="valley-floor path as two single-color polylines (before / after the snap) "
                      "for a gradient stroke; lambda runs 0 -> lambda_pre on the first and "
                      "lambda_snap -> lambda_post on the second",
                extra={"lambda_ranges": [[float(ridge.pre[1][0]), float(ridge.pre[1][-1])],
                                         [float(ridge.post[1][0]), float(ridge.post[1][-1])]]})
        for variant in ("bare", "axes", "text"):
            fig = plt.figure(figsize=(W, H))
            ax = draw_surface(fig, sc, ridge, cfg, variant)
            fig.canvas.draw()
            pts = surface_annotations(fig, ax, sc, ridge)
            stem = f"a_surface{suffix}_{variant}"
            ex.save(fig, stem, variant={"bare": "bare", "axes": "graphic", "text": "label"}[variant],
                    placement=dict(plc, alternative=alt),
                    data_scale={"projection": "orthographic", **pa["surface"],
                                "energy_clip_EI_over_L": round(ridge.clip, 5)},
                    notes="E(s, lambda) surface (rasterized); balls, dashed links along the "
                          "valley floor, process arrows and the saddle line (barrier, up to "
                          "the fold) as vectors; _axes adds the axis lines and ticks, _text is "
                          "all the text (same canvas, no graphics): stack it on _bare")
            ex.annotate(stem, pts)
        singles = [{"panel": "a", "x_in": rc["column_x_in"], "y_in": y, "anchor": "lower-left",
                    "alternative": True} for y in rc["column_y_in"]]
        export_rods(ex, sc, cfg, suffix,
                    {"panel": "a", "x_in": rc["placement_in"][0], "y_in": rc["placement_in"][1],
                     "anchor": "lower-left", "alternative": True,
                     "note": "strip variant; the main layout uses the a_rods_column"},
                    singles)
        export_rod_column(ex, sc, cfg, suffix, alt)
    export_colorbars(ex, cfg)
    (ddir / "landscape_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print_report(summary)
    print(f"\n{len(ex.saved)} components written to {rel(ex.dirs['components'])}; "
          f"data in {rel(ddir)}")


if __name__ == "__main__":
    main()
