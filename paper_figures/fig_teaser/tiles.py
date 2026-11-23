"""Panel (b): four square task tiles rendered from standard rollouts.

    python -m paper_figures.fig01.tiles                    # data/rollouts/<task>/, placeholders if missing
    python -m paper_figures.fig01.tiles --provisional      # also build + render *_provisional stand-ins
    python -m paper_figures.fig01.tiles --provisional --force   # rebuild the stand-ins

Each task in ``panel_b.tasks`` is read from ``data/rollouts/<task>/`` (format:
:mod:`paper_figures.common.rollout_io`). A missing task becomes a placeholder
tile (dashed light gray border, task name centered). Every tile is exported as
``b_tile_<name>_bare`` (graphics only) and ``b_tile_<name>_label`` (the task
name only, on the same canvas: stack it on the bare tile), at 0.82 x 0.82 in.

Rendering: 2D rollouts are drawn in the plane, 3D ones with an orthographic
projection (no axes, panes or grid). A few snapshots in the lambda colors
with increasing opacity, the final shape thicker, the gripper path as a
lambda-colored line, obstacles in gray, contacting edges of the final shape in
the accent color, a gripper glyph at the final pose and the fixed clamp.
"""
from __future__ import annotations

import paper_figures  # noqa: F401  (src/ on sys.path, BLAS threads)

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
import yaml                                                      # noqa: E402
from matplotlib.collections import LineCollection               # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection          # noqa: E402

from paper_figures import style                                  # noqa: E402
from paper_figures.common import draw, rollout_io                # noqa: E402
from paper_figures.common.export import Exporter, data_to_inches  # noqa: E402
from paper_figures.common.paths import ROLLOUTS_DIR, rel         # noqa: E402

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_cfg(path=CONFIG_PATH):
    return yaml.safe_load(Path(path).read_text())


def slot_placement(cfg, slot):
    W = cfg["figure"]["size_in"][0]
    t = cfg["panel_b"]["tile_in"]
    n = len(cfg["panel_b"]["tasks"])
    gap = (W - n * t) / (n - 1)
    return {"panel": "b", "slot": slot, "x_in": round(slot * (t + gap), 4),
            "y_in": cfg["panel_b"]["y_in"], "anchor": "lower-left"}


def snapshot_steps(M, n):
    """``n`` snapshot steps plus the final one, evenly spread."""
    idx = np.unique(np.linspace(0, M - 1, n + 1).round().astype(int))
    return idx[:-1], int(idx[-1])


# --------------------------------------------------------------------- 2D

def _bounds_2d(r, scene):
    pts = [r["nodes"].reshape(-1, 2), r["gripper_pose"][:, :2]]
    for ob in scene.get("obstacles", []):
        if ob["type"] == "circle":
            c, rad = np.asarray(ob["center"]), ob["radius"]
            pts.append(np.array([c - rad, c + rad]))
        elif ob["type"] == "pipe":
            pts.append(draw.pipe_wall_polygon(ob, ob.get("wall", 0.02)))
    pts = np.vstack(pts)
    return pts.min(axis=0), pts.max(axis=0)


def render_2d(fig, r, scene, tcfg):
    tile = fig.get_size_inches()[0]
    lo, hi = _bounds_2d(r, scene)
    span = float(np.max(hi - lo))
    half = 0.5 * span / (1.0 - 2.0 * tcfg["margin_in"] / tile)
    c = 0.5 * (lo + hi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect("equal", adjustable="box")
    scale = tile / (2 * half)                     # inches per meter
    in2m = 1.0 / scale

    for i, ob in enumerate(scene.get("obstacles", [])):
        if ob["type"] == "circle":
            draw.circle_obstacle(ax, ob["center"], ob["radius"], gid=f"obstacle_{i + 1}")
        elif ob["type"] == "pipe":
            draw.pipe_obstacle(ax, ob, wall=ob.get("wall", 0.02), gid=f"obstacle_{i + 1}")

    nodes, lam = r["nodes"], r["lam"]
    snaps, last = snapshot_steps(len(lam), tcfg["n_snapshots"])
    a0, a1 = tcfg["snapshot_alpha"]
    for j, k in enumerate(snaps):
        a = a0 + (a1 - a0) * j / max(len(snaps) - 1, 1)
        draw.polyline(ax, nodes[k], color=style.lambda_color(lam[k]), lw=style.LW_DATA_THIN,
                      alpha=a, gid=f"rod_lam_{lam[k]:.2f}", zorder=3)
    g = r["gripper_pose"]
    draw.lambda_path(ax, g[:, :2], lam, lw=style.LW_DATA_THIN, gid="gripper_path", zorder=2)
    draw.polyline(ax, nodes[last], color=style.lambda_color(lam[last]), lw=style.LW_DATA_THICK,
                  gid=f"rod_final_lam_{lam[last]:.2f}", zorder=4)
    if "contact_mask" in r and r["contact_mask"][last].any():
        e = np.nonzero(r["contact_mask"][last])[0]
        lc = LineCollection([nodes[last, k:k + 2] for k in e], colors=style.ACCENT,
                            linewidths=style.LW_DATA_THICK, capstyle="round", zorder=5,
                            clip_on=False)
        lc.set_gid("contact_edges")
        ax.add_collection(lc)
    fe = scene.get("fixed_end")
    if fe:
        draw.clamp_glyph(ax, fe["position"], fe["tangent"], length=0.035 * in2m,
                         width=0.07 * in2m, gid="fixed_clamp", zorder=6)
    end = nodes[last, -1] if scene.get("gripper_end", "last") == "last" else nodes[last, 0]
    th = g[last, 2]
    into_rod = -np.array([np.cos(th), np.sin(th)])
    draw.gripper_glyph(ax, end, into_rod, size=tcfg["gripper_in"] * in2m, gid="gripper")
    fig.canvas.draw()
    pts = {"gripper_final": (*data_to_inches(fig, ax, *end), {"data_m": [float(v) for v in end]}),
           "gripper_start": (*data_to_inches(fig, ax, *g[0, :2]),
                             {"data_m": [float(v) for v in g[0, :2]]})}
    return ax, {"scale_in_per_m": round(scale, 5), "view_m": {
        "xlim": [float(c[0] - half), float(c[0] + half)],
        "ylim": [float(c[1] - half), float(c[1] + half)]}}, pts


# --------------------------------------------------------------------- 3D

def _gripper_lines_3d(p, R, size):
    """Two-jaw glyph: U opening toward the rod (-x of the gripper frame), jaws along y."""
    t, n = -R[:, 0], R[:, 1]
    a, b, c, w = 0.55 * size, 0.55 * size, 0.6 * size, 0.5 * size

    def P(u, v):
        return p + u * t + v * n

    return [np.array([P(b, w), P(-a, w), P(-a, -w), P(b, -w)]),
            np.array([P(-a, 0.0), P(-a - c, 0.0)])]


def _obstacle_3d(ax, ob, gid):
    kw = dict(color=style.OBSTACLE_FILL, edgecolor="none", linewidth=0, shade=True,
              antialiased=True, rasterized=False)
    if ob["type"] == "sphere":
        X, Y, Z = draw.sphere_mesh(ob["center"], ob["radius"])
        surfs = [ax.plot_surface(X, Y, Z, **kw)]
    elif ob["type"] in ("cylinder", "capsule"):
        X, Y, Z = draw.cylinder_mesh(ob["p0"], ob["p1"], ob["radius"])
        surfs = [ax.plot_surface(X, Y, Z, **kw)]
        if ob["type"] == "capsule":
            for end in ("p0", "p1"):
                surfs.append(ax.plot_surface(*draw.sphere_mesh(ob[end], ob["radius"]), **kw))
    elif ob["type"] == "plane":
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        p, nrm = np.asarray(ob["point"], float), np.asarray(ob["normal"], float)
        nrm /= np.linalg.norm(nrm)
        e1 = np.cross(nrm, [1.0, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nrm, e1)
        a, b = ob.get("size", [0.5, 0.5])
        quad = [p + sa * a / 2 * e1 + sb * b / 2 * e2 for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        pc = Poly3DCollection([quad], facecolor=style.OBSTACLE_FILL, edgecolor=style.OBSTACLE_EDGE,
                              linewidth=style.LW_AUX)
        ax.add_collection3d(pc, autolim=False)
        surfs = [pc]
    else:
        return []
    for s in surfs:
        s.set_gid(gid)
    return surfs


def _obstacle_points(ob):
    if ob["type"] == "sphere":
        c, r = np.asarray(ob["center"], float), ob["radius"]
        return np.array([c - r, c + r])
    if ob["type"] in ("cylinder", "capsule"):
        p0, p1, r = np.asarray(ob["p0"], float), np.asarray(ob["p1"], float), ob["radius"]
        return np.array([p0 - r, p0 + r, p1 - r, p1 + r])
    if ob["type"] == "plane":
        return np.asarray(ob["point"], float)[None]
    return np.zeros((0, 3))


def render_3d(fig, r, scene, tcfg):
    tile = fig.get_size_inches()[0]
    nodes, lam, g = r["nodes"], r["lam"], r["gripper_pose"]
    pts = np.vstack([nodes.reshape(-1, 3), g[:, :3]]
                    + [_obstacle_points(ob) for ob in scene.get("obstacles", [])])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.04 * float(np.max(hi - lo))
    lo, hi = lo - pad, hi + pad
    cam = scene.get("camera", {})
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.computed_zorder = False
    ax.view_init(elev=cam.get("elev", 20.0), azim=cam.get("azim", -60.0))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo, zoom=1.0)

    for i, ob in enumerate(scene.get("obstacles", [])):
        _obstacle_3d(ax, ob, f"obstacle_{i + 1}")
    snaps, last = snapshot_steps(len(lam), tcfg["n_snapshots"])
    a0, a1 = tcfg["snapshot_alpha"]
    for j, k in enumerate(snaps):
        a = a0 + (a1 - a0) * j / max(len(snaps) - 1, 1)
        (ln,) = ax.plot(*nodes[k].T, color=style.lambda_color(lam[k]), lw=style.LW_DATA_THIN,
                        alpha=a, zorder=3, solid_capstyle="round", solid_joinstyle="round")
        ln.set_gid(f"rod_lam_{lam[k]:.2f}")
    pieces, lams = draw.lambda_pieces(g[:, :3], lam)
    lc = Line3DCollection(pieces, colors=style.lambda_color(lams), linewidths=style.LW_DATA_THIN,
                          capstyle="round", joinstyle="round", zorder=2)
    lc.set_gid("gripper_path")
    ax.add_collection3d(lc, autolim=False)
    (fin,) = ax.plot(*nodes[last].T, color=style.lambda_color(lam[last]), lw=style.LW_DATA_THICK,
                     zorder=4, solid_capstyle="round", solid_joinstyle="round")
    fin.set_gid(f"rod_final_lam_{lam[last]:.2f}")
    if "contact_mask" in r and r["contact_mask"][last].any():
        e = np.nonzero(r["contact_mask"][last])[0]
        cl = Line3DCollection([nodes[last, k:k + 2] for k in e], colors=style.ACCENT,
                              linewidths=style.LW_DATA_THICK, capstyle="round", zorder=5)
        cl.set_gid("contact_edges")
        ax.add_collection3d(cl, autolim=False)

    # fit: scale the box so the projected content fills the tile, then center it
    def projected_bbox():
        fig.canvas.draw()
        xy = np.array([data_to_inches(fig, ax, *p) for p in pts])
        return xy.min(axis=0), xy.max(axis=0)

    m = tcfg["margin_in"]
    zoom = 1.0
    for _ in range(3):
        b0, b1 = projected_bbox()
        zoom *= (tile - 2 * m) / float(np.max(b1 - b0))
        ax.set_box_aspect(hi - lo, zoom=zoom)
    b0, b1 = projected_bbox()
    shift = (tile / 2 - 0.5 * (b0 + b1)) / tile
    ax.set_position([shift[0], shift[1], 1, 1])
    fig.canvas.draw()

    # length scale of the orthographic view: a meter in the view plane, in inches
    p0 = 0.5 * (lo + hi)
    ex = np.cross([0, 0, 1.0], [np.cos(np.radians(cam.get("azim", -60.0))),
                                np.sin(np.radians(cam.get("azim", -60.0))), 0.0])
    ex = ex / np.linalg.norm(ex) * 0.1
    a_in, b_in = np.array(data_to_inches(fig, ax, *p0)), np.array(data_to_inches(fig, ax, *(p0 + ex)))
    scale = float(np.linalg.norm(b_in - a_in) / 0.1)
    in2m = 1.0 / scale

    fe = scene.get("fixed_end")
    if fe:
        p, t = np.asarray(fe["position"], float), np.asarray(fe["tangent"], float)
        (cl,) = ax.plot(*np.array([p, p - 0.035 * in2m * t]).T, color=style.CLAMP,
                        lw=0.07 * 72, solid_capstyle="butt", zorder=6)
        cl.set_gid("fixed_clamp")
    R = rollout_io.matrix_from_quat(g[last, 3:])
    end = nodes[last, -1] if scene.get("gripper_end", "last") == "last" else nodes[last, 0]
    for k, xyz in enumerate(_gripper_lines_3d(end, R, tcfg["gripper_in"] * in2m)):
        (gl,) = ax.plot(*xyz.T, color=style.CLAMP, lw=style.LW_DATA_THIN, zorder=7,
                        solid_capstyle="round", solid_joinstyle="round")
        gl.set_gid(f"gripper_{k}")
    draw.unclip(ax)
    fig.canvas.draw()
    pts_out = {"gripper_final": (*data_to_inches(fig, ax, *end), {"data_m": [float(v) for v in end]}),
               "gripper_start": (*data_to_inches(fig, ax, *g[0, :3]),
                                 {"data_m": [float(v) for v in g[0, :3]]})}
    return ax, {"scale_in_per_m": round(scale, 5), "projection": "orthographic",
                "camera": {"elev": cam.get("elev", 20.0), "azim": cam.get("azim", -60.0),
                           "zoom": round(zoom, 4)}}, pts_out


# ------------------------------------------------------------------ export

def tile_label(ax_fig, text, tcfg, tile):
    m = tcfg["margin_in"] / tile
    t = ax_fig.text(m * 0.6, 1 - m * 0.6, text, ha="left", va="top", fontsize=style.FONT_SIZE,
                    color=style.INK, transform=ax_fig.transAxes)
    t.set_gid("tile_label")


def export_tile(ex, cfg, name, task, slot, task_dir, status):
    tcfg = cfg["panel_b"]
    tile = tcfg["tile_in"]
    r, scene = rollout_io.read_rollout(task_dir)
    title = tcfg["titles"].get(task, scene.get("title", task))
    for variant in ("bare", "label"):
        fig = plt.figure(figsize=(tile, tile))
        render = render_2d if scene["dim"] == 2 else render_3d
        ax, scale, pts = render(fig, r, scene, tcfg)
        if variant == "label":
            for a in fig.axes:               # text only: the drawing is the bare tile
                a.set_visible(False)
            lab = fig.add_axes([0, 0, 1, 1])
            lab.set_axis_off()
            tile_label(lab, title, tcfg, tile)
        stem = f"b_tile_{name}_{variant}"
        ex.save(fig, stem, variant=variant, placement=slot_placement(cfg, slot),
                data_scale=scale, status=status,
                notes=f"{title}: rendered from {rel(task_dir)}; {scene.get('source', '')}",
                extra={"source_rollout": rel(task_dir), "task": task})
        if variant == "bare":
            ex.annotate(stem, pts, note="gripper positions (start and final) in this tile")


def export_placeholder(ex, cfg, task, slot):
    tcfg = cfg["panel_b"]
    tile = tcfg["tile_in"]
    title = tcfg["titles"].get(task, task)
    for variant in ("bare", "label"):
        fig = plt.figure(figsize=(tile, tile))
        draw.placeholder(fig, title, with_text=variant == "label",
                         with_border=variant == "bare")
        ex.save(fig, f"b_tile_{task}_{variant}", variant=variant,
                placement=slot_placement(cfg, slot), status="placeholder",
                notes=f"{title}: no rollout in {rel(ROLLOUTS_DIR / task)} yet",
                extra={"task": task})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--provisional", action="store_true",
                    help="build and render the *_provisional stand-ins")
    ap.add_argument("--force", action="store_true", help="rebuild provisional rollouts")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    style.apply()
    ex = Exporter(cfg["figure"]["name"], cfg["figure"]["size_in"])
    tasks = cfg["panel_b"]["tasks"]
    print("panel (b) tiles")
    for slot, task in enumerate(tasks):
        task_dir = ROLLOUTS_DIR / task
        if (task_dir / "rollout.npz").exists():
            export_tile(ex, cfg, task, task, slot, task_dir, "final")
        else:
            export_placeholder(ex, cfg, task, slot)
    if args.provisional:
        from paper_figures.fig01 import provisional
        tol = cfg["panel_b"]["contact_tol"]
        for slot, task in enumerate(tasks):
            if task not in provisional.BUILDERS:
                print(f"  {task}: no provisional source (placeholder only)")
                continue
            d = provisional.BUILDERS[task](cfg["provisional"][task], tol, force=args.force)
            export_tile(ex, cfg, f"{task}_provisional", task, slot, d, "provisional")
    print(f"{len(ex.saved)} components written to {rel(ex.dirs['components'])}")


if __name__ == "__main__":
    main()
