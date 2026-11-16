"""Drawing primitives shared by every figure.

All artists are created with ``clip_on=False`` and a meaningful ``gid`` so the
SVG layers can be found in Illustrator. Sizes of glyphs (clamps, grippers)
are given in data units; callers convert from inches with their scale.
"""
from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, PathPatch, Polygon
from matplotlib.path import Path as MPath

from paper_figures import style


# ------------------------------------------------------------------- lines

def polyline(ax, xy, *, color, lw, gid, alpha=1.0, zorder=2, ls="-"):
    xy = np.asarray(xy, dtype=float)
    (line,) = ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, alpha=alpha, ls=ls,
                      zorder=zorder, solid_capstyle="round", solid_joinstyle="round",
                      clip_on=False)
    line.set_gid(gid)
    return line


def lambda_pieces(xy, lam, max_segments=150):
    """Split a polyline into at most ``max_segments`` pieces (every vertex kept),
    each with the mean ``lam`` of its vertices."""
    xy = np.asarray(xy, dtype=float)
    lam = np.asarray(lam, dtype=float)
    n_seg = len(xy) - 1
    groups = np.array_split(np.arange(n_seg), min(max_segments, n_seg))
    pieces = [xy[g[0]:g[-1] + 2] for g in groups if len(g)]
    lams = [float(lam[g[0]:g[-1] + 2].mean()) for g in groups if len(g)]
    return pieces, np.array(lams)


def lambda_path(ax, xy, lam, *, lw, gid, max_segments=150, zorder=3, alpha=1.0):
    """A polyline colored by ``lam`` along it: a LineCollection of <= ``max_segments``
    pieces with round caps, so the joints are seamless."""
    pieces, lams = lambda_pieces(xy, lam, max_segments)
    colors = style.lambda_color(lams)
    if alpha != 1.0:
        colors[:, 3] = alpha
    lc = LineCollection(pieces, colors=colors, linewidths=lw, capstyle="round",
                        joinstyle="round", zorder=zorder, clip_on=False)
    lc.set_gid(gid)
    ax.add_collection(lc)
    return lc


def ball(ax, x, y, *, color, gid, size_pt=4.2, edge=style.INK, edge_lw=style.LW_AUX,
         zorder=6):
    (m,) = ax.plot([x], [y], ls="none", marker="o", ms=size_pt, mfc=color, mec=edge,
                   mew=edge_lw, zorder=zorder, clip_on=False)
    m.set_gid(gid)
    return m


# ------------------------------------------------------------------ glyphs

def _frame(t):
    t = np.asarray(t, dtype=float)
    t = t / np.linalg.norm(t)
    return t, np.array([-t[1], t[0]])


def clamp_glyph(ax, p, t, *, length, width, gid, color=style.CLAMP, zorder=5, alpha=1.0):
    """Filled rectangle behind the rod end ``p``; ``t`` points into the rod."""
    t, n = _frame(t)
    p = np.asarray(p, dtype=float)
    back = p - length * t
    corners = [p + 0.5 * width * n, back + 0.5 * width * n,
               back - 0.5 * width * n, p - 0.5 * width * n]
    poly = Polygon(corners, closed=True, fc=color, ec="none", alpha=alpha, zorder=zorder,
                   clip_on=False)
    poly.set_gid(gid)
    ax.add_patch(poly)
    return poly


def gripper_outline(p, t, size):
    """Two-jaw gripper around the rod end ``p`` (``t`` into the rod), as polylines
    in the local frame scaled by ``size`` (jaw opening)."""
    t, n = _frame(t)
    p = np.asarray(p, dtype=float)
    a, b, c, w = 0.55 * size, 0.55 * size, 0.6 * size, 0.5 * size

    def P(u, v):
        return p + u * t + v * n

    jaws = [P(b, w), P(-a, w), P(-a, -w), P(b, -w)]     # U opening toward the rod
    stem = [P(-a, 0.0), P(-a - c, 0.0)]
    return [np.array(jaws), np.array(stem)]


def gripper_glyph(ax, p, t, *, size, gid, color=style.CLAMP, lw=style.LW_DATA_THIN,
                  zorder=7):
    lines = []
    for k, xy in enumerate(gripper_outline(p, t, size)):
        lines.append(polyline(ax, xy, color=color, lw=lw, gid=f"{gid}_{k}", zorder=zorder))
    return lines


# --------------------------------------------------------------- obstacles

def circle_obstacle(ax, center, radius, *, gid, zorder=1):
    c = Circle(center, radius, fc=style.OBSTACLE_FILL, ec=style.OBSTACLE_EDGE,
               lw=style.LW_AUX, zorder=zorder, clip_on=False)
    c.set_gid(gid)
    ax.add_patch(c)
    return c


def circle_cap(ax, center, radius, y_max, *, gid, zorder=1, n=120):
    """A circle cut at ``y = y_max`` without a clipping mask: the circular segment
    below the cut is filled, and only its arc is outlined (the cut is not an edge
    of the obstacle). Falls back to the full circle when it fits."""
    cx, cy = float(center[0]), float(center[1])
    if cy + radius <= y_max:
        return circle_obstacle(ax, center, radius, gid=gid, zorder=zorder)
    h = np.clip((y_max - cy) / radius, -1.0, 1.0)
    a0 = np.arcsin(h)                      # angle of the cut, from the +x axis
    th = np.linspace(np.pi - a0, 2 * np.pi + a0, n)
    arc = np.column_stack([cx + radius * np.cos(th), cy + radius * np.sin(th)])
    fill = Polygon(arc, closed=True, fc=style.OBSTACLE_FILL, ec="none", zorder=zorder,
                   clip_on=False)
    fill.set_gid(f"{gid}_fill")
    ax.add_patch(fill)
    edge = polyline(ax, arc, color=style.OBSTACLE_EDGE, lw=style.LW_AUX, gid=f"{gid}_edge",
                    zorder=zorder)
    edge.set_solid_capstyle("butt")
    return fill, edge


def pipe_wall_polygon(params, wall):
    """Outline of the pipe walls (a band of thickness ``wall`` around the
    channel, open at the entry face) as one simple polygon, (n, 2)."""
    P0 = np.asarray(params["entry_position"], dtype=float)
    d1 = np.asarray(params["entry_direction"], dtype=float)
    d2 = np.asarray(params["exit_direction"], dtype=float)
    d1, d2 = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
    r, L1, L2 = (float(params[k]) for k in ("inner_radius", "first_segment_length",
                                             "second_segment_length"))
    if params.get("geometry_type", "l_bend_2d") == "l_corner_2d":
        # (a, b): a along d1 from the entry face, b along d2 (the second leg)
        def ring(o):
            return [(0.0, -r - o), (L1 + r + o, -r - o), (L1 + r + o, L2 + o),
                    (L1 - r - o, L2 + o), (L1 - r - o, r + o), (0.0, r + o)]
        outer, inner = ring(wall), ring(0.0)
        ab = outer + inner[::-1]
        return np.array([P0 + a * d1 + b * d2 for a, b in ab])
    # rounded elbow: offset the sampled centerline
    Rb = float(params["bend_radius"])
    s_b0, s_b1 = L1, L1 + 0.5 * np.pi * Rb
    C = P0 + L1 * d1 + Rb * d2
    P2 = P0 + L1 * d1 + Rb * (d1 + d2)
    s = np.linspace(0.0, s_b1 + L2, 240)
    pts, tng = [], []
    for v in s:
        if v <= s_b0:
            pts.append(P0 + v * d1)
            tng.append(d1)
        elif v <= s_b1:
            ph = (v - s_b0) / Rb
            pts.append(C + Rb * (np.cos(ph) * -d2 + np.sin(ph) * d1))
            tng.append(np.sin(ph) * d2 + np.cos(ph) * d1)
        else:
            pts.append(P2 + (v - s_b1) * d2)
            tng.append(d2)
    pts, tng = np.array(pts), np.array(tng)
    nrm = np.stack([-tng[:, 1], tng[:, 0]], axis=1)
    cap = np.linspace(0.0, np.pi, 30)

    def side(o):
        left, right = pts + o * nrm, pts - o * nrm
        end = pts[-1] + o * (np.cos(cap)[:, None] * nrm[-1] + np.sin(cap)[:, None] * d2)
        return np.vstack([left, end, right[::-1]])

    outer, inner = side(r + wall), side(r)
    return np.vstack([outer, inner[::-1]])


def pipe_obstacle(ax, params, *, wall, gid, zorder=1):
    poly = pipe_wall_polygon(params, wall)
    patch = Polygon(poly, closed=True, fc=style.OBSTACLE_FILL, ec=style.OBSTACLE_EDGE,
                    lw=style.LW_AUX, joinstyle="miter", zorder=zorder, clip_on=False)
    patch.set_gid(gid)
    ax.add_patch(patch)
    return patch


def placeholder(fig, name, *, with_text):
    """Dashed light gray border inside the canvas, task name centered (7 pt)."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    w_in = fig.get_size_inches()[0]
    inset = 0.5 * style.LW_AUX * style.PT / w_in
    rect = MPath([(inset, inset), (1 - inset, inset), (1 - inset, 1 - inset),
                  (inset, 1 - inset), (inset, inset)], closed=True)
    patch = PathPatch(rect, fc="none", ec=style.PLACEHOLDER, lw=style.LW_AUX,
                      ls=(0, (3, 2)), clip_on=False)
    patch.set_gid("placeholder_border")
    ax.add_patch(patch)
    if with_text:
        t = ax.text(0.5, 0.5, name, ha="center", va="center", fontsize=style.FONT_SIZE,
                    color="#8C8C8C")
        t.set_gid("placeholder_label")
    return ax


# -------------------------------------------------------------- 3D helpers

def unclip(ax):
    """No clipping masks: 3D artists clip to the axes box unless told otherwise."""
    for a in list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.texts):
        a.set_clip_on(False)


def sphere_mesh(center, radius, n=24):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2 + 1)
    c = np.asarray(center, dtype=float)
    return (c[0] + radius * np.outer(np.cos(u), np.sin(v)),
            c[1] + radius * np.outer(np.sin(u), np.sin(v)),
            c[2] + radius * np.outer(np.ones_like(u), np.cos(v)))


def cylinder_mesh(p0, p1, radius, n=24, caps=False):
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    a = p1 - p0
    L = np.linalg.norm(a)
    a = a / L
    ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(a, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    th = np.linspace(0, 2 * np.pi, n)
    h = np.array([0.0, L])
    ring = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    X = p0[None, None, :] + h[None, :, None] * a + radius * ring[:, None, :]
    return X[..., 0], X[..., 1], X[..., 2]
