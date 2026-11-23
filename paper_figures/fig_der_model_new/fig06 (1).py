#!/usr/bin/env python3
"""Fig. 6 -- Discrete elastic rod and six-coordinate gripper boundary convention.

IEEE T-RO single-column figure (3.5 in wide), editable text in PDF/SVG.
Layout: (a) on the first row (full width); (b) and (c) side by side below.
Fonts : plain text and panel labels in Arial; math in STIX (Times-like),
        consistent with source/TRO_figure_style.md.

Conventions (consistent with the paper):
    R = Rz(alpha) Ry(-beta) Rx(gamma), beta > 0 is elevation above horizontal
    t = (cos(alpha)cos(beta), sin(alpha)cos(beta), sin(beta))
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent

# ------------------------------------------------------------------ style
INK, MUTED, LIGHT = "#262626", "#666666", "#A8ADB2"
BLUE, RED, GREEN = "#0072B2", "#C0392B", "#009E73"     # Bishop / material / reference n
MECH_FACE, MECH_EDGE = "#E6E8EB", "#4B5563"
E3 = np.array([0.0, 0.0, 1.0])

FS = 7.0          # labels / annotations
FS_PANEL = 8.0    # (a), (b), (c) + titles

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": FS,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "text.usetex": False, "axes.unicode_minus": False,
    "hatch.linewidth": 0.4,
})


# ------------------------------------------------------------------ geometry
def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def bishop_frame(points):
    e = np.diff(points, axis=0)
    t = e / np.linalg.norm(e, axis=1)[:, None]
    b = [unit(np.cross(E3, t[0]))]
    for ta, tb in zip(t[:-1], t[1:]):
        axis = np.cross(ta, tb)
        s, c = np.linalg.norm(axis), np.dot(ta, tb)
        if s < 1e-11:
            if c < 0:
                raise ValueError("antiparallel edge tangents")
            d = b[-1].copy()
        else:
            axis /= s
            d = c * b[-1] + s * np.cross(axis, b[-1]) + (1 - c) * np.dot(axis, b[-1]) * axis
        b.append(unit(d - np.dot(d, tb) * tb))
    b = np.asarray(b)
    ref_end = unit(np.cross(E3, t[-1]))
    psi = np.arctan2(np.dot(np.cross(ref_end, b[-1]), t[-1]), np.dot(ref_end, b[-1]))
    return t, b, psi


_U = unit([0.92, 0.34, 0.0])
_V = unit([-0.31, 0.84, 0.56])
_V = unit(_V - np.dot(_V, _U) * _U)


def camera(points):
    points = np.asarray(points)
    return np.stack((points @ _U, points @ _V), axis=-1)


# ------------------------------------------------------------------ drawing helpers
def arrow(ax, begin, end, color=INK, lw=0.9, size=6, z=15):
    ax.annotate("", xy=end, xytext=begin, zorder=z,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                mutation_scale=size, shrinkA=0, shrinkB=0))


def put(ax, xy, s, offset=(0, 0), color=INK, ha="center", va="center", size=FS):
    ax.text(xy[0] + offset[0], xy[1] + offset[1], s, fontsize=size, color=color,
            ha=ha, va=va, zorder=25)


def node(ax, pt, clamped=True, size=3.4):
    ax.plot(pt[0], pt[1], marker="o", ms=size, ls="none",
            mfc=INK if clamped else "white", mec=INK, mew=0.8, zorder=12)


def draw_fixed_clamp(ax, p, direction, length=0.11, halfwidth=0.07):
    p = np.asarray(p, float); u = unit(direction); n = np.array([-u[1], u[0]])
    front = p - 0.01 * u; rear = front - length * u
    ax.add_patch(Polygon([rear - 0.05 * u - 1.4 * halfwidth * n, rear - 1.4 * halfwidth * n,
                          rear + 1.4 * halfwidth * n, rear - 0.05 * u + 1.4 * halfwidth * n],
                         closed=True, fc="none", ec=INK, lw=0.5, hatch="////", zorder=2))
    ax.add_patch(Polygon([rear - halfwidth * n, front - halfwidth * n,
                          front + halfwidth * n, rear + halfwidth * n],
                         closed=True, fc=MECH_FACE, ec=MECH_EDGE, lw=0.7, zorder=4))


def draw_gripper(ax, p, tangent, depth, gap, jaw, back, mount, lw=0.7):
    p = np.asarray(p, float); u = unit(tangent); v = np.array([-u[1], u[0]])
    L = lambda x, y: p + x * u + y * v
    quad = lambda pts, fc: ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=MECH_EDGE,
                                                lw=lw, zorder=5))
    quad([L(back, -0.65 * gap), L(back + mount, -0.65 * gap),
          L(back + mount, 0.65 * gap), L(back, 0.65 * gap)], "#D1D5DB")
    quad([L(0, -gap - jaw), L(back, -gap - jaw), L(back, gap + jaw), L(0, gap + jaw)], MECH_FACE)
    quad([L(-depth, gap), L(0, gap), L(0, gap + jaw), L(-depth, gap + jaw)], MECH_FACE)
    quad([L(-depth, -gap - jaw), L(0, -gap - jaw), L(0, -gap), L(-depth, -gap)], MECH_FACE)


def arc(ax, center, a0, a1, radius, color, lw=0.8, head=True):
    ph = np.linspace(a0, a1, 80)
    c = np.asarray(center) + radius * np.c_[np.cos(ph), np.sin(ph)]
    ax.plot(c[:, 0], c[:, 1], color=color, lw=lw, zorder=11)
    if head:
        arrow(ax, c[-6], c[-1], color=color, lw=lw, size=5, z=12)


def panel_label(fig, x, y, tag, title):
    fig.text(x, y, tag, fontsize=FS_PANEL, fontweight="bold", color=INK, va="top")
    fig.text(x + 0.055, y, title, fontsize=FS_PANEL, color=INK, va="top")


def rod_points(N, depth_amp=0.18):
    """Illustrative 3-D centerline, designed in screen coordinates.

    The curve is specified by its projection (X, Y) on the drawing plane plus a
    depth D along the viewing direction, P = X*U + Y*V + D*W, so the 2-D shape is
    controlled exactly. The first edge points straight up on the page (along V);
    in world coordinates its elevation is about 34 deg, away from the singular
    chart beta = +-pi/2. Nodes are equally spaced in arc length.
    """
    from scipy.interpolate import CubicSpline
    W = np.cross(_U, _V)
    ctrl = np.array([(0.00, 0.00), (0.00, 0.20), (0.12, 0.37), (0.34, 0.45), (0.58, 0.40),
                     (0.92, 0.06), (1.24, -0.10), (1.54, -0.03), (1.80, 0.17)])
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ctrl, axis=0), axis=1))]
    spl = CubicSpline(seg, ctrl, bc_type=((1, (0.0, 1.0)), "natural"))
    q = np.linspace(0, seg[-1], 4000)
    xyq = spl(q)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyq, axis=0), axis=1))]
    sq = np.interp(np.linspace(0, arc[-1], N), arc, q)
    XY = spl(sq)
    XY[1, 0] = XY[0, 0]                                  # first edge exactly vertical
    u = sq / seg[-1]
    D = depth_amp * np.sin(np.pi * u) ** 2               # zero slope at the clamp
    D[1] = D[0]
    return XY[:, :1] * _U + XY[:, 1:] * _V + D[:, None] * W


# ================================================================== figure
W, H = 3.50, 3.40
fig = plt.figure(figsize=(W, H), facecolor="white")
fy = lambda inch: inch / H        # vertical position in inches -> figure fraction

# ------------------------------------------------------------------ (a)
panel_label(fig, 0.01, fy(H - 0.02), "(a)", "Rod centerline and frames")
a = fig.add_axes([0.0, fy(1.83), 1.0, fy(H - 1.83 - 0.20)]); a.axis("off")

N = 11
s = np.linspace(0, 1, N)
points = rod_points(N)
t, b, psi = bishop_frame(points)
gamma = np.deg2rad(129.0)
theta = np.arange(N - 1) / (N - 2) * (gamma - psi)
m1 = np.cos(theta[:, None]) * b + np.sin(theta[:, None]) * np.cross(t, b)
xy = camera(points)


# flattened ground projection + drop lines (depth cue). U is horizontal, so the
# drop lines stay vertical; the footprint is foreshortened by SHADOW_K so that it
# reads as lying on a low-angle ground plane without enlarging the panel.
SHADOW_K, SHADOW_TOP = 0.30, -0.30
pg = points.copy(); pg[:, 2] = 0.0
gy = camera(pg)[:, 1]
xyg = np.c_[xy[:, 0], SHADOW_TOP + SHADOW_K * (gy - gy.max())]
a.plot(xyg[:, 0], xyg[:, 1], color=LIGHT, ls=(0, (1, 1.6)), lw=0.7, zorder=1)
for i in (0, 1, 5, N - 2, N - 1):
    a.plot([xy[i, 0], xyg[i, 0]], [xy[i, 1], xyg[i, 1]], color=LIGHT,
           ls=(0, (1, 1.6)), lw=0.6, zorder=1)

a.plot(xy[:, 0], xy[:, 1], color=INK, lw=1.3, solid_capstyle="round",
       solid_joinstyle="round", zorder=3)
draw_fixed_clamp(a, xy[0], xy[1] - xy[0])
draw_gripper(a, xy[-1], xy[-1] - xy[-2], depth=0.22, gap=0.045, jaw=0.028,
             back=0.06, mount=0.09)
for i, pt in enumerate(xy):
    node(a, pt, i in (0, 1, N - 2, N - 1))

# node / edge labels (placed away from frame vectors)
put(a, xy[0], r"$\mathbf{x}_0$", (-0.13, 0.02))
put(a, xy[1], r"$\mathbf{x}_1$", (-0.13, 0.0))
put(a, xy[5], r"$\mathbf{x}_i$", (-0.12, -0.03))
put(a, xy[N - 2], r"$\mathbf{x}_{N-2}$", (0.17, -0.11))
put(a, xy[N - 1], r"$\mathbf{x}_{N-1}$", (0.02, 0.18))
e_mid = (xy[6] + xy[7]) / 2
put(a, e_mid, r"$\mathbf{e}^{j}$", (-0.07, -0.11))

# frames on one interior edge and the gripper edge
j = 6
base = (points[j] + points[j + 1]) / 2
for vec, L, col, lab, off in [(b[j], 0.40, BLUE, r"$\mathbf{b}^{j}$", (0.0, 0.09)),
                               (m1[j], 0.55, RED, r"$\mathbf{m}_1^{j}$", (0.14, -0.02))]:
    q0, q1 = camera(base), camera(base + L * unit(vec))
    arrow(a, q0, q1, col, lw=0.9, size=5.5)
    put(a, q1, lab, off, color=col)

j = N - 2
base = (points[j] + points[j + 1]) / 2
for vec, L, col, lab, off in [(b[j], 0.30, BLUE, r"$\mathbf{b}^{N-2}$", (-0.02, 0.10)),
                               (m1[j], 0.36, RED, r"$\mathbf{m}_1^{N-2}$", (0.30, -0.03))]:
    q0, q1 = camera(base), camera(base + L * unit(vec))
    arrow(a, q0, q1, col, lw=0.9, size=5.5)
    put(a, q1, lab, off, color=col)

# gravity
g0 = np.array([-0.42, 0.62]); arrow(a, g0, g0 + [0, -0.26], MUTED, lw=0.8, size=5)
put(a, g0 + [0, -0.13], r"$g$", (0.07, 0), color=MUTED, ha="left")

put(a, xy[0], "fixed clamp", (-0.07, -0.22), color=MUTED, ha="right")
put(a, xy[-1], "gripper", (0.24, 0.10), color=MUTED, ha="left")

a.set_xlim(-0.62, 2.50); a.set_ylim(xyg[:, 1].min() - 0.05, 0.66)
a.set_aspect("equal", adjustable="box")

# legend (one row, under panel a)
ly = fy(1.767)
items = [("o", INK, "clamped node"), ("o", "white", "free node"),
         ("-", BLUE, "Bishop"), ("-", RED, "material")]
x = 0.07
for kind, col, lab in items:
    if kind == "o":
        fig.lines.append(Line2D([x + 0.012], [ly], transform=fig.transFigure, marker="o",
                                ms=3.4, mfc=col, mec=INK, mew=0.8, ls="none"))
        x += 0.032
    else:
        fig.lines.append(Line2D([x, x + 0.05], [ly, ly], transform=fig.transFigure,
                                lw=1.1, color=col))
        x += 0.062
    fig.text(x, ly, lab, fontsize=FS, va="center", color=INK)
    x += 0.045 + 0.0155 * len(lab)

# separator between rows
fig.lines.append(Line2D([0.02, 0.98], [fy(1.615), fy(1.615)], transform=fig.transFigure,
                        lw=0.4, color="#C8C8C8"))

# ------------------------------------------------------------------ (b)
panel_label(fig, 0.01, fy(1.558), "(b)", "Six-DOF gripper")
bax = fig.add_axes([0.0, fy(0.247), 0.50, fy(1.14)]); bax.axis("off")

alpha, beta, roll = np.deg2rad(35.0), np.deg2rad(28.0), np.deg2rad(65.0)
tb = np.array([np.cos(alpha) * np.cos(beta), np.sin(alpha) * np.cos(beta), np.sin(beta)])
nb = unit(np.cross(E3, tb))
mb = np.cos(roll) * nb + np.sin(roll) * np.cross(tb, nb)
p = 0.93 * tb; xprev = p - 0.35 * tb; O = np.zeros(3)
viewB = lambda v: camera(v) * np.array([9.0, 6.75])

for ax3, lab, off in [((1, 0, 0), r"$\mathbf{e}_x$", (0.25, -0.30)),
                      ((0, 1, 0), r"$\mathbf{e}_y$", (0.20, 0.25)),
                      ((0, 0, 1), r"$\mathbf{e}_z$", (-0.35, 0.10))]:
    q0, q1 = viewB(O), viewB(0.22 * np.array(ax3, float))
    arrow(bax, q0, q1, LIGHT, lw=0.8, size=5)
    put(bax, q1, lab, off, color=MUTED)

origin, pxy, prevxy = viewB(O), viewB(p), viewB(xprev)
ph = viewB(0.93 * np.cos(beta) * np.array([np.cos(alpha), np.sin(alpha), 0.0]))
bax.plot(*np.c_[origin, ph], ls="--", color=LIGHT, lw=0.6, zorder=1)
bax.plot(*np.c_[ph, pxy], ls=":", color=LIGHT, lw=0.6, zorder=1)
bax.plot(*np.c_[origin, prevxy], color=INK, lw=0.6, ls=(0, (3, 2)), zorder=3)
bax.plot(*np.c_[prevxy, pxy], color=INK, lw=1.5, solid_capstyle="round", zorder=4)
draw_gripper(bax, pxy, pxy - prevxy, depth=2.1, gap=0.28, jaw=0.14, back=0.45, mount=0.55)
node(bax, prevxy); node(bax, pxy)

put(bax, prevxy, r"$\mathbf{x}_{N-2}$", (-0.60, 0.45))
put(bax, pxy, r"$\mathbf{p}=\mathbf{x}_{N-1}$", (0.10, 0.75))

# edge length dimension
ed = unit(pxy - prevxy); nrm = np.array([-ed[1], ed[0]])
q1, q2 = prevxy - 0.75 * nrm, pxy - 0.75 * nrm
bax.plot(*np.c_[q1, q2], color=MUTED, lw=0.5)
for q in (q1, q2):
    bax.plot(*np.c_[q - 0.13 * nrm, q + 0.13 * nrm], color=MUTED, lw=0.5)
put(bax, (q1 + q2) / 2, r"$\bar e$", (0.18, -0.28), color=MUTED)

# director arrows at the gripper edge
pivot = 0.5 * (xprev + p)
for vec, col, lab, off in [(nb, GREEN, r"$\mathbf{n}$", (0.05, 0.28)),
                           (mb, RED, r"$\mathbf{m}_1$", (-0.30, 0.15))]:
    q0, q1 = viewB(pivot), viewB(pivot + 0.24 * unit(vec))
    arrow(bax, q0, q1, col, lw=0.9, size=5.5)
    put(bax, q1, lab, off, color=col)

put(bax, origin + 0.60 * (prevxy - origin), r"$\mathbf{t}(\alpha,\beta)$", (-0.15, 0.50))
a0 = np.arctan2(*(ph - origin)[::-1]); a1 = np.arctan2(*(pxy - origin)[::-1])
arc(bax, origin, a0, a1, 2.6, INK, lw=0.6)
am = 0.5 * (a0 + a1)
put(bax, origin + 2.6 * np.array([np.cos(am), np.sin(am)]), r"$\beta$", (0.30, -0.02))

bax.set_xlim(-0.9, 9.6); bax.set_ylim(-1.2, 5.6)
bax.set_aspect("equal", adjustable="box")

fig.text(0.25, fy(0.209), r"$\mathbf{u}=(\mathbf{p},\alpha,\beta,\gamma)$", ha="center", va="center")
fig.text(0.25, fy(0.068), r"$\mathbf{R}=\mathbf{R}_z(\alpha)\,\mathbf{R}_y(-\beta)\,\mathbf{R}_x(\gamma)$",
         ha="center", va="center")

# vertical separator
fig.lines.append(Line2D([0.505, 0.505], [fy(0.076), fy(1.52)], transform=fig.transFigure,
                        lw=0.4, color="#C8C8C8"))

# ------------------------------------------------------------------ (c)
panel_label(fig, 0.52, fy(1.558), "(c)", "Holonomy at gripper")
cax = fig.add_axes([0.52, fy(0.285), 0.47, fy(1.14)]); cax.axis("off")
arc(cax, (0, 0), np.deg2rad(-12), np.deg2rad(192), 1.0, "#C4C4C4", lw=0.5, head=False)
cax.add_patch(Circle((0, 0), 0.06, fc="white", ec=INK, lw=0.7, zorder=9))
cax.plot(0, 0, ".", color=INK, ms=1.8, zorder=10)

for ang, col, lab, off in [(0.0, GREEN, r"$\mathbf{n}$", (0.14, 0.0)),
                           (psi, BLUE, r"$\mathbf{b}^{N-2}$", (0.10, 0.13)),
                           (gamma, RED, r"$\mathbf{m}_1^{N-2}$", (-0.22, 0.12))]:
    end = 0.92 * np.array([np.cos(ang), np.sin(ang)])
    arrow(cax, (0, 0), end, col, lw=0.9, size=5.5)
    put(cax, end, lab, off, color=col)

arc(cax, (0, 0), 0.0, psi, 0.42, BLUE)
arc(cax, (0, 0), psi, gamma, 0.62, INK)
arc(cax, (0, 0), 0.0, gamma, 1.22, RED, lw=0.7)
pl = lambda r, a_: r * np.array([np.cos(a_), np.sin(a_)])
put(cax, pl(0.30, psi / 2), r"$\Psi$", color=BLUE)
put(cax, pl(0.47, 0.5 * (psi + gamma)) + [-0.02, 0.0], r"$\theta^{N-2}$", color=INK)
put(cax, pl(1.36, 0.22 * gamma), r"$\gamma$", color=RED)

cax.set_xlim(-1.45, 1.55); cax.set_ylim(-0.35, 1.55)
cax.set_aspect("equal", adjustable="box")

fig.text(0.755, fy(0.209), r"view along $\mathbf{t}^{N-2}$ (out of page)", ha="center", va="center",
         color=MUTED)
fig.text(0.755, fy(0.068), r"$\theta^{N-2}=\gamma-\Psi$", ha="center", va="center")

# ------------------------------------------------------------------ save
if __name__ == "__main__":
    for ext in ("pdf", "svg"):
        dst = OUT / f"fig06_rod_model.{ext}"
        fig.savefig(dst, facecolor="white")
        print("saved:", dst)
    print(f"Psi={np.rad2deg(psi):.2f} deg, gamma={np.rad2deg(gamma):.2f} deg, "
          f"theta_end={np.rad2deg(theta[-1]):.2f} deg")
