#!/usr/bin/env python3
"""Fig. 6 -- Edge-based smooth contact energy (Sec. VI-B), peg and 90-deg pipe.

IEEE T-RO single-column figure (3.5 in wide); editable text in PDF/SVG.

Model (manuscript Eq. (41)-(42)):
    E_c = sum_{j,a,o} w_ja * phi_delta(d_o(p_ja)),   p_ja = (1-s_a) x_j + s_a x_{j+1}
    phi_delta(d) = k_c/2 * [delta * log(1 + exp(-d/delta))]^2
    d_peg(p)  = ||(I - a a^T)(p - c)|| - (rho + r)
    d_pipe(p) = sdist_tau(p, A) - r      (A: allowed space, smoothed junction)
Quadrature: 4-point Gauss-Legendre on each edge, w_ja = e_bar * w_a.

Layout: (a) peg | (b) pipe on the first row; (c) the integrand along the two
highlighted edges, with the Gauss points, on the second row.
Fonts : plain text Arial, math STIX (see source/TRO_figure_style.md, Sec. 6).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.cm as cm

OUT = Path(__file__).resolve().parent

# ------------------------------------------------------------------ style
INK, MUTED, LIGHT = "#262626", "#666666", "#B5BAC0"
SOLID_FACE, SOLID_EDGE = "#B9BEC4", "#4B5563"     # rigid obstacles
ROD_TUBE = "#D6E4F0"
SHOW_TUBE = False                                  # draw the rod as a tube of radius r                               # rod of radius r
BLUE, GREEN = "#0072B2", "#009E73"                 # nodal force / clearance gradient
C_PEG, C_PIPE = "#0072B2", "#D55E00"               # curves in (c)
FS, FS_PANEL = 7.0, 8.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5,
    "ytick.labelsize": FS - 0.5, "legend.fontsize": FS,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "text.usetex": False, "axes.unicode_minus": False,
    "savefig.dpi": 600,
})

# energy colormap: white -> yellow -> red (sequential, prints in grayscale)
CMAP = LinearSegmentedColormap.from_list(
    "phi", ["#FFFFFF", "#FEE8A0", "#FDB863", "#E8603C", "#B2182B", "#67001F"])
PHI_MAX = 4.0
NORM = PowerNorm(gamma=0.5, vmin=0.0, vmax=PHI_MAX)

# ------------------------------------------------------------------ model
DELTA = 0.07        # smoothing length
R_ROD = 0.12        # rod radius r
RHO = 0.85          # peg radius
W_PIPE = 0.55       # half-width of each pipe passage
TAU = 0.10          # junction smoothing length

_gx, _gw = np.polynomial.legendre.leggauss(4)
S_A = 0.5 * (_gx + 1.0)            # 0.069, 0.330, 0.670, 0.931
W_A = 0.5 * _gw                    # 0.174, 0.326, 0.326, 0.174  (times e_bar)


def phi_n(d):
    """phi_delta / (k_c delta^2) = 0.5 * log(1 + exp(-d/delta))^2."""
    return 0.5 * np.logaddexp(0.0, -np.asarray(d) / DELTA) ** 2


def dphi_n(d):
    """d(phi_n)/dd (units 1/delta^2 * delta^2 / length)."""
    h = np.logaddexp(0.0, -np.asarray(d) / DELTA)
    sig = 0.5 * (1.0 - np.tanh(np.asarray(d) / (2 * DELTA)))
    return -h * sig / DELTA


def d_peg(P):
    P = np.atleast_2d(P)
    return np.linalg.norm(P, axis=1) - (RHO + R_ROD)          # c = 0, a = e_z


R_OUT = 0.22        # fillet radius of the outer wall at the bend


def _pipe_boundary(step=1.5e-3, far=6.0):
    """Wall curves of the allowed space A (L-shaped passage with rounded corners).

    A = horizontal passage {|y| < W} (open to the left) joined at a 90-deg bend
    to a vertical passage {|x| < W} (open to the top). The junction (inner
    corner) is smoothed with radius TAU; the outer corner with radius R_OUT.
    """
    W = W_PIPE
    def line(p0, p1):
        n = max(2, int(np.linalg.norm(np.subtract(p1, p0)) / step))
        return np.linspace(p0, p1, n)
    def fillet(c, R, a0, a1):
        t = np.linspace(a0, a1, max(8, int(abs(a1 - a0) * R / step)))
        return np.c_[c[0] + R * np.cos(t), c[1] + R * np.sin(t)]
    inner = np.vstack([line((-far, W), (-W - TAU, W)),
                       fillet((-W - TAU, W + TAU), TAU, -np.pi / 2, 0.0),
                       line((-W, W + TAU), (-W, far))])
    outer = np.vstack([line((-far, -W), (W - R_OUT, -W)),
                       fillet((W - R_OUT, -W + R_OUT), R_OUT, -np.pi / 2, 0.0),
                       line((W, -W + R_OUT), (W, far))])
    return inner, outer


from scipy.spatial import cKDTree            # noqa: E402
from matplotlib.path import Path as MPath    # noqa: E402
_INNER, _OUTER = _pipe_boundary()
_BND = np.vstack([_INNER, _OUTER])
_TREE = cKDTree(_BND)
_POLY = MPath(np.vstack([_OUTER, _INNER[::-1]]))      # closed outline of A


def _pipe_query(P):
    P = np.atleast_2d(P).astype(float)
    dist, idx = _TREE.query(P)
    sign = np.where(_POLY.contains_points(P), 1.0, -1.0)
    return sign * dist, P - _BND[idx], sign, dist


def sdist_pipe(P):
    """Signed distance to the wall of A (positive inside A)."""
    return _pipe_query(P)[0]


def d_pipe(P):
    return sdist_pipe(P) - R_ROD


def grad_pipe(P):
    _, v, sign, dist = _pipe_query(P)
    return sign[:, None] * v / np.maximum(dist, 1e-12)[:, None]


def grad(f, P, h=1e-5):
    P = np.atleast_2d(P).astype(float)
    g = np.zeros_like(P)
    for k in range(2):
        e = np.zeros(2); e[k] = h
        g[:, k] = (f(P + e) - f(P - e)) / (2 * h)
    return g


def grad_peg(P):
    P = np.atleast_2d(P)
    return P / np.linalg.norm(P, axis=1)[:, None]


GRADS = {}


def edge_terms(f, xj, xk):
    """Quadrature points, clearances, energy and nodal forces of one edge."""
    P = (1 - S_A)[:, None] * xj + S_A[:, None] * xk
    d = f(P)
    ebar = np.linalg.norm(xk - xj)
    gd = GRADS[f](P)
    coef = (ebar * W_A * dphi_n(d))[:, None] * gd           # dE/dp_a (normalized)
    Fj = -np.sum((1 - S_A)[:, None] * coef, axis=0)
    Fk = -np.sum(S_A[:, None] * coef, axis=0)
    total = np.sum(W_A * phi_n(d))                          # sum_a w_ja phi / (k_c d^2 e_bar)
    return P, d, gd, Fj, Fk, total


GRADS.update({d_peg: grad_peg, d_pipe: grad_pipe})

# ------------------------------------------------------------------ helpers
def arrow(ax, p0, p1, color, lw=0.9, ms=6, z=20):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=z))


def put(ax, xy, s, off=(0, 0), color=INK, ha="center", va="center", size=FS, z=30):
    ax.text(xy[0] + off[0], xy[1] + off[1], s, color=color, ha=ha, va=va,
            fontsize=size, zorder=z)


def energy_field(ax, f, extent, n=700, inside=None):
    x = np.linspace(extent[0], extent[1], n); y = np.linspace(extent[2], extent[3], n)
    X, Y = np.meshgrid(x, y)
    phi = phi_n(f(np.c_[X.ravel(), Y.ravel()])).reshape(X.shape)
    rgba = CMAP(NORM(np.clip(phi, 0, PHI_MAX)))
    rgba[..., 3] = np.clip(np.sqrt(phi / 0.02), 0, 1)       # fade to transparent
    if inside is not None:                                    # hide the field inside solids
        rgba[..., 3] *= inside(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    ax.imshow(rgba, extent=extent, origin="lower", interpolation="bilinear",
              zorder=1, rasterized=True)


def rod(ax, nodes, hi):
    """Rod tube (radius r), centerline, nodes; highlighted edge hi = (j, j+1)."""
    if SHOW_TUBE:   # rod cross-section of radius r (off: d_o = 0 already shows r)
        ax.plot(nodes[:, 0], nodes[:, 1], color=ROD_TUBE, lw=2 * R_ROD * PT_PER_UNIT,
                alpha=0.35, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.plot(nodes[:, 0], nodes[:, 1], color=MUTED, lw=0.6, zorder=4)
    ax.plot(nodes[hi, 0], nodes[hi, 1], color=INK, lw=1.5, zorder=5,
            solid_capstyle="round")
    ax.plot(nodes[:, 0], nodes[:, 1], "o", ms=3.0, mfc="white", mec=INK, mew=0.7, zorder=8)


def gauss_points(ax, P, d, z=9):
    for p, w, dd in zip(P, W_A, d):
        ax.plot(*p, marker="D", ms=2.0 + 6.0 * w, mfc=CMAP(NORM(min(phi_n(dd), PHI_MAX))),
                mec=INK, mew=0.5, zorder=z)


def panel_label(fig, x, y, tag, title):
    fig.text(x, y, tag, fontsize=FS_PANEL, fontweight="bold", va="top")
    fig.text(x + 0.062, y, title, fontsize=FS_PANEL, va="top")


# ================================================================== figure
W, H = 3.50, 4.15
fig = plt.figure(figsize=(W, H), facecolor="white")
fy = lambda inch: inch / H
AX_W_IN = 1.66                                   # drawing width of (a), (b) in inches

# ------------------------------------------------------------------ (a) peg
panel_label(fig, 0.01, fy(H - 0.02), "(a)", "Cylindrical peg")
xa = (-1.55, 1.55); ya = (-1.40, 1.50)
a = fig.add_axes([0.005, fy(2.32), AX_W_IN / W, fy(1.50)]); a.axis("off")
a.set_xlim(*xa); a.set_ylim(*ya); a.set_aspect("equal", adjustable="box")
PT_PER_UNIT = AX_W_IN * 72 / (xa[1] - xa[0])

energy_field(a, d_peg, (*xa, *ya))
a.add_patch(Circle((0, 0), RHO, fc=SOLID_FACE, ec=SOLID_EDGE, lw=0.7, zorder=6))
a.add_patch(Circle((0, 0), RHO + R_ROD, fc="none", ec=INK, lw=0.6, ls=(0, (3, 2)), zorder=7))

ang = np.deg2rad([202, 162, 126, 54, 18, -22])
rad = np.array([1.38, 1.17, 1.11, 1.11, 1.17, 1.38])
nodes_a = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
ja = 2
rod(a, nodes_a, [ja, ja + 1])
Pa, da, gda, Fja, Fka, Sa = edge_terms(d_peg, nodes_a[ja], nodes_a[ja + 1])
gauss_points(a, Pa, da)

# axis out of page at c
a.add_patch(Circle((0, 0), 0.07, fc="white", ec=INK, lw=0.6, zorder=10))
a.plot(0, 0, ".", color=INK, ms=1.5, zorder=11)
put(a, (0, 0), r"$\mathbf{c},\,\mathbf{a}$", (0.14, -0.06), ha="left")
# rho
th = np.deg2rad(-150)
arrow(a, (0, 0), RHO * np.array([np.cos(th), np.sin(th)]), INK, lw=0.6, ms=5, z=12)
put(a, 0.5 * RHO * np.array([np.cos(th), np.sin(th)]), r"$\rho$", (0.04, -0.13))
# r: dimension between rho and rho + r
th = np.deg2rad(-48)
u = np.array([np.cos(th), np.sin(th)])
arrow(a, (RHO + R_ROD + 0.22) * u, (RHO + R_ROD) * u, INK, lw=0.5, ms=4, z=12)
arrow(a, (RHO - 0.22) * u, RHO * u, INK, lw=0.5, ms=4, z=12)
put(a, (RHO + R_ROD + 0.30) * u, r"$r$", (0.03, 0.0))
put(a, (0, -(RHO + R_ROD)), r"$d_{\mathrm{peg}}=0$", (0.0, -0.17))
# d_peg at the Gauss point with the deepest penetration
k = int(np.argmin(da))
a.plot([0, Pa[k, 0]], [0, Pa[k, 1]], color=MUTED, lw=0.5, ls=(0, (1, 1.5)), zorder=12)
put(a, Pa[k], r"$\mathbf{p}_{ja}$", (0.0, 0.18))
# nodal forces
sc = 0.42 / max(np.linalg.norm(Fja), np.linalg.norm(Fka))
for xn, F in ((nodes_a[ja], Fja), (nodes_a[ja + 1], Fka)):
    arrow(a, xn, xn + sc * F, BLUE, lw=0.9, ms=5.5, z=21)
put(a, nodes_a[ja], r"$\mathbf{x}_j$", (-0.20, -0.08))
put(a, nodes_a[ja + 1], r"$\mathbf{x}_{j+1}$", (0.30, -0.08))

# ------------------------------------------------------------------ (b) pipe
panel_label(fig, 0.515, fy(H - 0.02), "(b)", "90° pipe (section)")
xb = (-1.30, 0.95); yb = (-1.00, 1.30)
b = fig.add_axes([0.515, fy(2.32), AX_W_IN / W, fy(1.50)]); b.axis("off")
b.set_xlim(*xb); b.set_ylim(*yb); b.set_aspect("equal", adjustable="box")
PT_PER_UNIT = AX_W_IN * 72 / (xb[1] - xb[0])

energy_field(b, d_pipe, (*xb, *yb), inside=lambda P: (sdist_pipe(P) > 0).astype(float))
# pipe wall = complement of A within a wall thickness, drawn by contour fill
gx = np.linspace(*xb, 800); gyy = np.linspace(*yb, 800)
GX, GY = np.meshgrid(gx, gyy)
SD = sdist_pipe(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
b.contourf(GX, GY, SD, levels=[-0.22, 0.0], colors=[SOLID_FACE], zorder=6)
b.contour(GX, GY, SD, levels=[0.0], colors=[SOLID_EDGE], linewidths=0.7, zorder=7)
b.contour(GX, GY, SD, levels=[R_ROD], colors=[INK], linewidths=0.6,
          linestyles=[(0, (3, 2))], zorder=7)

nodes_b = np.array([[-1.55, -0.28], [-1.17, -0.33], [-0.78, -0.38], [-0.38, -0.41],
                    [0.03, -0.415], [0.415, -0.03], [0.41, 0.37], [0.37, 0.77], [0.33, 1.17],
                    [0.30, 1.57]])
jb = 4
rod(b, nodes_b, [jb, jb + 1])
Pb, db, gdb, Fjb, Fkb, Sb = edge_terms(d_pipe, nodes_b[jb], nodes_b[jb + 1])
gauss_points(b, Pb, db)
# clearance gradient at the last Gauss point
k = 0
gnorm = gdb[k] / np.linalg.norm(gdb[k])
arrow(b, Pb[k], Pb[k] + 0.32 * gnorm, GREEN, lw=0.9, ms=5.5, z=22)
put(b, Pb[k] + 0.32 * gnorm, r"$\nabla d_{\mathrm{pipe}}(\mathbf{p}_{ja})$", (-0.04, 0.03),
    color=GREEN, ha="right")
put(b, nodes_b[jb], r"$\mathbf{x}_j$", (-0.15, -0.09))
put(b, nodes_b[jb + 1], r"$\mathbf{x}_{j+1}$", (-0.22, 0.13))

put(b, (-0.55, 0.22), r"$\mathcal{A}$", size=FS + 1)
put(b, (-0.75, -0.88), "pipe wall", color=MUTED)
arrow(b, (-0.95, 0.30), (-1.27, 0.30), MUTED, lw=0.5, ms=4, z=12)
put(b, (-0.95, 0.30), "open", (0.03, 0.0), color=MUTED, ha="left")
# junction smoothing
jc = np.array([-W_PIPE - TAU, W_PIPE + TAU]) + TAU * np.array([np.cos(-np.pi / 4), np.sin(-np.pi / 4)])
b.annotate(r"junction ($\tau$)", xy=jc, xytext=(-1.26, 1.18),
           fontsize=FS, color=MUTED, ha="left", va="center", zorder=30,
           arrowprops=dict(arrowstyle="-", lw=0.4, color=MUTED, shrinkA=1, shrinkB=1))
put(b, (-W_PIPE + R_ROD, 0.72), r"$d_{\mathrm{pipe}}=0$", (0.04, 0.0), ha="left")

# ------------------------------------------------------------------ colorbar
cax = fig.add_axes([0.30, fy(2.13), 0.40, fy(0.05)])
cb = fig.colorbar(cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal",
                  ticks=[0, 0.5, 1, 2, 4])
cb.outline.set_linewidth(0.4); cb.ax.tick_params(width=0.4, length=2, labelsize=FS - 0.5)
cb.ax.set_xticklabels(["0", "0.5", "1", "2", "4"])
fig.text(0.29, fy(2.155), r"$\phi_\delta(d_o)/(k_c\delta^2)$", ha="right", va="center")

# ------------------------------------------------------------------ (c) integrand
panel_label(fig, 0.01, fy(1.88), "(c)", "Contact energy along the highlighted edge")
c = fig.add_axes([0.13, fy(0.80), 0.84, fy(0.90)])
s = np.linspace(0, 1, 400)
for nodes, j, f, col, lab, S in ((nodes_a, ja, d_peg, C_PEG, "peg (a)", Sa),
                                 (nodes_b, jb, d_pipe, C_PIPE, "pipe (b)", Sb)):
    P = (1 - s)[:, None] * nodes[j] + s[:, None] * nodes[j + 1]
    y = phi_n(f(P))
    c.fill_between(s, 0, y, color=col, alpha=0.12, lw=0, zorder=1)
    c.plot(s, y, color=col, lw=1.0, zorder=3,
           label=lab + r",  $\Sigma_a\, w_{ja}\,\phi_\delta/(k_c\delta^2\bar e)=$" + f"{S:.2f}")
    yq = phi_n(f((1 - S_A)[:, None] * nodes[j] + S_A[:, None] * nodes[j + 1]))
    for sa, wa, yy in zip(S_A, W_A, yq):
        c.plot([sa, sa], [0, yy], color=col, lw=0.5, zorder=2)
        c.plot(sa, yy, "D", ms=2.0 + 6.0 * wa, mfc="white", mec=col, mew=0.8, zorder=4)
c.set_xlim(0, 1); c.set_ylim(0, 1.32)
c.set_xticks([0, 0.25, 0.5, 0.75, 1]); c.set_xticklabels(["0", "", "0.5", "", "1"])
c.set_yticks([0, 0.4, 0.8, 1.2])
c.set_xlabel(r"edge coordinate $s$", labelpad=1)
c.set_ylabel(r"$\phi_\delta/(k_c\delta^2)$", labelpad=2)
c.legend(loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.1,
         labelspacing=0.25, ncol=1, bbox_to_anchor=(0.0, 1.04))

# ------------------------------------------------------------------ shared legend
items = [
    Line2D([], [], ls="none", marker="o", ms=3.0, mfc="white", mec=INK, mew=0.7),
    Line2D([], [], color=INK, lw=1.5),
    Line2D([], [], ls="none", marker="D", ms=4.5, mfc="#FDB863", mec=INK, mew=0.5),
    Line2D([], [], color=INK, lw=0.6, ls=(0, (3, 2))),
    Line2D([], [], color=BLUE, lw=0.9),
    Line2D([], [], color=GREEN, lw=0.8),
]
labels = ["rod node", r"edge $(\mathbf{x}_j,\mathbf{x}_{j+1})$",
          r"Gauss point $\mathbf{p}_{ja}$ (size $\propto w_{ja}$)",
          r"$d_o=0$ (offset by $r$)", r"nodal force $-\nabla_{\mathbf{x}}E_c$",
          r"clearance gradient $\nabla d_o$"]
fig.legend(items, labels, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=1.6,
           handletextpad=0.4, fontsize=FS - 0.5)

if __name__ == "__main__":
    for ext in ("pdf", "svg"):
        dst = OUT / f"fig_contact_energy.{ext}"
        fig.savefig(dst, facecolor="white")
        print("saved:", dst)
    print(f"peg : d/delta at Gauss pts = {np.round(da / DELTA, 2)}, sum = {Sa:.3f}")
    print(f"pipe: d/delta at Gauss pts = {np.round(db / DELTA, 2)}, sum = {Sb:.3f}")
