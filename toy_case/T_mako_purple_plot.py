"""
Two-panel figure:
  (a) Potential landscape with ML-conference-friendly colormap (mako-like),
      light/white for deepest minima, showing principal-direction arrows
      of T_{(q,z)} at the two stable equilibria.
  (b) Variables along the path q1* -> saddle -> q2*:
      - condition number kappa(Hessian)
      - ||T|| = 1/sigma_min
      - eigenvalues lambda_min, lambda_max
      - U(q) for reference
      Showing how all STO-relevant quantities blow up at the saddle
      (singular-locus crossing) and recover in each basin.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

# Scale factor relative to original 16" wide figure
S = 5.5 / 16.0   # ≈ 0.344

# ===== Modified Müller-Brown (A_2 = 0, two minima) =====
A  = np.array([-200.0, 0.0, -170.0, 15.0])
a  = np.array([-1.0, -1.0, -6.5, 0.7])
b  = np.array([0.0, 0.0, 11.0, 0.6])
c_ = np.array([-10.0, -10.0, -6.5, 0.7])
x0 = np.array([1.0, 0.0, -0.5, -1.0])
y0 = np.array([0.0, 0.5, 1.5, 1.0])

def U(q1, q2):
    r = 0.0
    for i in range(4):
        r = r + A[i]*np.exp(a[i]*(q1-x0[i])**2 + b[i]*(q1-x0[i])*(q2-y0[i])
                             + c_[i]*(q2-y0[i])**2)
    return r

def hess_fields(Xg, Yg):
    Uxx = np.zeros_like(Xg); Uyy = np.zeros_like(Xg); Uxy = np.zeros_like(Xg)
    for i in range(4):
        dx = Xg - x0[i]; dy = Yg - y0[i]
        phi = A[i]*np.exp(a[i]*dx**2 + b[i]*dx*dy + c_[i]*dy**2)
        ex = 2*a[i]*dx + b[i]*dy
        ey = 2*c_[i]*dy + b[i]*dx
        Uxx = Uxx + phi*(ex*ex + 2*a[i])
        Uyy = Uyy + phi*(ey*ey + 2*c_[i])
        Uxy = Uxy + phi*(ex*ey + b[i])
    return Uxx, Uyy, Uxy

def Hessian(q1, q2):
    Uxx, Uyy, Uxy = hess_fields(np.array([[q1]]), np.array([[q2]]))
    return np.array([[Uxx[0,0], Uxy[0,0]], [Uxy[0,0], Uyy[0,0]]])

# Landmarks
minima = [(-0.558, 1.442), (0.641, 0.009)]
saddle = (-0.673, 0.417)

# ===== ML-friendly colormap: Mako with Viridis-style deep purple start =====
from matplotlib.colors import LinearSegmentedColormap as _LSC
mako_anchors = [
    (0.00, '#440154'),
    (0.15, '#471567'),
    (0.30, '#482878'),
    (0.40, '#3E3B8F'),
    (0.50, '#3B5299'),
    (0.58, '#3A6A9B'),
    (0.66, '#3F8196'),
    (0.73, '#4B9888'),
    (0.80, '#63AF7B'),
    (0.86, '#89C56D'),
    (0.91, '#B7D972'),
    (0.95, '#E0E97B'),
    (0.98, '#F7FA9C'),
    (1.00, '#FFFFFF'),
]
ml_cmap = _LSC.from_list(
    'mako_viridis_start',
    [(pos, color) for pos, color in mako_anchors], N=256)

# ===== Grid =====
xx = np.linspace(-1.7, 1.2, 400)
yy = np.linspace(-0.4, 2.2, 400)
Xg, Yg = np.meshgrid(xx, yy)
Z = U(Xg, Yg)
Zc = np.clip(Z, -150, 100)

# ==========================================================================
# FIGURE
# ==========================================================================
fig = plt.figure(figsize=(5.5, 2.23))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.28)

# ------------- Panel (a): landscape with T-direction arrows -------------
ax = fig.add_subplot(gs[0, 0])

levels = np.linspace(-150, 100, 40)
cs = ax.contourf(Xg, Yg, Zc, levels=levels, cmap=ml_cmap, extend='both')
ax.contour(Xg, Yg, Zc, levels=18, colors='white',
           linewidths=0.35 * S / S,   # keep thin contour lines as-is
           alpha=0.35)

# Basin separatrix
sep_level = U(np.array([[saddle[0]]]), np.array([[saddle[1]]]))[0, 0]
ax.contour(Xg, Yg, Z, levels=[sep_level], colors='#ff4d6d',
           linewidths=round(1.8 * S, 2), linestyles=':', zorder=3, alpha=0.7)

# --- Principal direction arrows of T at each minimum ---
# original lw=3.0, mutation_scale=22/15
def draw_principal_arrows(ax, p, color_soft, color_stiff,
                           length_soft=0.30, length_stiff=0.14, lw=3.0):
    H = Hessian(*p)
    _, V = np.linalg.eigh(H)
    v_soft  = V[:, 0]
    v_stiff = V[:, 1]

    for vec, length, color, width, mscale in [
        (v_soft,  length_soft,  color_soft,  lw,         round(22 * S, 1)),
        (v_stiff, length_stiff, color_stiff, lw * 0.7,   round(15 * S, 1)),
    ]:
        tail = (p[0] - length * vec[0], p[1] - length * vec[1])
        head = (p[0] + length * vec[0], p[1] + length * vec[1])
        ax.annotate('', xy=head, xytext=tail,
                    arrowprops=dict(arrowstyle='<->', color=color,
                                    lw=width, shrinkA=0, shrinkB=0,
                                    mutation_scale=mscale),
                    zorder=12)

c1_soft = '#00e0ff'; c1_stiff = '#4da8ff'
c2_soft = '#ff4ac2'; c2_stiff = '#ff8aa0'
draw_principal_arrows(ax, minima[0], c1_soft, c1_stiff,
                       length_soft=0.28, length_stiff=0.09,
                       lw=round(3.0 * S, 2))
draw_principal_arrows(ax, minima[1], c2_soft, c2_stiff,
                       length_soft=0.28, length_stiff=0.11,
                       lw=round(3.0 * S, 2))

# Landmarks  (original markersize 14 / 16, edgewidth 1.5)
for m in minima:
    ax.plot(m[0], m[1], 'o', color='red',
            markersize=round(14 * S, 1),
            markeredgecolor='black', markeredgewidth=round(1.5 * S, 2),
            zorder=13)
ax.plot(saddle[0], saddle[1], '^', color='orange',
        markersize=round(16 * S, 1),
        markeredgecolor='black', markeredgewidth=round(1.5 * S, 2),
        zorder=13)

# Labels
ax.text(minima[0][0]+0.25, minima[0][1]-0.12,
        r'$q^\star_1$', fontsize=7, color='white', fontweight='bold')
ax.text(minima[1][0]+0.20, minima[1][1]-0.12,
        r'$q^\star_2$', fontsize=7, color='white', fontweight='bold')
ax.text(saddle[0]+0.10, saddle[1]-0.12, 'saddle',
        fontsize=6, color='#1a1a1a', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.85,
                  edgecolor='orange', lw=round(1.0 * S, 2)))

# T-operator annotations
ax.annotate(r'$\mathcal{T}_{(q^\star_1, z)}$',
            xy=(minima[0][0]-0.15, minima[0][1]+0.15),
            xytext=(-1.58, 2.00),
            fontsize=6, color='#006fa6', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                      ec='#006fa6', lw=round(1.6 * S, 2), alpha=0.92),
            arrowprops=dict(arrowstyle='->', color='white',
                            lw=round(1.5 * S, 2)))
ax.annotate(r'$\mathcal{T}_{(q^\star_2, z)}$',
            xy=(minima[1][0]+0.15, minima[1][1]+0.10),
            xytext=(-0.25, 0.90),
            fontsize=6, color='#b5006e', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                      ec='#b5006e', lw=round(1.6 * S, 2), alpha=0.92),
            arrowprops=dict(arrowstyle='->', color='white',
                            lw=round(1.5 * S, 2)))

# ----- Sample path -----
n_path = 300
t_path = np.linspace(0.0, 1.0, n_path)
path_x = ((1 - t_path)**2 * minima[0][0]
          + 2 * (1 - t_path) * t_path * saddle[0]
          + t_path**2 * minima[1][0])
path_y = ((1 - t_path)**2 * minima[0][1]
          + 2 * (1 - t_path) * t_path * saddle[1]
          + t_path**2 * minima[1][1])

ax.plot(path_x, path_y, '-', color='white',
        lw=round(2.5 * S, 2), alpha=0.7, zorder=4)
ax.plot(path_x, path_y, '--', color='#ff4d6d',
        lw=round(1.8 * S, 2), alpha=0.95, zorder=5)
ax.plot(path_x[0],  path_y[0],  'D', color='#ff4d6d',
        markersize=round(6 * S, 1),
        markeredgecolor='white', markeredgewidth=round(1.0 * S, 2), zorder=6)
ax.plot(path_x[-1], path_y[-1], 'D', color='#ff4d6d',
        markersize=round(6 * S, 1),
        markeredgecolor='white', markeredgewidth=round(1.0 * S, 2), zorder=6)

ax.set_xlim(-1.7, 1.2); ax.set_ylim(-0.4, 2.2)
ax.set_xlabel(r'$q_1$', fontsize=7)
ax.set_ylabel(r'$q_2$', fontsize=7)
ax.set_title(r'(a) Potential landscape with $\mathcal{T}$ principal axes',
             fontsize=7)
ax.tick_params(labelsize=6)


# ------------- Panel (b): variables along the path -------------
ax2 = fig.add_subplot(gs[0, 1])

U_path = np.array([U(np.array([[px]]), np.array([[py]]))[0, 0]
                   for px, py in zip(path_x, path_y)])

lam_min_path  = np.zeros(n_path)
lam_max_path  = np.zeros(n_path)
sigma_min_path = np.zeros(n_path)
for i in range(n_path):
    H = Hessian(path_x[i], path_y[i])
    eigs = np.linalg.eigvalsh(H)
    lam_min_path[i]   = eigs[0]
    lam_max_path[i]   = eigs[1]
    sigma_min_path[i] = min(abs(eigs[0]), abs(eigs[1]))

T_norm_path = 1.0 / np.maximum(sigma_min_path, 1e-6)
kappa_path  = (np.maximum(np.abs(lam_min_path), np.abs(lam_max_path))
               / np.maximum(sigma_min_path, 1e-6))

crossings = []
for i in range(1, n_path):
    if lam_min_path[i-1] * lam_min_path[i] < 0:
        crossings.append(i)

s = t_path

l1, = ax2.semilogy(s, kappa_path, color='#d62728',
                    lw=round(2.5 * S, 2),
                    label=r'$\kappa(\nabla^2 U)$')
l2, = ax2.semilogy(s, T_norm_path * 1e3, color='#9467bd',
                    lw=round(2.5 * S, 2), linestyle='--',
                    label=r'$\|\mathcal{T}\| \times 10^3$')
ax2.set_ylabel(r'$\kappa$  /  $\|\mathcal{T}\|$ (log scale)',
               fontsize=7, color='#333333')
ax2.set_xlabel(r'Path parameter $s$: '
               r'$q^\star_1 \to \mathrm{saddle} \to q^\star_2$',
               fontsize=7)
ax2.tick_params(labelsize=6)

band_half_width = 0.04
for ci in crossings:
    ax2.axvspan(s[ci] - band_half_width, s[ci] + band_half_width,
                color='#ff4d6d', alpha=0.18, zorder=0)

for ci in crossings:
    ax2.axvline(s[ci], color='#ff4d6d', linestyle=':',
                lw=round(2.0 * S, 2), alpha=0.95, zorder=1)

if len(crossings) >= 1:
    ax2.text(s[crossings[0]], 4.0e5,
             r'$\lambda_{\min}=0$',
             ha='center', va='top', fontsize=5.5, color='#ff4d6d',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.1', fc='white',
                       ec='#ff4d6d', lw=round(1.2 * S, 2)))
    if len(crossings) > 1:
        ax2.text(s[crossings[-1]], 4.0e5,
                 r'$\lambda_{\min}=0$',
                 ha='center', va='top', fontsize=5.5, color='#ff4d6d',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.1', fc='white',
                           ec='#ff4d6d', lw=round(1.2 * S, 2)))

ax2r = ax2.twinx()
l3, = ax2r.plot(s, lam_min_path, color='#2ca02c',
                 lw=round(2.2 * S, 2),
                 label=r'$\lambda_{\min}(\nabla^2 U)$')
l4, = ax2r.plot(s, lam_max_path, color='#1f77b4',
                 lw=round(1.6 * S, 2), alpha=0.8,
                 label=r'$\lambda_{\max}(\nabla^2 U)$')
ax2r.axhline(0, color='black', lw=round(0.6 * S, 2),
             linestyle='-', alpha=0.5)
ax2r.set_ylabel(r'Eigenvalues $\lambda$ (linear)',
                fontsize=7, color='#333333')
ax2r.tick_params(labelsize=6)

# Mark endpoints  (original markersize 11, edgewidth 1.3)
ax2.plot(0, kappa_path[0], 'o', color='red',
         markersize=round(11 * S, 1),
         markeredgecolor='black', markeredgewidth=round(1.3 * S, 2),
         zorder=10)
ax2.plot(1, kappa_path[-1], 'o', color='red',
         markersize=round(11 * S, 1),
         markeredgecolor='black', markeredgewidth=round(1.3 * S, 2),
         zorder=10)
ax2.annotate(r'$q^\star_1$', xy=(0, kappa_path[0]),
             xytext=(0.02, 5), fontsize=7, color='red', fontweight='bold')
ax2.annotate(r'$q^\star_2$', xy=(1, kappa_path[-1]),
             xytext=(0.93, 5), fontsize=7, color='red', fontweight='bold')

ax2.set_xlim(0, 1)
ax2.set_ylim(1, 5e5)
ax2.set_title(r'(b) STO-relevant quantities along the path', fontsize=7)
ax2.grid(True, alpha=0.25, which='both', linestyle=':')

from matplotlib.patches import Patch
all_lines = [l1, l2, l3, l4,
             Patch(facecolor='#ff4d6d', alpha=0.25,
                   label='invalidation band')]
ax2.legend(handles=all_lines,
           loc='upper center', fontsize=5.5, framealpha=0.95,
           bbox_to_anchor=(0.5, 0.99), ncol=1,
           handlelength=0.8, handletextpad=0.15,
           labelspacing=0.08, borderpad=0.2,
           columnspacing=0.4)

plt.tight_layout()
plt.savefig('T_operator_ml_v3.png', dpi=160, bbox_inches='tight')
plt.savefig('T_operator_ml_v3.pdf', bbox_inches='tight')
print("Saved: T_operator_ml_v3.png / .pdf")

print("\n=== Numerical summary along path ===")
for name, idx in [('q1*', 0), ('saddle (nearest)', n_path // 2),
                   ('q2*', n_path - 1)]:
    print(f"  {name}:  kappa = {kappa_path[idx]:.2e},  "
          f"||T|| = {T_norm_path[idx]:.3e},  "
          f"lam_min = {lam_min_path[idx]:+.2e}")
max_k = np.max(kappa_path[np.isfinite(kappa_path)])
print(f"  max kappa along path: {max_k:.2e}")
