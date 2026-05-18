"""
Combined visualization v2:
  Background = ||T|| heatmap (magnitude)
  Overlay    = ORIENTATION ellipses (normalized so each ellipse has the
               same area; shape captures anisotropy and principal axes)

Key fix: ellipses no longer shrink in stiff regions. Each ellipse has
axes proportional to (1/sqrt(lam_i)) but then globally normalized so
the product of axes is constant. This way:
  - ellipse ORIENTATION = eigenvectors of T (= eigenvectors of Hessian)
  - ellipse ASPECT RATIO = anisotropy of T
  - ellipse SIZE = constant (for visual clarity)
  - magnitude of T is conveyed by the background colour instead
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

# ---- Modified Müller-Brown (A_2 = 0) ----
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

# ---- Grid ----
xx = np.linspace(-1.7, 1.2, 400)
yy = np.linspace(-0.4, 2.2, 400)
# xx = np.linspace(-3, 3, 400)
# yy = np.linspace(-3, 3, 400)
Xg, Yg = np.meshgrid(xx, yy)
Z = U(Xg, Yg); Zc = np.clip(Z, -150, 100)

Uxx, Uyy, Uxy = hess_fields(Xg, Yg)
tr = Uxx + Uyy
disc = np.sqrt(np.maximum((Uxx - Uyy)**2 / 4 + Uxy**2, 0.0))
lam1 = tr/2 + disc
lam2 = tr/2 - disc

sigma_min = np.minimum(np.abs(lam1), np.abs(lam2))
T_norm = 1.0 / np.maximum(sigma_min, 1e-4)
pd_mask = (lam1 > 0) & (lam2 > 0)
non_pd = ~pd_mask

minima = [(-0.558, 1.442), (0.641, 0.009)]
saddle = (-0.673, 0.417)

# ====================================================
# FIGURE
# ====================================================
fig, ax = plt.subplots(1, 1, figsize=(11.5, 9.8))

# Background heatmap of ||T||
T_vis = np.clip(T_norm, 1e-4, 1e2)
im = ax.pcolormesh(Xg, Yg, T_vis, cmap='viridis',
                   norm=LogNorm(vmin=1e-3, vmax=5e-2), shading='auto',
                   zorder=1)
cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label(r'$\|\mathcal{T}_{(q,z)}\| = 1/\sigma_{\min}(\nabla^2 U)$  (log)',
               fontsize=12)

# Non-PD overlay
overlay = np.zeros(non_pd.shape + (4,))
overlay[non_pd] = [0.08, 0.08, 0.08, 0.68]
ax.imshow(overlay, extent=[xx[0], xx[-1], yy[0], yy[-1]],
          origin='lower', aspect='auto', zorder=2)

# Singular locus
ax.contour(Xg, Yg, lam2, levels=[0], colors='#ff2a6d',
           linewidths=3.0, linestyles='-', zorder=3)

# Faint potential contours
ax.contour(Xg, Yg, Zc, levels=15, colors='white', linewidths=0.5,
           alpha=0.25, zorder=3)

# ---- ORIENTATION ELLIPSES: normalized area ----
def draw_orientation_ellipse(ax, p, color, target_radius=0.09,
                              face_alpha=0.30, lw=1.6, zorder=5):
    """Ellipse with principal axes from Hessian eigenvectors but area
    normalized. target_radius = geometric mean of half-axes."""
    H = Hessian(*p)
    eigs, V = np.linalg.eigh(H)
    if not (eigs[0] > 0 and eigs[1] > 0):
        return False
    # Raw compliance half-axes
    aa_raw = 1.0 / np.sqrt(eigs[0])
    bb_raw = 1.0 / np.sqrt(eigs[1])
    # Normalize so geometric mean = target_radius
    g = np.sqrt(aa_raw * bb_raw)
    aa = aa_raw / g * target_radius
    bb = bb_raw / g * target_radius
    angle = np.degrees(np.arctan2(V[1,0], V[0,0]))
    e = Ellipse(p, width=2*aa, height=2*bb, angle=angle,
                facecolor=color, alpha=face_alpha,
                edgecolor=color, lw=lw, zorder=zorder)
    ax.add_patch(e)
    # White outline for visibility
    e2 = Ellipse(p, width=2*aa, height=2*bb, angle=angle,
                 facecolor='none', edgecolor='white', lw=0.5,
                 alpha=0.5, zorder=zorder+0.3)
    ax.add_patch(e2)
    ax.plot(p[0], p[1], '.', color='white', markersize=3, zorder=zorder+1)
    return True

# Colors per branch
c1 = '#00e0ff'   # cyan for branch 1
c2 = '#ff4ac2'   # magenta for branch 2

# Sample 3x3 grid in each basin
def sample_pts(center, half=0.28, n=3):
    pts = []
    for dq1 in np.linspace(-half, half, n):
        for dq2 in np.linspace(-half, half, n):
            pts.append((center[0]+dq1, center[1]+dq2))
    return pts

for p in sample_pts(minima[0], half=0.30, n=3):
    draw_orientation_ellipse(ax, p, c1, target_radius=0.075,
                              face_alpha=0.20, lw=1.3, zorder=5)
for p in sample_pts(minima[1], half=0.30, n=3):
    draw_orientation_ellipse(ax, p, c2, target_radius=0.075,
                              face_alpha=0.20, lw=1.3, zorder=5)

# Big prominent ellipses at the minima
draw_orientation_ellipse(ax, minima[0], c1, target_radius=0.14,
                          face_alpha=0.55, lw=3.2, zorder=8)
draw_orientation_ellipse(ax, minima[1], c2, target_radius=0.14,
                          face_alpha=0.55, lw=3.2, zorder=8)

# Landmarks
for m in minima:
    ax.plot(m[0], m[1], 'o', color='red', markersize=16,
            markeredgecolor='black', markeredgewidth=1.6, zorder=12)
ax.plot(saddle[0], saddle[1], '^', color='orange', markersize=18,
        markeredgecolor='black', markeredgewidth=1.6, zorder=12)

# Branch labels
ax.annotate(r'$\mathcal{T}_{(q^\star_1,z)}$',
            xy=minima[0], xytext=(-1.45, 1.95),
            fontsize=16, color=c1, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='black',
                      ec=c1, lw=1.8, alpha=0.82),
            arrowprops=dict(arrowstyle='->', color=c1, lw=2.0))
ax.annotate(r'$\mathcal{T}_{(q^\star_2,z)}$',
            xy=minima[1], xytext=(0.40, 0.68),
            fontsize=16, color=c2, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='black',
                      ec=c2, lw=1.8, alpha=0.82),
            arrowprops=dict(arrowstyle='->', color=c2, lw=2.0))

# Singular locus annotation
ax.annotate('Singular locus\n'
            r'$\lambda_{\min}(\nabla^2 U)=0$' + '\n'
            r'$\|\mathcal{T}\| \to \infty$' + '\n'
            r'$\Rightarrow$ STO invalidates',
            xy=(saddle[0]+0.27, saddle[1]-0.05),
            xytext=(-0.10, 1.52),
            fontsize=10.5, color='#ff2a6d', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', fc='white',
                      ec='#ff2a6d', lw=1.5),
            arrowprops=dict(arrowstyle='->', color='#ff2a6d', lw=1.5))

# Non-PD region label (arrow pointing into the actual non-PD grey area)
ax.annotate('Non-PD region\n($\\nabla^2 U$ indefinite)',
            xy=(-1.35, 0.05),
            xytext=(-1.65, 2.05),
            fontsize=10, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='black',
                      ec='white', lw=1.0, alpha=0.78),
            va='top',
            arrowprops=dict(arrowstyle='->', color='white', lw=1.2))

# Principal direction arrows at the big ellipses (extra clarity)
for m, col in [(minima[0], c1), (minima[1], c2)]:
    H = Hessian(*m)
    eigs, V = np.linalg.eigh(H)
    # Softest direction = V[:,0]
    v_soft = V[:, 0]
    L = 0.18
    ax.annotate('', xy=(m[0] + L*v_soft[0], m[1] + L*v_soft[1]),
                xytext=(m[0] - L*v_soft[0], m[1] - L*v_soft[1]),
                arrowprops=dict(arrowstyle='<->', color=col, lw=1.8,
                                shrinkA=0, shrinkB=0, alpha=0.9),
                zorder=10)

# Landmark labels
ax.text(minima[0][0]+0.18, minima[0][1]-0.15,
        r'$q^\star_1$', fontsize=15, color='white', fontweight='bold')
ax.text(minima[1][0]+0.15, minima[1][1]-0.12,
        r'$q^\star_2$', fontsize=15, color='white', fontweight='bold')
ax.text(saddle[0]-0.28, saddle[1]-0.04, 'saddle',
        fontsize=11, color='white', fontweight='bold')

# Legend
legend_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='red',
           markeredgecolor='black', markersize=12, lw=0,
           label=r'Stable equilibrium $q^\star$'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='orange',
           markeredgecolor='black', markersize=13, lw=0, label='Saddle'),
    Line2D([0],[0], color='#ff2a6d', lw=3, label=r'Singular locus ($\lambda_{\min}=0$)'),
    Line2D([0],[0], color=c1, lw=2.8, label='Branch 1 ellipse'),
    Line2D([0],[0], color=c2, lw=2.8, label='Branch 2 ellipse'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=10,
          framealpha=0.92, facecolor='white', edgecolor='black')

# Bottom caption
ax.text(0.02, 0.01,
        r'Colour: magnitude $\|\mathcal{T}\|$  |  '
        r'Ellipse orientation and aspect: principal axes of $\mathcal{T}$  '
        r'(size normalized)',
        fontsize=9.5, color='white', fontweight='bold',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.8))

ax.set_xlim(-1.7, 1.2); ax.set_ylim(-0.4, 2.2)
# ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.set_xlabel(r'$q_1$', fontsize=13)
ax.set_ylabel(r'$q_2$', fontsize=13)
ax.set_title(r'Equilibrium tangent operator $\mathcal{T}_{(q,z)}$: '
             r'magnitude (colour) and principal structure (ellipses)',
             fontsize=12.5, pad=10)

plt.tight_layout()
plt.savefig('T_combined_v2.png', dpi=160, bbox_inches='tight')
# plt.savefig('T_combined_v2.pdf', bbox_inches='tight')
print("Saved: T_combined_v2.png / .pdf")
