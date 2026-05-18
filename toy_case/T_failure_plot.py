"""
Corrected visualization: the STO 'failure set' is a 1D curve
(lambda_min = 0), not a 2D region.

Changes vs. previous version:
  - Remove the large grey overlay on the indefinite region 
    (it doesn't represent STO failure, just Hessian indefiniteness).
  - Keep the full ||T|| heatmap everywhere (including outside PD region) 
    to show how quickly it diverges.
  - Add explicit iso-contours of ||T|| to make 'invalidation thresholds' 
    visible as 1D curves.
  - Clearly label the singular locus as the true failure boundary.
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

xx = np.linspace(-1.7, 1.2, 400)
yy = np.linspace(-0.4, 2.2, 400)
Xg, Yg = np.meshgrid(xx, yy)
Z = U(Xg, Yg); Zc = np.clip(Z, -150, 100)

Uxx, Uyy, Uxy = hess_fields(Xg, Yg)
tr = Uxx + Uyy
disc = np.sqrt(np.maximum((Uxx - Uyy)**2 / 4 + Uxy**2, 0.0))
lam1 = tr/2 + disc
lam2 = tr/2 - disc

sigma_min = np.minimum(np.abs(lam1), np.abs(lam2))
T_norm = 1.0 / np.maximum(sigma_min, 1e-4)

minima = [(-0.558, 1.442), (0.641, 0.009)]
saddle = (-0.673, 0.417)

# ====================================================
# FIGURE
# ====================================================
fig, ax = plt.subplots(1, 1, figsize=(11.5, 9.8))

# ---- Background: ||T|| heatmap everywhere (no masking) ----
T_vis = np.clip(T_norm, 1e-4, 1e2)
im = ax.pcolormesh(Xg, Yg, T_vis, cmap='viridis',
                   norm=LogNorm(vmin=1e-3, vmax=1e-1), shading='auto',
                   zorder=1)
cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label(r'$\|\mathcal{T}_{(q,z)}\| = 1/\sigma_{\min}(\nabla^2 U)$  (log)',
               fontsize=12)

# ---- Faint potential contours ----
ax.contour(Xg, Yg, Zc, levels=15, colors='white', linewidths=0.5,
           alpha=0.28, zorder=2)

# ---- Iso-contours of ||T|| (invalidation thresholds) ----
# These 1D curves are where STO's kappa_max would trigger
T_levels = [5e-3, 1e-2, 3e-2]
cs_T = ax.contour(Xg, Yg, T_norm, levels=T_levels,
                   colors=['#ffffff', '#ffcc00', '#ff7700'],
                   linewidths=[1.3, 1.8, 2.3],
                   linestyles=['--', '-.', '-'],
                   alpha=0.9, zorder=3)
# Label the contours
fmt = {5e-3: r'$\|\mathcal{T}\|=5\cdot 10^{-3}$',
       1e-2: r'$\|\mathcal{T}\|=10^{-2}$',
       3e-2: r'$\|\mathcal{T}\|=3\cdot 10^{-2}$'}
ax.clabel(cs_T, inline=True, fontsize=8.5, fmt=fmt, inline_spacing=5)

# ---- THE SINGULAR LOCUS (true STO failure boundary) ----
ax.contour(Xg, Yg, lam2, levels=[0], colors='#ff1e5a',
           linewidths=3.5, linestyles='-', zorder=4)

# ---- Compliance ellipses (orientation only, size normalized) ----
def draw_orientation_ellipse(ax, p, color, target_radius=0.09,
                              face_alpha=0.30, lw=1.6, zorder=5):
    H = Hessian(*p)
    eigs, V = np.linalg.eigh(H)
    if not (eigs[0] > 0 and eigs[1] > 0):
        return False
    aa_raw = 1.0 / np.sqrt(eigs[0])
    bb_raw = 1.0 / np.sqrt(eigs[1])
    g = np.sqrt(aa_raw * bb_raw)
    aa = aa_raw / g * target_radius
    bb = bb_raw / g * target_radius
    angle = np.degrees(np.arctan2(V[1,0], V[0,0]))
    e = Ellipse(p, width=2*aa, height=2*bb, angle=angle,
                facecolor=color, alpha=face_alpha,
                edgecolor=color, lw=lw, zorder=zorder)
    ax.add_patch(e)
    e2 = Ellipse(p, width=2*aa, height=2*bb, angle=angle,
                 facecolor='none', edgecolor='white', lw=0.5,
                 alpha=0.5, zorder=zorder+0.3)
    ax.add_patch(e2)
    ax.plot(p[0], p[1], '.', color='white', markersize=3, zorder=zorder+1)
    return True

c1 = '#00e0ff'
c2 = '#ff4ac2'

def sample_pts(center, half=0.28, n=3):
    pts = []
    for dq1 in np.linspace(-half, half, n):
        for dq2 in np.linspace(-half, half, n):
            pts.append((center[0]+dq1, center[1]+dq2))
    return pts

for p in sample_pts(minima[0], half=0.30, n=3):
    draw_orientation_ellipse(ax, p, c1, target_radius=0.075,
                              face_alpha=0.18, lw=1.2, zorder=6)
for p in sample_pts(minima[1], half=0.30, n=3):
    draw_orientation_ellipse(ax, p, c2, target_radius=0.075,
                              face_alpha=0.18, lw=1.2, zorder=6)

# Large prominent ellipses at minima
draw_orientation_ellipse(ax, minima[0], c1, target_radius=0.14,
                          face_alpha=0.55, lw=3.2, zorder=8)
draw_orientation_ellipse(ax, minima[1], c2, target_radius=0.14,
                          face_alpha=0.55, lw=3.2, zorder=8)

# Principal (softest) direction arrows
for m, col in [(minima[0], c1), (minima[1], c2)]:
    H = Hessian(*m)
    eigs, V = np.linalg.eigh(H)
    v_soft = V[:, 0]
    L = 0.18
    ax.annotate('', xy=(m[0] + L*v_soft[0], m[1] + L*v_soft[1]),
                xytext=(m[0] - L*v_soft[0], m[1] - L*v_soft[1]),
                arrowprops=dict(arrowstyle='<->', color=col, lw=2.0,
                                shrinkA=0, shrinkB=0, alpha=0.9),
                zorder=10)

# Landmarks
for m in minima:
    ax.plot(m[0], m[1], 'o', color='red', markersize=16,
            markeredgecolor='black', markeredgewidth=1.6, zorder=12)
ax.plot(saddle[0], saddle[1], '^', color='orange', markersize=18,
        markeredgecolor='black', markeredgewidth=1.6, zorder=12)

# Branch annotations
ax.annotate(r'$\mathcal{T}_{(q^\star_1,z)}$',
            xy=minima[0], xytext=(-1.50, 1.95),
            fontsize=16, color=c1, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='black',
                      ec=c1, lw=1.8, alpha=0.82),
            arrowprops=dict(arrowstyle='->', color=c1, lw=2.0))
ax.annotate(r'$\mathcal{T}_{(q^\star_2,z)}$',
            xy=minima[1], xytext=(0.40, 0.70),
            fontsize=16, color=c2, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='black',
                      ec=c2, lw=1.8, alpha=0.82),
            arrowprops=dict(arrowstyle='->', color=c2, lw=2.0))

# Annotation for singular locus (the REAL failure set)
ax.annotate('TRUE STO failure set:\n'
            'the 1D curve '
            r'$\{q : \lambda_{\min}(\nabla^2 U)=0\}$' + '\n'
            r'Here $\|\mathcal{T}\| \to \infty$.',
            xy=(-0.92, 1.84), 
            xytext=(-0.12, 2.04),
            fontsize=10, color='#ff1e5a', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', fc='white',
                      ec='#ff1e5a', lw=1.5),
            arrowprops=dict(arrowstyle='->', color='#ff1e5a', lw=1.5))

# Landmark text labels
ax.text(minima[0][0]+0.18, minima[0][1]-0.15,
        r'$q^\star_1$', fontsize=15, color='white', fontweight='bold')
ax.text(minima[1][0]+0.15, minima[1][1]-0.13,
        r'$q^\star_2$', fontsize=15, color='white', fontweight='bold')
ax.text(saddle[0]-0.28, saddle[1]-0.04, 'saddle',
        fontsize=11, color='white', fontweight='bold')

# Legend
legend_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='red',
           markeredgecolor='black', markersize=12, lw=0,
           label=r'Stable equilibrium $q^\star$ (STO anchor)'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='orange',
           markeredgecolor='black', markersize=13, lw=0, label='Saddle'),
    Line2D([0],[0], color='#ff1e5a', lw=3.5,
           label=r'Singular locus: $\lambda_{\min}=0$ (true failure)'),
    Line2D([0],[0], color='#ff7700', lw=2.3, linestyle='-',
           label=r'$\|\mathcal{T}\|$ iso-contour (invalidation)'),
    Line2D([0],[0], color=c1, lw=2.8, label='Branch 1 ellipse'),
    Line2D([0],[0], color=c2, lw=2.8, label='Branch 2 ellipse'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=9.5,
          framealpha=0.92, facecolor='white', edgecolor='black')

ax.text(0.02, 0.01,
        r'Colour = $\|\mathcal{T}\|$  |  '
        r'Ellipse = principal structure of $\mathcal{T}$  |  '
        r'Red curve = true STO failure set',
        fontsize=9.5, color='white', fontweight='bold',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.82))

ax.set_xlim(-1.7, 1.2); ax.set_ylim(-0.4, 2.2)
ax.set_xlabel(r'$q_1$', fontsize=13)
ax.set_ylabel(r'$q_2$', fontsize=13)
ax.set_title(r'STO failure is a 1D curve, not a 2D region: '
             r'$\|\mathcal{T}\|$ diverges only on $\lambda_{\min}(\nabla^2 U)=0$',
             fontsize=12, pad=10)

plt.tight_layout()
plt.savefig('/home/claude/T_failure_corrected.png', dpi=160, bbox_inches='tight')
plt.savefig('/home/claude/T_failure_corrected.pdf', bbox_inches='tight')
print("Saved: T_failure_corrected.png / .pdf")
