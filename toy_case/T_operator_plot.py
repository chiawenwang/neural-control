"""
Cleaner visualization of T_{(q,z)} on the modified Müller-Brown potential.

Three panels, each with a clear message:
  (a) Potential landscape with branches = basins of attraction
  (b) log(||T||) heatmap -- the key STO quantity, with 'singular locus' 
      (lambda_min = 0) marked. Outside this locus T is not well-defined 
      as a stable tangent operator.
  (c) Hessian (compliance) ellipses at selected anchor points on each 
      branch -- showing T is a DIFFERENT operator on each branch.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

# Modified Müller-Brown
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

# Grid
xx = np.linspace(-1.7, 1.2, 350)
yy = np.linspace(-0.4, 2.2, 350)
Xg, Yg = np.meshgrid(xx, yy)
Z = U(Xg, Yg); Zc = np.clip(Z, -150, 100)
Uxx, Uyy, Uxy = hess_fields(Xg, Yg)
tr = Uxx + Uyy
det = Uxx*Uyy - Uxy**2
disc = np.sqrt(np.maximum((Uxx - Uyy)**2 / 4 + Uxy**2, 0.0))
lam1 = tr/2 + disc   # larger eigenvalue
lam2 = tr/2 - disc   # smaller eigenvalue

# Signed minimum eigenvalue (can be negative in saddle region)
lam_min_signed = lam2
# Operator norm ||T|| = 1/|sigma_min(H)|; but for a valid branch, need H PD,
# so min(|eig|) is what matters.
sigma_min = np.minimum(np.abs(lam1), np.abs(lam2))
T_norm = 1.0 / np.maximum(sigma_min, 1e-4)
# Mask out regions where H is not positive definite
pd_mask = (lam1 > 0) & (lam2 > 0)

# Landmarks
minima = [(-0.558, 1.442), (0.641, 0.009)]
saddle = (-0.673, 0.417)

# ====================================================
# FIGURE  --  three panels in a row
# ====================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

# --------- Panel (a): landscape with basins of attraction ---------
ax = axes[0]
levels = np.linspace(-150, 100, 30)
cs = ax.contourf(Xg, Yg, Zc, levels=levels, cmap=cm.plasma, extend='both')
ax.contour(Xg, Yg, Zc, levels=15, colors='k', linewidths=0.3, alpha=0.35)
plt.colorbar(cs, ax=ax, shrink=0.82, label=r'$U(q)$')

for k, m in enumerate(minima):
    ax.plot(m[0], m[1], 'o', color='red', markersize=14,
            markeredgecolor='black', markeredgewidth=1.4, zorder=5)
    ax.annotate(r'$q^\star_{%d}$ (branch %d)' % (k+1, k+1),
                xy=m, xytext=(m[0] + (-0.05 if k==0 else 0.05),
                              m[1] + (-0.25 if k==0 else -0.25)),
                fontsize=12, color='white', fontweight='bold',
                ha='center')
ax.plot(saddle[0], saddle[1], '^', color='orange', markersize=16,
        markeredgecolor='black', markeredgewidth=1.4, zorder=5)
ax.annotate('saddle', xy=saddle, xytext=(saddle[0]-0.32, saddle[1]+0.15),
            fontsize=11, color='white', fontweight='bold')

# Basin separatrix (approximate: contour of U through saddle, restricted
# to the region between the two minima)
sep_level = U(np.array([[saddle[0]]]), np.array([[saddle[1]]]))[0,0]
ax.contour(Xg, Yg, Z, levels=[sep_level], colors='cyan', linewidths=2,
           linestyles='--')

ax.set_xlim(-1.7, 1.2); ax.set_ylim(-0.4, 2.2)
ax.set_xlabel(r'$q_1$', fontsize=12); ax.set_ylabel(r'$q_2$', fontsize=12)
ax.set_title('(a) Two equilibrium branches = two basins', fontsize=12)

# Custom legend
ax.legend(handles=[
    Line2D([0],[0], color='cyan', linestyle='--', lw=2, label='Basin boundary (separatrix)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='red',
           markeredgecolor='black', markersize=10, lw=0, label='Stable equilibrium'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='orange',
           markeredgecolor='black', markersize=11, lw=0, label='Saddle'),
], loc='upper right', fontsize=9, framealpha=0.92)

# --------- Panel (b): ||T|| heatmap ---------
ax = axes[1]

# Fill the entire region with log(||T||), but visually distinguish
# the PD region from the indefinite region.
# Approach: render ||T|| everywhere, then overlay a hatched mask on non-PD region.
T_vis = np.clip(T_norm, 1e-4, 1e2)

# Full heatmap
im = ax.pcolormesh(Xg, Yg, T_vis, cmap='viridis',
                   norm=LogNorm(vmin=1e-3, vmax=1e-1), shading='auto')
plt.colorbar(im, ax=ax, shrink=0.82,
             label=r'$\|\mathcal{T}_{(q,z)}\| = 1/|\sigma_{\min}(\nabla^2 U)|$  (log)')

# Overlay non-PD region with hatching
# Create a binary mask and hatch it
non_pd = ~pd_mask
# White semi-transparent overlay on non-PD region
overlay = np.zeros(non_pd.shape + (4,))
overlay[non_pd] = [1, 1, 1, 0.55]
ax.imshow(overlay, extent=[xx[0], xx[-1], yy[0], yy[-1]],
          origin='lower', aspect='auto', zorder=3)

# Singular locus: lambda_min = 0 boundary
ax.contour(Xg, Yg, lam2, levels=[0], colors='red',
           linewidths=2.8, linestyles='-', zorder=4)

# Potential contours for reference
ax.contour(Xg, Yg, Zc, levels=12, colors='white', linewidths=0.4,
           alpha=0.35, zorder=2)

# Landmarks
for k, m in enumerate(minima):
    ax.plot(m[0], m[1], 'o', color='red', markersize=14,
            markeredgecolor='black', markeredgewidth=1.4, zorder=10)
ax.plot(saddle[0], saddle[1], '^', color='orange', markersize=16,
        markeredgecolor='black', markeredgewidth=1.4, zorder=10)

# Text annotations
ax.text(-1.6, 2.05, 'Non-PD region:\n$\\nabla^2 U$ indefinite,\n'
                    '$\\mathcal{T}$ not a valid\ntangent operator',
        fontsize=10, color='black',
        bbox=dict(boxstyle='round,pad=0.35', fc='white',
                  ec='red', lw=1.3), va='top')

ax.annotate('Singular locus:\n$\\lambda_{\\min}(\\nabla^2 U) = 0$\n'
            r'$\|\mathcal{T}\| \to \infty$',
            xy=(saddle[0]+0.15, saddle[1]-0.3),
            xytext=(0.15, 1.6),
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', fc='white',
                      ec='red', lw=1.3),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.3))

ax.text(minima[0][0]+0.45, minima[0][1]+0.15,
        f'$\\|\\mathcal{{T}}\\|\\approx${1/min(np.abs(np.linalg.eigvalsh(Hessian(*minima[0])))):.1e}',
        fontsize=9.5, color='white', fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))
ax.text(minima[1][0]-0.40, minima[1][1]+0.05,
        f'$\\|\\mathcal{{T}}\\|\\approx${1/min(np.abs(np.linalg.eigvalsh(Hessian(*minima[1])))):.1e}',
        fontsize=9.5, color='white', fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))

ax.set_xlim(-1.7, 1.2); ax.set_ylim(-0.4, 2.2)
ax.set_xlabel(r'$q_1$', fontsize=12); ax.set_ylabel(r'$q_2$', fontsize=12)
ax.set_title(r'(b) $\|\mathcal{T}_{(q,z)}\|$ — branch-local magnitude', fontsize=12)

# --------- Panel (c): Hessian ellipses on each branch ---------
ax = axes[2]

# Faint background
ax.contourf(Xg, Yg, Zc, levels=levels, cmap=cm.plasma, alpha=0.55)
ax.contour(Xg, Yg, Zc, levels=18, colors='k', linewidths=0.3, alpha=0.3)

# Hatch non-PD region
overlay2 = np.zeros(non_pd.shape + (4,))
overlay2[non_pd] = [0.3, 0.3, 0.3, 0.55]
ax.imshow(overlay2, extent=[xx[0], xx[-1], yy[0], yy[-1]],
          origin='lower', aspect='auto', zorder=2)

# Draw T-ellipses at points within each basin
# Branch 1 (upper left): sample around min_1
branch1_pts = []
for dq1 in [-0.25, 0, 0.25]:
    for dq2 in [-0.25, 0, 0.25]:
        p = (minima[0][0]+dq1, minima[0][1]+dq2)
        H = Hessian(*p)
        eigs = np.linalg.eigvalsh(H)
        if eigs[0] > 0 and eigs[1] > 0:   # PD check
            branch1_pts.append(p)

branch2_pts = []
for dq1 in [-0.25, 0, 0.25]:
    for dq2 in [-0.22, 0, 0.22]:
        p = (minima[1][0]+dq1, minima[1][1]+dq2)
        H = Hessian(*p)
        eigs = np.linalg.eigvalsh(H)
        if eigs[0] > 0 and eigs[1] > 0:
            branch2_pts.append(p)

def draw_T_ellipse(ax, p, color, scale=0.18, alpha=0.7, lw=1.8, zorder=5):
    H = Hessian(*p)
    eigs, V = np.linalg.eigh(H)
    if not (eigs[0] > 0 and eigs[1] > 0):
        return
    # Compliance ellipse: axes proportional to 1/sqrt(eig)
    aa = scale / np.sqrt(eigs[0])
    bb = scale / np.sqrt(eigs[1])
    angle = np.degrees(np.arctan2(V[1,0], V[0,0]))
    e = Ellipse(p, width=2*aa, height=2*bb, angle=angle,
                facecolor=color, alpha=alpha*0.35,
                edgecolor=color, lw=lw, zorder=zorder)
    ax.add_patch(e)
    ax.plot(p[0], p[1], '.', color=color, markersize=5, zorder=zorder+1)

for p in branch1_pts:
    draw_T_ellipse(ax, p, '#00aaff', alpha=0.6, lw=1.4)
for p in branch2_pts:
    draw_T_ellipse(ax, p, '#ff66cc', alpha=0.6, lw=1.4)

# Big ellipses at the two minima
for m, col in zip(minima, ['#0066dd', '#dd0099']):
    draw_T_ellipse(ax, m, col, scale=0.28, alpha=1.0, lw=2.5, zorder=8)

# Landmarks on top
for k, m in enumerate(minima):
    ax.plot(m[0], m[1], 'o', color='red', markersize=13,
            markeredgecolor='black', markeredgewidth=1.4, zorder=12)
ax.plot(saddle[0], saddle[1], '^', color='orange', markersize=16,
        markeredgecolor='black', markeredgewidth=1.4, zorder=12)

# Branch annotations
ax.annotate(r'Branch 1: $\mathcal{T}_{(q^\star_1,z)}$',
            xy=minima[0], xytext=(-1.55, 0.75),
            fontsize=11, color='#0066dd', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#0066dd', lw=1.4))
ax.annotate(r'Branch 2: $\mathcal{T}_{(q^\star_2,z)}$' + '\n'
            r'($\neq \mathcal{T}_{(q^\star_1,z)}$: different' + '\n'
            'shape & axis orientation)',
            xy=minima[1], xytext=(-0.2, 0.7),
            fontsize=11, color='#dd0099', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#dd0099', lw=1.4))

# Legend box
ax.text(-1.65, -0.3,
        r'Ellipse axes = principal directions of $\mathcal{T}=(\nabla^2 U)^{-1}$'
        '\n'
        'Shading = non-PD region (STO invalid)',
        fontsize=9.5, color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.8))

ax.set_xlim(-1.7, 1.2); ax.set_ylim(-0.4, 2.2)
ax.set_xlabel(r'$q_1$', fontsize=12); ax.set_ylabel(r'$q_2$', fontsize=12)
ax.set_title(r'(c) $\mathcal{T}$ is branch-local: different operator on each basin',
             fontsize=12)

plt.tight_layout()
plt.savefig('T_operator_v2.png', dpi=160, bbox_inches='tight')
# plt.savefig('T_operator_v2.pdf', bbox_inches='tight')
print("Saved: T_operator_v2.png / .pdf")

# Numerical summary
print("\nNumerical summary at landmarks:")
for name, pt in [('min_1', minima[0]), ('min_2', minima[1]), ('saddle', saddle)]:
    H = Hessian(*pt)
    eigs, V = np.linalg.eigh(H)
    print(f"  {name} @ ({pt[0]:+.2f}, {pt[1]:+.2f}):")
    print(f"    eigenvalues(H)     = [{eigs[0]:+.2f}, {eigs[1]:+.2f}]")
    if eigs[0] > 0 and eigs[1] > 0:
        print(f"    condition number   = {eigs[1]/eigs[0]:.2f}")
        print(f"    ||T|| = 1/lam_min  = {1/eigs[0]:.3e}")
        print(f"    principal dir(soft) = ({V[0,0]:+.3f}, {V[1,0]:+.3f})")
    else:
        print(f"    H is indefinite: no valid T operator")
