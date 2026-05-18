"""
Science-style figure (single submitted figure, width = 5.5 inch).

Style choices:
  - Font: Arial (or Liberation Sans, the metric-compatible substitute), 8pt
  - Math rendered with matplotlib mathtext using Helvetica/sans serif glyphs
    so that math letters match body font (DejaVu Sans fallback for symbols
    Liberation Sans does not provide).
  - Single-figure width 5.5 in, three horizontal panels.
  - Color palette (Science-friendly, colorblind-safe-ish):
        branch 1 (B1) : steel blue       #2E5A88
        branch 2 (B2) : olive             #6E8B3D     (NOT red)
        saddle / unstable (S) : red       #C0322B
        gridlines / contours : light gray #B7B7B7
  - All labels placed in axis-external or guaranteed-empty regions to
    prevent occlusion of data.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================== STYLE ==============================
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica',
                         'DejaVu Sans'],
    'font.size': 8,
    # Use mathtext rendered with stixsans so math symbols align with sans body
    'mathtext.fontset': 'stixsans',
    'mathtext.default': 'it',
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.6,
    'pdf.fonttype': 42,    # embed TrueType (editable in Illustrator)
    'ps.fonttype':  42,
})

# ============================== MODEL ==============================
A_  = np.array([-200.0, 0.0, -170.0, 15.0])
a_  = np.array([-1.0, -1.0, -6.5, 0.7])
b_  = np.array([0.0, 0.0, 11.0, 0.6])
c_  = np.array([-10.0, -10.0, -6.5, 0.7])
x0_ = np.array([1.0, 0.0, -0.5, -1.0])
y0_ = np.array([0.0, 0.5, 1.5, 1.0])

def U_scalar(q):
    q1, q2 = q
    r = 0.0
    for i in range(4):
        r += A_[i]*np.exp(a_[i]*(q1-x0_[i])**2 + b_[i]*(q1-x0_[i])*(q2-y0_[i])
                          + c_[i]*(q2-y0_[i])**2)
    return r

def U_grid(Xg, Yg):
    r = np.zeros_like(Xg)
    for i in range(4):
        r += A_[i]*np.exp(a_[i]*(Xg-x0_[i])**2 + b_[i]*(Xg-x0_[i])*(Yg-y0_[i])
                          + c_[i]*(Yg-y0_[i])**2)
    return r

def gradU(q):
    q1, q2 = q
    g = np.zeros(2)
    for i in range(4):
        dx = q1 - x0_[i]; dy = q2 - y0_[i]
        phi = A_[i] * np.exp(a_[i]*dx*dx + b_[i]*dx*dy + c_[i]*dy*dy)
        g[0] += phi * (2*a_[i]*dx + b_[i]*dy)
        g[1] += phi * (2*c_[i]*dy + b_[i]*dx)
    return g

def hessU(q):
    q1, q2 = q
    H = np.zeros((2, 2))
    for i in range(4):
        dx = q1 - x0_[i]; dy = q2 - y0_[i]
        phi = A_[i] * np.exp(a_[i]*dx*dx + b_[i]*dx*dy + c_[i]*dy*dy)
        ex = 2*a_[i]*dx + b_[i]*dy
        ey = 2*c_[i]*dy + b_[i]*dx
        H[0, 0] += phi * (ex*ex + 2*a_[i])
        H[1, 1] += phi * (ey*ey + 2*c_[i])
        H[0, 1] += phi * (ex*ey + b_[i])
    H[1, 0] = H[0, 1]
    return H

minima_init = [np.array([-0.558, 1.442]), np.array([0.641, 0.009])]
saddle      = np.array([-0.673, 0.417])

H_sad = hessU(saddle)
eig_sad, V_sad = np.linalg.eigh(H_sad)
c_vec = V_sad[:, np.argmin(eig_sad)]   # unstable mode at saddle

# ====================== ARCLENGTH CONTINUATION ======================
def trace_branch_general(q_start, direction_sign, ds=1.5, max_steps=4000,
                          min_abs_lam=0.05):
    """Trace any critical-point branch (stable or unstable). Stops at
    Morse-index change, near-degenerate critical point, or repeated
    corrector failure."""
    q = q_start.copy(); z = 0.0
    pts = []
    H = hessU(q); eigs = np.linalg.eigvalsh(H)
    pts.append((z, q[0], q[1], eigs[0], eigs[1]))
    n_neg_init = int((eigs < 0).sum())
    prev_t = None; cur_ds = ds; ds_min = 1e-3
    for step in range(max_steps):
        H = hessU(q)
        try:
            v = -np.linalg.solve(H, c_vec)
        except np.linalg.LinAlgError:
            break
        t_q = v; t_z = 1.0
        norm = np.sqrt(t_q @ t_q + t_z * t_z)
        t_q /= norm; t_z /= norm
        if step == 0:
            t_q *= direction_sign; t_z *= direction_sign
        else:
            if t_q @ prev_t[0] + t_z * prev_t[1] < 0:
                t_q = -t_q; t_z = -t_z
        prev_t = (t_q, t_z)
        q_pred = q + cur_ds * t_q; z_pred = z + cur_ds * t_z
        q_n = q_pred.copy(); z_n = z_pred
        ok = False
        for _ in range(60):
            F = gradU(q_n) + z_n * c_vec
            N = (q_n - q_pred) @ t_q + (z_n - z_pred) * t_z
            res = np.concatenate([F, [N]])
            if np.linalg.norm(res) < 1e-9:
                ok = True; break
            Hn = hessU(q_n)
            J = np.zeros((3, 3))
            J[:2, :2] = Hn; J[:2, 2] = c_vec
            J[2,  :2] = t_q; J[2, 2] = t_z
            try:
                delta = np.linalg.solve(J, -res)
            except np.linalg.LinAlgError:
                break
            q_n += delta[:2]; z_n += delta[2]
        if not ok:
            cur_ds *= 0.5
            if cur_ds < ds_min: break
            continue
        H_new = hessU(q_n); eigs_new = np.linalg.eigvalsh(H_new)
        if int((eigs_new < 0).sum()) != n_neg_init:
            break
        ms = min(abs(eigs_new[0]), abs(eigs_new[1]))
        if ms < min_abs_lam:
            pts.append((z_n, q_n[0], q_n[1], eigs_new[0], eigs_new[1]))
            break
        q = q_n; z = z_n
        pts.append((z, q[0], q[1], eigs_new[0], eigs_new[1]))
        if ms < 3.0:    cur_ds = max(cur_ds * 0.4, ds_min)
        elif ms < 15.0: cur_ds = max(cur_ds * 0.7, ds_min)
        else:           cur_ds = min(cur_ds * 1.05, ds)
    return np.array(pts)


def trace_branch_arclen(q_start, direction_sign, ds=2.0, max_steps=4000,
                          lam_floor=0.05, lam_alert=20.0):
    """Stable (PD) branch only. Detects fold by lam_min approaching 0
    and rebounding (which means we've crossed onto the unstable branch)."""
    q = q_start.copy(); z = 0.0
    pts = []
    H = hessU(q); eigs = np.linalg.eigvalsh(H)
    pts.append((z, q[0], q[1], eigs[0], eigs[1]))
    prev_tangent = None; cur_ds = ds; ds_min = 1e-3
    last_lam_min = eigs.min(); seen_alert = False
    for step in range(max_steps):
        H = hessU(q)
        try:
            v = -np.linalg.solve(H, c_vec)
        except np.linalg.LinAlgError:
            break
        t_q = v; t_z = 1.0
        norm = np.sqrt(t_q @ t_q + t_z * t_z)
        t_q /= norm; t_z /= norm
        if step == 0:
            t_q *= direction_sign; t_z *= direction_sign
        else:
            if t_q @ prev_tangent[0] + t_z * prev_tangent[1] < 0:
                t_q = -t_q; t_z = -t_z
        prev_tangent = (t_q, t_z)
        q_pred = q + cur_ds * t_q; z_pred = z + cur_ds * t_z
        q_n = q_pred.copy(); z_n = z_pred
        ok = False
        for _ in range(60):
            F = gradU(q_n) + z_n * c_vec
            N = (q_n - q_pred) @ t_q + (z_n - z_pred) * t_z
            res = np.concatenate([F, [N]])
            if np.linalg.norm(res) < 1e-9:
                ok = True; break
            Hn = hessU(q_n)
            J = np.zeros((3, 3))
            J[:2, :2] = Hn; J[:2, 2] = c_vec
            J[2,  :2] = t_q; J[2, 2] = t_z
            try:
                delta = np.linalg.solve(J, -res)
            except np.linalg.LinAlgError:
                break
            q_n += delta[:2]; z_n += delta[2]
        if not ok:
            cur_ds *= 0.5
            if cur_ds < ds_min: break
            continue
        H_new = hessU(q_n); eigs_new = np.linalg.eigvalsh(H_new)
        lm = eigs_new.min()
        if seen_alert and lm > last_lam_min * 1.05: break
        if lm < lam_alert: seen_alert = True
        q = q_n; z = z_n; last_lam_min = lm
        pts.append((z, q[0], q[1], eigs_new[0], eigs_new[1]))
        if lm < lam_floor: break
        if lm < 3.0:    cur_ds = max(cur_ds * 0.4, ds_min)
        elif lm < 15.0: cur_ds = max(cur_ds * 0.7, ds_min)
        else:           cur_ds = min(cur_ds * 1.05, ds)
    return np.array(pts)


B1_pos_arc = trace_branch_arclen(minima_init[0], +1)
B1_neg_arc = trace_branch_arclen(minima_init[0], -1)
B2_pos_arc = trace_branch_arclen(minima_init[1], +1)
B2_neg_arc = trace_branch_arclen(minima_init[1], -1)
S_pos_arc  = trace_branch_general(saddle, +1, ds=1.5)
S_neg_arc  = trace_branch_general(saddle, -1, ds=1.5)

def assemble(pos_arr, neg_arr):
    if len(pos_arr) and len(neg_arr):
        return np.vstack([neg_arr[::-1], pos_arr[1:]])
    if len(pos_arr): return pos_arr
    return neg_arr[::-1]

B1 = assemble(B1_pos_arc, B1_neg_arc)
B2 = assemble(B2_pos_arc, B2_neg_arc)
S_branch = assemble(S_pos_arc, S_neg_arc)

B1_left  = B1[0];  B1_right = B1[-1]
B2_left  = B2[0];  B2_right = B2[-1]

# Path along the critical-point manifold: B1 (z=0 -> fold1) -> S (fold1->fold2) -> B2 (fold2 -> z=0)
P1   = B1_pos_arc.copy()
P_S  = np.vstack([S_pos_arc[::-1], S_neg_arc[1:]])
P2   = B2_neg_arc[::-1].copy()
path_q   = np.vstack([P1[:, 1:3], P_S[1:, 1:3], P2[1:, 1:3]])
path_lam = np.vstack([P1[:, 3:5], P_S[1:, 3:5], P2[1:, 3:5]])
path_z   = np.concatenate([P1[:, 0], P_S[1:, 0], P2[1:, 0]])
n_path   = len(path_q)
idx_fold1 = len(P1) - 1
idx_fold2 = len(P1) + len(P_S) - 2
arc_raw = np.zeros(n_path)
for i in range(1, n_path):
    arc_raw[i] = arc_raw[i-1] + np.linalg.norm(path_q[i] - path_q[i-1])
arc = arc_raw / arc_raw[-1]
s_fold1 = arc[idx_fold1]
s_fold2 = arc[idx_fold2]

lam_min_path = path_lam[:, 0]
lam_max_path = path_lam[:, 1]
sigma_min_path = np.minimum(np.abs(lam_min_path), np.abs(lam_max_path))
kappa_path  = (np.maximum(np.abs(lam_min_path), np.abs(lam_max_path))
               / np.maximum(sigma_min_path, 1e-6))

# ============================== COLORS ==============================
C_B1     = '#2E5A88'   # steel blue (stable)
C_B2     = '#6E8B3D'   # olive green (stable, NOT red)
C_S      = '#C0322B'   # red — reserved for unstable / fold
C_GRID   = '#B7B7B7'
C_TEXT   = '#1a1a1a'
C_B1_BG  = '#DCE5EE'
C_B2_BG  = '#E5EAD8'
C_S_BG   = '#F4DAD7'

# ============================== FIGURE ==============================
# Width: 5.5 in (Science fits between 1- and 2-column).
# Height tuned for 3 horizontal panels with room for axis labels & captions.
fig = plt.figure(figsize=(5.5, 2.15))
gs = fig.add_gridspec(1, 3,
                       width_ratios=[1.0, 1.18, 0.95],
                       wspace=0.55,
                       left=0.085, right=0.945, top=0.78, bottom=0.19)

# ---------------- Panel (a) ----------------
ax = fig.add_subplot(gs[0, 0])

xx = np.linspace(-1.7, 1.2, 300)
yy = np.linspace(-0.4, 2.2, 300)
Xg, Yg = np.meshgrid(xx, yy)
Z = U_grid(Xg, Yg)
Zc = np.clip(Z, -150, 50)
ax.contour(Xg, Yg, Zc, levels=np.linspace(-150, 50, 12),
           colors=C_GRID, linewidths=0.35, alpha=0.55, zorder=1)
sep_level = U_scalar(saddle)
ax.contour(Xg, Yg, Z, levels=[sep_level], colors=C_GRID,
           linewidths=0.55, alpha=0.85, zorder=1.5, linestyles='dashed')

# Branches
ax.plot(B1[:, 1], B1[:, 2], '-', color=C_B1, lw=1.4, zorder=5,
    solid_capstyle='round')
ax.plot(B2[:, 1], B2[:, 2], '-', color=C_B2, lw=1.4, zorder=5,
    solid_capstyle='round')
ax.plot(S_branch[:, 1], S_branch[:, 2], linestyle=(0, (3.5, 2)),
    color=C_S, lw=1.1, zorder=4, dash_capstyle='round')

legend_handles_a = [
    plt.Line2D([0], [0], color=C_B1, lw=1.4, label=r'$\mathcal{B}_1$'),
    plt.Line2D([0], [0], color=C_B2, lw=1.4, label=r'$\mathcal{B}_2$'),
    plt.Line2D([0], [0], marker='^', color=C_S, lw=0,
           markersize=4.0, markeredgecolor='white',
           markeredgewidth=0.5, label=r'$\mathcal{S}$'),
]
ax.legend(handles=legend_handles_a,
      loc='lower center',
      bbox_to_anchor=(0.5, 1.04),
      ncol=3, fontsize=6.5, frameon=False,
      handlelength=1.4, handletextpad=0.35,
      columnspacing=0.8, borderpad=0.0)

# z=0 markers (the original minima, the saddle)
ax.plot(minima_init[0][0], minima_init[0][1], 'o', color=C_B1,
        markersize=3.5, markeredgecolor='white', markeredgewidth=0.5,
        zorder=10)
ax.plot(minima_init[1][0], minima_init[1][1], 'o', color=C_B2,
        markersize=3.5, markeredgecolor='white', markeredgewidth=0.5,
        zorder=10)
ax.plot(saddle[0], saddle[1], '^', color=C_S, markersize=3.8,
        markeredgecolor='white', markeredgewidth=0.5, zorder=10)

# Fold endpoints
for end in [B1_left, B1_right]:
    ax.plot(end[1], end[2], 'X', color=C_S, markersize=3.5,
            markeredgecolor='white', markeredgewidth=0.5, zorder=11)
for end in [B2_left, B2_right]:
    ax.plot(end[1], end[2], 'X', color=C_S, markersize=3.5,
            markeredgecolor='white', markeredgewidth=0.5, zorder=11)

# In-axes labels — placed in known empty regions
ax.text(minima_init[0][0] - 0.18, minima_init[0][1] + 0.22,
        r'$q_1^{*}$', fontsize=8, color=C_B1, ha='right')
ax.text(minima_init[1][0] + 0.12, minima_init[1][1] + 0.20,
        r'$q_2^{*}$', fontsize=8, color=C_B2)

ax.set_xlim(-1.7, 1.25); ax.set_ylim(-0.4, 2.25)
ax.set_xlabel(r'$q_1$', labelpad=1)
ax.set_ylabel(r'$q_2$', labelpad=1)
ax.set_xticks([-1.5, -0.5, 0.5])
ax.set_yticks([0, 1, 2])
# Panel-letter label, placed outside data area in the upper-left
ax.text(-0.32, 1.05, '(a)', transform=ax.transAxes, fontsize=8,
        fontweight='bold', va='bottom')

# ---------------- Panel (b) ----------------
ax2 = fig.add_subplot(gs[0, 1])

# Saddle-region background only
ax2.axvspan(s_fold1, s_fold2, color=C_S_BG, alpha=0.45, zorder=0)
# Fold lines
ax2.axvline(s_fold1, color=C_S, linestyle=':', lw=0.7, alpha=0.85, zorder=2)
ax2.axvline(s_fold2, color=C_S, linestyle=':', lw=0.7, alpha=0.85, zorder=2)

l1, = ax2.semilogy(arc, kappa_path, color='black', lw=1.1,
                    label=r'$\kappa(\nabla^{\!2} U)$', zorder=4)
# Filled area under kappa makes the divergence spikes visually prominent
ax2.fill_between(arc, 1, kappa_path, color='black', alpha=0.08, zorder=3)
ax2.set_ylabel(r'$\kappa$  (log)', labelpad=1)

# Annotate the two divergence peaks with ↑∞
ax2.set_xlim(0, 1)
ax2.set_ylim(1, 5e5)
for sx in (s_fold1, s_fold2):
    ax2.annotate(r'$\uparrow\!\infty$',
                 xy=(sx, 5e5), xytext=(sx, 5e5),
                 ha='center', va='top', fontsize=6, color='black')

ax2r = ax2.twinx()
l3, = ax2r.plot(arc, lam_min_path, color='#555555', lw=0.9,
                 linestyle=(0, (4, 2)), label=r'$\lambda_{\min}$')
ax2r.axhline(0, color='black', lw=0.4, alpha=0.6)
ax2r.set_ylabel(r'$\lambda_{\min}$', labelpad=1)
ax2r.tick_params(labelsize=7, pad=1)

# Region letters at top of plot
y_top = 2.5e5
ax2.text(s_fold1 / 2, y_top, r'$\mathcal{B}_1$',
         ha='center', va='top', fontsize=7, color=C_B1)
ax2.text((s_fold1 + s_fold2) / 2, y_top, r'$\mathcal{S}$',
         ha='center', va='top', fontsize=7, color=C_S)
ax2.text((s_fold2 + 1) / 2, y_top, r'$\mathcal{B}_2$',
         ha='center', va='top', fontsize=7, color=C_B2)

ax2.set_xlabel(r'arc length  $s$', labelpad=1)
ax2.set_xticks([0, 0.5, 1])
# Panel letter
ax2.text(-0.32, 1.05, '(b)', transform=ax2.transAxes, fontsize=8,
         fontweight='bold', va='bottom')

# Combined legend ABOVE the panel (b), centered, 2x2 layout fitting inside (b)
combined = [l1, l3]
leg_b = ax2.legend(handles=combined,
                   loc='lower center',
                   bbox_to_anchor=(0.5, 1.04),
                   ncol=2, fontsize=6.5, frameon=False,
                   handlelength=1.2, handletextpad=0.3,
                   columnspacing=0.8, labelspacing=0.25,
                   borderpad=0.0)

# ---------------- Panel (c) ----------------
ax3 = fig.add_subplot(gs[0, 2])

proj_B1 = B1[:, 1:3] @ c_vec
proj_B2 = B2[:, 1:3] @ c_vec
proj_S  = S_branch[:, 1:3] @ c_vec
z_B1 = B1[:, 0]; z_B2 = B2[:, 0]; z_S = S_branch[:, 0]

z_overlap_lo = max(z_B1[0], z_B2[0])
z_overlap_hi = min(z_B1[-1], z_B2[-1])
ax3.axvspan(z_overlap_lo, z_overlap_hi, color='#EEEEEE',
            alpha=1.0, zorder=0)

ax3.plot(z_B1, proj_B1, '-', color=C_B1, lw=1.4, zorder=5)
ax3.plot(z_B2, proj_B2, '-', color=C_B2, lw=1.4, zorder=5)
ax3.plot(z_S,  proj_S, linestyle=(0, (3.5, 2)), color=C_S, lw=1.1,
         zorder=4, dash_capstyle='round')

# Fold endpoint markers
for end in [B1_left, B1_right, B2_left, B2_right]:
    ax3.plot(end[0], np.dot(end[1:3], c_vec), 'X', color=C_S,
             markersize=3.5, markeredgecolor='white',
             markeredgewidth=0.5, zorder=10)
# z=0 references
ax3.plot(0, np.dot(minima_init[0], c_vec), 'o', color=C_B1,
         markersize=3.0, markeredgecolor='white', markeredgewidth=0.4,
         zorder=11)
ax3.plot(0, np.dot(minima_init[1], c_vec), 'o', color=C_B2,
         markersize=3.0, markeredgecolor='white', markeredgewidth=0.4,
         zorder=11)
ax3.plot(0, np.dot(saddle, c_vec), '^', color=C_S, markersize=3.4,
         markeredgecolor='white', markeredgewidth=0.4, zorder=11)
ax3.axvline(0, color=C_GRID, lw=0.4, alpha=0.7, zorder=1)

ax3.set_xlabel(r'load parameter  $z$', labelpad=1)
ax3.set_ylabel(r'$\mathbf{c}^{\!\top}\!q^{*}$',
               labelpad=2)
ax3.set_xticks([-500, 0, 500])
ax3.set_yticks([-0.5, 0, 0.5, 1.0, 1.5])
# Panel letter
ax3.text(-0.32, 1.05, '(c)', transform=ax3.transAxes, fontsize=8,
         fontweight='bold', va='bottom')

# Compact legend OUTSIDE plot, top-right of panel (c)
legend_handles = [
    plt.Line2D([0], [0], color=C_B1, lw=1.4, label=r'$\mathcal{B}_1$'),
    plt.Line2D([0], [0], color=C_B2, lw=1.4, label=r'$\mathcal{B}_2$'),
    plt.Line2D([0], [0], color=C_S, lw=1.1,
               linestyle=(0, (3.5, 2)),  label=r'$\mathcal{S}$'),
    plt.Line2D([0], [0], marker='X', color=C_S, lw=0,
               markersize=3.5, markeredgecolor='white',
               markeredgewidth=0.5, label='fold'),
]
leg_c = ax3.legend(handles=legend_handles,
                   loc='lower center',
                   bbox_to_anchor=(0.5, 1.04),
                   ncol=2, fontsize=6.5, frameon=False,
                   handlelength=1.2, handletextpad=0.3,
                   columnspacing=0.8, labelspacing=0.25,
                   borderpad=0.0)

# Annotation of fold z-values placed in clear white space within bistable window
B1_fold_z = B1_right[0]; B2_fold_z = B2_left[0]
ax3.text(-180, 1.05,
         f'$z_{{c1}}\\!\\approx\\!{B1_fold_z:.0f}$',
         fontsize=6.5, color=C_S, ha='left', va='center')
ax3.text(60, 0.05,
         f'$z_{{c2}}\\!\\approx\\!{B2_fold_z:.0f}$',
         fontsize=6.5, color=C_S, ha='left', va='center')

# ---------------- Save ----------------
plt.savefig('T_branches_science.pdf', bbox_inches='tight',
            pad_inches=0.02)
plt.savefig('T_branches_science.png', dpi=600,
            bbox_inches='tight', pad_inches=0.02)
print("Saved Science-style figure.")
print(f"B1 fold:  z = {B1_right[0]:+.1f}")
print(f"B2 fold:  z = {B2_left[0]:+.1f}")
print(f"Bistable window:  z in [{z_overlap_lo:.0f}, {z_overlap_hi:.0f}]")
