#!/usr/bin/env python3
"""IEEE T-RO-style Fig. 6, Single-Column Optimized.

Reproducible geometry, NO image-generation dependency and NO project imports.
Conventions (consistent with the revised paper):
    R = Rz(alpha) Ry(-beta) Rx(gamma), beta > 0 is elevation above horizontal
    t = (cos(alpha)cos(beta), sin(alpha)cos(beta), sin(beta))
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
from matplotlib.lines import Line2D

# 强制输出到当前运行目录
OUT = Path('.')

# T-RO Professional Color Palette
INK = '#272727'
BLUE = '#1D66AA'
RED = '#BB4139'
GREEN = '#237A59'
GRAY = '#939A9F'
MUTED = '#595959'

# Colors for mechanical entities and shadows
MECH_FACE = '#E5E7EB'
MECH_EDGE = '#374151'
SHADOW_LINE = '#A3A8AD'

E3 = np.array([0., 0., 1.])

# IEEE 单栏标准配置 (8pt base font size)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 7.5, # 适合单栏排版的字号
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'axes.unicode_minus': False,
    'hatch.linewidth': 0.5,
})

def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)

def bishop_frame(points):
    e = np.diff(points, axis=0)
    t = e / np.linalg.norm(e, axis=1)[:, None]
    b = [unit(np.cross(E3, t[0]))]
    for ta, tb in zip(t[:-1], t[1:]):
        axis = np.cross(ta, tb)
        s = np.linalg.norm(axis)
        c = np.dot(ta, tb)
        if s < 1e-11:
            if c < 0: raise ValueError('antiparallel edge tangents')
            d = b[-1].copy()
        else:
            axis /= s
            d = c*b[-1] + s*np.cross(axis, b[-1]) + (1-c)*np.dot(axis, b[-1])*axis
        d = d - np.dot(d, tb)*tb
        b.append(unit(d))
    b = np.asarray(b)
    ref_end = unit(np.cross(E3, t[-1]))
    psi = np.arctan2(np.dot(np.cross(ref_end, b[-1]), t[-1]), np.dot(ref_end, b[-1]))
    return t, b, psi

def camera(points):
    U = unit([0.92, 0.34, 0.])
    V = unit([-0.31, 0.84, 0.56])
    V = unit(V - np.dot(V, U)*U)
    points = np.asarray(points)
    return np.stack((points@U, points@V), axis=-1)

def arrow(ax, begin, end, color=INK, lw=0.9, size=6, z=15):
    ax.annotate('', xy=end, xytext=begin,
                arrowprops={'arrowstyle':'-|>', 'lw':lw, 'color':color,
                            'mutation_scale':size, 'shrinkA':0, 'shrinkB':0}, zorder=z)

def vector(ax, p, v, L, color, label=None, offset=(0,0), fontsize=7.5):
    start = camera(p)
    end = camera(np.asarray(p) + L*unit(v))
    arrow(ax, start, end, color, lw=1.0, size=6)
    if label:
        ax.text(end[0]+offset[0], end[1]+offset[1], label,
                color=color, fontsize=fontsize, zorder=20,
                ha='left' if offset[0] >= 0 else 'right', va='center')

def draw_fixed_clamp(ax, p, direction, length=0.12, halfwidth=0.08):
    p = np.asarray(p, float)
    u = unit(direction)
    n = np.array([-u[1], u[0]])
    front = p - 0.012*u
    rear = front - length*u
    # 底部接地阴影线
    ground_pts = [rear - 0.06*u - halfwidth*1.4*n, rear - halfwidth*1.4*n,
                  rear + halfwidth*1.4*n, rear - 0.06*u + halfwidth*1.4*n]
    ax.add_patch(Polygon(ground_pts, closed=True, facecolor='none', edgecolor=INK, lw=0.6, hatch='////', zorder=2))
    # 夹具金属块
    block_pts = [rear-halfwidth*n, front-halfwidth*n, front+halfwidth*n, rear+halfwidth*n]
    ax.add_patch(Polygon(block_pts, closed=True, facecolor=MECH_FACE, edgecolor=MECH_EDGE, lw=0.9, zorder=4))

def draw_gripper(ax, p, tangent, depth=0.25, gap=0.05, jaw_width=0.03, back_offset=0.07, mount_length=0.1):
    p = np.asarray(p, float)
    u = unit(tangent)
    v = np.array([-u[1], u[0]])
    def loc(x, y): return p + x*u + y*v
    lw = 0.8 
    # Mount
    ax.add_patch(Polygon([loc(back_offset, -gap*0.65), loc(back_offset+mount_length, -gap*0.65),
                          loc(back_offset+mount_length, gap*0.65), loc(back_offset, gap*0.65)], 
                         closed=True, facecolor='#D1D5DB', edgecolor=MECH_EDGE, lw=lw, zorder=4))
    # Base
    ax.add_patch(Polygon([loc(0, -gap-jaw_width), loc(back_offset, -gap-jaw_width),
                          loc(back_offset, gap+jaw_width), loc(0, gap+jaw_width)], 
                         closed=True, facecolor=MECH_FACE, edgecolor=MECH_EDGE, lw=lw, zorder=5))
    # Jaws
    ax.add_patch(Polygon([loc(-depth, gap), loc(0, gap), loc(0, gap+jaw_width), loc(-depth, gap+jaw_width)], 
                         closed=True, facecolor=MECH_FACE, edgecolor=MECH_EDGE, lw=lw, zorder=5))
    ax.add_patch(Polygon([loc(-depth, -gap-jaw_width), loc(0, -gap-jaw_width), loc(0, -gap), loc(-depth, -gap)], 
                         closed=True, facecolor=MECH_FACE, edgecolor=MECH_EDGE, lw=lw, zorder=5))

def node(ax, pt, clamped=True, size=4.0):
    ax.plot(pt[0], pt[1], marker='o', ms=size, ls='none', mfc=INK if clamped else 'white', mec=INK, mew=0.9, zorder=12)

def put(ax, xy, s, offset=(0,0), size=7.5, color=INK, ha='center'):
    ax.text(xy[0]+offset[0], xy[1]+offset[1], s, fontsize=size, color=color, ha=ha, va='center', zorder=25)

def frame_panel(fig, rect, tag, heading):
    x, y, w, h = rect
    fig.patches.append(Rectangle((x,y),w,h, transform=fig.transFigure, facecolor='none', edgecolor='#C0C0C0', lw=.6))
    fig.text(x+.015, y+h-.035, tag, fontsize=8, fontweight='bold', color=INK)
    fig.text(x+.070, y+h-.035, heading, fontsize=7.5, color=INK)

def arc_between(ax, center, a0, a1, radius, color, lw=0.9):
    ph = np.linspace(a0,a1,64)
    coords = center + radius*np.c_[np.cos(ph),np.sin(ph)]
    ax.plot(coords[:,0], coords[:,1], color=color, lw=lw, zorder=11)
    arrow(ax, coords[-5], coords[-1], color=color, lw=lw, size=5, z=12)


# ==============================================================
# 绝对单栏尺寸: 3.5英寸 宽 x 2.6英寸 高 (维持原 a左右, bc上下的经典布局)
# ==============================================================
fig = plt.figure(figsize=(3.50, 2.60), facecolor='white')

# --------------------------------------------------------------
# (a) 左侧面板 (占用左侧 55% 宽度)
# --------------------------------------------------------------
frame_panel(fig, (.01, .01, .54, .98), '(a)', 'Rod centerline and frames')
a = fig.add_axes([.01, .15, .54, .82]) # 底部留白给图例
a.axis('off')

N = 11
s = np.linspace(0, 1, N)
points = np.c_[1.85*s, 0.57*np.sin(2*np.pi*s)+0.10*s, 0.68*s + 0.26*np.sin(np.pi*s)**2]
t, b, psi = bishop_frame(points)
gamma = np.deg2rad(129.)
theta = np.arange(N-1)/(N-2)*(gamma-psi)
m1 = np.cos(theta[:,None])*b + np.sin(theta[:,None])*np.cross(t, b)
xy = camera(points)

# --- 无冲突空间投影 (仅保留虚线) ---
z_g = -0.55  # 投影面更低，彻底避开上方文字
points_g = points.copy(); points_g[:, 2] = z_g
xy_g = camera(points_g)
a.plot(xy_g[:,0], xy_g[:,1], color=SHADOW_LINE, ls=':', lw=0.9, zorder=1)

# 下垂辅助线
for idx in [0, 1, 5, N-2, N-1]:
    a.plot([xy[idx,0], xy_g[idx,0]], [xy[idx,1], xy_g[idx,1]], color=SHADOW_LINE, ls=':', lw=0.8, zorder=1)

# 重力箭头
mid_0 = (points[0] + points[1]) / 2
xy_mid_0 = camera(mid_0)
xy_g_arrow = camera(mid_0 - np.array([0, 0, 0.35]))
arrow(a, xy_mid_0, xy_g_arrow, color=MUTED, lw=0.9, size=5, z=2)
put(a, xy_g_arrow, r'$g$', offset=(-0.05, -0.05), size=7.5, color=MUTED)
# -----------------------------------

# DER 风格实线杆
a.plot(xy[:,0],xy[:,1], color=INK, lw=1.5, solid_capstyle='round', solid_joinstyle='round', zorder=3)
draw_fixed_clamp(a, xy[0], xy[1]-xy[0], length=.12, halfwidth=.08)
draw_gripper(a, xy[-1], xy[-1]-xy[-2], depth=.25, gap=.05, jaw_width=.03, back_offset=.07, mount_length=.1)

# 绘制节点
for i,pt in enumerate(xy): node(a,pt,i in (0,1,N-2,N-1))

# 防遮挡文字布局：将易冲突的标签全部放在杆的上方
put(a, xy[0], r'$\mathbf{x}_0$', (-0.12, 0.12))
put(a, xy[1], r'$\mathbf{x}_1$', (0.12, -0.05))
put(a, xy[5], r'$\mathbf{x}_i$', (-0.10, 0.15))       # 移到上方防遮挡
put(a, xy[N-2], r'$\mathbf{x}_{N-2}$', (-0.08, 0.15)) # 移到上方防遮挡
put(a, xy[N-1], r'$\mathbf{x}_{N-1}$', (0.22, 0.15))
put(a, (xy[2]+xy[3])/2, r'$\mathbf{e}^{j}$', (0, 0.15))

# 空间参考系标记
base=(points[0]+points[1])/2
vector(a,base,b[0],.28,BLUE)
put(a,camera(base+.28*b[0]),r'$\mathbf{b}^{0}=\mathbf{n}(\mathbf{t}^{0})=\mathbf{m}_1^{0}$', (-.05,.12), ha='left')

j=3; base=(points[j]+points[j+1])/2
vector(a,base,b[j],.3,BLUE,r'$\mathbf{b}^{j}$',(.05,.08))
vector(a,base,m1[j],.35,RED,r'$\mathbf{m}^{j}_{1}$',(.05,-.05))

j=N-2; base=(points[j]+points[j+1])/2
vector(a,base,b[j],.3,BLUE,r'$\mathbf{b}^{N-2}$',(-.05,.08))
vector(a,base,m1[j],.45,RED,r'$\mathbf{m}^{N-2}_{1}$',(.05,.02))

put(a,xy[0], 'fixed clamp',(-.25,-.25), color=MUTED)
put(a,xy[-1], 'gripper',(.15,-.30), color=MUTED)

# >>> 修复了投影截断问题：将 Y轴下界从 -0.85 扩大到 -1.15 <<<
a.set_xlim(-.40, 2.25); a.set_ylim(-1.15, 1.15) 
a.set_aspect('equal', adjustable='box')

# 极简图例 (基于 Figure 绝对坐标，永不重叠)
fig.text(.02, .04, '●', color=INK, fontsize=7)
fig.text(.05, .04, 'clamped')
fig.text(.18, .04, '○', color=INK, fontsize=7)
fig.text(.21, .04, 'free')
fig.lines.append(Line2D([.31, .34], [.05, .05], transform=fig.transFigure, lw=1.2, color=BLUE))
fig.text(.35, .04, 'Bishop')
fig.lines.append(Line2D([.45, .48], [.05, .05], transform=fig.transFigure, lw=1.2, color=RED))
fig.text(.49, .04, 'material')


# --------------------------------------------------------------
# (b) 右上方面板
# --------------------------------------------------------------
frame_panel(fig, (.56, .52, .43, .47), '(b)', 'Six-DOF gripper')
bax = fig.add_axes([.57, .52, .41, .45])
bax.axis('off')
alpha=np.deg2rad(35.); beta=np.deg2rad(28.); roll=np.deg2rad(65.)
tb=np.array([np.cos(alpha)*np.cos(beta),np.sin(alpha)*np.cos(beta),np.sin(beta)])
nb=unit(np.cross(E3,tb))
mb=np.cos(roll)*nb+np.sin(roll)*np.cross(tb,nb)
p=.93*tb; xprev=p-.35*tb; O=np.zeros(3)

def viewB(v): return camera(v)*np.array([9.0, 6.75]) + np.array([0.6, 0.8])
def bvec(p0,v,L,color,label,shift=(0,0)):
    q0=viewB(p0); q1=viewB(p0+L*unit(v))
    arrow(bax,q0,q1,color,lw=0.9,size=6)
    put(bax,q1,label,shift,color=color)

for axis,lab,ofs in [(np.array([1.,0,0]),r'$\mathbf{e}_x$',(.0,-.45)), (np.array([0,1.,0]),r'$\mathbf{e}_y$',(.15,.25)), (np.array([0,0,1.]),r'$\mathbf{e}_z$',(-.35,.15))]:
    bvec(O,axis,.25,GRAY,lab,ofs)
    
origin=viewB(O); pxy=viewB(p); prevxy=viewB(xprev)
ph=viewB(.93*np.cos(beta)*np.array([np.cos(alpha),np.sin(alpha),0.]))

bax.plot([origin[0],ph[0]],[origin[1],ph[1]],ls='--',color=GRAY,lw=.6,zorder=1)
bax.plot([ph[0],pxy[0]],[ph[1],pxy[1]],ls=':',color=GRAY,lw=.6,zorder=1)
bax.plot([origin[0],prevxy[0]],[origin[1],prevxy[1]],color=INK,lw=1.0,zorder=3)
bax.plot([prevxy[0],pxy[0]],[prevxy[1],pxy[1]],color=INK,lw=1.8, solid_capstyle='round',zorder=4)

draw_gripper(bax, pxy, pxy-prevxy, depth=2.40, gap=.30, jaw_width=.15, back_offset=.50, mount_length=.60)
node(bax,prevxy,True); node(bax,pxy,True)

put(bax,prevxy,r'$\mathbf{x}_{N-2}$',(-.7,.5))
put(bax,pxy,r'$\mathbf{p}=\mathbf{x}_{N-1}$',(1.2,-.7))

ed=unit(pxy-prevxy); normal=np.array([-ed[1],ed[0]])
off=-.58*normal; q1=prevxy+off; q2=pxy+off
bax.plot([q1[0],q2[0]],[q1[1],q2[1]],color=MUTED,lw=.6,zorder=2)
for q in (q1,q2):
    d=.15*normal; bax.plot([q[0]-d[0],q[0]+d[0]],[q[1]-d[1],q[1]+d[1]], color=MUTED,lw=.6,zorder=2)
put(bax,(q1+q2)/2,r'$\bar e$',(0,-.30))

pivot=.73*xprev+.27*p
bvec(pivot,nb,.25,GREEN,r'$\mathbf{n}$',(.10,.30))
bvec(pivot,mb,.25,RED,r'$\mathbf{m}_1$',(-.10,.25))
put(bax,origin+.37*(prevxy-origin),r'$\mathbf{t}(\alpha,\beta)$',(.50,.50))

a0=np.arctan2(ph[1]-origin[1],ph[0]-origin[0])
a1=np.arctan2(pxy[1]-origin[1],pxy[0]-origin[0])
arc_between(bax,origin,a0,a1,3.7,INK,lw=.7)
put(bax,origin+(3.7*np.array([np.cos((a0+a1)/2),np.sin((a0+a1)/2)])), r'$\beta>0$',(.30,-.65))

bax.set_xlim(-.5, 10.0); bax.set_ylim(-1.0, 6.5)
bax.set_aspect('equal',adjustable='box')

fig.text(.57, .54, r'$\mathbf{R}=\mathbf{R}_z(\alpha)\mathbf{R}_y(-\beta)\mathbf{R}_x(\gamma)$')
fig.text(.78, .54, r'$\mathbf{u}=(\mathbf{p},\alpha,\beta,\gamma)$')


# --------------------------------------------------------------
# (c) 右下方面板
# --------------------------------------------------------------
frame_panel(fig, (.56, .01, .43, .49), '(c)', 'Holonomy at gripper')
cax = fig.add_axes([.57, .03, .41, .44])
cax.axis('off')
cax.add_patch(Circle((0,0),1.0,fc='none',ec='#B7B7B7',lw=.6, ls=(0,(4,3)),zorder=1))
cax.add_patch(Circle((0,0),.05,fc='white',ec=INK,lw=.8,zorder=9))
cax.plot(0,0,'.',color=INK,ms=2,zorder=10)

for angle,color,label,length,offset in [(0.,GREEN,r'$\mathbf{n}$',.91,(.20,-.05)), (psi,BLUE,r'$\mathbf{b}^{N-2}$',.89,(.20,.10)), (gamma,RED,r'$\mathbf{m}_1^{N-2}$',.90,(-.40,.40))]:
    end=length*np.array([np.cos(angle),np.sin(angle)])
    arrow(cax,(0,0),end,color,lw=1.0,size=6)
    put(cax,end,label,offset,color=color)

arc_between(cax,np.array([0.,0.]),0.,psi,.64,BLUE,0.9)
arc_between(cax,np.array([0.,0.]),psi,gamma,.40,INK,0.9)
arc_between(cax,np.array([0.,0.]),0.,gamma,1.08,RED,0.9)
put(cax,(.65,.25),r'$\Psi$',color=BLUE)
put(cax,(-.05,.70),r'$\theta^{N-2}$',color=INK)
put(cax,(.70,1.15),r'$\gamma$',color=RED)

cax.text(0, -1.15, r'$\mathbf{t}^{N-2}\ \mathrm{(out\ of\ page)}$', ha='center', fontsize=7.5)
cax.text(0, -1.40, r'$\theta^{N-2}=\gamma-\Psi$', color=INK, ha='center', fontsize=8.0)

cax.set_xlim(-1.50, 1.45); cax.set_ylim(-1.60, 1.40)
cax.set_aspect('equal')


for ext in ('png','pdf','svg'):
    destination=OUT/f'fig06_rod_model.{ext}'
    fig.savefig(destination,dpi=350 if ext=='png' else 200, facecolor='white',transparent=False)
    print('saved:',destination)
print(f'Angle check: Psi={np.rad2deg(psi):.2f}°, gamma={np.rad2deg(gamma):.2f}°, theta_end={np.rad2deg(theta[-1]):.2f}°')
plt.close(fig)