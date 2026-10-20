"""Validate the rod--pipe contact model and the 90-degree insertion task.

Run this after touching LPipe2D (src/contact.py), the free-tip grasp mode, or
the pipe task::

    python scripts/check_pipe_contact.py                 # A-D, F (a few minutes)
    python scripts/check_pipe_contact.py --with-training # + E: full optimization

Checks, in order:
  A. geometry -- signed clearance inside/outside every piece, penetration
     depth, contact normals, and an edge that cuts the inner corner of the
     bend between two nodes that are both inside
  B. gradients -- clearance, contact force and Jacobian vs finite differences,
     the contact loss, and the adjoint dL/du through a contacting equilibrium
  C. simple insertion -- push along the straight first segment with a tilted
     gripper, so the rod rides the channel wall
  D. bend crossing -- a scripted straight push takes the tip round an elbow
     (l_bend_2d); for a sharp corner (l_corner_2d) it must instead buckle and
     jam at the dead point, not be driven through the wall
  E. target reaching -- scripts/train_pipe.py end to end (--with-training)
  F. contact ablation -- the same push with contact off goes through the wall;
     with the pipe present but contact off the rollout is bit-identical to no
     pipe at all
"""
import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from blas_threads import limit_blas_threads
limit_blas_threads()

import numpy as np

from config import load_config
from contact import LPipe2D
from env_factory import build_env, reset_env, initial_control, task_target, node_position
from implicit_grad_tools import implicit_final_control_grad

CONFIG = _ROOT / 'configs' / 'task_90deg_pipe.yaml'
TOL_GRAD = 1e-6      # analytic vs central differences, relative
TOL_HESS = 1e-4      # vs second differences

results = []


def report(name, err, tol, fmt=".3e"):
    ok = err < tol
    results.append(ok)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:56s} {err:{fmt}}  (tol {tol:g})")


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:56s} {detail}")


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def scripted_push(env, feed, n_steps, y=0.0, theta=0.0):
    """Open-loop straight push of the gripper by ``feed`` along -x."""
    u0 = np.array([0, 0, 0, 1.0, 0, 0], dtype=float)
    q_list, conv = [], True
    for k in range(n_steps):
        a = (k + 1) / n_steps
        u = u0 + a * np.array([0, 0, 0, -feed, y, theta])
        _, _, info = env.step(u)
        q_list.append(info["q"])
        conv &= info["converged"]
    return q_list, conv, u


def max_pen(pipe, q_list):
    return max(pipe.detect_dlo_pipe_contact(q)["max_penetration"] for q in q_list)


parser = argparse.ArgumentParser()
parser.add_argument("--with-training", action="store_true",
                    help="also run Test E: scripts/train_pipe.py end to end")
args = parser.parse_args()

cfg = load_config(CONFIG)
tp = cfg.task.pipe
env = build_env(cfg)
reset_env(env, cfg)
pipe = env.pipe
target = task_target(cfg)
tip = int(tp.tip_node)
s_goal = pipe.arclength(target)[0]
feed_needed = s_goal - pipe.arclength(env.nodes[tip])[0]

# ----------------------------------------------------------- A. geometry
print("\nA. contact geometry -- signed clearance, normals, edges")
s_vals = np.linspace(0.03, pipe.length - 0.01, 60)
if pipe.corner:
    # Inside the square junction the clearance is to the corner walls, not r_eff.
    s_vals = s_vals[(np.abs(s_vals - pipe.L1) > pipe.r_in + 5e-3)
                    & (s_vals < pipe.length - pipe.r_in - 5e-3)]   # flat closed end
C = np.array([pipe.centerline_point(s) for s in s_vals])
report("centerline: g = inner_radius - rod_radius",
       float(np.abs(pipe.compute_signed_clearance(C) - pipe.r_eff).max()), 1e-9)

# Lateral offsets across the wall, on every piece (incl. inner/outer bend wall).
ds = 1e-7
T = np.array([pipe.centerline_point(s + ds) - pipe.centerline_point(s - ds) for s in s_vals])
T /= np.linalg.norm(T, axis=1, keepdims=True)
Nrm = np.stack([-T[:, 1], T[:, 0]], 1)
ok_sign, err_depth, err_nrm = True, 0.0, 0.0
for side in (1, -1):
    for off, sgn in ((pipe.r_eff - 1e-3, 1), (pipe.r_eff + 1e-3, -1)):
        P = C + side * off * Nrm
        g, G = pipe.compute_signed_clearance(P, derivs=1)
        ok_sign &= bool(np.all(np.sign(g) == sgn))
        err_depth = max(err_depth, float(np.abs(g - sgn * 1e-3).max()))
        # The wall pushes back toward the centerline.
        err_nrm = max(err_nrm, float(np.abs(G - (-side) * Nrm).max()))
check("inside (+) / outside (-) on both walls of all three pieces", ok_sign)
report("penetration depth = -g (1 mm through the wall)", err_depth, 1e-6)
report("contact normal = grad g points back into the pipe", err_nrm, 1e-6)
check("free space in front of the entry face is allowed",
      pipe.compute_signed_clearance(pipe.P0 - 0.1 * pipe.d1 + 0.2 * Nrm[0])[0] > 0)
check("solid block beside the channel is a wall",
      pipe.compute_signed_clearance(pipe.P0 + 0.1 * pipe.d1 + (pipe.r_in + 0.03) * Nrm[0])[0] < 0)
check("closed far end is a wall",
      pipe.compute_signed_clearance(pipe.P3 + 0.02 * pipe.d2)[0] < 0)
s_mid = 0.5 * (pipe.s_bend_start + pipe.s_bend_end)
check("piece labels follow the centerline",
      [pipe.piece_of(pipe.centerline_point(s)) for s in (0.3, s_mid, pipe.length - 0.05)]
      == ["first_segment", "bend", "second_segment"])

# An edge that cuts the inner corner of the turn between two nodes inside it.
if pipe.corner:
    # One node in each channel, 2 mm clear of its inner wall; the chord between
    # them crosses the convex inner corner Q of the solid block.
    Q = pipe.P1 - pipe.r_in * pipe.d1 + pipe.r_in * pipe.d2
    c2 = pipe.r_rod + 2e-3
    nodes = np.array([Q - 0.03 * pipe.d1 - c2 * pipe.d2, Q + 0.03 * pipe.d2 + c2 * pipe.d1])
    away = (pipe.d1 - pipe.d2) / np.sqrt(2.0)       # off the block, into the junction
    feature = "inner corner"
else:
    rho = pipe.Rb - pipe.r_eff + 2e-3      # nodes 2 mm clear of the inner wall
    ang = np.array([np.pi / 8, 3 * np.pi / 8])
    nodes = pipe.C + rho * (np.cos(ang)[:, None] * -pipe.d2 + np.sin(ang)[:, None] * pipe.d1)
    mid = nodes.mean(0) - pipe.C
    away = mid / np.linalg.norm(mid)                 # radially outward
    feature = "inner wall of the bend"
q_edge = nodes.reshape(-1)
kw = dict(entry_position=pipe.P0, entry_direction=pipe.d1, exit_direction=pipe.d2,
          inner_radius=pipe.r_in, bend_radius=pipe.Rb, first_segment_length=pipe.L1,
          second_segment_length=pipe.L2, rod_radius=pipe.r_rod, nv=2,
          deltaL=float(np.linalg.norm(nodes[1] - nodes[0])), k_contact=pipe.k_contact,
          smoothing=pipe.eps, corner_smoothing=pipe.tau, geometry_type=pipe.geometry_type)
edge4 = LPipe2D(samples_per_edge=4, **kw)
edge1 = LPipe2D(samples_per_edge=1, **kw)
g_nodes = edge4.compute_signed_clearance(nodes, smooth=False)
det = edge4.detect_dlo_pipe_contact(q_edge)
print(f"       both nodes inside (g = {1e3 * g_nodes.min():.2f} mm), chord "
      f"{1e3 * det['max_penetration']:.1f} mm into the {feature}")
check("nodes alone see no contact", g_nodes.min() > 0 and edge1.compute_contact_energy(q_edge) < 1e-12,
      f"E(nodes only) = {edge1.compute_contact_energy(q_edge):.2e} J")
check("edge sampling detects the cut and resists it", edge4.compute_contact_energy(q_edge) > 0,
      f"E(4/edge) = {edge4.compute_contact_energy(q_edge):.3e} J")
F = edge4.compute_contact_force(q_edge).reshape(2, 2)
check(f"...pushing the edge off the {feature}", float(F.sum(0) @ away) > 0)

# ----------------------------------------------------------- B. gradients
print("\nB. contact gradients vs finite differences")
rng = np.random.default_rng(3)
# Points spread over the walls, the two entry lips and both corners of the
# turn (where the smoothed max / min act).
lat = Nrm[0]
turn = pipe.centerline_point(s_mid)
pts = np.vstack([
    C[rng.integers(0, len(C), 20)] + rng.normal(0, pipe.r_eff, (20, 2)),
    pipe.P0 + pipe.r_in * lat + rng.normal(0, 1e-3, (10, 2)),
    pipe.P0 - pipe.r_in * lat + rng.normal(0, 1e-3, (10, 2)),
    turn + rng.normal(0, pipe.r_in, (20, 2)),
])
h = 1e-6
g0, G0, H0 = pipe.compute_signed_clearance(pts, derivs=2)
Gfd = np.zeros_like(G0)
Hfd = np.zeros_like(H0)
for c in range(2):
    d = np.zeros(2); d[c] = h
    Gfd[:, c] = (pipe.compute_signed_clearance(pts + d) - pipe.compute_signed_clearance(pts - d)) / (2 * h)
    Hfd[:, :, c] = (pipe.compute_signed_clearance(pts + d, derivs=1)[1]
                    - pipe.compute_signed_clearance(pts - d, derivs=1)[1]) / (2 * h)
report("clearance gradient (turn, walls, entry lips)", rel(G0, Gfd), TOL_GRAD)
report("clearance Hessian", rel(H0, Hfd), TOL_HESS)

# A real contacting configuration: the tip pushed into the turn until it
# rides the outer wall.
reset_env(env, cfg)
q_list, _, u_mid = scripted_push(env, pipe.s_bend_end + 0.01 - pipe.arclength(env.nodes[tip])[0], 60)
q_c = q_list[-1]
state_c = env.get_state()
n_act = pipe.detect_dlo_pipe_contact(q_c)["n_contacts"]
print(f"       test state: tip at s = {pipe.arclength(q_c[:2])[0]:.3f} ({pipe.piece_of(q_c[:2])}), "
      f"{n_act} contact samples")
F_an = pipe.compute_contact_force(q_c)
F_fd = np.zeros_like(F_an)
hq = 1e-7
for i in range(q_c.size):
    d = np.zeros(q_c.size); d[i] = hq
    F_fd[i] = -(pipe.compute_contact_energy(q_c + d) - pipe.compute_contact_energy(q_c - d)) / (2 * hq)
report("contact force = -dE_c/dq", rel(F_an, F_fd), 1e-5)
J_an = pipe.compute_contact_jacobian(q_c)
J_fd = np.zeros_like(J_an)
for i in range(q_c.size):
    d = np.zeros(q_c.size); d[i] = hq
    J_fd[:, i] = (pipe.compute_contact_force(q_c + d) - pipe.compute_contact_force(q_c - d)) / (2 * hq)
report("contact Jacobian = dF_c/dq", rel(J_an, J_fd), TOL_HESS)
report("contact Jacobian symmetric", rel(J_an, J_an.T), 1e-12)

# Put the tip 2 mm through the wall it leans on.
g_tip, G_tip = pipe.compute_signed_clearance(q_c[:2], derivs=1)
q_pen = q_c.copy()
q_pen[:2] -= (g_tip[0] + 2e-3) * G_tip[0]
Lc, dLc = pipe.compute_contact_loss(q_pen, 0.0)
dLc_fd = np.zeros_like(dLc)
for i in range(dLc.size):
    d = np.zeros(q_pen.size); d[i] = hq
    dLc_fd[i] = (pipe.compute_contact_loss(q_pen + d)[0] - pipe.compute_contact_loss(q_pen - d)[0]) / (2 * hq)
check("contact loss > 0 when penetrating", Lc > 0, f"L = {Lc:.3e}")
report("contact loss gradient", rel(dLc, dLc_fd), 1e-5)

# Adjoint dL/du through a contacting equilibrium, vs re-solving with u +- h.
env.tol = 1e-9          # tight solves, so the differences are not solver noise
L_of = lambda q: float(np.sum((node_position(q, 2, tip) - target) ** 2))   # noqa: E731
env.set_state(state_c)
_, _, info = env.step(u_mid)
a_free = np.zeros(len(env.free_index))
p_tip = node_position(info["q"], 2, tip)
for c in range(2):
    a_free[np.searchsorted(env.free_index, 2 * tip + c)] = 2 * (p_tip[c] - target[c])
g_an = implicit_final_control_grad(info["J"], env.free_index, env.fixed_index, a_free, u_mid,
                                   env.deltaL, dim=2, boundary_rows=env.boundary_rows)
g_fd = np.zeros(6)
hu = 1e-5
for i in range(3, 6):
    d = np.zeros(6); d[i] = hu
    env.set_state(state_c); Lp = L_of(env.step(u_mid + d)[2]["q"])
    env.set_state(state_c); Lm = L_of(env.step(u_mid - d)[2]["q"])
    g_fd[i] = (Lp - Lm) / (2 * hu)
print("       component   adjoint        finite-diff")
for i, nm in zip(range(3, 6), ("x_r", "y_r", "th_r")):
    print(f"       {nm:9s} {g_an[i]:+13.6f} {g_fd[i]:+13.6f}")
report("adjoint dL/du through contact (right end)", rel(g_an[3:], g_fd[3:]), 1e-4)
# Same equilibrium, Jacobian without the contact block: the gradient is wrong,
# i.e. contact is genuinely carried by the gradient, not just by the loss.
J_nc = info["J"] + pipe.compute_contact_jacobian(info["q"])
g_nc = implicit_final_control_grad(J_nc, env.free_index, env.fixed_index, a_free, u_mid,
                                   env.deltaL, dim=2, boundary_rows=env.boundary_rows)
check("dropping the contact Jacobian changes the gradient",
      rel(g_nc[3:], g_fd[3:]) > 1e-2, f"error without it: {rel(g_nc[3:], g_fd[3:]):.2e}")
# What makes train.adjoint_mode: local exact here: frictionless and
# quasi-static, the state after a step depends on that step's control only, so
# perturbing an *earlier* control (and then returning to the same later ones)
# must leave the final state where it was.
env.set_state(state_c)
q_ref = env.step(u_mid)[2]["q"]
q_ref = env.step(u_mid + np.array([0, 0, 0, -2e-3, 0, 0]))[2]["q"]
env.set_state(state_c)
env.step(u_mid + np.array([0, 0, 0, 1e-3, 1e-3, 1e-2]))        # perturbed earlier step
q_pert = env.step(u_mid + np.array([0, 0, 0, -2e-3, 0, 0]))[2]["q"]
report("history independence: earlier control leaves final state [m]",
       float(np.abs(q_pert - q_ref).max()), 1e-6)
env.tol = env.energy.stiffness_scale / env.rod_length ** 2 * 1e-3

# ----------------------------------------------------------- C. straight insertion
print("\nC. simple insertion along the straight first segment (tilted gripper)")
reset_env(env, cfg)
# Stay clear of the turn: at most half the remaining straight length.
feed_C = min(0.10, 0.5 * (pipe.s_bend_start - pipe.arclength(env.nodes[tip])[0]))
q_list, conv, u_end = scripted_push(env, feed_C, 20, y=0.0, theta=0.2)
pen = max_pen(pipe, q_list)
n_c = pipe.detect_dlo_pipe_contact(q_list[-1])["n_contacts"]
check("all equilibrium solves converged", conv)
check("rod touches the wall / lip", n_c > 0, f"{n_c} contact samples")
check("tip still in the first segment", pipe.piece_of(q_list[-1][:2]) == "first_segment")
# The rod carries the push without stretching: every edge keeps its length.
# (Tip advance = gripper advance only holds in a bore tight enough to keep the
# rod straight; in a wide one, tilting the gripper swings the whole rod.)
edge_strain = max(float(np.abs(np.linalg.norm(np.diff(q.reshape(-1, 2), axis=0), axis=1)
                               / env.deltaL - 1.0).max()) for q in q_list)
report("inextensible: max edge strain", edge_strain, 1e-3)
report("max penetration [m]", pen, float(tp.max_penetration))

# ----------------------------------------------------------- D. bend crossing
reset_env(env, cfg)
q_push, conv, _ = scripted_push(env, feed_needed, env.Nsteps)
s_tr = np.array([pipe.arclength(q[:2])[0] for q in q_push])
pen_on = max_pen(pipe, q_push)
if pipe.corner:
    # A straight push into a sharp, frictionless corner meets the outer wall
    # head-on: there is no force component to turn the tip, so the rod
    # buckles -- the dead point. Which way it then goes (jammed in the outer
    # corner, or down the outer wall into the second segment) depends on the
    # buckling branch and the bore, so that is reported, not asserted. What
    # must hold is that it buckles (a straight rod held on the unstable branch
    # would be driven through the wall) and does not pass through it.
    print("\nD. 90-degree corner, scripted straight push -- the dead point")
    lateral = max(float(np.abs((q.reshape(-1, 2) - pipe.P0) @ pipe.d2).max()) for q in q_push)
    check("all equilibrium solves converged", conv)
    check("the rod buckles instead of staying straight", lateral > 1e-3,
          f"max lateral deflection {1e3 * lateral:.1f} mm")
    check("no pass-through: penetration below the rod radius", pen_on < pipe.r_rod,
          f"{1e3 * pen_on:.2f} mm (task limit {1e3 * float(tp.max_penetration):.2f} mm)")
    print(f"       outcome: tip {'passes the corner' if s_tr.max() > pipe.s_bend_end else 'jams at the corner'}"
          f" (s_max = {s_tr.max():.3f}, turn ends at {pipe.s_bend_end:.3f})")
else:
    print("\nD. 90-degree bend crossing, scripted open-loop push")
    check("all equilibrium solves converged", conv)
    check("tip passes the end of the bend", s_tr.max() > pipe.s_bend_end,
          f"s_max = {s_tr.max():.3f} > {pipe.s_bend_end:.3f}")
    check("tip ends in the second segment", pipe.piece_of(q_push[-1][:2]) == "second_segment")
    report("max penetration over the push [m]", pen_on, float(tp.max_penetration))
print(f"       straight push alone: tip error {np.linalg.norm(q_push[-1][:2] - target):.4f} m"
      " (no optimization)")

# ----------------------------------------------------------- F. ablation
print("\nF. contact ablation")
cfg_off = load_config(CONFIG)
cfg_off["contact"]["enabled"] = False
env_off = build_env(cfg_off)
reset_env(env_off, cfg_off)
q_off, conv_off, _ = scripted_push(env_off, feed_needed, env_off.Nsteps)
pen_off = max_pen(env_off.pipe, q_off)
tip_on, tip_off = q_push[-1][:2], q_off[-1][:2]
print(f"       final tip  contact ON ({tip_on[0]:+.3f}, {tip_on[1]:+.3f})   "
      f"OFF ({tip_off[0]:+.3f}, {tip_off[1]:+.3f})")
check("contact OFF: the rod goes straight through the wall", pen_off > 0.05,
      f"max penetration {pen_off:.3f} m")
check("contact changes the forward state", np.linalg.norm(tip_on - tip_off) > 0.1,
      f"|tip_on - tip_off| = {np.linalg.norm(tip_on - tip_off):.3f} m")

cfg_none = load_config(CONFIG)
cfg_none["contact"]["enabled"] = False
cfg_none["pipe"]["enabled"] = False
env_none = build_env(cfg_none)
reset_env(env_none, cfg_none)
q_none, _, _ = scripted_push(env_none, feed_needed, env_none.Nsteps)
report("pipe present + contact OFF == no pipe (max |dq|)",
       float(max(np.abs(a - b).max() for a, b in zip(q_off, q_none))), 1e-15)

# ----------------------------------------------------------- E. training
if args.with_training:
    print("\nE. target reaching -- scripts/train_pipe.py (full optimization)")
    out = subprocess.run([sys.executable, str(_ROOT / 'scripts' / 'train_pipe.py'),
                          '--config', str(CONFIG)], capture_output=True, text=True)
    tail = [l for l in out.stdout.splitlines() if l.strip()][-14:]
    print("\n".join("       " + l for l in tail))
    check("train_pipe.py reports SUCCESS", "RESULT: SUCCESS" in out.stdout,
          "" if out.returncode == 0 else f"(exit {out.returncode}) {out.stderr[-300:]}")
else:
    print("\nE. target reaching -- skipped (pass --with-training)")

print("\n" + "=" * 72)
print(f"{sum(results)}/{len(results)} checks passed"
      f"{'' if all(results) else '  <-- SOMETHING IS WRONG'}")
sys.exit(0 if all(results) else 1)
