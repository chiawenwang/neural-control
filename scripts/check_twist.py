"""Validate the torsion model end to end.

Run this after touching src/rodframes.py, the twist energy, or the twist
boundary map::

    python scripts/check_twist.py

Everything the adjoint relies on is checked against finite differences.  The
exact Hessian matters in particular: an approximate one still lets Newton
converge but silently corrupts the training gradient, so it is checked
directly rather than inferred from the solver behaving.

Checks, in order:
  1. the Bishop holonomy H -- value, gradient and Hessian vs finite differences
  2. the full twisting energy -- force and Jacobian vs finite differences
  3. GJ = 0 reduces exactly to the twist-free der3d model
  4. the 14x12 boundary map d(q_b)/d(u) vs finite differences
  5. the adjoint dL/du vs finite differences -- including the yaw columns,
     which is what decides whether yaw is a usable control input, and over
     several steps, which the 2008 frame makes exact (there is no frame state)
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from blas_threads import limit_blas_threads
limit_blas_threads()

import numpy as np

from rodframes import holonomy, holonomy_grad_hess
from energy import DERElasticEnergy3DTwist, DERElasticEnergy3D
from SimulatorEnv_3D import SimulatorEnv_3D
from implicit_grad_tools import boundary_node_jacobian_3d_twist, implicit_final_control_grad
from env_factory import dloss_dq_free_node, node_tracking_loss

TOL_GRAD = 1e-6      # analytic vs central differences
TOL_HESS = 1e-4      # vs second differences, whose own truncation floor is ~1e-6

results = []


def report(name, err, tol):
    ok = err < tol
    results.append(ok)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:52s} {err:.3e}  (tol {tol:g})")


# ------------------------------------------------------------- 1. holonomy
print("\n1. Bishop holonomy H -- value and derivatives vs finite differences")
rng = np.random.default_rng(7)
nv, L = 8, 1.0
dL = L / (nv - 1)
s = np.arange(nv) * dL
pos0 = np.stack([s, np.zeros(nv), np.zeros(nv)], 1)
planar = np.stack([s, np.zeros(nv), 0.15 * np.sin(np.pi * s / L)], 1)
report("H = 0 on the straight rod", abs(holonomy(pos0)), 1e-12)
report("H = 0 on a planar (vertical) rod", abs(holonomy(planar)), 1e-12)

p = (pos0 + 0.03 * rng.normal(size=(nv, 3))).reshape(-1)
H, g_an, Hs_an = holonomy_grad_hess(p.reshape(-1, 3))
print(f"       bent, non-planar rod: H = {H:+.4f} rad  "
      f"(a planar rod has none -- this is the writhe term)")


def h_of(z):
    return holonomy(z.reshape(-1, 3))


eps = 1e-7
g_fd = np.zeros(3 * nv)
for k in range(3 * nv):
    d = np.zeros(3 * nv); d[k] = eps
    g_fd[k] = (h_of(p + d) - h_of(p - d)) / (2 * eps)
report("grad(H)", float(np.abs(g_an - g_fd).max() / max(np.abs(g_fd).max(), 1e-30)),
       TOL_GRAD)

h2 = 1e-4
Hs_fd = np.zeros((3 * nv, 3 * nv))
for a in range(3 * nv):
    da = np.zeros(3 * nv); da[a] = h2
    for b in range(a, 3 * nv):
        db = np.zeros(3 * nv); db[b] = h2
        Hs_fd[a, b] = Hs_fd[b, a] = (
            h_of(p + da + db) - h_of(p + da - db)
            - h_of(p - da + db) + h_of(p - da - db)) / (4 * h2 * h2)
report("hess(H)", float(np.abs(Hs_an - Hs_fd).max() / max(np.abs(Hs_fd).max(), 1e-30)),
       TOL_HESS)


# ------------------------------------------------------- 2/3. twisting energy
print("\n2. twisting energy -- force and Jacobian vs finite differences")
e_tw = DERElasticEnergy3DTwist(EI=1.0e3, EA=1.0e5, deltaL=dL, nv=nv)
q = np.concatenate([p, [0.2, -0.5]])        # positions + (yaw_l, yaw_r)
errs = e_tw.check_gradients(q, eps=1e-6, verbose=False)
report("dE/dq   (force)", errs["gradient_rel_err"], TOL_GRAD)
report("d2E/dq2 (Jacobian)", errs["hessian_rel_err"], TOL_GRAD)
print(f"       GJ = {e_tw.GJ:.4g} = EI / (1 + nu);  DOFs = {e_tw.n_dof} = 3*nv + 2;"
      f"  total twist Phi = {e_tw.total_twist(q):+.4f} rad")

print("\n3. GJ = 0 must reduce exactly to the twist-free der3d model")
e_0 = DERElasticEnergy3DTwist(EI=1.0e3, EA=1.0e5, deltaL=dL, nv=nv, GJ=0.0)
e_3 = DERElasticEnergy3D(EI=1.0e3, EA=1.0e5, deltaL=dL)
F0, _ = e_0.force_and_jacobian(q)
F3, _ = e_3.force_and_jacobian(q[:3 * nv])
report("position force matches der3d", float(np.abs(F0[:3 * nv] - F3).max()), 1e-12)
report("twist force is identically zero", float(np.abs(F0[3 * nv:]).max()), 1e-12)


# ------------------------------------------------------ 4. boundary map
print("\n4. boundary map d(q_b)/d(u), 14 x 12")
env = SimulatorEnv_3D(nv=15, rod_length=1.0, twist=True,
                      R_outer=0.003, r_inner=0.0, E_al=2e9, rho_al=1200)
u = np.array([0.03, -0.02, 0.05, 0.3, -0.2, 0.7, 0.9, 0.1, -0.04, 0.25, 0.4, -0.6])
J_an = boundary_node_jacobian_3d_twist(u, env.deltaL)
J_fd = np.zeros_like(J_an)
eps = 1e-7
for i in range(12):
    d = np.zeros(12); d[i] = eps
    J_fd[:, i] = (env._control_node_from_pose(u + d)
                  - env._control_node_from_pose(u - d)) / (2 * eps)
report("d(q_b)/d(u)", float(np.abs(J_an - J_fd).max()), TOL_GRAD)


# ------------------------------------------------------ 5. adjoint dL/du
print("\n5. adjoint dL/du vs finite differences  (the yaw columns are the point)")
KW = dict(nv=15, dt=0.01, rod_length=1.0, total_time=0.05, twist=True,
          R_outer=0.003, r_inner=0.0, E_al=2e9, rho_al=1200)
target = np.array([0.45, 0.05, 0.10])


# The default Newton tolerance (EI / L^2 * 1e-3) is far looser than the 1e-6
# probe below: it leaves the equilibrium uncertain by more than the difference
# being measured, and the comparison then reports the solver's slop rather than
# the adjoint's error. Tighten it so this checks the math.
TOL_FACTOR = 1e-6


def run(u_vec, steps=1):
    """Hold the control constant so the final state is a function of u alone.

    With the 2008 frame there is no frame state to carry, so this is true for
    any number of steps: every step re-solves the same equilibrium problem.
    """
    env = SimulatorEnv_3D(**KW, max_newton_iter=2000)
    env.tol *= TOL_FACTOR
    env.reset()
    info = None
    for _ in range(steps):
        _, _, info = env.step(u_vec)
    return env, info


u0 = np.array([0.0, 0, 0, 0, 0, 0.0, 0.92, 0.05, 0.03, 0.2, 0.3, 0.8])
env, info = run(u0)
a_free = dloss_dq_free_node(info["q"], env.free_index, 3, env.mid_node, target)
g_an = implicit_final_control_grad(info["J"], env.free_index, env.fixed_index,
                                   a_free, u0, env.deltaL, dim=3, twist=True)

g_fd = np.zeros(12)
h = 1e-6
for i in range(6, 12):                     # left end is clamped; vary the right one
    d = np.zeros(12); d[i] = h
    _, ip = run(u0 + d)
    _, im = run(u0 - d)
    Lp, _ = node_tracking_loss(ip["q"], 3, env.mid_node, target)
    Lm, _ = node_tracking_loss(im["q"], 3, env.mid_node, target)
    g_fd[i] = (Lp - Lm) / (2 * h)

np.set_printoptions(precision=6, suppress=True)
names = ["x_r", "y_r", "z_r", "th_r", "ph_r", "yaw_r"]
print("       component   adjoint        finite-diff")
for k, nm in enumerate(names):
    print(f"       {nm:9s} {g_an[6 + k]:+13.6f} {g_fd[6 + k]:+13.6f}")
scale = max(np.abs(g_fd[6:]).max(), 1e-30)
report("dL/du over the whole right end",
       float(np.abs(g_an[6:] - g_fd[6:]).max() / scale), TOL_GRAD)
yaw_err = abs(g_an[11] - g_fd[11]) / max(abs(g_fd[11]), 1e-30)
report("dL/d(yaw) specifically", float(yaw_err), 1e-4)

# The 2010 reference frames were carried forward by parallel transport once a
# step was committed, so a perturbed control followed a slightly different frame
# path -- a dependence the single-step implicit gradient could not see, and the
# check below measured how big it was.  The 2008 frame is a function of the
# positions, so there is no such drift and this is a plain equality.
n_steps = 3
env_m, info_m = run(u0, n_steps)
a_m = dloss_dq_free_node(info_m["q"], env_m.free_index, 3, env_m.mid_node, target)
g_m = implicit_final_control_grad(info_m["J"], env_m.free_index, env_m.fixed_index,
                                  a_m, u0, env_m.deltaL, dim=3, twist=True)
g_fdm = np.zeros(12)
for i in range(6, 12):
    d = np.zeros(12); d[i] = h
    _, ipm = run(u0 + d, n_steps)
    _, imm = run(u0 - d, n_steps)
    Lp, _ = node_tracking_loss(ipm["q"], 3, env_m.mid_node, target)
    Lm, _ = node_tracking_loss(imm["q"], 3, env_m.mid_node, target)
    g_fdm[i] = (Lp - Lm) / (2 * h)
report(f"dL/du after {n_steps} steps (no frame history)",
       float(np.abs(g_m[6:] - g_fdm[6:]).max() / max(np.abs(g_fdm[6:]).max(), 1e-30)),
       TOL_GRAD)

if abs(g_fd[11]) < 1e-9:
    print("       WARNING: dL/d(yaw) is numerically zero -- yaw is not "
          "influencing the shape, so the bend/twist coupling is not active.")
else:
    print(f"       dL/d(yaw) = {g_fd[11]:.4g}, comfortably nonzero: yaw is a live "
          f"control input.")

print("\n" + "=" * 72)
print(f"{sum(results)}/{len(results)} checks passed"
      f"{'' if all(results) else '  <-- SOMETHING IS WRONG'}")
sys.exit(0 if all(results) else 1)
