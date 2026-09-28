"""Validate the torsion model end to end.

Run this after touching src/rodframes.py, the twist energy, or the twist
boundary map::

    python scripts/check_twist.py

Everything the adjoint relies on is checked against finite differences.  The
exact Hessian matters in particular: an approximate one still lets Newton
converge but silently corrupts the training gradient, so it is checked
directly rather than inferred from the solver behaving.

Checks, in order:
  1. reference twist psi -- gradient and Hessian vs finite differences
  2. the full twisting energy -- force and Jacobian vs finite differences
  3. GJ = 0 reduces exactly to the twist-free der3d model
  4. the 14x12 boundary map d(q_b)/d(u) vs finite differences
  5. the adjoint dL/du vs finite differences -- including the yaw columns,
     which is what decides whether yaw is a usable control input
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from blas_threads import limit_blas_threads
limit_blas_threads()

import numpy as np

from rodframes import RodFrames, psi_grad_hess
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


# ---------------------------------------------------------------- 1. psi
print("\n1. reference twist psi -- derivatives vs finite differences")
rng = np.random.default_rng(7)
nv, L = 8, 1.0
dL = L / (nv - 1)
pos0 = np.stack([np.arange(nv) * dL, np.zeros(nv), np.zeros(nv)], 1)
frames = RodFrames(nv)
frames.reset(pos0)
print(f"       straight rod: max |psi| = {np.abs(frames.reference_twist(pos0)).max():.1e}"
      f"  (parallel transport along a straight rod closes up exactly)")

pos1 = pos0 + 0.02 * rng.normal(size=(nv, 3))
frames.advance(pos1)
# Evaluate away from the stored configuration: the re-adaptation term in
# grad(psi) is invisible at the stored tangents and only shows up here.
p = pos1.reshape(-1) + 0.01 * rng.normal(size=3 * nv)


def psi_i(z, i):
    return frames.reference_twist(z.reshape(-1, 3))[i - 1]


worst_g = worst_h = 0.0
for i in range(1, nv - 1):
    idx = np.arange(3 * (i - 1), 3 * (i + 2))
    h = 1e-7
    g_fd = np.zeros(9)
    for k in range(9):
        d = np.zeros(3 * nv); d[idx[k]] = h
        g_fd[k] = (psi_i(p + d, i) - psi_i(p - d, i)) / (2 * h)
    P = p.reshape(-1, 3)
    g_an, H_an = psi_grad_hess(P[i - 1], P[i], P[i + 1],
                               frames.stored_tangents[i - 1], frames.stored_tangents[i])
    h2 = 1e-4
    H_fd = np.zeros((9, 9))
    for a in range(9):
        for b in range(9):
            da = np.zeros(3 * nv); da[idx[a]] = h2
            db = np.zeros(3 * nv); db[idx[b]] = h2
            H_fd[a, b] = (psi_i(p + da + db, i) - psi_i(p + da - db, i)
                          - psi_i(p - da + db, i) + psi_i(p - da - db, i)) / (4 * h2 * h2)
    worst_g = max(worst_g, np.abs(g_an - g_fd).max() / max(np.abs(g_fd).max(), 1e-30))
    worst_h = max(worst_h, np.abs(H_an - H_fd).max() / max(np.abs(H_fd).max(), 1e-30))

report("grad(psi), worst interior node", worst_g, TOL_GRAD)
report("hess(psi), worst interior node", worst_h, TOL_HESS)


# ------------------------------------------------------- 2/3. twisting energy
print("\n2. twisting energy -- force and Jacobian vs finite differences")
e_tw = DERElasticEnergy3DTwist(EI=1.0e3, EA=1.0e5, deltaL=dL, nv=nv, frames=frames)
q = np.concatenate([p, 0.3 * rng.normal(size=nv - 1)])
errs = e_tw.check_gradients(q, eps=1e-6, verbose=False)
report("dE/dq   (force)", errs["gradient_rel_err"], TOL_GRAD)
report("d2E/dq2 (Jacobian)", errs["hessian_rel_err"], TOL_GRAD)
print(f"       GJ = {e_tw.GJ:.4g} = EI / (1 + nu);  DOFs = {e_tw.n_dof} = 4*nv - 1")

print("\n3. GJ = 0 must reduce exactly to the twist-free der3d model")
e_0 = DERElasticEnergy3DTwist(EI=1.0e3, EA=1.0e5, deltaL=dL, nv=nv, frames=frames, GJ=0.0)
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

    One step by default: the material frames advance only *after* a step is
    committed, so over a single step they are identical for every perturbed
    control and the final state is a clean function of u.  Over several steps
    the frames follow the path, which the single-step implicit gradient does
    not model -- that drift is measured separately below.
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
# How much does the frame history add over a longer rollout?  The frames are
# carried forward by parallel transport once a step is committed, so a
# perturbed control follows a slightly different frame path -- a dependence the
# single-step implicit gradient does not see.  This is the same class of
# approximation multistep_adjoint_gradient already makes for the equilibrium
# branch; this measures how big the frame part of it is.
g_multi = None
for n_steps in (3,):
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
    drift = float(np.abs(g_m[6:] - g_fdm[6:]).max() / max(np.abs(g_fdm[6:]).max(), 1e-30))
    print(f"       frame-history drift over {n_steps} steps: {drift:.2e} relative "
          f"(single-step is exact; this is informational, not a failure)")

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
