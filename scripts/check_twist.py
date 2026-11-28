"""Validate the torsion model (``der3d_twist``) against MAE 263F.

Run this after touching src/rodframes.py or the twist energy::

    python scripts/check_twist.py [--ref263f PATH]

``PATH`` is a clone of github.com/StructuresComp/mae-263F (default
``../mae-263F`` next to this repository); its ``src/der/der_functions.py`` is
the reference the energy is compared with.

The test rod has a curved, non-planar rest shape (so kappaBar and the rest
lengths are non-trivial), an accepted state ``q_ref`` away from the rest shape,
and is evaluated at ``q != q_ref``, so the time-parallel transport and the
reference twist are exercised.

Checks, in order:
  a. force and Jacobian vs 263F getFs + getFb + getFt (after the DOF permutation)
  b. reference frame and reference twist vs 263F computeTimeParallel + getRefTwist
     (and the reset frame vs computeSpaceParallel)
  c. after commit, the gradient vs central differences of the energy at q = q_ref
  d. at q = q_ref, the symmetric part 0.5 (H + H^T) vs second differences of the
     energy (H itself is not symmetric; its antisymmetric part is reported)
  e. the energy at q is unchanged by commit(q)
"""
import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from blas_threads import limit_blas_threads
limit_blas_threads()

import numpy as np

from energy import DERElasticEnergy3DTwist
from rodframes import gripper_director, tangents

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--ref263f", default=str(_ROOT.parent / "mae-263F"),
                help="clone of github.com/StructuresComp/mae-263F")
args = ap.parse_args()
ref_dir = Path(args.ref263f) / "src" / "der"
if not (ref_dir / "der_functions.py").exists():
    sys.exit(f"263F reference not found: {ref_dir / 'der_functions.py'} "
             f"(clone github.com/StructuresComp/mae-263F and pass --ref263f)")
os.environ.setdefault("MPLBACKEND", "Agg")      # der_functions imports pyplot
sys.path.insert(0, str(ref_dir))
import der_functions as R                         # noqa: E402

TOL_EXACT = 1e-12    # same arithmetic, different code: rounding only
TOL_GRAD = 1e-6      # analytic vs central differences
TOL_HESS = 1e-4      # vs second differences, whose own truncation floor is ~1e-6

results = []


def report(name, err, tol):
    ok = err < tol
    results.append(ok)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:56s} {err:.3e}  (tol {tol:g})")


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


# ------------------------------------------------------------------ the rod
rng = np.random.default_rng(7)
nv = 8
ne = nv - 1
EI, EA = 1.0, 20.0          # comparable stretch and bend/twist entries in H
s = np.linspace(0.0, 1.0, nv)
rest = np.stack([s, 0.1 * np.sin(np.pi * s), 0.05 * s ** 2], 1)

e = DERElasticEnergy3DTwist(EI=EI, EA=EA, deltaL=1.0 / ne, nv=nv)
e.reset_frame(rest)
n_pos = e.n_pos

# Accepted state away from the rest shape, then the evaluation point.
q1 = np.concatenate([(rest + 0.03 * rng.normal(size=rest.shape)).reshape(-1),
                     0.5 * rng.normal(size=ne)])
e.commit(q1)
q2 = q1 + np.concatenate([0.02 * rng.normal(size=n_pos), 0.3 * rng.normal(size=ne)])

# 263F layout: x[4c:4c+3] = node c, x[4c+3] = theta^c.  q_ours = x[perm].
perm = np.concatenate([np.array([4 * c, 4 * c + 1, 4 * c + 2]) for c in range(nv)]
                      + [4 * np.arange(ne) + 3])


def to263(q):
    x = np.zeros(4 * nv - 1)
    x[perm] = q
    return x


# ------------------------------------------------- b. frames (checked first)
print("\nb. reference frame and reference twist vs 263F")
a1_s, a2_s = R.computeSpaceParallel(gripper_director(tangents(rest)[0]),
                                    to263(np.concatenate([rest.reshape(-1), np.zeros(ne)])))
from rodframes import space_parallel                                    # noqa: E402
a1_o, a2_o = space_parallel(gripper_director(tangents(rest)[0]), rest)
report("reset frame (a1, a2) vs computeSpaceParallel",
       max(rel(a1_o, a1_s), rel(a2_o, a2_s)), TOL_EXACT)

state = e.get_frame_state()
x2 = to263(q2)
a1_R, a2_R = R.computeTimeParallel(state["a1_ref"].copy(),
                                   to263(np.concatenate([state["q_ref"].reshape(-1),
                                                         np.zeros(ne)])), x2)
tan_R = R.computeTangent(x2)
ref_R = R.getRefTwist(a1_R, tan_R, state["refTwist_ref"].copy())
a1, a2, ref, m1, m2 = e.frames(q2)
report("a1 vs computeTimeParallel", rel(a1, a1_R), TOL_EXACT)
report("a2 vs computeTimeParallel", rel(a2, a2_R), TOL_EXACT)
report("refTwist vs getRefTwist", rel(ref, ref_R), TOL_EXACT)
report("frame state untouched by evaluation",
       float(np.abs(e.get_frame_state()["refTwist_ref"] - state["refTwist_ref"]).max()),
       1e-300)
print(f"       max |refTwist| = {np.abs(ref).max():.3e} rad")

# ------------------------------------------------------- a. force, Jacobian
print("\na. force and Jacobian vs 263F getFs + getFb + getFt")
m1_R, m2_R = R.computeMaterialDirectors(a1_R, a2_R, x2[3::4])
kappaBar_R = R.getKappa(to263(np.concatenate([rest.reshape(-1), np.zeros(ne)])),
                        a1_s, a2_s)
refLen = np.linalg.norm(np.diff(rest, axis=0), axis=1)
vor = np.r_[0.5 * refLen[0], 0.5 * (refLen[:-1] + refLen[1:]), 0.5 * refLen[-1]]
Fs, Js = R.getFs(x2, EA, refLen)
Fb, Jb = R.getFb(x2, m1_R, m2_R, kappaBar_R, EI, vor)
Ft, Jt = R.getFt(x2, ref_R, np.zeros(nv), e.GJ, vor)
F_R = (Fs + Fb + Ft)[perm]
J_R = (Js + Jb + Jt)[np.ix_(perm, perm)]
F, J = e.force_and_jacobian(q2)
report("kappaBar vs getKappa (rest shape)", rel(e.kappaBar, kappaBar_R), TOL_EXACT)
report("force F = -dE/dq", rel(F, F_R), TOL_EXACT)
report("Jacobian J = -d2E/dq2 (as is, non-symmetric)", rel(J, J_R), TOL_EXACT)
print(f"       |kappaBar|max = {np.abs(e.kappaBar).max():.3e};  DOFs = {e.n_dof} = 3*nv + ne")

# ------------------------------------------------------------- e. commit
print("\ne. commit(q) leaves the energy at q unchanged")
E_before = e.energy(q2)
e.commit(q2)
E_after = e.energy(q2)
report("|E_after - E_before| / |E|", abs(E_after - E_before) / abs(E_before), TOL_EXACT)

# --------------------------------------------- c. gradient at q = q_ref
print("\nc. gradient vs central differences of the energy, at q = q_ref")
g = e.gradient(q2)
eps = 1e-6
g_fd = np.zeros_like(g)
for k in range(q2.size):
    d = np.zeros_like(q2); d[k] = eps
    g_fd[k] = (e.energy(q2 + d) - e.energy(q2 - d)) / (2 * eps)
report("dE/dq", rel(g, g_fd), TOL_GRAD)
report("dE/dq, position rows", rel(g[:n_pos], g_fd[:n_pos]), TOL_GRAD)
report("dE/dq, theta rows", rel(g[n_pos:], g_fd[n_pos:]), TOL_GRAD)

# ----------------------------------------------- d. Hessian at q = q_ref
print("\nd. symmetric part of the Hessian vs second differences, at q = q_ref")
H = e.hessian(q2)
Hsym = 0.5 * (H + H.T)
h2 = 1e-4
n = q2.size
H_fd = np.zeros((n, n))
for a in range(n):
    da = np.zeros(n); da[a] = h2
    for b in range(a, n):
        db = np.zeros(n); db[b] = h2
        H_fd[a, b] = H_fd[b, a] = (
            e.energy(q2 + da + db) - e.energy(q2 + da - db)
            - e.energy(q2 - da + db) + e.energy(q2 - da - db)) / (4 * h2 * h2)
report("0.5 (H + H^T)", rel(Hsym, H_fd), TOL_HESS)
report("0.5 (H + H^T), theta-theta block",
       rel(Hsym[n_pos:, n_pos:], H_fd[n_pos:, n_pos:]), TOL_HESS)
report("0.5 (H + H^T), position-theta block",
       rel(Hsym[:n_pos, n_pos:], H_fd[:n_pos, n_pos:]), TOL_HESS)
print(f"       for scale: H itself vs second differences   {rel(H, H_fd):.3e}"
      f"   |H - H^T| / |H| = {np.abs(H - H.T).max() / np.abs(H).max():.3e}")

print("\n" + "=" * 72)
print(f"{sum(results)}/{len(results)} checks passed"
      f"{'' if all(results) else '  <-- SOMETHING IS WRONG'}")
sys.exit(0 if all(results) else 1)
