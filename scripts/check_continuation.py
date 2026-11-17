"""Regression check of the quasi-static stepping (src/continuation.py).

Run this after touching src/continuation.py, src/newton.py or the
environments' step()::

    python scripts/check_continuation.py

1. Compression.  configs/example_3d_twist.yaml, nv = 21: the rod is settled
   at its initial control u0 (the equilibrium under gravity), then the right
   end is pushed 0.3 m along -x in N = 10, 50 and 200 steps.  Every step has
   to converge to a stable equilibrium, the rod has to bend *down* (mid node
   at z ~ -0.30 m), and the three final shapes have to agree: the realized
   branch is set by the boundary path, not by the number of steps.  For
   comparison the old flow is run as well -- no equilibrium at u0, every
   option off, N = 10 -- which lands on the compressed straight rod or on an
   arch against gravity, both unstable.
2. Snap-through.  From the arch of (1) the end tangent is turned down, which
   has no stable continuation on that arch: the rod flips to the upward arch.
   With N = 25 ... 200 steps the flip is flagged as a snap (by the energy
   descent or by the displacement indicator), every step is stable, and the
   final shapes agree.
3. Bit-identity.  With every option off a step is the single warm-started
   Newton solve it used to be: the state and Jacobian equal a direct call of
   newton.solve_equilibrium exactly.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from blas_threads import limit_blas_threads
limit_blas_threads()

import time

import numpy as np

from config import load_config
from env_factory import build_env, reset_env, settle_initial, initial_control
from newton import is_stable, solve_equilibrium

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:64s} {detail}")


def make_env(N, options=None):
    cfg = load_config(_ROOT / "configs/example_3d_twist.yaml")
    cfg["rod"]["nv"] = 21
    cfg["rod"]["total_time"] = N * cfg["rod"]["dt"]
    cfg["workspace"]["enabled"] = False
    if options is not None:
        cfg["newton"].update(options)
    env = build_env(cfg)
    reset_env(env, cfg)
    return cfg, env


def drive(env, u0, path, N):
    """Step along ``u0 + path(k/N)`` on the right end; per-step diagnostics."""
    rows = []
    for k in range(1, N + 1):
        u = u0.copy()
        u[6:] += path(k / N)
        _, _, info = env.step(u)
        rows.append({"converged": info["converged"], "stable": info["stable"],
                     "substeps": info["substeps"], "snap": info["snap"], "snap_by": info["snap_by"],
                     "iterations": info["newton_iterations"]})
    return rows


def summarize(rows, env):
    P = env.positions()
    return {"converged": all(r["converged"] for r in rows),
            "unstable": sum(1 for r in rows if r["stable"] is False),
            "stable_checked": all(r["stable"] is not None for r in rows),
            "max_substeps": max(r["substeps"] for r in rows),
            "snap_steps": [k + 1 for k, r in enumerate(rows) if r["snap"]],
            "iterations": sum(r["iterations"] for r in rows),
            "mid": P[env.mid_node], "shape": P}


OFF = {"initial_equilibrium": False, "continuation": False, "stability_check": False,
       "snap_rule": False}

# ------------------------------------------------------------ 1. compression
print("\n1. compression by 0.3 m from the equilibrium at u0: stable, downward, N-independent")
compress = lambda l: np.array([-0.3 * l, 0, 0, 0, 0, 0])
shapes = {}
for N in (10, 50, 200):
    cfg, env = make_env(N)
    t0 = time.time()
    info0 = settle_initial(env, cfg)
    u0 = initial_control(cfg)
    s = summarize(drive(env, u0, compress, N), env)
    shapes[N] = s["shape"]
    print(f"       N={N:4d}: settled mid z={env.positions()[env.mid_node][2]:+.4f} (start "
          f"{'stable' if info0['stable'] else 'UNSTABLE'}), final mid "
          f"{np.array2string(s['mid'], precision=4)}, max substeps {s['max_substeps']}, "
          f"snaps {s['snap_steps'] or 'none'}, {s['iterations']} Newton iterations, "
          f"{time.time() - t0:.1f}s")
    check(f"N={N}: every step converged and was checked stable",
          s["converged"] and s["unstable"] == 0 and s["stable_checked"],
          f"unstable steps {s['unstable']}")
    check(f"N={N}: the rod bends down, mid z in [-0.32, -0.28] m", -0.32 < s["mid"][2] < -0.28,
          f"z = {s['mid'][2]:+.4f}")
    check(f"N={N}: no snap-through", not s["snap_steps"])
ref = shapes[200]
for N in (10, 50):
    d = float(np.linalg.norm(shapes[N] - ref, axis=1).max())
    check(f"N={N} and N=200 end in the same shape (max node distance < 1 mm)", d < 1e-3,
          f"{d * 1e3:.4f} mm")

# the old flow, for the record
cfg, env = make_env(10, OFF)
u0 = initial_control(cfg)
rows = drive(env, u0, compress, 10)
P = env.positions()
stable_old = is_stable(env._solve(env.q, env._control_node_from_pose(env.u_prev))[2], env.free_index)
print(f"       old flow (no equilibrium at u0, every option off), N=10: final mid "
      f"{np.array2string(P[env.mid_node], precision=4)}, arc length "
      f"{np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)):.4f} m, "
      f"{'stable' if stable_old else 'unstable'} end state -- "
      f"{'the compressed straight rod' if abs(P[env.mid_node][2]) < 0.02 else 'an arch against gravity' if P[env.mid_node][2] > 0 else 'a downward arch'}")

# ---------------------------------------------------------- 2. snap-through
print("\n2. the arch flips when its end tangent is turned down: a snap, N-independent landing")
flip = lambda l: np.array([-0.3 * min(2 * l, 1.0), 0, 0, 0, -0.6 * max(2 * l - 1.0, 0.0), 0])
shapes = {}
for N in (25, 50, 100, 200):
    cfg, env = make_env(N)
    t0 = time.time()
    settle_initial(env, cfg)
    u0 = initial_control(cfg)
    rows = drive(env, u0, flip, N)
    s = summarize(rows, env)
    shapes[N] = s["shape"]
    by = sorted({r["snap_by"] for r in rows if r["snap"]})
    print(f"       N={N:4d}: final mid {np.array2string(s['mid'], precision=4)}, snap at steps "
          f"{s['snap_steps'] or 'none'} ({', '.join(by) if by else '-'}), max substeps "
          f"{s['max_substeps']}, {s['iterations']} Newton iterations, {time.time() - t0:.1f}s")
    check(f"N={N}: every step converged and was checked stable",
          s["converged"] and s["unstable"] == 0 and s["stable_checked"])
    check(f"N={N}: the flip is flagged as a snap-through", bool(s["snap_steps"]))
    check(f"N={N}: the rod ends on the upward arch, mid z in [0.28, 0.32] m",
          0.28 < s["mid"][2] < 0.32, f"z = {s['mid'][2]:+.4f}")
ref = shapes[200]
for N in (25, 50, 100):
    d = float(np.linalg.norm(shapes[N] - ref, axis=1).max())
    check(f"N={N} and N=200 land on the same shape (max node distance < 1 mm)", d < 1e-3,
          f"{d * 1e3:.4f} mm")

# ------------------------------------------------------------ 3. bit-identity
print("\n3. every option off: a step is the single Newton solve")
cfg, env = make_env(5, OFF)
u0 = initial_control(cfg)
same = True
for k in range(1, 6):
    u = u0.copy()
    u[6] -= 0.02 * k
    q_old = env.q.copy()
    node = env._control_node_from_pose(u)
    q_start = q_old.copy()
    q_start[env.fixed_index] = node
    q_ref, flag_ref, J_ref, _ = solve_equilibrium(
        q_start, energy=env.energy, W=env.W, free_index=env.free_index, dim=3, tol=env.tol,
        obstacles=env.obstacles, n_pos=env.n_pos, max_iter=env.max_newton_iter, q_old=q_old,
        u_old=env.u, dt=env.dt, m=env.m, mMat=env.mMat, use_inertia=False,
        line_search=cfg["newton"]["line_search"], max_backtrack=cfg["newton"]["max_backtrack"],
        nonmonotone_memory=cfg["newton"]["nonmonotone_memory"], reg=cfg["newton"]["reg"])
    _, _, info = env.step(u)
    same &= (np.array_equal(info["q"], q_ref) and np.array_equal(info["J"], J_ref)
             and info["converged"] == (flag_ref > 0) and info["substeps"] == 1
             and info["stable"] is None and not info["snap"])
check("state and Jacobian equal a direct solve_equilibrium call, bit for bit", same)

print("\n" + "=" * 72)
print(f"{sum(results)}/{len(results)} checks passed"
      f"{'' if all(results) else '  <-- SOMETHING IS WRONG'}")
sys.exit(0 if all(results) else 1)
