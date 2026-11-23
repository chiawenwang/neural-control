"""Panel (a) physics: realized path and constrained landscape ``E(s; lambda)``.

Setup (``config.yaml: model``): ``SimulatorEnv_2D``, both ends clamped at a
distance ``(1 - eps) L``, so the strip buckles into a bistable arch; the
homotopy rotates the right clamp, ``theta_r(lambda) = lambda * theta_max``.

Steps (each logged, results cached under ``paper_figures/cache/`` by a hash
of the physics settings):

1. both arches from sine-shaped guesses; convergence and stability checked;
2. the rotation direction and critical angle ``theta_c`` at which the upper
   branch ends: coarse sweep both ways, then bisection; ``theta_max = 1.3 theta_c``;
3. the realized quasi-static path, ``M`` warm-started steps from the upper arch
   (s, E, Newton iterations, lowest Hessian eigenvalue, landscape curvature at
   the state, nodes); exactly one snap is asserted;
4. the lower branch over the same steps (the other well before the snap);
5. ``E(s; lambda)`` for the curve and surface lambdas: the midpoint y pinned
   at ``s``, the other DOFs solved by ``solve_equilibrium``, continuation from
   each well outward in both directions, merged by the lowest energy; every
   local extremum is released and matched to an unconstrained equilibrium.

Energies are reported in ``EI/L`` and ``s`` in ``L``.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time

import numpy as np

from paper_figures.common import sim_adapters as sa

CACHE_VERSION = "fig01-landscape-v3"


# ------------------------------------------------------------------- setup

def physics_spec(cfg):
    """The settings the computation depends on (hashed into the cache key)."""
    lcfg = {k: v for k, v in cfg["landscape"].items() if k != "workers"}
    return {"version": CACHE_VERSION, "model": cfg["model"], "homotopy": cfg["homotopy"],
            "landscape": lcfg, "obstacle": cfg["obstacle"]}


def spec_hash(spec) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def build_arch(model, obstacle=None):
    obs = []
    if obstacle is not None:
        obs = [sa.make_circle(obstacle["center"], obstacle["radius"], obstacle["k_contact"])]
    return sa.PlanarArch(material=model["material"], nv=model["nv"], length=model["length"],
                         eps=model["eps"], gravity=model["gravity"],
                         max_iter=model["newton_max_iter"], obstacles=obs,
                         kick_amp=model["kick_amp"],
                         descent_max_iter=model["descent_max_iter"])


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------- branches

def initial_branches(arch):
    """Upper and lower arches at theta_r = 0 from ``+/- A sin^2`` guesses."""
    out = {}
    for name, sign in (("upper", +1), ("lower", -1)):
        q0, amp = arch.initial_guess(sign)
        r = arch.solve(q0, 0.0)
        if not r.ok or r.min_eig <= 0.0:
            raise RuntimeError(f"{name} arch did not converge to a stable state "
                               f"(ok={r.ok}, min_eig={r.min_eig})")
        y = arch.nodes(r.q)[:, 1]
        out[name] = {"q": r.q, "s": arch.s_of(r.q), "iterations": r.iterations,
                     "min_eig": r.min_eig, "guess_amplitude": amp,
                     "asymmetry": float(np.abs(y - y[::-1]).max()),
                     "E": arch.total_energy(r.q) / arch.E_unit}
        log(f"  {name} arch: s = {out[name]['s']:+.5f} L, E = {out[name]['E']:.4f} EI/L, "
            f"{r.iterations} Newton iterations, min eig {r.min_eig:.3e}, "
            f"asymmetry {out[name]['asymmetry']:.1e} m")
    return out


def find_theta_c(arch, q_up, h, direction=None):
    """Rotation direction and critical angle of the upper branch.

    Both directions are swept; the one whose branch ends first is chosen,
    unless ``direction`` (+1 / -1) is imposed -- as for the obstacle scenario,
    which reuses the free scenario's rotation.
    """
    step, jump = h["sweep_step"], h["jump_tol"] * arch.L
    found = {}
    for d in (+1, -1):
        q, s_prev, th_prev = q_up, arch.s_of(q_up), 0.0
        found[d] = None
        for i in range(1, int(round(h["sweep_limit"] / step)) + 1):
            th = d * i * step
            r = arch.solve(q, th, kicks=0, fallback=False)
            if not r.ok or abs(arch.s_of(r.q) - s_prev) > jump:
                found[d] = (th_prev, th, q)
                break
            q, s_prev, th_prev = r.q, arch.s_of(r.q), th
        log(f"  sweep {'+' if d > 0 else '-'}: " +
            (f"upper branch ends between {found[d][0]:+.3f} and {found[d][1]:+.3f} rad"
             if found[d] else f"no snap up to {d * h['sweep_limit']:+.3f} rad"))
    snapping = {d: b for d, b in found.items() if b is not None}
    if direction is not None:
        snapping = {d: b for d, b in snapping.items() if d == direction}
    if not snapping:
        raise RuntimeError("no rotation direction snaps the arch")
    d = min(snapping, key=lambda k: abs(snapping[k][1]))
    th_a, th_b, q_a = snapping[d]
    s_a = arch.s_of(q_a)
    n_bis = 0
    while abs(th_b - th_a) > h["bisection_tol"]:
        th_m = 0.5 * (th_a + th_b)
        r = arch.solve(q_a, th_m, max_iter=50, kicks=0, fallback=False)
        n_bis += 1
        if r.ok and abs(arch.s_of(r.q) - s_a) < jump:
            th_a, q_a, s_a = th_m, r.q, arch.s_of(r.q)
        else:
            th_b = th_m
    r = arch.solve(q_a, th_a, max_iter=50, kicks=0, fallback=False)
    out = {"direction": int(d), "theta_c": float(th_a), "theta_c_bracket": [float(th_a), float(th_b)],
           "s_at_fold": arch.s_of(r.q), "min_eig_at_fold": float(r.min_eig),
           "E_at_fold": arch.total_energy(r.q) / arch.E_unit, "nodes_at_fold": arch.nodes(r.q),
           "kappa_at_fold": arch.schur_curvature(r.J) * arch.L ** 3 / arch.env.EI,
           "bisection_steps": n_bis,
           "no_snap": [int(k) for k, b in found.items() if b is None],
           "other_direction_branch_end": {int(k): [float(b[0]), float(b[1])]
                                          for k, b in found.items() if b is not None and k != d}}
    log(f"  theta_c = {th_a:+.6f} rad ({np.degrees(th_a):+.3f} deg), bracket width "
        f"{abs(th_b - th_a):.1e}, {n_bis} bisection steps; at theta_c-: s = {out['s_at_fold']:+.4f}, "
        f"min eig {out['min_eig_at_fold']:.3e}")
    return out


def rollout(arch, q_start, thetas, label):
    """Warm-started quasi-static path; every state stable (see ``PlanarArch.solve``)."""
    rec = {k: [] for k in ("s", "E", "E_el", "iterations", "first_attempt", "attempts",
                           "min_eig", "kappa", "nodes", "contact")}
    q = q_start
    t0 = time.time()
    for k, th in enumerate(thetas):
        r = arch.solve(q, th)
        if not r.ok:
            raise RuntimeError(f"{label}: step {k} (theta {th:.5f}) did not converge: {r.attempts}")
        q = r.q
        rec["s"].append(arch.s_of(q))
        rec["E"].append(arch.total_energy(q) / arch.E_unit)
        rec["E_el"].append(arch.elastic_energy(q) / arch.E_unit)
        rec["iterations"].append(r.iterations)
        rec["first_attempt"].append(r.attempts[0][1])
        rec["attempts"].append("+".join(a[0] for a in r.attempts))
        rec["min_eig"].append(r.min_eig)
        rec["kappa"].append(arch.schur_curvature(r.J) * arch.L ** 3 / arch.env.EI)
        rec["nodes"].append(arch.nodes(q).copy())
        rec["contact"].append(arch.contact_energy(q) / arch.E_unit)
    out = {k: (np.array(v) if k != "attempts" else v) for k, v in rec.items()}
    log(f"  {label}: {len(thetas)} states in {time.time() - t0:.1f} s, "
        f"Newton iterations median {np.median(out['iterations']):.0f}, max {out['iterations'].max()}")
    return out


def detect_snap(path, h, L):
    ds = np.abs(np.diff(path["s"]))
    its = path["iterations"][1:]
    med_ds, med_it = float(np.median(ds)), float(np.median(its))
    jump = (ds > h["snap_ratio"] * med_ds) & (ds > h["snap_min_jump"] * L)
    spike = its > h["snap_ratio"] * max(med_it, 1.0)
    steps = np.nonzero(jump | spike)[0] + 1
    return steps, {"median_ds": med_ds, "median_iterations": med_it,
                   "jump_steps": (np.nonzero(jump)[0] + 1).tolist(),
                   "spike_steps": (np.nonzero(spike)[0] + 1).tolist()}


# -------------------------------------------------------------- landscape

_ARCH = {}


def _worker_arch(model, obstacle):
    key = json.dumps([model, obstacle], sort_keys=True, default=float)
    if key not in _ARCH:
        _ARCH[key] = build_arch(model, obstacle)
    return _ARCH[key]


def _predict(arch, q, s0, J, s1):
    """First-order continuation predictor in s, from the constrained Jacobian."""
    cf = arch.free_pinned
    q1 = q.copy()
    q1[arch.pin] = s1
    try:
        dq = -np.linalg.solve(J[np.ix_(cf, cf)], J[cf, arch.pin]) * (s1 - s0)
    except np.linalg.LinAlgError:
        return q1
    if np.abs(dq).max() < 0.05 * arch.L:
        q1[cf] += dq
    return q1


def landscape_job(job):
    """``E(s; lambda)`` at one lambda: sweeps from each well, merged, validated."""
    arch = _worker_arch(job["model"], job["obstacle"])
    th, grid = job["theta"], np.asarray(job["s_grid"])
    pts, fails, wells = [], [], []
    t0 = time.time()
    for wi, qw in enumerate(job["wells"]):
        r0 = arch.solve(qw, th)
        sw = arch.s_of(r0.q)
        wells.append({"s": sw, "E": arch.total_energy(r0.q) / arch.E_unit,
                      "min_eig": r0.min_eig, "q": r0.q})
        pts.append(dict(i=-1, s=sw, E=wells[-1]["E"], dEds=0.0, eig=r0.min_eig, its=0,
                        well=wi, dir=0, q=r0.q))
        for d in (+1, -1):
            idx = np.nonzero(grid > sw + 1e-12)[0] if d > 0 else np.nonzero(grid < sw - 1e-12)[0][::-1]
            q_prev, s_prev, J_prev = r0.q, sw, r0.J
            for i in idx:
                s = float(grid[i])
                r = arch.solve(_predict(arch, q_prev, s_prev, J_prev, s), th, s=s,
                               max_iter=job["max_iter"])
                if not r.ok:
                    fails.append({"s": s, "well": wi, "dir": d, "attempts": r.attempts})
                    continue
                f = arch.residual(r.q)
                pts.append(dict(i=int(i), s=s, E=arch.total_energy(r.q) / arch.E_unit,
                                dEds=float(f[arch.pin] / arch.F_unit), eig=r.min_eig,
                                its=r.iterations, well=wi, dir=d, q=r.q))
                q_prev, s_prev, J_prev = r.q, s, r.J

    # merge: lowest energy per grid point, plus the wells themselves
    best = {}
    for p in pts:
        if p["i"] >= 0 and (p["i"] not in best or p["E"] < best[p["i"]]["E"]):
            best[p["i"]] = p
    grid_E = np.full(len(grid), np.nan)
    for i, p in best.items():
        grid_E[i] = p["E"]
    curve = sorted(list(best.values()) + [p for p in pts if p["i"] < 0], key=lambda p: p["s"])

    extrema = validate_extrema(arch, th, curve, job["match_tol"])
    by_well = {}
    for p in pts:
        by_well.setdefault(f"{p['well']}{'+' if p['dir'] >= 0 else '-'}", []).append(
            (p["s"], p["E"], p["dEds"]))
    return {
        "k": job["k"], "lam": job["lam"], "theta": th,
        "grid_E": grid_E,
        "curve_s": np.array([p["s"] for p in curve]),
        "curve_E": np.array([p["E"] for p in curve]),
        "curve_dEds": np.array([p["dEds"] for p in curve]),
        "curve_eig": np.array([p["eig"] for p in curve]),
        "curve_src": np.array([p["well"] * 10 + p["dir"] for p in curve]),
        "curve_is_well": np.array([p["i"] < 0 for p in curve]),
        "wells": [{k: v for k, v in w.items() if k != "q"} for w in wells],
        "sweeps": {k: np.array(sorted(v)) for k, v in by_well.items()},
        "extrema": extrema,
        "n_fail": len(fails),
        "fails": [{"s": f["s"], "well": f["well"], "dir": f["dir"]} for f in fails],
        "n_points": len(pts),
        "seconds": time.time() - t0,
    }


def validate_extrema(arch, th, curve, tol):
    """Release every interior local min / max of the merged curve and match it
    to an unconstrained equilibrium (stable for minima, a saddle for maxima)."""
    E = np.array([p["E"] for p in curve])
    out = []
    for i in range(1, len(curve) - 1):
        is_min = E[i] < E[i - 1] and E[i] <= E[i + 1]
        is_max = E[i] > E[i - 1] and E[i] >= E[i + 1]
        if not (is_min or is_max):
            continue
        p = curve[i]
        if is_min:
            r = arch.solve(p["q"], th)
        else:
            r = arch.solve(p["q"], th, kicks=0, fallback=False)
        rec = {"kind": "min" if is_min else "max", "s": p["s"], "E": p["E"],
               "is_well_point": p["i"] < 0, "converged": bool(r.ok)}
        if r.ok:
            s_eq, E_eq = arch.s_of(r.q), arch.total_energy(r.q) / arch.E_unit
            # a maximum that is a crossing of two continuation branches is a kink,
            # not a stationary point: the slope does not change sign smoothly there
            same_branch = (curve[i - 1]["well"], curve[i - 1]["dir"]) == (p["well"], p["dir"]) \
                == (curve[i + 1]["well"], curve[i + 1]["dir"])
            rec.update({"s_eq": s_eq, "E_eq": E_eq, "min_eig_eq": r.min_eig,
                        "ds": abs(s_eq - p["s"]) / arch.L, "dE": abs(E_eq - p["E"]),
                        "smooth": bool(same_branch or p["i"] < 0)})
            rec["match"] = bool(rec["ds"] <= tol[0] and rec["dE"] <= tol[1]
                                and (r.min_eig > 0) == is_min)
        else:
            rec["match"] = False
        out.append(rec)
    return out


# ------------------------------------------------------------------ driver

def lambda_lists(lc, k_snap, M):
    """Curve lambdas (on the step grid) and the surface lambda grid."""
    lam_pre = (k_snap - 1) / M
    delta = lc["post_snap_delta"]
    k_post = M if delta is None else min(M, int(round((k_snap / M + delta) * M)))
    fixed = [x for x in lc["curves_fixed"] if x < lam_pre - 0.05]
    n_early = lc["n_curves"] - 2
    if len(fixed) < n_early:
        fixed = list(np.linspace(0.0, lam_pre, n_early + 1)[:-1])
    k_curves = sorted({int(round(x * M)) for x in fixed[:n_early]} | {k_snap - 1, k_post})
    k_surf = sorted({int(round(x * M)) for x in np.linspace(0.0, k_post / M, lc["surface_n_lambda"])}
                    | set(k_curves))
    return k_curves, k_surf


def scenario(cfg, name, obstacle, theta_max=None, s_max=None):
    """Branches, theta_c, realized + lower paths and the landscape jobs of one scenario."""
    log(f"[{name}] building the arch" + (f" with obstacle {obstacle}" if obstacle else ""))
    arch = build_arch(cfg["model"], obstacle)
    h, lc = cfg["homotopy"], cfg["landscape"]
    res = {"name": name, "obstacle": obstacle}
    if obstacle is not None:
        q_test = arch.initial_guess(+1)[0]
        q_test[1::2] *= 1.25           # push nodes into the circle for a contact check
        res["contact_fd_error"] = sa.check_contact_energy(arch.obstacles, q_test)
        log(f"  contact energy vs src/contact.py forces: max rel error {res['contact_fd_error']:.2e}")
    res["branches"] = initial_branches(arch)
    res["theta_c"] = find_theta_c(arch, res["branches"]["upper"]["q"], h,
                                  direction=None if theta_max is None else int(np.sign(theta_max)))
    if theta_max is None:
        theta_max = h["theta_max_factor"] * res["theta_c"]["theta_c"]
    res["theta_max"] = float(theta_max)
    M = h["steps"]
    lam = np.arange(M + 1) / M
    thetas = lam * theta_max
    res["lam"], res["theta"] = lam, thetas
    log(f"  theta_max = {theta_max:+.6f} rad; rollout over {M} steps")
    res["path"] = rollout(arch, res["branches"]["upper"]["q"], thetas, "realized path")
    steps, info = detect_snap(res["path"], h, arch.L)
    res["snap_detection"] = info
    if len(steps) != 1:
        raise RuntimeError(f"[{name}] expected exactly one snap, found steps {steps.tolist()} ({info})")
    k = int(steps[0])
    res["k_snap"] = k
    res["lam_snap"] = float(lam[k])
    p = res["path"]
    log(f"  snap at step {k}: lambda {lam[k - 1]:.3f} -> {lam[k]:.3f}, s {p['s'][k - 1]:+.4f} -> "
        f"{p['s'][k]:+.4f}, Newton iterations {p['iterations'][k]} ({p['attempts'][k]}) vs median "
        f"{info['median_iterations']:.0f}; min eig before {p['min_eig'][k - 1]:.3e}")
    res["lower"] = rollout(arch, res["branches"]["lower"]["q"], thetas, "lower branch")
    lo = res["lower"]
    res["lower_jumps"] = int(np.sum(np.abs(np.diff(lo["s"])) > h["jump_tol"] * arch.L))
    res["post_snap_vs_lower"] = float(np.abs(p["nodes"][k:] - lo["nodes"][k:]).max())
    log(f"  lower branch jumps: {res['lower_jumps']}; realized after the snap vs lower branch: "
        f"max node distance {res['post_snap_vs_lower']:.2e} m")
    k_curves, k_surf = lambda_lists(lc, k, M)
    res["k_curves"], res["k_surface"] = k_curves, k_surf
    if s_max is None:                  # 1.2 x the arch amplitude (shared by both scenarios)
        s_max = lc["s_max_factor"] * abs(res["branches"]["upper"]["s"])
    res["s_grid"] = np.linspace(-s_max, s_max, lc["n_s"])
    jobs = []
    for kk in sorted(set(k_curves) | set(k_surf)):
        wells = [lo["nodes"][kk].reshape(-1)]
        if kk < k:
            wells.insert(0, p["nodes"][kk].reshape(-1))
        jobs.append({"scenario": name, "k": kk, "lam": float(lam[kk]), "theta": float(thetas[kk]),
                     "wells": wells, "s_grid": res["s_grid"], "model": cfg["model"],
                     "obstacle": obstacle, "max_iter": lc["max_iter"],
                     "match_tol": lc["min_match_tol"]})
    return arch, res, jobs


def compute(cfg, workers=None):
    """Both scenarios; returns ``{name: result}``."""
    t0 = time.time()
    arch, free, jobs = scenario(cfg, "free", None)
    results = {"free": free}
    all_jobs = list(jobs)
    ob = cfg["obstacle"]
    if ob.get("enabled", False):
        A0 = free["branches"]["upper"]["s"]
        spec = {"center": [ob["center_x"] * arch.ell, ob["bottom"] * A0 + ob["radius"] * arch.L],
                "radius": ob["radius"] * arch.L, "k_contact": ob["k_contact"]}
        _, obst, jobs = scenario(cfg, "obstacle", spec, theta_max=free["theta_max"],
                                 s_max=float(free["s_grid"][-1]))
        results["obstacle"] = obst
        all_jobs += jobs
    n = workers or cfg["landscape"].get("workers", 1)
    n = max(1, min(int(n), len(all_jobs), mp.cpu_count()))
    log(f"landscapes: {len(all_jobs)} lambda values on {n} processes")
    t1 = time.time()
    if n > 1:
        with mp.get_context("fork").Pool(n) as pool:
            outs = pool.map(landscape_job, all_jobs, chunksize=1)
    else:
        outs = [landscape_job(j) for j in all_jobs]
    for j, o in zip(all_jobs, outs):
        results[j["scenario"]].setdefault("landscapes", {})[o["k"]] = o
    for name, res in results.items():
        ls = res["landscapes"]
        n_fail = sum(o["n_fail"] for o in ls.values())
        n_pts = sum(o["n_points"] for o in ls.values())
        log(f"  [{name}] {len(ls)} landscapes, {n_pts} constrained states, {n_fail} failed points "
            f"(dropped), {sum(o['seconds'] for o in ls.values()):.0f} s CPU")
    log(f"computation done in {time.time() - t0:.0f} s (landscapes {time.time() - t1:.0f} s)")
    return results
