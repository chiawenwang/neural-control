"""Task P: 3D path tracking with contact (paper Sec. VII), trajectory objective.

The rod's mid node follows a reference path ``p`` over the whole motion; the
loss is a trajectory objective (paper eq. (objectives)),

    L = sum_{k=1..M} dlam |x_mid,k - p(lambda_k)|^2,   dlam = 1/M,

so ``sqrt(L)`` is the tracking RMS over the steps.  Every step carries a loss
term: the state-based gradient takes one adjoint solve per step (the costate
recursion), the frozen-tangent proxy accumulates the future losses in state
space.  What differs from the shape task S (scripts/train_shape.py), by the
paper's rule for trajectory objectives (Sec. III-C, IV-E): a step flagged as
a snap-through makes the rollout infeasible -- the rod must stay on the branch
of the initial equilibrium.  L-BFGS treats such a trial as ``f = +inf`` (its
line search shortens the step), Adam goes back to the last feasible iterate;
the common initialization must be snap-free; and "admissible" requires no
flagged step.  Everything else (optimizers and their settings, budgets,
stopping, contact model, Newton guards, admissibility otherwise) is that of S.

Targets (``task.path_targets``): built like the shape targets of S.  The
reference tracker (src/path_tracking.py: damped least squares with the exact
per-step mid-node sensitivity) follows a half sine on the free rod; that
designed motion, rolled out with the peg(s), gives the reference path
(``build: contact``), reachable by construction.  ``targets`` checks the
design criteria: the start rod clear of the pegs, no peg touched in the first
quarter, the required pegs touched at the end, no snap, penetration, no cap or
self-contact, the mid node itself kept off the pegs.

Common initialization: the designed motion perturbed by
``lambda e + sin(pi lambda) b`` per seed (``task.path_init``), scaled so that
the initial tracking RMS lies in ``rms_band``, admissible and snap-free,
passing the pegs on the side of the designed motion and touching the required
pegs; the network (or the rates) is fitted to it.

Methods (paper Table methods, column P): ours (state, L-BFGS, net); state,
Adam; frozen tangent with L-BFGS and with Adam; direct rates (state, L-BFGS);
end pose only, tuned (L-BFGS) over the last 20 / 60 steps; greedy (the
reference tracker with the step cap of our solutions, deterministic).
No "no obstacles" or "aimed" baseline for P.

Usage
-----
    bash scripts/run_path_batch.sh P1 runs/path_p1 6                 # the whole batch
    bash scripts/run_path_batch.sh P1 runs/smoke 2 "0" "--epochs 2"  # smoke test

    C=configs/task_path_peg_3d.yaml
    python scripts/train_path.py targets --config $C --out DIR [--targets P1]
    python scripts/train_path.py check   --config $C --out DIR --target P1
    python scripts/train_path.py train   --config $C --out DIR --target P1 --optimizer lbfgs --seed 0
    python scripts/train_path.py jobs    --config $C --out DIR --targets P1 --seeds 0 1 2 3 4 --baselines
    python scripts/train_path.py greedy  --config $C --out DIR --target P1        # after the main runs
    python scripts/train_path.py verify  --config $C --out DIR
    python scripts/train_path.py report  --config $C --out DIR
    python scripts/train_path.py figure  --config $C --npz RUN.npz [--snapshots 3]

Outputs of ``train`` / ``greedy`` (OUT/<name>.json/.npz, and a copy in the run
folder): the summary with the log of every evaluation (json); the reference
path, the commands and rod states of the initial and final rollouts, the mid
node's trajectory and error per step, contact per peg and edge per step, snap
flags, holonomy and twist per step, parameters at the start, best and end
(npz).
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'scripts'))

from blas_threads import limit_blas_threads
limit_blas_threads()

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env_factory import build_env, reset_env, env_dim, env_twist, end_control_dim, initial_control, \
    material_summary
import traj_opt as T
import path_tracking as PT
from runlog import RunLogger
from utils import maybe_save_trajectory_gif, draw_obstacles, paper_scene
import tro_style as ts
from train_shape import (experiment_config as _shape_config, opt_settings, workspace_bounds,
                         rates_of, peg_sides, designed_commands, _json_default, _axes3d, INK, INK2, COLORS,
                         GRAD_METHODS)

DEFAULT_CONFIG = _ROOT / "configs" / "task_path_peg_3d.yaml"


# ---------------------------------------------------------------- config

def experiment_config(path):
    """The config with the defaults of S (train_shape.experiment_config), and the
    trajectory objective's snap rule on unless the config says otherwise."""
    cfg = _shape_config(path)
    if cfg["train"].get("reject_snap") is None:
        cfg["train"]["reject_snap"] = True
    return cfg


def make_env(cfg, pegs=True):
    """Environment at the initial equilibrium, ``(env, checkpoint)``.

    The equilibrium at u0 is found by continuation from the undeformed
    (straight) rod with the pegs removed -- the pegs are placed after the rod
    is set up -- and then solved once more at u0 with them, so that the
    checkpoint's Hessian includes the contact.  (A peg on the concave side of
    the start arc lies in the region the straight rod sweeps on its way to the
    start pose; with it in place the continuation would push the rod out of
    plane.)  The targets require the start rod to be clear of every peg, so
    the equilibrium itself is that of the scene with the pegs.
    """
    from env_factory import settle_initial
    if not pegs:
        import copy
        cfg = copy.deepcopy(cfg)
        cfg["peg"]["enabled"] = False
    env = build_env(cfg)
    reset_env(env, cfg)
    obs = list(env.obstacles)
    env.obstacles[:] = [o for o in obs if o not in T.pegs_of(env)]
    settle_initial(env, cfg)
    if len(env.obstacles) != len(obs):
        env.obstacles[:] = obs
        info = env.settle(initial_control(cfg))
        if not info["converged"] or info.get("stable") is False:
            raise RuntimeError("the start equilibrium is lost when the pegs are added")
    return env, env.get_state()


def path_specs(cfg):
    named = (cfg.get("task") or {}).get("path_targets") or {}
    if not named:
        raise SystemExit("this config has no task.path_targets (see configs/task_path_peg_3d.yaml)")
    return {n: dict(v) for n, v in named.items()}


def check_names(names, specs):
    bad = [n for n in (names or []) if n not in specs]
    if bad:
        raise SystemExit(f"unknown target(s) {bad}; this config has {sorted(specs)}")
    return list(names or specs)


def tracker_opts(cfg):
    o = dict((cfg.get("task") or {}).get("path_tracker") or {})
    return {"damping": float(o.get("damping", 0.03)), "angle_scale": float(o.get("angle_scale", 0.2)),
            "max_step": float(o.get("max_step", 0.03)), "alphas": tuple(o.get("alphas", (1.0, 0.5, 0.25)))}


def mid_traj(env, q_list, node):
    return np.array([env.positions(q)[node] for q in q_list])


def contact_stats(rec, req=None):
    """Per-step contact of the required pegs: fraction of steps, transitions, last step."""
    tch = rec["touching"].any(axis=2)                     # (M, npeg)
    if req is not None and len(req):
        tch = tch[:, list(req)]
    any_t = tch.any(axis=1)
    return {"contact_fraction": float(any_t.mean()) if len(any_t) else 0.0,
            "contact_fraction_per_peg": tch.mean(axis=0).tolist() if len(any_t) else [],
            "contact_transitions": int(np.sum(any_t[1:] != any_t[:-1])) if len(any_t) > 1 else 0,
            "touching_at_end": bool(any_t[-1]) if len(any_t) else False}


# ---------------------------------------------------------------- targets

def cmd_targets(args):
    """Build the reference paths and check the design criteria (cf. train_shape targets)."""
    cfg = experiment_config(args.config)
    env, cp = make_env(cfg)
    env_free, cp_free = make_env(cfg, pegs=False)
    u0 = initial_control(cfg)
    M, node = env.Nsteps, env.mid_node
    k = end_control_dim(env_dim(cfg), env_twist(cfg))
    ws = workspace_bounds(cfg, k)
    to = tracker_opts(cfg)
    out = Path(args.out) / "targets"
    out.mkdir(parents=True, exist_ok=True)
    specs = path_specs(cfg)
    names = check_names(args.targets, specs)
    pegs = T.pegs_of(env)
    q0 = cp["q"]
    m0 = env.positions(q0)[node]
    start_clear = [p.report(q0)["min_clearance"] for p in pegs]
    print(f"Rod: {material_summary(cfg)}\nHorizon: {M} steps, nv = {env.nv}, mid node {node} at "
          f"{np.round(m0, 4).tolist()}, {len(pegs)} peg(s), contact model {cfg.peg.get('model', 'smooth')}")
    print(f"Start rod: clearance to the pegs {[round(c * 1e3, 1) for c in start_clear]} mm (rod surface), "
          f"holonomy {float(env.energy.holonomy(q0)):+.3f} rad")
    for name in names:
        spec = specs[name]
        req = [int(i) for i in spec.get("touch", range(len(pegs)))]
        build = spec.get("build", "contact")
        t0 = time.time()
        if "via" in spec:
            # the designed motion is given in pose space (via points, as the shape targets of S);
            # the reference is its mid-node trajectory with the pegs
            U = designed_commands(u0, spec, M)
            build = "contact"
            free_d = T.rollout(env_free, cp_free, U, keep_J="last")
            sine = (mid_traj(env_free, free_d["q_list"], node) if free_d["ok"]
                    else np.full((M, 3), np.nan))
            tr = {"rms": 0.0, "max": 0.0, "held": 0, "rejected": {}, "time": 0.0}
        else:
            sine = PT.reference_path(spec, m0, M)
            # the designed motion: the tracker on the free rod (build: contact) or with the pegs (build: path)
            e_tr, c_tr = (env_free, cp_free) if build == "contact" else (env, cp)
            tr = PT.track_reference(e_tr, c_tr, u0, sine, node, cap=to["max_step"], damping=to["damping"],
                                    angle_scale=to["angle_scale"], alphas=to["alphas"], ws_bounds=ws,
                                    reject_snap=True, pegs=T.pegs_of(e_tr))
            if not tr["ok"]:
                raise RuntimeError(f"target {name}: the reference tracker stopped at step {tr['steps']}")
            U = tr["U"]
        res = T.rollout(env, cp, U, keep_J="last", record=True, reject_snap=True)
        if not res["ok"]:
            raise RuntimeError(f"target {name}: the designed motion fails with the pegs ({res['reason']})")
        mids = mid_traj(env, res["q_list"], node)
        ref = mids if build == "contact" else sine
        err = np.linalg.norm(mids - ref, axis=1)
        free = T.rollout(env_free, cp_free, U, keep_J="last")
        mids_free = mid_traj(env_free, free["q_list"], node) if free["ok"] else None
        rec = res["rec"]
        cs = contact_stats(rec, req)
        onset = [None if o is None else int(o) for o in res["onset"]]
        mid_clear = min(_point_clearance(p, mids) for p in pegs) if pegs else np.inf
        dev_contact = (float(np.sqrt(np.mean(np.sum((mids_free - ref) ** 2, axis=1))))
                       if mids_free is not None else None)
        criteria = {
            "start_clearance_ge_5mm": bool(min(start_clear) >= args.start_clearance),
            "no_contact_first_25pct": bool(all(onset[i] is not None and onset[i] >= int(args.free_start * M)
                                               for i in req)),
            "required_pegs_touch_at_end": bool(all(res["final_touching"][i] for i in req)),
            "contact_fraction_ge_30pct": bool(cs["contact_fraction"] >= args.min_contact),
            "mid_node_off_pegs_ge_15mm": bool(mid_clear >= args.mid_clearance),
            "max_penetration_le_0.5mm": bool(res["max_penetration"] <= 5e-4),
            "no_cap_contact": not res["end_contact"],
            "no_self_contact": bool(res["min_self_clearance"] >= 0.0),
            "no_snap": res["snaps"] == 0,
            "in_workspace": bool(T.workspace_violation(U, len(u0) // 2, ws) <= 1e-9),
            "designed_tracks_ref": bool(np.sqrt(np.mean(err ** 2)) <= float(cfg.train.solve_rms)),
        }
        rec_json = {"name": name, "description": spec.get("description", PT.describe(spec)),
                    "spec": spec, "build": build, "require_touch": req, "N": M, "nv": int(env.nv),
                    "mid_node": int(node), "m0": m0.tolist(), "config": str(args.config),
                    "contact_model": cfg.peg.get("model", "smooth"),
                    "smoothing": float(cfg.peg.get("smoothing", 1e-4)),
                    "tracker": {**to, "rms_on_sine": tr["rms"], "max_on_sine": tr["max"], "held": tr["held"],
                                "rejected": tr["rejected"], "time": tr["time"]},
                    "designed_rms_vs_ref": float(np.sqrt(np.mean(err ** 2))),
                    "start_clearance": start_clear, "onset_step": onset, **cs,
                    "final_force": res["final_force"], "max_force": float(np.nanmax(rec["force"])),
                    "max_penetration": res["max_penetration"], "max_strain": res["max_strain"],
                    "max_abs_holonomy": res["max_abs_holonomy"], "min_self_clearance": res["min_self_clearance"],
                    "mid_clearance": mid_clear, "snap_steps": res["snap_steps"],
                    "max_substeps": res["max_substeps"], "newton_iterations": res["newton_iterations"],
                    "rollout_time": time.time() - t0, "u_final": U[-1].tolist(),
                    "path_length": float(np.sum(np.linalg.norm(np.diff(np.vstack([m0, ref]), axis=0), axis=1))),
                    "ref_vs_sine_rms": float(np.sqrt(np.mean(np.sum((ref - sine) ** 2, axis=1)))),
                    "ref_vs_sine_max": float(np.linalg.norm(ref - sine, axis=1).max()),
                    "free_rod_vs_ref_rms": dev_contact,
                    "criteria": criteria, "criteria_ok": bool(all(criteria.values()))}
        np.savez_compressed(out / f"{name}.npz", ref=ref, sine=sine, U_ref=U, m0=m0,
                            shapes=np.array([env.positions(q) for q in res["q_list"]]),
                            mids_free=mids_free if mids_free is not None else np.zeros((0, 3)),
                            min_clearance=rec["min_clearance"], force=rec["force"], touching=rec["touching"])
        json.dump(rec_json, open(out / f"{name}.json", "w"), indent=1, default=_json_default)
        print(f"[{name}] {rec_json['description']}\n   build {build}: tracker on the sine rms "
              f"{tr['rms'] * 1e3:.3f} mm ({tr['held']} held steps); designed motion vs reference "
              f"{rec_json['designed_rms_vs_ref'] * 1e3:.3f} mm; reference vs {'free-rod path' if 'via' in spec else 'sine'} rms "
              f"{rec_json['ref_vs_sine_rms'] * 1e3:.1f} mm (max {rec_json['ref_vs_sine_max'] * 1e3:.1f})\n"
              f"   path length {rec_json['path_length'] * 1e3:.0f} mm; contact from step {onset} "
              f"({cs['contact_fraction'] * 100:.0f} % of the steps, {cs['contact_transitions']} transitions, "
              f"at the end {res['final_touching']}), final force {np.round(res['final_force'], 2).tolist()} N, "
              f"max penetration {res['max_penetration'] * 1e3:.3f} mm, mid node >= {mid_clear * 1e3:.0f} mm "
              f"from the pegs, snaps {res['snap_steps'] or 'none'}, self clearance "
              f"{res['min_self_clearance'] * 1e3:.1f} mm\n   the same motion without the pegs: mid node "
              f"{('%.1f mm' % (dev_contact * 1e3)) if dev_contact is not None else 'failed'} (rms) from the "
              f"reference\n   criteria: "
              f"{'all OK' if rec_json['criteria_ok'] else {k: v for k, v in criteria.items() if not v}}",
              flush=True)
    plot_targets(out, names, env)


def _point_clearance(peg, pts):
    """Distance of points (n, 3) from the surface of a y-axis peg (capsule ends)."""
    pts = np.asarray(pts, dtype=float)
    yc = np.clip(pts[:, 1], peg.cy - peg.half_length, peg.cy + peg.half_length)
    d = np.stack([pts[:, 0] - peg.cx, pts[:, 1] - yc, pts[:, 2] - peg.cz], axis=1)
    return float((np.linalg.norm(d, axis=1) - peg.radius).min())


def plot_targets(out, names, env=None):
    fig = plt.figure(figsize=(5.2 * len(names), 4.8))
    for i, name in enumerate(names):
        z = np.load(out / f"{name}.npz")
        ax = fig.add_subplot(1, len(names), i + 1, projection="3d")
        S = z["shapes"]
        for k in range(0, len(S), max(1, len(S) // 8)):
            ax.plot(S[k, :, 0], S[k, :, 1], S[k, :, 2], color=INK2, lw=0.6, alpha=0.35)
        ax.plot(S[-1, :, 0], S[-1, :, 1], S[-1, :, 2], color=INK, lw=1.6)
        for P, c, lab in ((z["sine"], "#9a9a9a", "sine"), (z["ref"], "#d55e00", "reference")):
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color=c, lw=1.4, label=lab)
        if env is not None:
            draw_obstacles(ax, env)
        _axes3d(ax, name, pts=S.reshape(-1, 3))
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "targets.png", dpi=140)
    plt.close(fig)


def load_target(out, name, cfg, env):
    f = Path(out) / "targets" / f"{name}.json"
    if not f.exists():
        raise SystemExit(f"no target {name} in {f.parent}; build it with the `targets` command")
    rec = json.load(open(f))
    z = np.load(f.with_suffix(".npz"))
    if rec["nv"] != env.nv or rec["N"] != env.Nsteps:
        raise SystemExit(f"target {name} was built for nv={rec['nv']}, N={rec['N']}; "
                         f"this config has nv={env.nv}, N={env.Nsteps}")
    if rec.get("contact_model") != cfg.peg.get("model", "smooth"):
        raise SystemExit(f"target {name} was built with another contact model: rebuild the targets")
    return z["ref"], z["U_ref"], rec


# ---------------------------------------------------------------- initialization

def initial_commands(cfg, U_ref, seed):
    """Designed motion + ``lambda e + sin(pi lambda) b`` (``task.path_init``), as in S."""
    si = (cfg.get("task") or {}).get("path_init") or {}
    sc_e = np.array([si.get("end_pos", 0.02)] * 3 + [si.get("end_ang", 0.05)] * 2 + [si.get("end_yaw", 0.05)])
    sc_b = np.array([si.get("bump_pos", 0.02)] * 3 + [si.get("bump_ang", 0.05)] * 2 + [si.get("bump_yaw", 0.05)])
    rng = np.random.default_rng(int(si.get("seed_base", 2000)) + int(seed))
    e = rng.standard_normal(6) * sc_e
    b = rng.standard_normal(6) * sc_b
    M = len(U_ref)
    lam = (np.arange(1, M + 1) / M)[:, None]
    k_end = U_ref.shape[1] // 2
    U = U_ref.copy()
    U[:, k_end:] += lam * e[None, :] + np.sin(np.pi * lam) * b[None, :]
    return U, {"end_offset": e.tolist(), "bump": b.tolist()}


def calibrated_init(cfg, out_dir, target, seed, env, cp, U_ref, ref, trec, u0):
    """The common initialization of (target, seed), calibrated and cached (cf. train_shape).

    Scaled (``c <- c * aim / rms``) until the initial tracking RMS lies in
    ``task.path_init.rms_band``.  Accepted only if the rollout is feasible
    for P -- every step converged and stable and NOT flagged as a snap -- is
    in the workspace (``train.workspace: reject``), has no self-contact and no
    penetration above ``train.max_penetration``, passes every peg on the side
    of the designed motion (at its first contact step and at the end), and,
    with ``require_contact``, touches every required peg during the motion and
    at its end.  Otherwise the amplitude shrinks or a new direction is drawn.
    Cached in OUT/inits/<target>_s<seed>.npz/.json.
    """
    d = Path(out_dir) / "inits"
    f = d / f"{target}_s{seed}.json"
    if f.exists():
        return np.load(f.with_suffix(".npz"))["U_init"], json.load(open(f))
    si = (cfg.get("task") or {}).get("path_init") or {}
    lo, hi = (float(v) for v in si.get("rms_band", [0.01, 0.02]))
    aim = 0.5 * (lo + hi)
    need_contact = bool(si.get("require_contact", True))
    req = trec.get("require_touch") or list(range(len(T.pegs_of(env))))
    k_end = len(u0) // 2
    ws = workspace_bounds(cfg, len(u0) - k_end) if cfg.train.workspace == "reject" else None
    obj = T.MidTracking(ref, env.mid_node)
    zt = np.load(Path(out_dir) / "targets" / f"{target}.npz")
    shapes_t = zt["shapes"]
    on = [o for o in trec["onset_step"] if o is not None]
    k_side = [min(on) if on else len(shapes_t) - 1, len(shapes_t) - 1]
    sides_t = [peg_sides(env, shapes_t[kk]) for kk in k_side]
    tried = []
    for attempt in range(12):
        U_unit, pert = initial_commands(cfg, U_ref, seed + 1000 * attempt)
        P = U_unit - U_ref
        c = 1.0
        for it in range(10):
            U = U_ref + c * P
            res = T.rollout(env, cp, U, keep_J="last", record=True, reject_snap=True)
            wsv = T.workspace_violation(U, k_end, ws)
            row = {"attempt": attempt, "c": c, "ok": res["ok"], "reason": res["reason"], "ws_violation": wsv}
            if (not res["ok"] or wsv > 1e-9 or res["min_self_clearance"] < 0
                    or res["max_penetration"] > float(cfg.train.max_penetration)):
                tried.append(row)
                c *= 0.7
                continue
            L, _, extra = obj(env, res, 1.0 / len(U))
            rms = float(np.sqrt(L))
            same = all(float(a @ b) > 0 for kk, st in zip(k_side, sides_t)
                       for a, b in zip(st, peg_sides(env, env.positions(res["q_list"][kk]))))
            touch = all(res["onset"][i] is not None and res["final_touching"][i] for i in req)
            row.update(rms=rms, same_side=same, touching=touch)
            tried.append(row)
            if lo <= rms <= hi and same and (touch or not need_contact):
                cs = contact_stats(res["rec"], req)
                rec = {"target": target, "seed": seed, "attempt": attempt, "scale": c, "rms": rms,
                       "rms_band": [lo, hi], **pert, "onset": res["onset"], **cs,
                       "max_penetration": res["max_penetration"], "tried": tried}
                d.mkdir(parents=True, exist_ok=True)
                # written atomically (tmp + rename): parallel jobs of one seed may race here
                tmp = d / f".{target}_s{seed}.{os.getpid()}"
                np.savez_compressed(f"{tmp}.npz", U_init=U)
                os.replace(f"{tmp}.npz", f.with_suffix(".npz"))
                json.dump(rec, open(f"{tmp}.json", "w"), indent=1, default=_json_default)
                os.replace(f"{tmp}.json", f)
                return U, rec
            if not same or (lo <= rms <= hi and need_contact and not touch):
                break
            c *= aim / max(rms, 1e-6)
    raise SystemExit(f"no admissible snap-free initialization in the band {lo * 1e3:.0f}-{hi * 1e3:.0f} mm "
                     f"for {target}, seed {seed}: {tried[-3:]}")


def cmd_inits(args):
    """Build (and cache) the common initialization of one (target, seed) before the jobs."""
    cfg = experiment_config(args.config)
    env, cp = make_env(cfg)
    ref, U_ref, trec = load_target(args.out, args.target, cfg, env)
    for seed in args.seeds:
        t0 = time.time()
        U, rec = calibrated_init(cfg, args.out, args.target, seed, env, cp, U_ref, ref, trec, initial_control(cfg))
        print(f"[init {args.target} s{seed}] rms {rec['rms'] * 1e3:.2f} mm (band "
              f"{rec['rms_band'][0] * 1e3:.0f}-{rec['rms_band'][1] * 1e3:.0f}), attempt {rec['attempt']}, "
              f"scale {rec['scale']:.3f}, contact {rec.get('contact_fraction', 0) * 100:.0f} % of the steps, "
              f"onset {rec['onset']}, {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- common setup

def setup(args, cfg):
    env, cp = make_env(cfg)
    ref, U_ref, trec = load_target(args.out, args.target, cfg, env)
    u0 = initial_control(cfg)
    k_end = len(u0) // 2
    k = end_control_dim(env_dim(cfg), env_twist(cfg))
    M = env.Nsteps
    seed = getattr(args, "seed", 0)
    calib = calibrated_init(cfg, args.out, args.target, seed, env, cp, U_ref, ref, trec, u0)
    if cfg.train.param == "endpose":
        window = int(getattr(args, "window", 20) or 20)
        param = T.EndPose(M, k, calib[0], u0, window)
        xi0 = np.zeros(k)
        init_info = {"init": "designed + end pose offset 0", "window": window,
                     "calibration": {kk: calib[1][kk] for kk in ("attempt", "scale", "rms", "rms_band")}}
        label = f"endpose{window}"
    else:
        param = T.make_parameterization(cfg.train.param, M, k, hidden=int(cfg.train.get("hidden", 64)),
                                        seed=seed)
        if cfg.train.init != "designed":
            raise SystemExit("task P uses the common initialization (train.init: designed)")
        U_init, rec = calib
        xi0, fit = param.from_rates(rates_of(U_init, u0, k_end, M))
        init_info = {"init": "designed",
                     "calibration": {kk: rec[kk] for kk in ("attempt", "scale", "rms", "rms_band")}, **fit}
        label = cfg.train.param
    return dict(env=env, checkpoint=cp, ref=ref, U_ref=U_ref, target_rec=trec, u0=u0, k_end=k_end, k=k,
                M=M, param=param, xi0=xi0, init_info=init_info, node=env.mid_node,
                objective=T.MidTracking(ref, env.mid_node), ws=workspace_bounds(cfg, k), label=label,
                req=trec.get("require_touch"))


def evaluation(S, U, reject_snap=False):
    """Roll the commands ``U`` out (recording everything) and measure the tracking.

    ``reject_snap=False`` here so that a final rollout with a snap is still
    measured; admissibility then reports it."""
    env = S["env"]
    res = T.rollout(env, S["checkpoint"], U, keep_J="last", record=True, reject_snap=reject_snap)
    res["ws_violation"] = T.workspace_violation(U, S["k_end"], S["ws"])
    out = {"U": U, "res": res, "ok": res["ok"]}
    if res["ok"]:
        L, _, extra = S["objective"](env, res, 1.0 / S["M"])
        out.update(loss=L, rms=float(np.sqrt(L)), mids=extra["mids"], err=extra["err"],
                   max_err=float(extra["err"].max()), end_err=float(extra["err"][-1]),
                   contact=contact_stats(res["rec"], S["req"]))
    return out


def admissible(ev, cfg):
    """Paper Sec. VII-A: as for S, plus no step flagged as a snap-through (P)."""
    r = ev["res"]
    checks = {"all_steps_stable": bool(r["ok"]),
              "in_workspace": cfg.train.workspace != "reject" or r.get("ws_violation", 0.0) <= 1e-9,
              "no_self_contact": r.get("min_self_clearance", np.inf) >= 0.0,
              "penetration_le_max": r["max_penetration"] <= float(cfg.train.max_penetration),
              "no_cap_contact": not r["end_contact"],
              "no_snap": r["snaps"] == 0}
    return all(checks.values()), checks


def commands_of(S, xi):
    return T.commands(S["u0"], S["param"].rates(xi), S["k_end"], 1.0 / S["M"])


# ---------------------------------------------------------------- saving (train and greedy)

def save_run(args, cfg, S, name, run_dir, summary, arrays, fin, init_eval=None):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for tag, ev in (("init", init_eval), ("final", fin)):
        if ev is None:
            continue
        for key, val in (ev["res"].get("rec") or {}).items():
            arrays[f"{tag}_{key}"] = val
        if ev["ok"]:
            arrays[f"{tag}_mids"] = ev["mids"]
            arrays[f"{tag}_err"] = ev["err"]
    for d in dict.fromkeys([Path(run_dir), out]):
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"{name}.npz", **arrays)
        json.dump(summary, open(d / f"{name}.json", "w"), indent=1, default=_json_default)
    env = S["env"]
    if fin["ok"]:
        plot_path_result(Path(run_dir) / f"{name}_path", env, S["ref"], fin["mids"], fin["res"]["q_list"],
                         S["checkpoint"]["q"], f"{name}: rms {fin['rms'] * 1e3:.2f} mm")
        plot_path_error(S["ref"], fin["err"], Path(run_dir) / f"{name}_error",
                        init_err=init_eval["err"] if (init_eval is not None and init_eval["ok"]) else None,
                        touching=fin["res"]["rec"]["touching"].any(axis=2).any(axis=1))
        if bool((cfg.get("output") or {}).get("save_gif", True)) and not getattr(args, "no_gif", False):
            maybe_save_trajectory_gif(cfg, fin["res"]["q_list"], env, S["ref"], save_path=str(run_dir),
                                      filename=f"{name}.gif", title=f"path tracking {name}",
                                      gripper_node=env.nv - 1)


def tracking_summary(prefix, ev):
    if not ev["ok"]:
        return {f"{prefix}_ok": False}
    r = ev["res"]
    return {f"{prefix}_ok": True, f"{prefix}_rms": ev["rms"], f"{prefix}_loss": ev["loss"],
            f"{prefix}_max_err": ev["max_err"], f"{prefix}_end_err": ev["end_err"],
            f"{prefix}_contact_fraction": ev["contact"]["contact_fraction"],
            f"{prefix}_contact_transitions": ev["contact"]["contact_transitions"],
            f"{prefix}_touching_at_end": ev["contact"]["touching_at_end"],
            f"{prefix}_onset": r["onset"], f"{prefix}_snaps": r["snaps"], f"{prefix}_snap_steps": r["snap_steps"],
            f"{prefix}_max_penetration": r["max_penetration"], f"{prefix}_max_strain": r["max_strain"],
            f"{prefix}_max_abs_holonomy": r["max_abs_holonomy"], f"{prefix}_ws_violation": r["ws_violation"],
            f"{prefix}_min_self_clearance": r["min_self_clearance"],
            f"{prefix}_final_touching": r.get("final_touching"), f"{prefix}_final_force": r.get("final_force")}


# ---------------------------------------------------------------- train

def cmd_train(args):
    cfg = experiment_config(args.config)
    tr = cfg["train"]
    for key in ("optimizer", "grad_method", "param", "epochs", "max_rollouts", "stop_rms", "workspace"):
        v = getattr(args, key, None)
        if v is not None:
            tr[key] = v
    if args.lr is not None:
        tr.setdefault("adam", {})
        tr["adam"]["lr"] = args.lr
    tr = cfg.train
    optimizer, gm = tr.optimizer, tr.grad_method
    if gm not in GRAD_METHODS:
        raise SystemExit(f"train.grad_method must be one of {GRAD_METHODS}")
    if not tr.reject_snap:
        print("WARNING: train.reject_snap is off -- the trajectory objective of P rejects snaps (paper)")
    np.random.seed(args.seed)
    S = setup(args, cfg)
    env = S["env"]
    name = f"{args.target}_{S['label']}_{gm}_{optimizer}{args.tag}_s{args.seed}"
    run = RunLogger(cfg, env=env, root=_ROOT, tag=name,
                    enabled=bool((cfg.get("output") or {}).get("run_log", True)))
    run_dir = Path(run.artifact_dir) if getattr(run, "artifact_dir", None) else Path(args.out)
    ocfg = opt_settings(cfg, optimizer)
    print(f"Task: 3D path tracking, target {args.target} -- {S['target_rec']['description']}")
    print(f"Trajectory: {S['label']} (P = {S['param'].P}), init {S['init_info']['init']}"
          + (f" (fit rms {S['init_info'].get('fit_rms_pose', 0):.2e})" if 'fit_rms_pose' in S['init_info'] else ""))
    print(f"Objective: trajectory (mid node, {S['M']} terms); snap-through: "
          f"{'rejected (infeasible trial)' if tr.reject_snap else 'admitted'}")
    print(f"Gradient: {gm}   Optimizer: {optimizer} {ocfg}")
    print(f"Stop: rms <= {float(tr.stop_rms) * 1e3:g} mm | {tr.epochs} updates | {tr.max_rollouts} rollouts "
          f"| {tr.max_failures} failed rollouts;  workspace: {tr.workspace}")
    init_eval = evaluation(S, commands_of(S, S["xi0"]))
    print(f"Initial trajectory: "
          + (f"rms {init_eval['rms'] * 1e3:.2f} mm, max {init_eval['max_err'] * 1e3:.2f} mm, contact "
             f"{init_eval['contact']['contact_fraction'] * 100:.0f} % of the steps, snaps {init_eval['res']['snaps']}"
             if init_eval["ok"] else "FAILED: " + init_eval["res"]["reason"]), flush=True)

    ev_fn = T.Evaluator(env, S["checkpoint"], S["param"], S["u0"], S["k_end"], objective=S["objective"],
                        reject_snap=bool(tr.reject_snap), grad_method=gm, jac_reg=float(tr.get("jac_reg", 0.0)),
                        ws_bounds=S["ws"], workspace=tr.workspace, stop_rms=float(tr.stop_rms),
                        max_rollouts=int(tr.max_rollouts), max_failures=int(tr.max_failures), tag=name)
    t0 = time.time()
    iters, final_ev = T.optimize(ev_fn, S["xi0"], optimizer, int(tr.epochs), ocfg, tag=name)
    t_train = time.time() - t0
    print(f"[{name}] stopped: {ev_fn.stop}", flush=True)
    xi_final = S["xi0"] if final_ev is None else final_ev["xi"]
    xi_best = S["xi0"] if ev_fn.best is None else ev_fn.best["xi"]
    fin = evaluation(S, commands_of(S, xi_final))
    adm, adm_checks = admissible(fin, cfg) if fin["ok"] else (False, {"all_steps_stable": False})
    solved = bool(fin["ok"] and adm and fin["rms"] <= float(tr.solve_rms))
    first_below = lambda thr: next((i for i, it in enumerate(iters) if it["rms"] <= thr), None)  # noqa: E731
    reasons = [r for r, ok in zip(ev_fn.log["reason"], ev_fn.log["ok"]) if not ok]
    summary = {
        "task": "path", "status": "done", "target_name": args.target, "config": str(args.config),
        "param": S["label"], "grad_method": gm, "optimizer": optimizer, "optimizer_settings": ocfg,
        "reject_snap": bool(tr.reject_snap), "seed": args.seed, "P": int(S["param"].P), "init": S["init_info"],
        "nv": int(env.nv), "N": int(S["M"]), "mid_node": int(S["node"]),
        "stop_rms": float(tr.stop_rms), "solve_rms": float(tr.solve_rms), "epochs_cap": int(tr.epochs),
        "max_rollouts": int(tr.max_rollouts), "max_failures": int(tr.max_failures),
        "workspace_mode": tr.workspace, "stop": ev_fn.stop, "train_time": t_train,
        "updates": len(iters) - 1 if iters else 0, "rollouts": ev_fn.n_rollouts,
        "evaluations": ev_fn.n_evals, "failures": ev_fn.n_failures,
        "failure_reasons": sorted(set(reasons)),
        "snap_rejections": int(sum(r.startswith("snap") for r in reasons)),
        "adjoint_solves": int(T.N_ADJOINT_SOLVES[0]),
        "grad_time_median": float(np.median(ev_fn.grad_time)) if ev_fn.grad_time else None,
        "init_rms": init_eval.get("rms"), **tracking_summary("init", init_eval),
        "final_rms": fin.get("rms"), **tracking_summary("final", fin),
        "best_rms": None if ev_fn.best is None else ev_fn.best["rms"],
        "solved": solved, "admissible": adm, "admissibility": adm_checks,
        "updates_to_2mm": first_below(2e-3), "updates_to_1mm": first_below(1e-3),
        "iterates": iters, "evals": {k: v for k, v in ev_fn.log.items()},
        "target": {k: S["target_rec"].get(k) for k in ("description", "build", "onset_step", "criteria_ok",
                                                         "contact_fraction", "path_length")},
    }
    arrays = {"ref": S["ref"], "U_ref": S["U_ref"], "xi_init": S["xi0"], "xi_final": xi_final, "xi_best": xi_best,
              "U_init": init_eval["U"], "U_final": fin["U"], "q_init": np.asarray(S["checkpoint"]["q"]),
              "q_traj_init": np.array(init_eval["res"]["q_list"]), "q_traj": np.array(fin["res"]["q_list"]),
              "eval_loss": np.array(ev_fn.log["loss"], float), "eval_rms": np.array(ev_fn.log["rms"], float),
              "eval_ok": np.array(ev_fn.log["ok"]), "eval_rollouts": np.array(ev_fn.log["rollouts"]),
              "eval_time": np.array(ev_fn.log["time"]), "eval_snaps": np.array(ev_fn.log["snaps"], float),
              "iter_rms": np.array([it["rms"] for it in iters], float),
              "iter_eval": np.array([it["eval"] for it in iters], int)}
    save_run(args, cfg, S, name, run_dir, summary, arrays, fin, init_eval)
    if len(iters) > 1:
        ts.loss_curve([it["rms"] * 1e3 for it in iters], run_dir / f"{name}_rms", xlabel="Update",
                      ylabel="Tracking RMS (mm)", title=f"{args.target}, {gm}, {optimizer}")
    run.finish(task="path", target_name=args.target, optimizer=optimizer, grad_method=gm, param=tr.param,
               final_error=fin.get("rms"), solved=solved, stop=ev_fn.stop, rollouts=ev_fn.n_rollouts,
               failures=ev_fn.n_failures)
    print(f"[{name}] final rms {('%.3f mm' % (fin['rms'] * 1e3)) if fin['ok'] else 'n/a'} "
          f"(best {('%.3f' % (ev_fn.best['rms'] * 1e3)) if ev_fn.best else 'n/a'} mm), solved {solved}, "
          f"{len(iters) - 1 if iters else 0} updates, {ev_fn.n_rollouts} rollouts, {ev_fn.n_failures} failed "
          f"({summary['snap_rejections']} by the snap rule), {t_train:.0f}s -> {run_dir}", flush=True)


# ---------------------------------------------------------------- greedy

def cmd_greedy(args):
    """Greedy baseline: the reference tracker (one-step horizon, exact per-step
    sensitivity, next reference point), its step cap the largest per-step pose
    change of our solutions (OUT/<target>_net_state_lbfgs_s*.npz; metres, angles
    times angle_scale) or ``--cap``.  Snap-flagged trials are rejected (task
    rule); a step with no admissible trial better than holding holds the pose.
    From the initial equilibrium; deterministic, one run per target.  The
    commands are replayed for the same data as a training run."""
    cfg = experiment_config(args.config)
    args.seed = 0
    env, cp = make_env(cfg)
    ref, U_ref, trec = load_target(args.out, args.target, cfg, env)
    u0 = initial_control(cfg)
    k_end = len(u0) // 2
    k = len(u0) - k_end
    to = tracker_opts(cfg)
    if args.cap is not None:
        cap, cap_src = float(args.cap), "--cap"
    else:
        files = sorted(Path(args.out).glob(f"{args.target}_net_state_lbfgs_s*.npz"))
        if files:
            caps = []
            for f in files:
                D = np.diff(np.vstack([u0[None, :], np.load(f)["U_final"]]), axis=0)[:, k_end:]
                D[:, 3:] *= to["angle_scale"]
                caps.append(float(np.abs(D).max()))
            cap, cap_src = max(caps), [f.name for f in files]
        else:
            cap, cap_src = to["max_step"], "path_tracker.max_step (no state/L-BFGS runs found yet)"
            print(f"NOTE: no {args.target}_net_state_lbfgs_s*.npz in {args.out}: cap = max_step {cap:g}; "
                  f"run greedy after the main runs for the paper's bound")
    print(f"Greedy on {args.target}: step cap {cap:.4g} (from {cap_src}), alphas {to['alphas']}, "
          f"damping {to['damping']}")
    ws = workspace_bounds(cfg, k)
    t0 = time.time()
    trk = PT.track_reference(env, cp, u0, ref, env.mid_node, cap=cap, damping=to["damping"],
                             angle_scale=to["angle_scale"], alphas=to["alphas"], ws_bounds=ws,
                             reject_snap=bool(cfg.train.reject_snap), pegs=T.pegs_of(env),
                             log=lambda m: print(m, flush=True), log_every=args.log_every)
    t_run = time.time() - t0
    U = trk["U"]
    name = f"{args.target}_greedy_state_greedy{args.tag}_s0"
    if len(U) == 0:
        raise SystemExit(f"[{name}] no step taken")
    S = dict(env=env, checkpoint=cp, ref=ref, U_ref=U_ref, k_end=k_end, M=env.Nsteps, ws=ws,
             objective=T.MidTracking(ref, env.mid_node), req=trec.get("require_touch"), node=env.mid_node)
    if len(U) < env.Nsteps:                 # the tracker stopped: hold the last pose (the replay fails there)
        print(f"[{name}] the tracker stopped at step {len(U)}")
        U = np.vstack([U, np.repeat(U[-1:], env.Nsteps - len(U), axis=0)])
    fin = evaluation(S, U)
    replay_dev = (float(np.abs(np.array(fin["res"]["q_list"][-1]) - trk["q_list"][-1]).max())
                  if fin["ok"] else None)
    adm, adm_checks = admissible(fin, cfg) if fin["ok"] else (False, {"all_steps_stable": False})
    solved = bool(fin["ok"] and adm and fin["rms"] <= float(cfg.train.solve_rms))
    summary = {
        "task": "path", "status": "done", "target_name": args.target, "config": str(args.config),
        "param": "greedy", "grad_method": "state", "optimizer": "greedy",
        "optimizer_settings": {**to, "cap": cap}, "reject_snap": bool(cfg.train.reject_snap), "seed": 0, "P": 0,
        "init": {"init": "u0 (no trajectory)"}, "nv": int(env.nv), "N": int(env.Nsteps),
        "mid_node": int(env.mid_node), "step_cap_source": cap_src, "stop": "end of the path" if trk["ok"]
        else f"stopped at step {trk['steps']}", "train_time": t_run, "updates": int(len(U)),
        "rollouts": float((len(to["alphas"]) + 1) * len(U) / env.Nsteps), "evaluations": int(len(U)),
        "failures": int(sum(v for kk, v in trk["rejected"].items() if kk != "workspace_clip")),
        "failure_reasons": [kk for kk, v in trk["rejected"].items() if v], "rejected": trk["rejected"],
        "held_steps": trk["held"], "snap_rejections": trk["rejected"]["snap"],
        "init_rms": None, "final_rms": fin.get("rms"), **tracking_summary("final", fin),
        "best_rms": fin.get("rms"), "replay_max_deviation": replay_dev,
        "solved": solved, "admissible": adm, "admissibility": adm_checks,
        "iterates": [{"eval": 0, "rms": fin.get("rms", np.nan)}], "evals": {},
        "target": {kk: trec.get(kk) for kk in ("description", "build", "onset_step", "criteria_ok")},
    }
    arrays = {"ref": ref, "U_ref": U_ref, "U_final": U, "q_init": np.asarray(cp["q"]),
              "q_traj": np.array(fin["res"]["q_list"]), "greedy_err": trk["err"], "greedy_alpha": trk["alpha"],
              "iter_rms": np.array([fin.get("rms", np.nan)]), "eval_rms": np.array([fin.get("rms", np.nan)]),
              "eval_ok": np.array([fin["ok"]]), "eval_rollouts": np.array([summary["rollouts"]]),
              "eval_time": np.array([t_run])}
    run_dir = Path(args.out)
    save_run(args, cfg, S, name, run_dir, summary, arrays, fin)
    print(f"[{name}] {len(U)} steps, {trk['held']} held, rejected {trk['rejected']}; final rms "
          f"{('%.3f mm' % (fin['rms'] * 1e3)) if fin['ok'] else 'n/a'}"
          + (f", max {fin['max_err'] * 1e3:.2f} mm" if fin["ok"] else "")
          + f", solved {solved}, admissible {adm}, replay deviation {replay_dev}, {t_run:.0f}s -> {run_dir}",
          flush=True)


# ---------------------------------------------------------------- check (H1)

def cmd_check(args):
    """Gradient check at the common initialization (paper Sec. VII-C, H1), trajectory objective.

    Directional derivatives of L along ``--dirs`` random unit directions of
    xi: state-based and frozen-tangent gradients against central differences
    of full rollouts (Newton tolerance tightened by ``--tol-factor``).  Also
    the cost of one gradient (time, adjoint solves: M for both with a
    trajectory objective) and the per-rate difference of the proxy from the
    state gradient before / after the first peg contact.
    """
    cfg = experiment_config(args.config)
    if args.param is not None:
        cfg["train"]["param"] = args.param
    S = setup(args, cfg)
    env = S["env"]
    env.tol *= float(args.tol_factor)
    rng = np.random.default_rng(args.seed + 77)
    rep = {"target": args.target, "param": cfg.train.param, "P": int(S["param"].P), "tol": env.tol,
           "h": args.h, "dirs": []}

    def mk(gm, param=None):
        return T.Evaluator(env, S["checkpoint"], param or S["param"], S["u0"], S["k_end"],
                           objective=S["objective"], reject_snap=True, grad_method=gm, ws_bounds=None,
                           workspace="off", stop_rms=-1, max_rollouts=10 ** 9, max_failures=10 ** 9,
                           verbose=False)
    evs = {}
    for gm in GRAD_METHODS:
        e = mk(gm)
        n0 = T.N_ADJOINT_SOLVES[0]
        evs[gm] = e(S["xi0"])
        if not evs[gm]["ok"]:
            raise SystemExit(f"initial rollout failed: {evs[gm]['reason']}")
        rep[f"{gm}_grad_time"] = e.grad_time[-1]
        rep[f"{gm}_adjoint_solves"] = T.N_ADJOINT_SOLVES[0] - n0
    print(f"[check {args.target}] initial rms {evs['state']['rms'] * 1e3:.2f} mm, P = {S['param'].P}, "
          f"Newton tol {env.tol:.2e}")
    print(f"   gradient cost: state {rep['state_grad_time'] * 1e3:.0f} ms / {rep['state_adjoint_solves']} solves; "
          f"proxy {rep['proxy_grad_time'] * 1e3:.0f} ms / {rep['proxy_adjoint_solves']} solves")
    plain = mk("state")
    for i in range(args.dirs):
        d = rng.standard_normal(S["param"].P)
        d /= np.linalg.norm(d)
        lp, lm = plain(S["xi0"] + args.h * d), plain(S["xi0"] - args.h * d)
        if not (lp["ok"] and lm["ok"]):
            print(f"   dir {i}: a perturbed rollout failed ({lp['reason'] or lm['reason']})")
            continue
        fd = (lp["loss"] - lm["loss"]) / (2 * args.h)
        row = {"fd": fd}
        for gm in GRAD_METHODS:
            gd = float(evs[gm]["g"] @ d)
            row[gm], row[f"{gm}_relerr"] = gd, abs(gd - fd) / max(abs(fd), 1e-300)
        rep["dirs"].append(row)
        print(f"   dir {i}: fd {fd:+.6e}  state {row['state']:+.6e} (rel err {row['state_relerr']:.2e})  "
              f"proxy {row['proxy']:+.6e} (rel err {row['proxy_relerr']:.2e})", flush=True)
    if rep["dirs"]:
        se, pe = [r["state_relerr"] for r in rep["dirs"]], [r["proxy_relerr"] for r in rep["dirs"]]
        rep.update(state_relerr_median=float(np.median(se)), state_relerr_max=float(np.max(se)),
                   proxy_relerr_median=float(np.median(pe)))
        print(f"   state: median rel err {np.median(se):.2e}, max {np.max(se):.2e};  proxy median {np.median(pe):.2e}")
    G = {}
    for gm in GRAD_METHODS:
        r = mk(gm, T.DirectRates(S["M"], S["k"]))(S["param"].rates(S["xi0"]).reshape(-1))
        if r["ok"]:
            G[gm] = r["g"].reshape(S["M"], S["k"])
    if len(G) == 2:
        rel = np.linalg.norm(G["proxy"] - G["state"], axis=1) / np.maximum(np.linalg.norm(G["state"], axis=1), 1e-300)
        on = [o for o in evs["state"]["res"]["onset"] if o is not None]
        split = min(on) if on else S["M"] // 2
        rep["per_rate"] = {"split_at": int(split), "rel": rel.tolist(),
                           "median_before": float(np.median(rel[:split])) if split else None,
                           "median_after": float(np.median(rel[split:])), "last": float(rel[-1])}
        print(f"   proxy vs state per rate: first contact at step {split}; median rel diff before "
              f"{rep['per_rate']['median_before']}, after {rep['per_rate']['median_after']:.3e}, "
              f"last rate {rel[-1]:.2e}")
    out = Path(args.out) / "checks"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"check_{args.target}_{cfg.train.param}_s{args.seed}.json"
    json.dump(rep, open(f, "w"), indent=1, default=_json_default)
    print(f"   -> {f}")


# ---------------------------------------------------------------- figures

def plot_path_result(path, env, ref, mids, q_list, q_init, title, n_snaps=0, view=(22, -58), workspace="off"):
    """Initial and final rod, the reference path and the mid node's path, pegs, gripper frames."""
    snaps = ()
    if n_snaps > 0 and len(q_list):
        idx = np.linspace(0, len(q_list) - 1, n_snaps + 2)[1:-1].round().astype(int)
        snaps = [q_list[i] for i in idx]
    return paper_scene(path, env, rod_init=q_init, rod_final=q_list[-1], rod_snaps=snaps,
                       rod_labels=("Initial rod", "Final rod"),
                       curves=[(ref, "reference", "Reference path"), (mids, "achieved", "Mid node")],
                       frames_q=q_list[-1], title=title, view=view, workspace=workspace)


def plot_path_error(ref, err, path, init_err=None, touching=None, title=None):
    """Tracking error |x_mid,k - p_k| (mm) per step, single column; shaded: rod on a peg."""
    err = 1e3 * np.asarray(err)
    k = np.arange(1, len(err) + 1)
    with ts.style():
        fig, ax = plt.subplots(figsize=(ts.COL_W, 1.35))
        if touching is not None and np.any(touching):
            t = np.asarray(touching, bool)
            ax.fill_between(k, 0, 1, where=t[:len(k)], transform=ax.get_xaxis_transform(),
                            color="#e8e8e8", lw=0, step="mid", label="On peg")
        if init_err is not None:
            ax.plot(k, 1e3 * np.asarray(init_err), color="#9a9a9a", lw=0.8, label="Initial")
        ax.plot(k, err, **ts.ROLE["achieved"], label="Final")
        rms = float(np.sqrt(np.mean(err ** 2)))
        ax.text(0.99, 0.92, f"RMS {rms:.2f} mm", transform=ax.transAxes, ha="right", va="top",
                fontsize=ts.FS_TICK)
        ax.set_xlim(0, len(err))
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"Step $k$")
        ax.set_ylabel("Error (mm)")
        ax.legend(fontsize=ts.FS_TICK, frameon=False, loc="upper left")
        if title:
            ax.set_title(title)
        ts.style_2d(ax)
        paths = ts.save(fig, path, width=ts.COL_W)
        plt.close(fig)
    return paths


def cmd_figure(args):
    """Re-render a finished run's figures from its .npz -- no training."""
    npz = Path(args.npz)
    z = np.load(npz)
    meta_f = npz.with_suffix(".json")
    meta = json.load(open(meta_f)) if meta_f.exists() else {}
    cfg = experiment_config(args.config)
    env = build_env(cfg)
    reset_env(env, cfg)
    if z["q_init"].size != env.n_dof:
        raise SystemExit(f"{npz} was run on a rod with {z['q_init'].size} DOFs; pass its --config")
    stem = Path(args.stem) if args.stem else npz.with_name(npz.stem)
    if "final_mids" in z.files and len(z["q_traj"]):
        title = None if args.no_title else f"Path tracking ({meta.get('target_name', npz.stem)})"
        plot_path_result(f"{stem}_path", env, z["ref"], z["final_mids"], list(z["q_traj"]), z["q_init"], title,
                         n_snaps=args.snapshots, view=tuple(args.view), workspace=args.workspace)
        plot_path_error(z["ref"], z["final_err"], f"{stem}_error",
                        init_err=z["init_err"] if "init_err" in z.files else None,
                        touching=z["final_touching"].any(axis=2).any(axis=1) if "final_touching" in z.files else None)
    if "iter_rms" in z.files and len(z["iter_rms"]) > 1:
        ts.loss_curve(list(1e3 * z["iter_rms"]), f"{stem}_rms", xlabel="Update", ylabel="Tracking RMS (mm)")


# ---------------------------------------------------------------- jobs / report

def cmd_jobs(args):
    """The batch: {state, proxy} x {adam, lbfgs} (network) per seed; with
    ``--baselines`` also direct rates (state, L-BFGS) and end pose only, tuned
    (state, L-BFGS, last 20 / 60 steps) per seed, and greedy last (its step
    cap comes from the state/L-BFGS runs)."""
    out = Path(args.out)
    cfg = experiment_config(args.config)
    names = check_names(args.targets, path_specs(cfg))
    base = f"python scripts/train_path.py train --config {args.config} --out {out}"
    lines = []
    for t in names:
        for gm in args.grad_methods:
            for o in args.optimizers:
                for s in args.seeds:
                    lines.append((f"{base} --target {t} --param net --grad-method {gm} --optimizer {o} --seed {s}",
                                  f"{t}_net_{gm}_{o}_s{s}"))
        if args.baselines:
            for s in args.seeds:
                lines.append((f"{base} --target {t} --param rates --grad-method state --optimizer lbfgs --seed {s}",
                              f"{t}_rates_state_lbfgs_s{s}"))
                for w in (20, 60):
                    lines.append((f"{base} --target {t} --param endpose --window {w} --grad-method state "
                                  f"--optimizer lbfgs --seed {s}", f"{t}_endpose{w}_state_lbfgs_s{s}"))
    # longest first: the network L-BFGS / Adam runs, then the rest
    print(f"mkdir -p {out}/logs")
    for cmd, name in lines:
        print(f"{cmd} > {out}/logs/{name}.log 2>&1")
    if args.baselines:
        print("# after the runs above (greedy's step cap is the largest step of the state/L-BFGS solutions):")
        for t in names:
            print(f"python scripts/train_path.py greedy --config {args.config} --out {out} --target {t} "
                  f"> {out}/logs/{t}_greedy.log 2>&1")


def load_runs(out):
    runs = []
    for js in sorted(Path(out).glob("*.json")):
        try:
            s = json.load(open(js))
        except (ValueError, OSError):
            continue
        if isinstance(s, dict) and s.get("task") == "path" and s.get("status") == "done":
            s["_name"] = js.stem
            runs.append(s)
    return runs


def report(out, config=None):
    out = Path(out)
    runs = load_runs(out)
    if not runs:
        raise SystemExit(f"no path runs in {out}")
    key = lambda r: (r["param"], r["grad_method"], r["optimizer"])  # noqa: E731
    f = lambda v: "-" if v is None else f"{v * 1e3:.3f}"  # noqa: E731
    L = ["# 3D path tracking with contact (task P)", "",
         "Solved: tracking RMS of the mid node over all steps <= solve_rms and the motion admissible "
         "(every step converged and stable, NO snap-through flagged, every pose in the workspace, "
         "max penetration <= train.max_penetration, no cap or self-contact). 'rollouts' counts every "
         "rollout, failed ones and line-search trials included; 'snap rej.' the trials rejected by the "
         "snap rule.", "",
         "| target | param | gradient | optimizer | seed | init rms [mm] | final rms [mm] | final max [mm] | "
         "end err [mm] | on peg [%] | solved | updates | rollouts | failed | snap rej. | max pen. [mm] | "
         "time [s] | stop |", "|" + "---|" * 18]
    for r in sorted(runs, key=lambda r: (r["target_name"], key(r), r["seed"])):
        L.append(f"| {r['target_name']} | {r['param']} | {r['grad_method']} | {r['optimizer']} | {r['seed']} | "
                 f"{f(r.get('init_rms'))} | {f(r.get('final_rms'))} | {f(r.get('final_max_err'))} | "
                 f"{f(r.get('final_end_err'))} | "
                 f"{'-' if r.get('final_contact_fraction') is None else '%.0f' % (100 * r['final_contact_fraction'])} | "
                 f"{'yes' if r['solved'] else 'no'} | {r['updates']} | {r['rollouts']:.0f} | {r['failures']} | "
                 f"{r.get('snap_rejections', 0)} | "
                 f"{'-' if r.get('final_max_penetration') is None else '%.3f' % (1e3 * r['final_max_penetration'])} | "
                 f"{r['train_time']:.0f} | {r['stop']} |")
    L += ["", "## Per method", "", "| param | gradient | optimizer | runs | solved | median final rms [mm] | "
          "median rollouts | failed rollouts (total) | snap rejections (total) |", "|" + "---|" * 9]
    for kk in sorted({key(r) for r in runs}):
        R = [r for r in runs if key(r) == kk]
        fr = [r["final_rms"] for r in R if r.get("final_rms") is not None]
        L.append(f"| {kk[0]} | {kk[1]} | {kk[2]} | {len(R)} | {sum(r['solved'] for r in R)} | "
                 f"{np.median(fr) * 1e3 if fr else float('nan'):.3f} | "
                 f"{np.median([r['rollouts'] for r in R]):.0f} | {sum(r['failures'] for r in R)} | "
                 f"{sum(r.get('snap_rejections', 0) for r in R)} |")
    targets = sorted({r["target_name"] for r in runs})
    fig, axes = plt.subplots(len(targets), 3, figsize=(15, 3.4 * len(targets)), squeeze=False)
    for i, t in enumerate(targets):
        for r in [r for r in runs if r["target_name"] == t]:
            z = np.load(out / f"{r['_name']}.npz")
            if r["param"] == "net":
                ok = z["eval_ok"]
                y = np.maximum(1e3 * np.minimum.accumulate(np.where(ok, z["eval_rms"], np.inf)), 1e-2)
                c = COLORS.get((r["grad_method"], r["optimizer"]), INK2)
                axes[i][0].plot(z["eval_rollouts"], y, color=c, lw=1.0, alpha=0.8)
                axes[i][1].plot(z["eval_time"], y, color=c, lw=1.0, alpha=0.8)
            if r["seed"] == 0 and "final_err" in z.files:
                c = COLORS.get((r["grad_method"], r["optimizer"]), INK2) if r["param"] == "net" else None
                ls = "-" if r["param"] == "net" else {"greedy": ":", "rates": "-."}.get(r["param"], "--")
                axes[i][2].plot(np.arange(1, len(z["final_err"]) + 1) / len(z["final_err"]),
                                1e3 * z["final_err"], lw=0.9, ls=ls, color=c,
                                label=f"{r['param']}, {r['grad_method']}, {r['optimizer']}")
        for ax, xl in zip(axes[i][:2], ("rollouts", "wall time [s]")):
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-2)
            ax.axhline(1e3 * float(runs[0].get("solve_rms", 2e-3)), color=INK2, lw=0.8, ls=":")
            ax.set_xlabel(xl)
            ax.set_ylabel("best tracking RMS so far [mm]")
            ax.set_title(t, loc="left", fontsize=9)
        axes[i][2].set_xlabel(r"$\lambda$")
        axes[i][2].set_ylabel("mid-node error, seed 0 [mm]")
        axes[i][2].legend(fontsize=6)
    handles = [plt.Line2D([], [], color=c, label=f"{g}, {o}") for (g, o), c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "curves.png", dpi=140)
    plt.close(fig)
    L += ["", "![curves](curves.png)", ""]
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nreport -> {out / 'report.md'}")


# ---------------------------------------------------------------- verify

_REQ_JSON = ("task", "status", "target_name", "param", "grad_method", "optimizer", "seed", "stop", "reject_snap",
             "train_time", "updates", "rollouts", "failures", "snap_rejections", "final_rms", "final_ok",
             "solved", "admissible", "admissibility", "iterates", "target")
_REQ_JSON_OK = ("final_max_err", "final_end_err", "final_contact_fraction", "final_contact_transitions",
                "final_touching_at_end", "final_onset", "final_snaps", "final_max_penetration",
                "final_max_strain", "final_max_abs_holonomy", "final_ws_violation", "final_final_touching",
                "final_final_force")
_REQ_NPZ_TRAIN = ("ref", "U_ref", "xi_init", "xi_final", "xi_best", "U_init", "U_final", "q_init",
                  "q_traj_init", "q_traj", "eval_loss", "eval_rms", "eval_ok", "eval_rollouts", "eval_time",
                  "eval_snaps", "iter_rms", "iter_eval", "init_mids", "init_err",
                  "init_min_clearance", "init_force", "init_touching", "init_snap", "init_substeps",
                  "init_holonomy", "init_twist", "final_min_clearance", "final_force", "final_touching",
                  "final_snap", "final_substeps", "final_holonomy", "final_twist", "final_mids", "final_err")
_REQ_NPZ_GREEDY = ("ref", "U_ref", "U_final", "q_init", "q_traj", "greedy_err", "greedy_alpha",
                   "final_min_clearance", "final_force", "final_touching", "final_snap", "final_mids", "final_err")


def cmd_verify(args):
    """Check every path run in OUT: complete, consistent, physically admissible as saved."""
    out = Path(args.out)
    bad = n = 0
    try:
        cfg = experiment_config(args.config)
        env_v = build_env(cfg)
        pegs_v = T.pegs_of(env_v)
    except Exception:                                   # noqa: BLE001
        env_v, pegs_v = None, []
    for js in sorted(out.glob("*.json")):
        try:
            s = json.load(open(js))
        except (ValueError, OSError):
            continue
        if not (isinstance(s, dict) and s.get("task") == "path"):
            continue
        n += 1
        problems = []
        greedy = s.get("optimizer") == "greedy"
        miss = [k for k in _REQ_JSON + (_REQ_JSON_OK if s.get("final_ok") else ()) if k not in s]
        if miss:
            problems.append(f"json misses {miss}")
        npz = js.with_suffix(".npz")
        if not npz.exists():
            problems.append("no npz")
        else:
            z = np.load(npz)
            req = _REQ_NPZ_GREEDY if greedy else _REQ_NPZ_TRAIN
            if not s.get("final_ok"):
                req = tuple(k for k in req if k not in ("final_mids", "final_err"))
            miss = [k for k in req if k not in z.files]
            if miss:
                problems.append(f"npz misses {miss}")
            M, nv, node = int(s.get("N", 0)), int(s.get("nv", 0)), int(s.get("mid_node", -1))
            if s.get("final_ok") and "q_traj" in z.files and z["q_traj"].shape[0] != M:
                problems.append(f"q_traj has {z['q_traj'].shape[0]} steps, expected {M}")
            if "ref" in z.files and z["ref"].shape != (M, 3):
                problems.append(f"ref has shape {z['ref'].shape}")
            if "U_final" in z.files and z["U_final"].shape[1] != 12:
                problems.append(f"U_final has {z['U_final'].shape[1]} controls")
            if not greedy and "xi_final" in z.files and z["xi_final"].size != int(s.get("P", -1)):
                problems.append(f"xi has {z['xi_final'].size} entries, P = {s.get('P')}")
            if s.get("final_ok") and "final_mids" in z.files:
                X = z["q_traj"][:, :3 * nv].reshape(len(z["q_traj"]), nv, 3)[:, node]
                if np.abs(X - z["final_mids"]).max() > 1e-12:
                    problems.append("final_mids != mid node of q_traj")
                rms = float(np.sqrt(np.mean(np.sum((X - z["ref"]) ** 2, axis=1))))
                if abs(rms - s["final_rms"]) > 1e-9:
                    problems.append(f"final_rms {s['final_rms']} != recomputed {rms}")
                if s.get("solved") and (s.get("final_snaps") or np.any(z["final_snap"])):
                    problems.append("solved with a snap-through")
            if "final_touching" in z.files and z["final_touching"].ndim == 3 and \
                    z["final_touching"].shape[1] != len(pegs_v or [0]):
                problems.append(f"final_touching shape {z['final_touching'].shape}")
            if env_v is not None and "q_traj" in z.files and len(z["q_traj"]) and z["q_init"].size == env_v.n_dof:
                Q = np.vstack([z["q_init"][None, :], z["q_traj"]])
                tun = [k for k in range(1, len(Q)) for p in pegs_v if p.crossed(Q[k - 1], Q[k])]
                if tun:
                    problems.append(f"rod passes through a peg at steps {sorted(set(tun))[:5]}")
                sc = min(T.self_clearance(env_v.positions(q), env_v.R_outer) for q in Q)
                if sc < 0:
                    problems.append(f"self-contact ({sc * 1e3:.1f} mm)")
        figs = list(out.glob(f"{js.stem}_path.pdf"))
        figs += list(Path(_ROOT / "experiments").glob(f"*/runs/*{js.stem}/{js.stem}_path.pdf"))
        if s.get("final_ok") and not figs:
            problems.append("no path figure (run `figure --npz` to make one)")
        bad += bool(problems)
        print(f"{'ok  ' if not problems else 'FAIL'} {js.stem}: final rms "
              f"{'-' if s.get('final_rms') is None else '%.3f mm' % (1e3 * s['final_rms'])}, solved {s.get('solved')}, "
              f"{s.get('rollouts')} rollouts, snap rejections {s.get('snap_rejections')}, stop: {s.get('stop')}"
              + ("" if not problems else "\n     " + "; ".join(problems)))
    print(f"{n} runs, {bad} with problems")
    if bad:
        raise SystemExit(1)


# ---------------------------------------------------------------- main

def main():
    cmds = ("train", "targets", "inits", "check", "jobs", "greedy", "report", "figure", "verify")
    argv = sys.argv[1:]
    cmd = argv.pop(0) if argv and argv[0] in cmds else "train"
    ap = argparse.ArgumentParser(description=f"3D path tracking with contact ({cmd})")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", required=cmd != "figure", help="experiment directory (targets/ lives in it)")
    if cmd in ("targets", "jobs"):
        ap.add_argument("--targets", nargs="+", default=None)
    if cmd == "targets":
        ap.add_argument("--start-clearance", dest="start_clearance", type=float, default=0.005,
                        help="[m] start rod surface to every peg")
        ap.add_argument("--free-start", dest="free_start", type=float, default=0.25,
                        help="share of the path, from the start, in which no required peg is touched")
        ap.add_argument("--min-contact", dest="min_contact", type=float, default=0.30,
                        help="share of the steps on the required pegs")
        ap.add_argument("--mid-clearance", dest="mid_clearance", type=float, default=0.015,
                        help="[m] the mid node's distance from every peg surface")
    if cmd == "inits":
        ap.add_argument("--target", required=True)
        ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    if cmd == "jobs":
        ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        ap.add_argument("--optimizers", nargs="+", choices=("adam", "lbfgs"), default=["adam", "lbfgs"])
        ap.add_argument("--grad-methods", nargs="+", choices=GRAD_METHODS, default=list(GRAD_METHODS))
        ap.add_argument("--baselines", action="store_true", help="also direct rates, end pose (tuned) and greedy")
    if cmd in ("train", "check"):
        ap.add_argument("--target", required=True)
        ap.add_argument("--seed", type=int, default=0, help="initialization seed")
        ap.add_argument("--param", choices=T.PARAMETERIZATIONS, default=None)
    if cmd == "train":
        ap.add_argument("--optimizer", choices=T.OPTIMIZERS, default=None)
        ap.add_argument("--grad-method", dest="grad_method", choices=GRAD_METHODS, default=None)
        ap.add_argument("--epochs", type=int, default=None, help="update cap (train.epochs)")
        ap.add_argument("--max-rollouts", dest="max_rollouts", type=int, default=None)
        ap.add_argument("--stop-rms", dest="stop_rms", type=float, default=None)
        ap.add_argument("--lr", type=float, default=None, help="Adam rate (train.adam.lr)")
        ap.add_argument("--workspace", choices=("reject", "record"), default=None)
        ap.add_argument("--tag", default="")
        ap.add_argument("--window", type=int, default=20, help="--param endpose: ramp steps")
        ap.add_argument("--no-gif", dest="no_gif", action="store_true")
    if cmd == "greedy":
        ap.add_argument("--target", required=True)
        ap.add_argument("--cap", type=float, default=None,
                        help="step cap (m, angles x angle_scale); default: the largest step of our L-BFGS runs")
        ap.add_argument("--epochs", type=int, default=None, help="ignored (the path has M steps); for EXTRA")
        ap.add_argument("--log-every", dest="log_every", type=int, default=25)
        ap.add_argument("--tag", default="")
        ap.add_argument("--no-gif", dest="no_gif", action="store_true")
    if cmd == "check":
        ap.add_argument("--dirs", type=int, default=4)
        ap.add_argument("--h", type=float, default=1e-6, help="central-difference step in xi")
        ap.add_argument("--tol-factor", dest="tol_factor", type=float, default=1e-3)
    if cmd == "figure":
        ap.add_argument("--npz", required=True)
        ap.add_argument("--stem", default=None)
        ap.add_argument("--view", type=float, nargs=2, default=(22.0, -58.0), metavar=("ELEV", "AZIM"))
        ap.add_argument("--snapshots", type=int, default=0)
        ap.add_argument("--workspace", choices=("auto", "fit", "clip", "off"), default="off")
        ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args(argv)
    {"targets": cmd_targets, "inits": cmd_inits, "check": cmd_check, "jobs": cmd_jobs, "greedy": cmd_greedy, "figure": cmd_figure,
     "verify": cmd_verify, "report": lambda a: report(a.out, a.config), "train": cmd_train}[cmd](args)


if __name__ == "__main__":
    main()
