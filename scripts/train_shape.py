"""Task S: 3D terminal shape formation against two pegs (paper Sec. VII).

Every free node of the rod reaches a target shape; the loss is terminal,

    L = 1/n_free * sum_i |x_i(1) - x_i^*|^2,      i over the free nodes,

and its gradient is the exact state-based one (one adjoint solve at the final
step) or, for the baseline, the frozen-tangent proxy.  The trajectory is the
network of the paper (``--param net``, needs torch) or the rates themselves
(``--param rates``, the "direct rates" baseline, numpy only).  Optimizers, as
in the RSS paper: ``adam`` (fixed rate, every update taken) and ``lbfgs``
(strong-Wolfe line search on full rollouts); see src/traj_opt.py.

A run stops at whichever comes first: the shape RMS of an admissible rollout
reaches ``train.stop_rms``, ``train.epochs`` updates, ``train.max_rollouts``
rollouts (every rollout counts), or ``train.max_failures`` failed rollouts.

Targets (``task.shape_targets`` of the config, e.g. P1-P3 of
configs/task_shape_pegs_3d.yaml) are the final equilibria of straight gripper
moves ``u0 -> u0 + du``, rolled out with the full physics; ``targets`` builds
them and checks the design criteria of the paper (no peg touched in the first
30 % of the motion, both touched at the end, start rod >= 3 cm from both).

Common initialization (``train.init: designed``): the designed motion of the
target, perturbed per seed (``task.shape_init``), to which the trajectory is
fitted; every method starts from the same one per (target, seed).

Usage
-----
The whole batch of one target, in order (targets, gradient check, Adam-rate
sweep, main runs and baselines in parallel, greedy, figures, verify, report):

    bash scripts/run_shape_batch.sh P1 runs/shape_p1 <nproc>
    bash scripts/run_shape_batch.sh P1 runs/smoke 2 "0" "--epochs 2"     # smoke test

Baselines (paper Sec. VII-A, Table methods): ``--param rates`` (direct
rates), ``--train-without-pegs`` (no obstacles), ``--param endpose --window W``
with ``--optimizer lbfgs`` (end pose, tuned) or ``--aim --optimizer none``
(end pose, aimed), the ``greedy`` command; ``--grad-method proxy`` (frozen
tangent).  The single commands:

    C=configs/task_shape_pegs_3d.yaml
    python scripts/train_shape.py targets --config $C --out DIR [--targets P1]
    python scripts/train_shape.py check   --config $C --out DIR --target P1 [--param rates]
    python scripts/train_shape.py train   --config $C --out DIR --target P1 --optimizer lbfgs --seed 0
    python scripts/train_shape.py sweep   --config $C --out DIR --targets P1          # Adam-rate sweep jobs
    python scripts/train_shape.py sweep   --config $C --out DIR --targets P1 --pick   # -> DIR/adam_lr.json
    python scripts/train_shape.py jobs    --config $C --out DIR --targets P1 --seeds 0 1 2 3 4
    python scripts/train_shape.py greedy  --config $C --out DIR --target P1           # after the main runs
    python scripts/train_shape.py verify  --config $C --out DIR                       # every run complete
    python scripts/train_shape.py report  --config $C --out DIR
    python scripts/train_shape.py figure  --config $C --npz RUN.npz [--view 22 -58] [--snapshots 3]

Outputs of ``train`` (OUT/<name>.json/.npz, the latest of each job, and a copy
in the run's own folder OUT/runs/...): the summary and every evaluation's log
(json), and the arrays (npz) -- parameters at the start, best and end, the
commands and rod states of the initial and final rollouts, their contact per
peg and edge, snap flags, holonomy and twist per step, and the per-node error.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from blas_threads import limit_blas_threads
limit_blas_threads()

import argparse
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import load_config
from env_factory import (
    build_env, reset_env, settle_initial, env_dim, env_twist, control_dim, end_control_dim,
    initial_control, material_summary,
)
import traj_opt as T
from runlog import RunLogger
from utils import maybe_save_trajectory_gif, draw_obstacles
import tro_style as ts

DEFAULT_CONFIG = _ROOT / "configs" / "task_shape_pegs_3d.yaml"
GRAD_METHODS = ("state", "proxy")


# ---------------------------------------------------------------- config

def experiment_config(path):
    """The config, with the shape experiment's defaults filled in."""
    cfg = load_config(path)
    tr = cfg["train"]
    defaults = {"param": "net", "init": "designed", "max_rollouts": 500, "max_failures": 20,
                "stop_rms": 1.0e-4, "solve_rms": 5.0e-3, "max_penetration": 5.0e-4,
                "workspace": "reject"}
    for k, v in defaults.items():
        if tr.get(k) is None:
            tr[k] = v
    if tr.get("optimizer") not in T.OPTIMIZERS:
        tr["optimizer"] = "adam"
    return cfg


def opt_settings(cfg, optimizer):
    """The ``train.adam`` / ``train.lbfgs`` block (Adam's lr falls back to ``train.lr``)."""
    tr = cfg["train"]
    block = dict(tr.get(optimizer) or {})
    if optimizer == "adam":
        block.setdefault("lr", tr.get("lr", 1e-3))
    return block


def workspace_bounds(cfg, k):
    """``(lo, hi)`` over the gripper's ``k`` controls from ``workspace:`` (yaw unbounded), or None."""
    ws = cfg.get("workspace") or {}
    if not ws.get("position_min_3d"):
        return None
    lo, hi = np.full(k, -np.inf), np.full(k, np.inf)
    lo[:3], hi[:3] = ws["position_min_3d"], ws["position_max_3d"]
    ang = np.array(ws["orientation_limits_3d"], dtype=float).reshape(-1, 2)
    lo[3:3 + len(ang)], hi[3:3 + len(ang)] = ang[:, 0], ang[:, 1]
    return lo, hi


def make_env(cfg, pegs=True):
    """Environment at the settled initial equilibrium: ``(env, checkpoint)``."""
    if not pegs:
        cfg = load_config_copy(cfg)
        cfg["peg"]["enabled"] = False
    env = build_env(cfg)
    reset_env(env, cfg)
    settle_initial(env, cfg)
    return env, env.get_state()


def load_config_copy(cfg):
    import copy
    return copy.deepcopy(cfg)


# ---------------------------------------------------------------- targets

def target_paths(cfg):
    """``{name: (description, spec)}`` of ``task.shape_targets``.

    A spec is either a straight move ``{du: [6]}`` of the gripper or a path
    through via points ``{via: [[lam, dx, dy, dz, dtheta, dphi, dyaw], ...]}``
    (increments over u0 at the given lambda, linear in between, lambda = 1
    last), optionally with ``touch`` (indices of the pegs the target must
    touch; default all).
    """
    named = (cfg.get("task") or {}).get("shape_targets") or {}
    if not named:
        raise SystemExit("this config has no task.shape_targets (see configs/task_shape_pegs_3d.yaml)")
    out = {}
    for name, t in named.items():
        spec = dict(t)
        if "via" not in spec and "du" not in spec:
            raise SystemExit(f"target {name}: give du (straight move) or via (via points)")
        desc = t.get("description", f"straight move du = {list(t['du'])}" if "du" in t
                     else f"path through {len(t['via'])} via points")
        out[name] = (desc, spec)
    return out


def check_names(names, paths):
    bad = [n for n in (names or []) if n not in paths]
    if bad:
        raise SystemExit(f"unknown target(s) {bad}; this config has {sorted(paths)}")
    return list(names or paths)


def designed_commands(u0, spec, M):
    """The designed motion of a target spec, ``u_k`` at ``lambda = (k+1)/M``:
    the straight move ``u0 + lambda du``, or the piecewise-linear path through
    the via points (increments over u0)."""
    k_end = len(u0) // 2
    U = np.repeat(np.asarray(u0, float)[None, :], M, axis=0)
    lam = np.arange(1, M + 1) / M
    if isinstance(spec, dict) and "via" in spec:
        V = np.asarray(spec["via"], dtype=float)
        knots = np.concatenate([[0.0], V[:, 0]])
        inc = np.vstack([np.zeros(V.shape[1] - 1), V[:, 1:]])
        if abs(knots[-1] - 1.0) > 1e-12 or np.any(np.diff(knots) <= 0):
            raise SystemExit("via points need increasing lambda ending at 1")
        for c in range(inc.shape[1]):
            U[:, k_end + c] += np.interp(lam, knots, inc[:, c])
        return U
    du = np.asarray(spec["du"] if isinstance(spec, dict) else spec, dtype=float)
    U[:, k_end:] += lam[:, None] * du[None, :]
    return U


def cmd_targets(args):
    cfg = experiment_config(args.config)
    env, checkpoint = make_env(cfg)
    env_free, cp_free = make_env(cfg, pegs=False)
    u0 = initial_control(cfg)
    M = env.Nsteps
    out = Path(args.out) / "targets"
    out.mkdir(parents=True, exist_ok=True)
    paths = target_paths(cfg)
    names = check_names(args.targets, paths)
    pegs = T.pegs_of(env)
    q0 = checkpoint["q"]
    start_clear = [p.report(q0)["min_clearance"] for p in pegs]
    print(f"Rod: {material_summary(cfg)}\nHorizon: {M} steps, nv = {env.nv}, {len(pegs)} peg(s), "
          f"contact model {cfg.peg.get('model', 'smooth')} (delta {cfg.peg.get('smoothing', 1e-4):g} m)")
    print(f"Start rod: clearance to the pegs {[round(c * 1e3, 1) for c in start_clear]} mm (rod surface)")
    for name in names:
        desc, spec = paths[name]
        U = designed_commands(u0, spec, M)
        req = [int(i) for i in spec.get("touch", range(len(pegs)))]
        t0 = time.time()
        res = T.rollout(env, checkpoint, U, keep_J="last", record=True)
        if not res["ok"]:
            raise RuntimeError(f"target {name}: designed motion failed ({res['reason']})")
        shape = env.positions(res["q_list"][-1])
        free = T.rollout(env_free, cp_free, U, keep_J="last")
        nodes = T.free_nodes(env)
        rms_free = (float(np.sqrt(np.mean(np.sum((env.positions(free["q_list"][-1])[nodes]
                                                   - shape[nodes]) ** 2, axis=1))))
                    if free["ok"] else None)
        rec = res["rec"]
        touching_nodes = []
        for j in range(len(pegs)):
            e = np.nonzero(rec["touching"][-1, j])[0]
            touching_nodes.append(sorted(set(e.tolist()) | set((e + 1).tolist())))
        onset = [None if o is None else int(o) for o in res["onset"]]
        criteria = {
            # rod SURFACE to peg surface, as stated in the paper (Sec. VII-A)
            "start_clearance_ge_27mm": bool(min(start_clear) >= 0.027),
            # the pegs the target needs (`touch`, default all): not touched in the
            # first 30 % of the designed motion, touched at its end
            "no_contact_first_30pct": bool(all(onset[i] is not None and onset[i] >= int(0.3 * M) for i in req)),
            "required_pegs_touch_at_end": bool(all(res["final_touching"][i] for i in req)),
            "max_penetration_le_0.5mm": bool(res["max_penetration"] <= 5e-4),
            "no_cap_contact": not res["end_contact"],
            "no_self_contact": bool(res["min_self_clearance"] >= 0.0),
            # a target may admit a snap-through on its designed motion (terminal objective)
            "no_snap": res["snaps"] == 0 or bool(spec.get("allow_snap", False)),
        }
        # How hard (not a criterion; the RSS paper kept a scene only if neither
        # came within two thresholds): the straight constant-rate move to the
        # same end pose, and the designed motion without the pegs.
        Us = designed_commands(u0, {"du": (U[-1] - u0)[len(u0) // 2:]}, M)
        rs = T.rollout(env, checkpoint, Us, keep_J="last")
        rms_straight = (float(np.sqrt(np.mean(np.sum((env.positions(rs["q_list"][-1])[nodes]
                                                      - shape[nodes]) ** 2, axis=1))))
                        if rs["ok"] else None)
        rec_json = {"name": name, "description": desc, "du": spec.get("du"), "via": spec.get("via"),
                    "require_touch": req, "allow_snap": bool(spec.get("allow_snap", False)),
                    "N": M, "nv": int(env.nv),
                    "config": str(args.config), "contact_model": cfg.peg.get("model", "smooth"),
                    "smoothing": float(cfg.peg.get("smoothing", 1e-4)),
                    "snap_steps": res["snap_steps"], "max_substeps": res["max_substeps"],
                    "newton_iterations": res["newton_iterations"], "rollout_time": time.time() - t0,
                    "u_final": U[-1].tolist(), "mid_final": shape[env.nv // 2].tolist(),
                    "start_clearance": start_clear, "onset_step": onset,
                    "final_force": res["final_force"], "touching_nodes_final": touching_nodes,
                    "max_penetration": res["max_penetration"], "max_strain": res["max_strain"],
                    "max_abs_holonomy": res["max_abs_holonomy"],
                    "rms_free_vs_target": rms_free, "min_self_clearance": res["min_self_clearance"],
                    "rms_straight_vs_target": rms_straight,
                    "straight_move": "ok" if rs["ok"] else rs["reason"],
                    "rms_start_vs_target": float(np.sqrt(np.mean(np.sum(
                        (env.positions(q0)[nodes] - shape[nodes]) ** 2, axis=1)))),
                    "criteria": criteria, "criteria_ok": bool(all(criteria.values()))}
        np.savez_compressed(out / f"{name}.npz", shape=shape, U_ref=U,
                            shapes=np.array([env.positions(q) for q in res["q_list"]]),
                            min_clearance=rec["min_clearance"], force=rec["force"],
                            touching=rec["touching"])
        json.dump(rec_json, open(out / f"{name}.json", "w"), indent=1, default=float)
        print(f"[{name}] {desc}\n   onset steps {onset}, final force {np.round(res['final_force'], 2).tolist()} N, "
              f"touching nodes {touching_nodes}, max penetration {res['max_penetration'] * 1e3:.3f} mm, "
              f"snaps {res['snap_steps'] or 'none'}, start->target rms {rec_json['rms_start_vs_target'] * 1e3:.1f} mm, "
              f"free (no pegs) end shape vs target {('%.1f mm' % (rms_free * 1e3)) if rms_free is not None else 'failed'}, "
              f"straight move to the end pose: {('%.1f mm' % (rms_straight * 1e3)) if rms_straight is not None else 'infeasible (' + rs['reason'] + ')'}, "
              f"self clearance {res['min_self_clearance'] * 1e3:.1f} mm"
              f"\n   criteria: {'all OK' if rec_json['criteria_ok'] else {k: v for k, v in criteria.items() if not v}}",
              flush=True)
    plot_targets(out, names, env)


def plot_targets(out, names, env=None):
    fig = plt.figure(figsize=(5.2 * len(names), 4.8))
    for i, name in enumerate(names):
        z = np.load(out / f"{name}.npz")
        ax = fig.add_subplot(1, len(names), i + 1, projection="3d")
        S = z["shapes"]
        for k in range(0, len(S), max(1, len(S) // 8)):
            ax.plot(S[k, :, 0], S[k, :, 1], S[k, :, 2], color=INK2, lw=0.6, alpha=0.35)
        ax.plot(z["shape"][:, 0], z["shape"][:, 1], z["shape"][:, 2], color=INK, lw=2.2)
        if env is not None:
            draw_obstacles(ax, env)
        _axes3d(ax, name, pts=S.reshape(-1, 3))
    fig.tight_layout()
    fig.savefig(out / "targets.png", dpi=140)
    plt.close(fig)


def built_targets(out):
    d = Path(out) / "targets"
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json") if (d / f"{p.stem}.npz").exists())


def load_target(out, name, cfg, env):
    f = Path(out) / "targets" / f"{name}.json"
    if not f.exists():
        raise SystemExit(f"no target {name} in {f.parent}; build it with the `targets` command")
    rec = json.load(open(f))
    z = np.load(f.with_suffix(".npz"))
    if rec["nv"] != env.nv or rec["N"] != env.Nsteps:
        raise SystemExit(f"target {name} was built for nv={rec['nv']}, N={rec['N']}; "
                         f"this config has nv={env.nv}, N={env.Nsteps}")
    model = cfg.peg.get("model", "smooth")
    if rec.get("contact_model", "node") != model or abs(rec.get("smoothing", -1)
                                                       - float(cfg.peg.get("smoothing", 1e-4))) > 0:
        raise SystemExit(f"target {name} was built with contact model {rec.get('contact_model', 'node')} "
                         f"(delta {rec.get('smoothing')}); this config uses {model}: rebuild the targets")
    return z["shape"], z["U_ref"], rec


# ---------------------------------------------------------------- initialization

def initial_commands(cfg, U_ref, seed):
    """The common initialization: the designed motion, perturbed per seed.

    ``U_init(lambda) = U_ref(lambda) + lambda * e + sin(pi lambda) * b`` on the
    gripper, with ``e`` and ``b`` drawn from N(0, diag(scale^2)) by a
    generator seeded with ``seed`` (``task.shape_init``: ``end_*`` scale e,
    ``bump_*`` scale b; position [m], tangent angles [rad], yaw [rad]).
    """
    si = (cfg.get("task") or {}).get("shape_init") or {}
    sc_e = np.array([si.get("end_pos", 0.02)] * 3 + [si.get("end_ang", 0.05)] * 2 + [si.get("end_yaw", 0.05)])
    sc_b = np.array([si.get("bump_pos", 0.02)] * 3 + [si.get("bump_ang", 0.05)] * 2 + [si.get("bump_yaw", 0.05)])
    rng = np.random.default_rng(int(si.get("seed_base", 1000)) + int(seed))
    e = rng.standard_normal(6) * sc_e
    b = rng.standard_normal(6) * sc_b
    M = len(U_ref)
    lam = (np.arange(1, M + 1) / M)[:, None]
    k_end = U_ref.shape[1] // 2
    U = U_ref.copy()
    U[:, k_end:] += lam * e[None, :] + np.sin(np.pi * lam) * b[None, :]
    return U, {"end_offset": e.tolist(), "bump": b.tolist()}


def rates_of(U, u0, k_end, M):
    """Rates ``r_k = (u_k - u_{k-1}) / dlam`` of the gripper, ``u_{-1} = u0``."""
    V = np.vstack([u0[None, :], U])[:, k_end:]
    return np.diff(V, axis=0) * M


def peg_sides(env, X):
    """Per peg, the unit vector from its axis to the closest rod point (x-z plane):
    which side of the peg the rod passes."""
    out = []
    for p in T.pegs_of(env):
        v = X[:, [0, 2]] - np.array([p.cx, p.cz])
        i = int(np.argmin(np.linalg.norm(v, axis=1)))
        out.append(v[i] / max(np.linalg.norm(v[i]), 1e-12))
    return out


def calibrated_init(cfg, out_dir, target, seed, env, checkpoint, U_ref, target_shape, nodes, u0):
    """The common initialization of (target, seed), calibrated and cached.

    Direction: ``initial_commands`` (seeded).  Amplitude: a factor ``c`` on the
    perturbation, adjusted until the initial shape RMS lies in
    ``task.shape_init.rms_band`` (aim: its middle), by the secant-like update
    ``c <- c * aim / rms`` on rollouts with the pegs.  Accepted only if the
    rollout is admissible (converged, stable, in the workspace), has no snap,
    passes every peg on the same side as the target, has no snap unless the
    target admits one (``allow_snap``), and -- with
    ``task.shape_init.require_contact`` (default, as in the RSS paper) --
    ends touching every peg; otherwise the next
    direction is drawn (``seed + 1000 * attempt``).  Cached in
    OUT/inits/<target>_s<seed>.npz/.json, so every method of this (target,
    seed) starts from the same commands.
    """
    d = Path(out_dir) / "inits"
    f = d / f"{target}_s{seed}.json"
    if f.exists():
        rec = json.load(open(f))
        return np.load(f.with_suffix(".npz"))["U_init"], rec
    si = (cfg.get("task") or {}).get("shape_init") or {}
    lo, hi = (float(v) for v in si.get("rms_band", [0.03, 0.05]))
    aim = 0.5 * (lo + hi)
    need_contact = bool(si.get("require_contact", True))
    trec = json.load(open(Path(out_dir) / "targets" / f"{target}.json"))
    req = trec.get("require_touch") or list(range(len(T.pegs_of(env))))
    k_end = len(u0) // 2
    ws = workspace_bounds(cfg, len(u0) - k_end) if cfg.train.workspace == "reject" else None
    side_t = peg_sides(env, target_shape)
    tried = []
    for attempt in range(12):
        U_unit, pert = initial_commands(cfg, U_ref, seed + 1000 * attempt)
        P = U_unit - U_ref
        c = 1.0
        for it in range(10):
            U = U_ref + c * P
            res = T.rollout(env, checkpoint, U, keep_J="last")
            wsv = T.workspace_violation(U, k_end, ws)
            row = {"attempt": attempt, "c": c, "ok": res["ok"], "reason": res["reason"],
                   "snaps": res["snaps"], "ws_violation": wsv}
            if (not res["ok"] or wsv > 1e-9 or (res["snaps"] and not trec.get("allow_snap"))
                    or res["min_self_clearance"] < 0):
                tried.append(row)
                c *= 0.7                       # too far: shrink and retry
                continue
            X = env.positions(res["q_list"][-1])
            rms = float(np.sqrt(np.mean(np.sum((X[nodes] - target_shape[nodes]) ** 2, axis=1))))
            same = all(float(a @ b) > 0 for a, b in zip(side_t, peg_sides(env, X)))
            touch = all((res.get("final_touching") or [True] * (max(req) + 1))[i] for i in req)
            row.update(rms=rms, same_side=same, touching=touch)
            tried.append(row)
            if lo <= rms <= hi and same and (touch or not need_contact):
                rec = {"target": target, "seed": seed, "attempt": attempt, "scale": c, "rms": rms,
                       "rms_band": [lo, hi], **pert, "final_touching": res.get("final_touching"),
                       "onset": res["onset"], "max_penetration": res["max_penetration"], "tried": tried}
                d.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(f.with_suffix(".npz"), U_init=U)
                json.dump(rec, open(f, "w"), indent=1, default=_json_default)
                return U, rec
            if not same or (lo <= rms <= hi and need_contact and not touch):
                break                          # wrong side / off the pegs: a new direction
            c *= aim / max(rms, 1e-6)
    raise SystemExit(f"no admissible initialization in the band {lo * 1e3:.0f}-{hi * 1e3:.0f} mm "
                     f"for {target}, seed {seed}: {tried[-3:]}")


def initial_parameters(cfg, param, u0, k_end, M, U_ref, seed, init, calib=None):
    if init == "designed":
        if calib is not None:
            U_init, rec = calib
            pert = {"calibration": {k: rec[k] for k in ("attempt", "scale", "rms", "rms_band")}}
        else:
            U_init, pert = initial_commands(cfg, U_ref, seed)
        xi0, fit = param.from_rates(rates_of(U_init, u0, k_end, M))
        return xi0, {"init": "designed", **pert, **fit}
    if init == "random":
        if param.kind == "rates":
            return np.zeros(param.P), {"init": "zero rates"}
        return param.get(), {"init": "random network (torch seed)"}
    raise SystemExit(f"--init must be designed or random, got {init!r}")


# ---------------------------------------------------------------- common setup

def setup(args, cfg):
    """Everything a run or a check needs, in one dict."""
    env, checkpoint = make_env(cfg, pegs=not getattr(args, "train_without_pegs", False))
    env_eval, cp_eval = (env, checkpoint) if not getattr(args, "train_without_pegs", False) \
        else make_env(cfg)
    target_shape, U_ref, target_rec = load_target(args.out, args.target, cfg, env_eval)
    u0 = initial_control(cfg)
    k_end = len(u0) // 2
    k = end_control_dim(env_dim(cfg), env_twist(cfg))
    M = env.Nsteps
    seed = getattr(args, "seed", 0)
    nodes = T.free_nodes(env_eval)
    calib = None
    if cfg.train.init == "designed" or cfg.train.param == "endpose":
        calib = calibrated_init(cfg, args.out, args.target, seed, env_eval, cp_eval, U_ref,
                                target_shape, nodes, u0)
    if cfg.train.param == "endpose":
        # End pose only: the common initial trajectory with its final pose moved
        # over its last `window` steps; aimed (--aim) at the designed end pose,
        # or tuned by the optimizer from the initial trajectory's own end pose.
        window = int(getattr(args, "window", 20) or 20)
        param = T.EndPose(M, k, calib[0], u0, window)
        aim = bool(getattr(args, "aim", False))
        xi0 = param.aimed(np.asarray(target_rec["u_final"], float)) if aim else np.zeros(k)
        init_info = {"init": "designed + end pose " + ("aimed" if aim else "offset 0"), "window": window,
                     "calibration": {kk: calib[1][kk] for kk in ("attempt", "scale", "rms", "rms_band")},
                     "aimed_offset": xi0.tolist() if aim else None}
        label = f"endpose{window}" + ("aimed" if aim else "")
    else:
        param = T.make_parameterization(cfg.train.param, M, k, hidden=int(cfg.train.get("hidden", 64)),
                                        seed=seed)
        xi0, init_info = initial_parameters(cfg, param, u0, k_end, M, U_ref, seed, cfg.train.init, calib)
        label = cfg.train.param
    return dict(env=env, checkpoint=checkpoint, env_eval=env_eval, cp_eval=cp_eval,
                target_shape=target_shape, U_ref=U_ref, target_rec=target_rec, u0=u0, k_end=k_end,
                k=k, M=M, param=param, xi0=xi0, init_info=init_info, nodes=nodes,
                ws=workspace_bounds(cfg, k), label=label)


def evaluation(S, xi, record=True):
    """Roll ``xi`` out on the evaluation scene (with the pegs) and measure it."""
    env, cp = S["env_eval"], S["cp_eval"]
    U = T.commands(S["u0"], S["param"].rates(xi), S["k_end"], 1.0 / S["M"])
    res = T.rollout(env, cp, U, keep_J="last", record=record)
    res["ws_violation"] = T.workspace_violation(U, S["k_end"], S["ws"])
    out = {"U": U, "res": res, "ok": res["ok"]}
    if res["ok"]:
        L, _, X = T.terminal_loss(env, res["q_list"][-1], S["target_shape"], S["nodes"])
        out.update(loss=L, rms=float(np.sqrt(L)), shape=X,
                   node_err=np.linalg.norm(X - S["target_shape"], axis=1))
    return out


def admissible(ev, cfg):
    """The paper's admissibility of a motion (Sec. VII-A, Evaluation)."""
    r = ev["res"]
    checks = {"all_steps_stable": bool(r["ok"]),
              "in_workspace": cfg.train.workspace != "reject" or r.get("ws_violation", 0.0) <= 1e-9,
              "no_self_contact": r.get("min_self_clearance", np.inf) >= 0.0,
              "penetration_le_max": r["max_penetration"] <= float(cfg.train.max_penetration),
              "no_cap_contact": not r["end_contact"]}
    return all(checks.values()), checks


# ---------------------------------------------------------------- train

def cmd_train(args):
    cfg = experiment_config(args.config)
    tr = cfg["train"]
    for key in ("optimizer", "grad_method", "param", "init", "epochs", "max_rollouts",
                "stop_rms", "workspace"):
        v = getattr(args, key, None)
        if v is not None:
            tr[key] = v
    if args.lr is not None:
        tr.setdefault("adam", {})
        tr["adam"]["lr"] = args.lr
    tr = cfg.train                      # attribute view of the updated block
    optimizer, gm = tr.optimizer, tr.grad_method
    if gm not in GRAD_METHODS:
        raise SystemExit(f"train.grad_method must be one of {GRAD_METHODS}")
    np.random.seed(args.seed)
    S = setup(args, cfg)
    env = S["env"]
    variant = "_nopegs" if args.train_without_pegs else ""
    name = f"{args.target}_{S['label']}_{gm}_{optimizer}{variant}{args.tag}_s{args.seed}"
    run = RunLogger(cfg, env=env, root=_ROOT, tag=name,
                    enabled=bool((cfg.get("output") or {}).get("run_log", True)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run.artifact_dir) if getattr(run, "artifact_dir", None) else out
    run_dir.mkdir(parents=True, exist_ok=True)
    ocfg = opt_settings(cfg, optimizer)

    print(f"Task: 3D shape formation, target {args.target} -- {S['target_rec']['description']}")
    print(f"Trajectory: {S['label']} (P = {S['param'].P}), init {S['init_info']['init']}"
          + (f" (fit rms {S['init_info'].get('fit_rms_pose', 0):.2e})" if tr.init == 'designed' else ""))
    print(f"Gradient: {gm}   Optimizer: {optimizer} {ocfg}")
    print(f"Stop: rms <= {float(tr.stop_rms) * 1e3:g} mm | {tr.epochs} updates | {tr.max_rollouts} rollouts "
          f"| {tr.max_failures} failed rollouts;  workspace: {tr.workspace}")
    print(f"Scene: {material_summary(cfg)}; {S['M']} steps, nv = {env.nv}, "
          f"{len(T.pegs_of(env))} peg(s) in training{' (trained WITHOUT pegs, evaluated with)' if variant else ''}")

    init_eval = evaluation(S, S["xi0"])
    print(f"Initial trajectory: {'rms %.2f mm' % (init_eval['rms'] * 1e3) if init_eval['ok'] else 'FAILED: ' + init_eval['res']['reason']}"
          + (f", pegs touched at the end {init_eval['res'].get('final_touching')}, onset {init_eval['res']['onset']}"
             if init_eval['ok'] else ""), flush=True)

    ev_fn = T.Evaluator(env, S["checkpoint"], S["param"], S["u0"], S["k_end"], S["target_shape"],
                        S["nodes"], grad_method=gm, jac_reg=float(tr.get("jac_reg", 0.0)),
                        ws_bounds=S["ws"], workspace=tr.workspace, stop_rms=float(tr.stop_rms),
                        max_rollouts=int(tr.max_rollouts), max_failures=int(tr.max_failures), tag=name)
    t0 = time.time()
    iters, final_ev = T.optimize(ev_fn, S["xi0"], optimizer, int(tr.epochs), ocfg, tag=name)
    t_train = time.time() - t0
    print(f"[{name}] stopped: {ev_fn.stop}", flush=True)

    xi_final = S["xi0"] if final_ev is None else final_ev["xi"]
    xi_best = S["xi0"] if ev_fn.best is None else ev_fn.best["xi"]
    fin = evaluation(S, xi_final)
    adm, adm_checks = admissible(fin, cfg) if fin["ok"] else (False, {"all_steps_stable": False})
    solved = bool(fin["ok"] and adm and fin["rms"] <= float(tr.solve_rms))
    first_below = lambda thr: next((i for i, it in enumerate(iters) if it["rms"] <= thr), None)  # noqa: E731

    summary = {
        "task": "shape", "status": "done", "target_name": args.target, "config": str(args.config),
        "param": S["label"], "grad_method": gm, "optimizer": optimizer, "optimizer_settings": ocfg,
        "trained_without_pegs": bool(variant), "seed": args.seed, "P": int(S["param"].P),
        "init": S["init_info"], "nv": int(env.nv), "N": int(S["M"]), "n_free_nodes": int(len(S["nodes"])),
        "stop_rms": float(tr.stop_rms), "solve_rms": float(tr.solve_rms), "epochs_cap": int(tr.epochs),
        "max_rollouts": int(tr.max_rollouts), "max_failures": int(tr.max_failures),
        "workspace_mode": tr.workspace, "stop": ev_fn.stop, "train_time": t_train,
        "updates": len(iters) - 1 if iters else 0, "rollouts": ev_fn.n_rollouts,
        "evaluations": ev_fn.n_evals, "failures": ev_fn.n_failures,
        "failure_reasons": sorted({r for r, ok in zip(ev_fn.log["reason"], ev_fn.log["ok"]) if not ok}),
        "adjoint_solves": int(T.N_ADJOINT_SOLVES[0]),
        "grad_time_median": float(np.median(ev_fn.grad_time)) if ev_fn.grad_time else None,
        "init_rms": init_eval.get("rms"), "init_ok": init_eval["ok"],
        "init_onset": init_eval["res"]["onset"], "init_final_touching": init_eval["res"].get("final_touching"),
        "final_rms": fin.get("rms"), "final_loss": fin.get("loss"), "final_ok": fin["ok"],
        # best_rms is measured in the TRAINING scene (without the pegs for the
        # no-obstacles baseline); final_* and solved in the evaluation scene.
        "best_rms": None if ev_fn.best is None else ev_fn.best["rms"],
        "solved": solved, "admissible": adm, "admissibility": adm_checks,
        "final_snaps": fin["res"]["snaps"], "final_snap_steps": fin["res"]["snap_steps"],
        "final_max_penetration": fin["res"]["max_penetration"], "final_max_strain": fin["res"]["max_strain"],
        "final_max_abs_holonomy": fin["res"]["max_abs_holonomy"], "final_ws_violation": fin["res"]["ws_violation"],
        "final_onset": fin["res"]["onset"], "final_touching": fin["res"].get("final_touching"),
        "final_force": fin["res"].get("final_force"),
        "snaps_total": int(np.nansum(ev_fn.log["snaps"])),
        "updates_to_5mm": first_below(5e-3), "updates_to_1mm": first_below(1e-3),
        "iterates": iters,
        "evals": {k: v for k, v in ev_fn.log.items()},
        "target": {k: S["target_rec"].get(k) for k in ("description", "du", "onset_step", "criteria_ok")},
    }
    arrays = {"target_shape": S["target_shape"], "U_ref": S["U_ref"],
              "xi_init": S["xi0"], "xi_final": xi_final, "xi_best": xi_best,
              "U_init": init_eval["U"], "U_final": fin["U"],
              "q_init": np.asarray(S["cp_eval"]["q"]),
              "q_traj_init": np.array(init_eval["res"]["q_list"]),
              "q_traj": np.array(fin["res"]["q_list"]),
              "eval_loss": np.array(ev_fn.log["loss"], float), "eval_rms": np.array(ev_fn.log["rms"], float),
              "eval_ok": np.array(ev_fn.log["ok"]), "eval_rollouts": np.array(ev_fn.log["rollouts"]),
              "eval_time": np.array(ev_fn.log["time"]),
              "iter_rms": np.array([it["rms"] for it in iters], float),
              "iter_eval": np.array([it["eval"] for it in iters], int)}
    for tag, ev in (("init", init_eval), ("final", fin)):
        rec = ev["res"].get("rec")
        if rec:
            for key, val in rec.items():
                arrays[f"{tag}_{key}"] = val
        if ev["ok"]:
            arrays[f"{tag}_shape"] = ev["shape"]
            arrays[f"{tag}_node_err"] = ev["node_err"]
    if fin["ok"]:
        arrays["final_shape"] = fin["shape"]
    for d in dict.fromkeys([run_dir, out]):
        np.savez_compressed(d / f"{name}.npz", **arrays)
        json.dump(summary, open(d / f"{name}.json", "w"), indent=1, default=_json_default)

    env_e = S["env_eval"]
    if fin["ok"]:
        plot_shape_result(run_dir / f"{name}_shape", S["target_shape"], fin["res"]["q_list"][-1], env_e,
                          f"{name}: rms {fin['rms'] * 1e3:.2f} mm", q_init=S["cp_eval"]["q"])
        plot_shape_error(S["target_shape"], fin["shape"], S["nodes"], run_dir / f"{name}_error")
    if len(iters) > 1:
        ts.loss_curve([it["rms"] * 1e3 for it in iters], run_dir / f"{name}_rms", xlabel="Update",
                      ylabel="Shape RMS (mm)", title=f"{args.target}, {gm}, {optimizer}")
    if fin["ok"]:
        maybe_save_trajectory_gif(cfg, fin["res"]["q_list"], env_e, None, save_path=str(run_dir),
                                  filename=f"{name}.gif", title=f"shape formation {name}",
                                  gripper_node=env_e.nv - 1, target_shape=S["target_shape"])
    run.finish(task="shape", target_name=args.target, optimizer=optimizer, grad_method=gm,
               param=tr.param, final_error=fin.get("rms"), solved=solved, stop=ev_fn.stop,
               rollouts=ev_fn.n_rollouts, failures=ev_fn.n_failures)
    print(f"[{name}] final rms {('%.3f mm' % (fin['rms'] * 1e3)) if fin['ok'] else 'n/a'} "
          f"(best {('%.3f' % (ev_fn.best['rms'] * 1e3)) if ev_fn.best else 'n/a'} mm), solved {solved}, "
          f"{len(iters) - 1 if iters else 0} updates, {ev_fn.n_rollouts} rollouts, {ev_fn.n_failures} failed, "
          f"max penetration {fin['res']['max_penetration'] * 1e3:.3f} mm, {t_train:.0f}s -> {run_dir}", flush=True)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ---------------------------------------------------------------- greedy baseline

def cmd_greedy(args):
    """Greedy baseline (paper Sec. VII-A): a one-step horizon with the exact
    per-step sensitivity.

    From the initial equilibrium at u0, every step moves the gripper by the
    increment that most lowers the task loss (mean squared error of the free
    nodes) at the next equilibrium: a Levenberg-Marquardt direction of the
    linearized shape map, bounded by the step cap, and the best of the trial
    lengths ``greedy.alphas``, each a real quasi-static step
    (src/greedy_control.py, schedule ``direct``).  The step cap is the largest
    per-step pose change of our solutions (OUT/<target>_net_state_lbfgs_s*.npz;
    positions in m, angles times ``greedy.angle_scale``), or ``--cap``.  It
    stops when the trust radius collapses, the loss stalls, or after
    ``--max-steps``.  Deterministic: one run per target.  The accepted
    commands are then replayed on the evaluation scene, which gives the same
    data as a training run (contact per step, snaps, penetration, figures).
    """
    from greedy_control import greedy_shape_control, DEFAULTS as GD
    cfg = experiment_config(args.config)
    env, cp = make_env(cfg)
    target_shape, U_ref, trec = load_target(args.out, args.target, cfg, env)
    u0 = initial_control(cfg)
    k_end = len(u0) // 2
    nodes = T.free_nodes(env)
    o = dict(GD, **(cfg.get("greedy") or {}))
    o.update(schedule="direct", reg=None, ref_final=None)
    if args.max_steps is not None:
        o["max_steps"] = int(args.max_steps)
    cap_src = "--cap"
    if args.cap is not None:
        cap = float(args.cap)
    else:
        files = sorted(Path(args.out).glob(f"{args.target}_net_state_lbfgs_s*.npz"))
        if files:
            caps = []
            for f in files:
                D = np.diff(np.vstack([u0[None, :], np.load(f)["U_final"]]), axis=0)[:, k_end:]
                D[:, 3:] *= float(o["angle_scale"])
                caps.append(float(np.abs(D).max()))
            cap, cap_src = max(caps), [f.name for f in files]
        else:
            cap, cap_src = float(o["step_cap"]), "greedy.step_cap (no state/L-BFGS runs found yet)"
            print(f"NOTE: no {args.target}_net_state_lbfgs_s*.npz in {args.out}: the step cap is "
                  f"greedy.step_cap = {cap:g}; run greedy after the main runs for the paper's bound")
    o["step_cap"] = cap
    cols = np.arange(k_end, len(u0))
    print(f"Greedy on {args.target}: step cap {cap:.4g} (from {cap_src}), max steps {o['max_steps']}, "
          f"alphas {o['alphas']}")
    t0 = time.time()
    res = greedy_shape_control(env, u0, env._last_J, lambda lam: target_shape, cols,
                               angle_cols=cols[3:], opts=o, nodes=nodes,
                               log=lambda m: print(m, flush=True), log_every=args.log_every)
    t_run = time.time() - t0
    U = np.asarray(res["u"])[1:]
    name = f"{args.target}_greedy_state_greedy{args.tag}_s0"
    out = Path(args.out)
    if len(U) == 0:
        raise SystemExit(f"[{name}] no step accepted ({res['stop']})")
    rep = T.rollout(env, cp, U, keep_J="last", record=True)
    ws = T.workspace_violation(U, k_end, workspace_bounds(cfg, len(cols)))
    rep["ws_violation"] = ws
    fin = {"U": U, "res": rep, "ok": rep["ok"]}
    if rep["ok"]:
        L, _, X = T.terminal_loss(env, rep["q_list"][-1], target_shape, nodes)
        fin.update(loss=L, rms=float(np.sqrt(L)), shape=X,
                   node_err=np.linalg.norm(X - target_shape, axis=1))
        replay_dev = float(np.abs(env.positions(rep["q_list"][-1]) - env.positions(res["q"][-1])).max())
    else:
        replay_dev = None
    adm, adm_checks = admissible(fin, cfg) if fin["ok"] else (False, {"all_steps_stable": False})
    solved = bool(fin["ok"] and adm and fin["rms"] <= float(cfg.train.solve_rms))
    rms_steps = np.asarray(res["rms_final"], float)
    trials = np.asarray(res["trials"], float)
    summary = {
        "task": "shape", "status": "done", "target_name": args.target, "config": str(args.config),
        "param": "greedy", "grad_method": "state", "optimizer": "greedy", "optimizer_settings": o,
        "trained_without_pegs": False, "seed": 0, "P": 0, "init": {"init": "u0 (no trajectory)"},
        "nv": int(env.nv), "N": int(env.Nsteps), "greedy_steps": int(len(U)), "step_cap_source": cap_src,
        "stop": res["stop"], "train_time": t_run, "updates": int(len(U)),
        "rollouts": float(np.sum(trials) / env.Nsteps), "env_steps": int(np.sum(trials)),
        "evaluations": int(np.sum(trials)), "failures": int(sum(res["rejected"].values())),
        "failure_reasons": [k for k, v in res["rejected"].items() if v], "rejected": res["rejected"],
        "snap_trials": int(res["snap_trials"]),
        "init_rms": float(rms_steps[0]), "init_ok": True,
        "final_rms": fin.get("rms"), "final_loss": fin.get("loss"), "final_ok": fin["ok"],
        "best_rms": float(np.nanmin(rms_steps)), "replay_max_deviation": replay_dev,
        "solved": solved, "admissible": adm, "admissibility": adm_checks,
        "final_snaps": rep["snaps"], "final_snap_steps": rep["snap_steps"],
        "final_max_penetration": rep["max_penetration"], "final_max_strain": rep["max_strain"],
        "final_max_abs_holonomy": rep["max_abs_holonomy"], "final_ws_violation": ws,
        "final_onset": rep["onset"], "final_touching": rep.get("final_touching"),
        "final_force": rep.get("final_force"), "snaps_total": rep["snaps"],
        "iterates": [{"eval": i, "rms": float(r)} for i, r in enumerate(rms_steps)],
        "evals": {"rms": rms_steps.tolist()},
        "target": {kk: trec.get(kk) for kk in ("description", "du", "onset_step", "criteria_ok")},
    }
    arrays = {"target_shape": target_shape, "U_ref": U_ref, "U_final": U, "q_init": np.asarray(cp["q"]),
              "q_traj": np.array(rep["q_list"]), "iter_rms": rms_steps,
              "eval_rms": rms_steps, "eval_loss": rms_steps ** 2, "eval_ok": np.ones(len(rms_steps), bool),
              "eval_rollouts": np.concatenate([[0.0], np.cumsum(trials[1:])]) / env.Nsteps,
              "eval_time": np.linspace(0.0, t_run, len(rms_steps)),
              "greedy_alpha": np.asarray(res["alpha"], float), "greedy_cap": np.asarray(res["cap"], float),
              "greedy_snap": np.asarray(res["snap"], bool), "greedy_holonomy": np.asarray(res["holonomy"], float)}
    for key, val in (rep.get("rec") or {}).items():
        arrays[f"final_{key}"] = val
    if fin["ok"]:
        arrays["final_shape"] = fin["shape"]
        arrays["final_node_err"] = fin["node_err"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", **arrays)
    json.dump(summary, open(out / f"{name}.json", "w"), indent=1, default=_json_default)
    if fin["ok"]:
        plot_shape_result(out / f"{name}_shape", target_shape, rep["q_list"][-1], env,
                          f"{name}: rms {fin['rms'] * 1e3:.2f} mm", q_init=cp["q"])
        plot_shape_error(target_shape, fin["shape"], nodes, out / f"{name}_error")
    print(f"[{name}] {len(U)} steps ({res['stop']}), final rms "
          f"{('%.3f mm' % (fin['rms'] * 1e3)) if fin['ok'] else 'n/a'}, solved {solved}, admissible {adm} "
          f"{ {k: v for k, v in adm_checks.items() if not v} or ''}, snaps {rep['snaps']}, "
          f"replay deviation {replay_dev}, {t_run:.0f}s -> {out}", flush=True)


# ---------------------------------------------------------------- check (H1 and the proxy error)

def cmd_check(args):
    """Gradient check at the common initialization (paper Sec. VII-C, H1).

    1. Directional derivatives of the loss along ``--dirs`` random unit
       directions of xi: the state-based gradient and the frozen-tangent proxy
       against central differences of full rollouts, with the Newton
       tolerance tightened by ``--tol-factor``.  Relative error |g.d - fd| / |fd|.
       The perturbed rollouts must snap at the same steps (same branch).
    2. Cost: wall time of one state and one proxy gradient, and their adjoint
       solves.
    3. Per rate: relative difference of the proxy from the state gradient,
       dL/dr_j, before and after the first peg contact (Sec. VII-F), with the
       pegs and -- ``--no-pegs-too`` -- without them.
    """
    cfg = experiment_config(args.config)
    for key in ("param", "init"):
        if getattr(args, key, None) is not None:
            cfg["train"][key] = getattr(args, key)
    args.train_without_pegs = False
    S = setup(args, cfg)
    env = S["env"]
    env.tol *= float(args.tol_factor)
    rng = np.random.default_rng(args.seed + 77)
    report = {"target": args.target, "param": cfg.train.param, "P": int(S["param"].P),
              "tol": env.tol, "h": args.h, "dirs": []}
    evs = {}
    for gm in GRAD_METHODS:
        e = T.Evaluator(env, S["checkpoint"], S["param"], S["u0"], S["k_end"], S["target_shape"],
                        S["nodes"], grad_method=gm, ws_bounds=None, workspace="off", stop_rms=-1,
                        max_rollouts=10 ** 9, max_failures=10 ** 9, verbose=False)
        n0 = T.N_ADJOINT_SOLVES[0]
        evs[gm] = e(S["xi0"])
        if not evs[gm]["ok"]:
            raise SystemExit(f"initial rollout failed: {evs[gm]['reason']}")
        report[f"{gm}_grad_time"] = e.grad_time[-1]
        report[f"{gm}_adjoint_solves"] = T.N_ADJOINT_SOLVES[0] - n0
    L0 = evs["state"]["loss"]
    base_snaps = evs["state"]["res"]["snap_steps"]
    print(f"[check {args.target}] initial rms {np.sqrt(L0) * 1e3:.2f} mm, snaps {base_snaps or 'none'}, "
          f"P = {S['param'].P}, Newton tol {env.tol:.2e}")
    print(f"   gradient cost: state {report['state_grad_time'] * 1e3:.1f} ms / {report['state_adjoint_solves']} solve(s); "
          f"proxy {report['proxy_grad_time'] * 1e3:.1f} ms / {report['proxy_adjoint_solves']} solve(s)")
    plain = T.Evaluator(env, S["checkpoint"], S["param"], S["u0"], S["k_end"], S["target_shape"],
                        S["nodes"], grad_method="state", workspace="off", stop_rms=-1,
                        max_rollouts=10 ** 9, max_failures=10 ** 9, verbose=False)
    for i in range(args.dirs):
        d = rng.standard_normal(S["param"].P)
        d /= np.linalg.norm(d)
        lp = plain(S["xi0"] + args.h * d)
        lm = plain(S["xi0"] - args.h * d)
        if not (lp["ok"] and lm["ok"]):
            print(f"   dir {i}: a perturbed rollout failed")
            continue
        fd = (lp["loss"] - lm["loss"]) / (2 * args.h)
        same = lp["res"]["snap_steps"] == base_snaps == lm["res"]["snap_steps"]
        row = {"fd": fd, "same_branch": bool(same)}
        for gm in GRAD_METHODS:
            gd = float(evs[gm]["g"] @ d)
            row[gm] = gd
            row[f"{gm}_relerr"] = abs(gd - fd) / max(abs(fd), 1e-300)
        report["dirs"].append(row)
        print(f"   dir {i}: fd {fd:+.6e}  state {row['state']:+.6e} (rel err {row['state_relerr']:.2e})  "
              f"proxy {row['proxy']:+.6e} (rel err {row['proxy_relerr']:.2e})  same snaps {same}", flush=True)
    if report["dirs"]:
        se = [r["state_relerr"] for r in report["dirs"]]
        pe = [r["proxy_relerr"] for r in report["dirs"]]
        report["state_relerr_median"], report["state_relerr_max"] = float(np.median(se)), float(np.max(se))
        report["proxy_relerr_median"] = float(np.median(pe))
        print(f"   state: median rel err {np.median(se):.2e}, max {np.max(se):.2e};  proxy median {np.median(pe):.2e}")

    # 3. per-rate proxy error, with (and optionally without) the pegs
    scenes = [("pegs", env, S["checkpoint"])]
    if args.no_pegs_too:
        env_f, cp_f = make_env(cfg, pegs=False)
        scenes.append(("no_pegs", env_f, cp_f))
    on_pegs = [o for o in evs["state"]["res"]["onset"] if o is not None]
    split = min(on_pegs) if on_pegs else S["M"] // 2      # first peg contact of the initial rollout
    for label, env_s, cp_s in scenes:
        G = {}
        onset = None
        for gm in GRAD_METHODS:
            e = T.Evaluator(env_s, cp_s, T.DirectRates(S["M"], S["k"]), S["u0"], S["k_end"],
                            S["target_shape"], S["nodes"], grad_method=gm, workspace="off", stop_rms=-1,
                            max_rollouts=10 ** 9, max_failures=10 ** 9, verbose=False)
            r = e(S["param"].rates(S["xi0"]).reshape(-1))
            if not r["ok"]:
                print(f"   ({label}: rollout failed, {r['reason']})")
                break
            G[gm] = r["g"].reshape(S["M"], S["k"])
            on = [o for o in r["res"]["onset"] if o is not None]
            onset = min(on) if on else None
        if len(G) < 2:
            continue
        rel = np.linalg.norm(G["proxy"] - G["state"], axis=1) / np.maximum(np.linalg.norm(G["state"], axis=1), 1e-300)
        ref = split
        before = rel[30:ref] if ref > 30 else rel[:ref]
        after = rel[ref:]
        report[f"per_rate_{label}"] = {"onset": onset, "split_at": int(ref), "rel": rel.tolist(),
                                       "median_before": float(np.median(before)) if before.size else None,
                                       "median_after": float(np.median(after)) if after.size else None,
                                       "last": float(rel[-1])}
        print(f"   proxy vs state per rate ({label}): first contact at step {onset}; median rel diff "
              f"steps 30..{ref}: {report[f'per_rate_{label}']['median_before']}, after: "
              f"{report[f'per_rate_{label}']['median_after']}, last rate: {rel[-1]:.2e}")
    out = Path(args.out) / "checks"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"check_{args.target}_{cfg.train.param}_s{args.seed}.json"
    json.dump(report, open(f, "w"), indent=1, default=_json_default)
    print(f"   -> {f}")


# ---------------------------------------------------------------- figures

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {("state", "lbfgs"): "#1baf7a", ("state", "adam"): "#2a78d6",
          ("proxy", "lbfgs"): "#eb6834", ("proxy", "adam"): "#b8860b"}


def _axes3d(ax, title, pts=None):
    ax.set_xlabel("x [m]", color=INK2, fontsize=8)
    ax.set_ylabel("y [m]", color=INK2, fontsize=8)
    ax.set_zlabel("z [m]", color=INK2, fontsize=8)
    if pts is not None:
        pts = np.asarray(pts, dtype=float)
        lo, hi = pts.min(0) - 0.08, pts.max(0) + 0.08
        c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
        h[1] = max(h[1], 0.25)
        lims = np.column_stack([c - h, c + h])
        for f, l in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lims):
            f(*l)
        try:
            ax.set_box_aspect(tuple(lims[:, 1] - lims[:, 0]))
        except AttributeError:
            pass
    ax.tick_params(labelsize=7, colors=INK2)
    ax.set_title(title, color=INK, fontsize=9)


def plot_shape_result(path, target_shape, q, env, title, q_init=None, q_snaps=None,
                      view=(22, -58), workspace="off", margin=0.04, limits=None):
    """Final rod against the target, with the pegs and gripper frames (IEEE T-RO style)."""
    from utils import paper_scene
    return paper_scene(path, env, rod_init=q_init, rod_final=q, rod_snaps=q_snaps,
                       rod_labels=("Initial rod", "Final shape"),
                       curves=[(target_shape, "reference", "Target shape")], frames_q=q,
                       title=title, view=view, workspace=workspace, margin=margin, limits=limits)


def plot_shape_error(target_shape, shape, nodes, path, title=None):
    """Per-node error |x_i - x_i*| (mm) along the rod, single column."""
    nodes = np.asarray(nodes)
    err = 1e3 * np.linalg.norm(np.asarray(shape)[nodes] - np.asarray(target_shape)[nodes], axis=1)
    with ts.style():
        fig, ax = plt.subplots(figsize=(ts.COL_W, 1.35))
        ax.plot(nodes, err, **ts.ROLE["achieved"])
        fin = err[np.isfinite(err)]
        if fin.size:
            rms = float(np.sqrt(np.mean(fin ** 2)))
            ax.axhline(rms, **ts.ROLE["reference"])
            ax.text(0.99, rms, f"RMS {rms:.2f} mm", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=ts.FS_TICK)
        ax.set_xlim(0, len(target_shape) - 1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"Node $i$")
        ax.set_ylabel("Error (mm)")
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
        raise SystemExit(f"{npz} was trained on a rod with {z['q_init'].size} DOFs; "
                         f"pass --config {meta.get('config', '<the run config>')}")
    target_shape, q_traj = z["target_shape"], z["q_traj"]
    snaps = None
    if args.snapshots > 0 and len(q_traj):
        idx = np.linspace(0, len(q_traj) - 1, args.snapshots + 2)[1:-1].round().astype(int)
        snaps = [q_traj[i] for i in idx]
    stem = Path(args.stem) if args.stem else npz.with_name(npz.stem)
    if "final_shape" in z.files:
        title = None if args.no_title else f"Shape formation ({meta.get('target_name', npz.stem)})"
        plot_shape_result(f"{stem}_shape", target_shape, q_traj[-1], env, title, q_init=z["q_init"],
                          q_snaps=snaps, view=tuple(args.view), workspace=args.workspace)
        plot_shape_error(target_shape, z["final_shape"], T.free_nodes(env), f"{stem}_error")
    if "iter_rms" in z.files and len(z["iter_rms"]) > 1:
        ts.loss_curve(list(1e3 * z["iter_rms"]), f"{stem}_rms", xlabel="Update", ylabel="Shape RMS (mm)")


# ---------------------------------------------------------------- jobs / report

def cmd_jobs(args):
    """Print the batch: the four main methods, and with ``--baselines`` the rest.

    Main ({state, proxy} x {adam, lbfgs}, the network): one line per seed.
    Baselines (state-based gradient): direct rates and no obstacles with
    L-BFGS; end pose tuned (L-BFGS) and aimed (no optimization) over the last
    20 and 60 steps -- per seed, since each starts from that seed's initial
    trajectory.  Greedy is deterministic and its step cap comes from the
    main L-BFGS runs, so it is printed last, to run after them.
    """
    out = Path(args.out)
    cfg = experiment_config(args.config)
    names = check_names(args.targets, target_paths(cfg))
    lr_file = out / "adam_lr.json"
    adam_lr = json.load(open(lr_file))["lr"] if lr_file.exists() else None
    base = f"python scripts/train_shape.py train --config {args.config} --out {out}"
    lines = []
    for t in names:
        for gm in args.grad_methods:
            for o in args.optimizers:
                for s in args.seeds:
                    name = f"{t}_{args.param}_{gm}_{o}{'_nopegs' if args.train_without_pegs else ''}_s{s}"
                    lines.append((f"{base} --target {t} --param {args.param} --grad-method {gm} --optimizer {o} "
                                  f"--seed {s}" + (f" --lr {adam_lr:g}" if (o == "adam" and adam_lr) else "")
                                  + (" --train-without-pegs" if args.train_without_pegs else ""), name))
        if args.baselines:
            for s in args.seeds:
                lines.append((f"{base} --target {t} --param rates --grad-method state --optimizer lbfgs --seed {s}",
                              f"{t}_rates_state_lbfgs_s{s}"))
                lines.append((f"{base} --target {t} --param net --grad-method state --optimizer lbfgs "
                              f"--train-without-pegs --seed {s}", f"{t}_net_state_lbfgs_nopegs_s{s}"))
                for w in (20, 60):
                    lines.append((f"{base} --target {t} --param endpose --window {w} --grad-method state "
                                  f"--optimizer lbfgs --seed {s}", f"{t}_endpose{w}_state_lbfgs_s{s}"))
                    lines.append((f"{base} --target {t} --param endpose --window {w} --aim --grad-method state "
                                  f"--optimizer none --seed {s}", f"{t}_endpose{w}aimed_state_none_s{s}"))
    print(f"mkdir -p {out}/logs")
    for cmd, name in lines:
        print(f"{cmd} > {out}/logs/{name}.log 2>&1")
    if args.baselines:
        print("# after the runs above (greedy's step cap is the largest step of the state/L-BFGS solutions):")
        for t in names:
            print(f"python scripts/train_shape.py greedy --config {args.config} --out {out} --target {t} "
                  f"> {out}/logs/{t}_greedy.log 2>&1")


#: Adam rates of the sweep (paper Sec. VII-A: the rate is chosen by a sweep on
#: the first seed of each task).
SWEEP_LRS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


def cmd_sweep(args):
    """The Adam-rate sweep (print the jobs) or its decision (``--pick``).

    Protocol, fixed before the runs: seed 0 of each target, the frozen tangent
    with Adam (the baseline's own optimizer; RSS paper), the full budget, the
    rates SWEEP_LRS.  Rule: the rate at which the most targets are solved;
    ties go to the lower median final RMS, then to the smaller rate.  The
    chosen rate is written to OUT/adam_lr.json and used by ``jobs`` for every
    Adam run (both gradients).  The sweep runs live in OUT/sweep.
    """
    out = Path(args.out)
    sw = out / "sweep"
    cfg = experiment_config(args.config)
    names = check_names(args.targets, target_paths(cfg))
    if not args.pick:
        print(f"mkdir -p {sw}/logs && cp -r {out}/targets {out}/inits {sw}/ 2>/dev/null; true")
        for t in names:
            for lr in SWEEP_LRS:
                tag = f"_lr{lr:g}"
                print(f"python scripts/train_shape.py train --config {args.config} --out {sw} --target {t} "
                      f"--param {args.param} --grad-method {args.grad_method} --optimizer adam --lr {lr:g} "
                      f"--tag {tag} --seed 0 > {sw}/logs/{t}_{args.grad_method}_adam{tag}_s0.log 2>&1")
        return
    runs = [r for r in load_runs(sw) if r["optimizer"] == "adam" and r["grad_method"] == args.grad_method
            and r["target_name"] in names]
    rows = []
    for lr in SWEEP_LRS:
        R = [r for r in runs if abs(r["optimizer_settings"]["lr"] - lr) <= 1e-12 * max(lr, 1)]
        if not R:
            continue
        fr = [r["final_rms"] if r["final_rms"] is not None else np.inf for r in R]
        rows.append({"lr": lr, "runs": len(R), "solved": int(sum(r["solved"] for r in R)),
                     "median_final_rms": float(np.median(fr)), "failures": int(sum(r["failures"] for r in R))})
    if not rows:
        raise SystemExit(f"no sweep runs in {sw}")
    best = sorted(rows, key=lambda r: (-r["solved"], r["median_final_rms"], r["lr"]))[0]
    for r in rows:
        print(f"lr {r['lr']:g}: {r['solved']}/{r['runs']} solved, median final rms "
              f"{r['median_final_rms'] * 1e3:.3f} mm, {r['failures']} failed rollouts"
              + ("   <- chosen" if r is best else ""))
    json.dump({"lr": best["lr"], "rule": "most solved, then lower median final rms, then smaller rate",
               "grad_method": args.grad_method, "seed": 0, "rows": rows},
              open(out / "adam_lr.json", "w"), indent=1)
    print(f"-> {out / 'adam_lr.json'}")


def load_runs(out):
    runs = []
    for js in sorted(Path(out).glob("*.json")):
        try:
            s = json.load(open(js))
        except (ValueError, OSError):
            continue
        if isinstance(s, dict) and s.get("task") == "shape" and s.get("status") == "done" and "iterates" in s:
            s["_name"] = js.stem
            runs.append(s)
    return runs


def report(out, config=None):
    out = Path(out)
    runs = load_runs(out)
    if not runs:
        raise SystemExit(f"no shape runs in {out}")
    key = lambda r: (r["param"], r["grad_method"], r["optimizer"], r["trained_without_pegs"])  # noqa: E731
    L = ["# 3D shape formation against two pegs", "",
         "Solved: final shape RMS <= solve_rms and the motion admissible (every step converged and stable, "
         "every pose in the workspace, max penetration <= train.max_penetration, no cap contact). "
         "'rollouts' counts every rollout, failed ones and line-search trials included.", "",
         "| target | param | gradient | optimizer | seed | init rms [mm] | final rms [mm] | best rms [mm] | solved | "
         "updates | rollouts | failed | snaps (final) | max pen. [mm] | time [s] | stop |",
         "|" + "---|" * 16]
    for r in sorted(runs, key=lambda r: (r["target_name"], key(r), r["seed"])):
        f = lambda v: "-" if v is None else f"{v * 1e3:.3f}"  # noqa: E731
        L.append(f"| {r['target_name']} | {r['param']}{' (no pegs)' if r['trained_without_pegs'] else ''} | "
                 f"{r['grad_method']} | {r['optimizer']} | {r['seed']} | {f(r['init_rms'])} | {f(r['final_rms'])} | "
                 f"{f(r['best_rms'])} | {'yes' if r['solved'] else 'no'} | {r['updates']} | {r['rollouts']} | "
                 f"{r['failures']} | {r['final_snaps']} | {r['final_max_penetration'] * 1e3:.3f} | "
                 f"{r['train_time']:.0f} | {r['stop']} |")
    L += ["", "## Per method", "", "| param | gradient | optimizer | runs | solved | median final rms [mm] | "
          "median rollouts | failed rollouts (total) |", "|" + "---|" * 8]
    for kk in sorted({key(r) for r in runs}):
        R = [r for r in runs if key(r) == kk]
        fr = [r["final_rms"] for r in R if r["final_rms"] is not None]
        L.append(f"| {kk[0]}{' (no pegs)' if kk[3] else ''} | {kk[1]} | {kk[2]} | {len(R)} | "
                 f"{sum(r['solved'] for r in R)} | {np.median(fr) * 1e3 if fr else float('nan'):.3f} | "
                 f"{np.median([r['rollouts'] for r in R]):.0f} | {sum(r['failures'] for r in R)} |")
    targets = sorted({r["target_name"] for r in runs})
    fig, axes = plt.subplots(len(targets), 2, figsize=(11, 3.4 * len(targets)), squeeze=False)
    for i, t in enumerate(targets):
        # the network runs only (the four methods; dashed: trained without the pegs)
        for r in [r for r in runs if r["target_name"] == t and r["param"] == "net"]:
            z = np.load(out / f"{r['_name']}.npz")
            ok = z["eval_ok"]
            y = np.maximum(1e3 * np.minimum.accumulate(np.where(ok, z["eval_rms"], np.inf)), 1e-2)
            c = COLORS.get((r["grad_method"], r["optimizer"]), INK2)
            ls = "--" if r["trained_without_pegs"] else "-"
            axes[i][0].plot(z["eval_rollouts"], y, color=c, lw=1.0, ls=ls, alpha=0.8)
            axes[i][1].plot(z["eval_time"], y, color=c, lw=1.0, ls=ls, alpha=0.8)
        for ax, xl in zip(axes[i], ("rollouts", "wall time [s]")):
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-2)
            ax.axhline(5.0, color=INK2, lw=0.8, ls=":")
            ax.set_xlabel(xl)
            ax.set_ylabel("best shape RMS so far [mm]")
            ax.set_title(t, loc="left", fontsize=9)
    handles = [plt.Line2D([], [], color=c, label=f"{g}, {o}") for (g, o), c in COLORS.items()]
    handles.append(plt.Line2D([], [], color=INK2, ls="--", label="state, lbfgs, trained without pegs"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "curves.png", dpi=140)
    plt.close(fig)
    L += ["", "![curves](curves.png)", ""]
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nreport -> {out / 'report.md'}")


# ---------------------------------------------------------------- verify

#: What every finished run must have saved (train runs; greedy saves a subset).
_REQ_JSON = ("task", "status", "target_name", "param", "grad_method", "optimizer", "seed", "stop",
             "train_time", "updates", "rollouts", "failures", "init_rms", "final_rms", "final_ok",
             "solved", "admissible", "admissibility", "final_snaps", "final_max_penetration",
             "final_max_strain", "final_max_abs_holonomy", "final_ws_violation", "final_onset",
             "final_touching", "final_force", "iterates", "target")
_REQ_NPZ_TRAIN = ("target_shape", "U_ref", "xi_init", "xi_final", "xi_best", "U_init", "U_final",
                  "q_init", "q_traj_init", "q_traj", "eval_loss", "eval_rms", "eval_ok",
                  "eval_rollouts", "eval_time", "iter_rms", "iter_eval",
                  "init_min_clearance", "init_force", "init_touching", "init_snap", "init_substeps",
                  "init_holonomy", "init_twist", "init_shape", "init_node_err",
                  "final_min_clearance", "final_force", "final_touching", "final_snap",
                  "final_substeps", "final_holonomy", "final_twist", "final_shape", "final_node_err")
_REQ_NPZ_GREEDY = ("target_shape", "U_ref", "U_final", "q_init", "q_traj", "iter_rms", "eval_rms",
                   "eval_ok", "eval_rollouts", "eval_time", "final_min_clearance", "final_force",
                   "final_touching", "final_snap", "final_shape", "final_node_err")


def cmd_verify(args):
    """Check every run in OUT: the summary and arrays are complete and consistent.

    Per run: the required json fields and npz arrays exist; the arrays have
    the shapes of the scene (M steps, nv nodes, n_u controls, P parameters);
    the final shape saved equals the last state of the saved trajectory and
    the recorded final RMS; the figures exist in the run folder (train runs)
    or in OUT (greedy).  Prints one line per run and a summary; exits with an
    error if anything is missing.
    """
    out = Path(args.out)
    bad = 0
    files = sorted(out.glob("*.json"))
    n = 0
    # the scene, for the physical checks of the saved trajectories (no rod
    # through a peg between consecutive states, no self-contact)
    try:
        env_v = build_env(experiment_config(args.config))
        pegs_v = T.pegs_of(env_v)
    except Exception:                                   # noqa: BLE001
        env_v, pegs_v = None, []
    for js in files:
        try:
            s = json.load(open(js))
        except (ValueError, OSError):
            continue
        if not (isinstance(s, dict) and s.get("task") == "shape"):
            continue
        n += 1
        problems = []
        greedy = s.get("optimizer") == "greedy"
        miss = [k for k in _REQ_JSON if k not in s]
        if miss:
            problems.append(f"json misses {miss}")
        npz = js.with_suffix(".npz")
        if not npz.exists():
            problems.append("no npz")
        else:
            z = np.load(npz)
            req = _REQ_NPZ_GREEDY if greedy else _REQ_NPZ_TRAIN
            if s.get("final_ok") is False:
                req = tuple(k for k in req if k not in ("final_shape", "final_node_err"))
            miss = [k for k in req if k not in z.files]
            if miss:
                problems.append(f"npz misses {miss}")
            M, nv = int(s.get("N", 0)), int(s.get("nv", 0))
            if "q_traj" in z.files and not greedy and z["q_traj"].shape[0] != M and s.get("final_ok"):
                problems.append(f"q_traj has {z['q_traj'].shape[0]} steps, expected {M}")
            if "U_final" in z.files and z["U_final"].shape[1] != 12:
                problems.append(f"U_final has {z['U_final'].shape[1]} controls")
            if not greedy and "xi_final" in z.files and z["xi_final"].size != int(s.get("P", -1)):
                problems.append(f"xi has {z['xi_final'].size} entries, P = {s.get('P')}")
            if "final_shape" in z.files and "q_traj" in z.files and len(z["q_traj"]):
                X = z["q_traj"][-1][:3 * nv].reshape(nv, 3)
                if np.abs(X - z["final_shape"]).max() > 1e-12:
                    problems.append("final_shape != last state of q_traj")
                nodes = np.arange(2, nv - 2)
                rms = float(np.sqrt(np.mean(np.sum((X[nodes] - z["target_shape"][nodes]) ** 2, axis=1))))
                if s.get("final_rms") is not None and abs(rms - s["final_rms"]) > 1e-9:
                    problems.append(f"final_rms {s['final_rms']} != recomputed {rms}")
            if "final_touching" in z.files and z["final_touching"].shape[1] != len(pegs_v or [0, 0]):
                problems.append(f"final_touching shape {z['final_touching'].shape}")
            if env_v is not None and "q_traj" in z.files and len(z["q_traj"]) and \
                    z["q_init"].size == env_v.n_dof:
                Q = np.vstack([z["q_init"][None, :], z["q_traj"]])
                tun = [k for k in range(1, len(Q)) for p in pegs_v if p.crossed(Q[k - 1], Q[k])]
                if tun:
                    problems.append(f"rod passes through a peg at steps {sorted(set(tun))[:5]}")
                sc = min(T.self_clearance(env_v.positions(q), env_v.R_outer) for q in Q)
                if sc < 0:
                    problems.append(f"self-contact ({sc * 1e3:.1f} mm)")
        # figures: next to the run (greedy, `figure`) or in its run folder (train)
        figs = list(out.glob(f"{js.stem}_shape.pdf"))
        figs += list(Path(_ROOT / "experiments").glob(f"*/runs/*{js.stem}/{js.stem}_shape.pdf"))
        if s.get("final_ok") and not figs:
            problems.append("no shape figure (run `figure --npz` to make one)")
        bad += bool(problems)
        print(f"{'ok  ' if not problems else 'FAIL'} {js.stem}: final rms "
              f"{'-' if s.get('final_rms') is None else '%.3f mm' % (1e3 * s['final_rms'])}, "
              f"solved {s.get('solved')}, {s.get('rollouts')} rollouts, stop: {s.get('stop')}"
              + ("" if not problems else "\n     " + "; ".join(problems)))
    print(f"{n} runs, {bad} with problems")
    if bad:
        raise SystemExit(1)


# ---------------------------------------------------------------- main

def main():
    cmds = ("train", "targets", "check", "jobs", "sweep", "greedy", "report", "figure", "verify")
    argv = sys.argv[1:]
    cmd = argv.pop(0) if argv and argv[0] in cmds else "train"
    ap = argparse.ArgumentParser(description=f"3D shape formation against pegs ({cmd})")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", required=cmd != "figure", help="experiment directory (targets/ lives in it)")
    if cmd in ("targets", "jobs", "sweep"):
        ap.add_argument("--targets", nargs="+", default=None)
    if cmd == "sweep":
        ap.add_argument("--param", choices=T.PARAMETERIZATIONS, default="net")
        ap.add_argument("--grad-method", dest="grad_method", choices=GRAD_METHODS, default="proxy")
        ap.add_argument("--pick", action="store_true", help="decide from the finished sweep runs")
    if cmd == "jobs":
        ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        ap.add_argument("--optimizers", nargs="+", choices=("adam", "lbfgs"), default=["adam", "lbfgs"])
        ap.add_argument("--grad-methods", nargs="+", choices=GRAD_METHODS, default=list(GRAD_METHODS))
        ap.add_argument("--param", choices=T.PARAMETERIZATIONS, default="net")
        ap.add_argument("--train-without-pegs", action="store_true")
        ap.add_argument("--baselines", action="store_true",
                        help="also direct rates, no obstacles, end pose (tuned, aimed) and greedy")
    if cmd in ("train", "check"):
        ap.add_argument("--target", required=True)
        ap.add_argument("--seed", type=int, default=0, help="initialization seed")
        ap.add_argument("--param", choices=T.PARAMETERIZATIONS, default=None)
        ap.add_argument("--init", choices=("designed", "random"), default=None)
    if cmd == "train":
        ap.add_argument("--optimizer", choices=T.OPTIMIZERS, default=None)
        ap.add_argument("--grad-method", dest="grad_method", choices=GRAD_METHODS, default=None)
        ap.add_argument("--epochs", type=int, default=None, help="update cap (train.epochs)")
        ap.add_argument("--max-rollouts", dest="max_rollouts", type=int, default=None)
        ap.add_argument("--stop-rms", dest="stop_rms", type=float, default=None,
                        help="[m] stop once an admissible rollout is this close (train.stop_rms)")
        ap.add_argument("--lr", type=float, default=None, help="Adam rate (train.adam.lr)")
        ap.add_argument("--workspace", choices=("reject", "record"), default=None)
        ap.add_argument("--train-without-pegs", action="store_true",
                        help="'No obstacles' baseline: train without the pegs, evaluate with them")
        ap.add_argument("--tag", default="", help="suffix of the run name (e.g. _lr1e-3 in a sweep)")
        ap.add_argument("--window", type=int, default=20,
                        help="--param endpose: steps over which the final pose is moved")
        ap.add_argument("--aim", action="store_true",
                        help="--param endpose: aim the end pose at the designed one (use --optimizer none)")
    if cmd == "greedy":
        ap.add_argument("--target", required=True)
        ap.add_argument("--cap", type=float, default=None,
                        help="step cap (normalized); default: the largest step of our L-BFGS solutions")
        ap.add_argument("--max-steps", "--epochs", dest="max_steps", type=int, default=None,
                        help="cap on greedy steps (greedy.max_steps); --epochs is an alias")
        ap.add_argument("--log-every", dest="log_every", type=int, default=10)
        ap.add_argument("--tag", default="")
    if cmd == "check":
        ap.add_argument("--dirs", type=int, default=4)
        ap.add_argument("--h", type=float, default=1e-6, help="central-difference step in xi")
        ap.add_argument("--tol-factor", dest="tol_factor", type=float, default=1e-3)
        ap.add_argument("--no-pegs-too", dest="no_pegs_too", action="store_true")
    if cmd == "figure":
        ap.add_argument("--npz", required=True)
        ap.add_argument("--stem", default=None)
        ap.add_argument("--view", type=float, nargs=2, default=(22.0, -58.0), metavar=("ELEV", "AZIM"))
        ap.add_argument("--snapshots", type=int, default=0)
        ap.add_argument("--workspace", choices=("auto", "fit", "clip", "off"), default="off")
        ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args(argv)
    {"targets": cmd_targets, "check": cmd_check, "jobs": cmd_jobs, "sweep": cmd_sweep, "greedy": cmd_greedy,
     "figure": cmd_figure, "verify": cmd_verify,
     "report": lambda a: report(a.out, a.config), "train": cmd_train}[cmd](args)


if __name__ == "__main__":
    main()
