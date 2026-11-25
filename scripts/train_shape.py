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
    C=configs/task_shape_pegs_3d.yaml
    python scripts/train_shape.py targets --config $C --out DIR [--targets P1]
    python scripts/train_shape.py check   --config $C --out DIR --target P1 [--param rates]
    python scripts/train_shape.py train   --config $C --out DIR --target P1 --optimizer lbfgs --seed 0
    python scripts/train_shape.py jobs    --config $C --out DIR --targets P1 --seeds 0 1 2 3 4
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
    """``{name: (description, du)}`` of ``task.shape_targets``."""
    named = (cfg.get("task") or {}).get("shape_targets") or {}
    if not named:
        raise SystemExit("this config has no task.shape_targets (see configs/task_shape_pegs_3d.yaml)")
    return {name: (t.get("description", f"straight move du = {list(t['du'])}"),
                   np.array(t["du"], dtype=float)) for name, t in named.items()}


def check_names(names, paths):
    bad = [n for n in (names or []) if n not in paths]
    if bad:
        raise SystemExit(f"unknown target(s) {bad}; this config has {sorted(paths)}")
    return list(names or paths)


def designed_commands(u0, du, M):
    """The designed straight move: ``u_k = u0 + (k+1)/M * du`` on the gripper."""
    k_end = len(u0) // 2
    U = np.repeat(u0[None, :], M, axis=0)
    U[:, k_end:] += (np.arange(1, M + 1) / M)[:, None] * du[None, :]
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
        desc, du = paths[name]
        U = designed_commands(u0, du, M)
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
            "start_clearance_ge_30mm": bool(min(start_clear) >= 0.03),
            "no_contact_first_30pct": bool(all(o is not None and o >= int(0.3 * M) for o in onset)),
            "both_touch_at_end": bool(all(res["final_touching"])),
            "max_penetration_le_0.5mm": bool(res["max_penetration"] <= 5e-4),
            "no_cap_contact": not res["end_contact"],
            "no_snap": res["snaps"] == 0,
        }
        rec_json = {"name": name, "description": desc, "du": du.tolist(), "N": M, "nv": int(env.nv),
                    "config": str(args.config), "contact_model": cfg.peg.get("model", "smooth"),
                    "smoothing": float(cfg.peg.get("smoothing", 1e-4)),
                    "snap_steps": res["snap_steps"], "max_substeps": res["max_substeps"],
                    "newton_iterations": res["newton_iterations"], "rollout_time": time.time() - t0,
                    "u_final": U[-1].tolist(), "mid_final": shape[env.nv // 2].tolist(),
                    "start_clearance": start_clear, "onset_step": onset,
                    "final_force": res["final_force"], "touching_nodes_final": touching_nodes,
                    "max_penetration": res["max_penetration"], "max_strain": res["max_strain"],
                    "max_abs_holonomy": res["max_abs_holonomy"],
                    "rms_free_vs_target": rms_free,
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
              f"free (no pegs) end shape vs target {('%.1f mm' % (rms_free * 1e3)) if rms_free is not None else 'failed'}"
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


def initial_parameters(cfg, param, u0, k_end, M, U_ref, seed, init):
    if init == "designed":
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
    param = T.make_parameterization(cfg.train.param, M, k, hidden=int(cfg.train.get("hidden", 64)),
                                    seed=seed)
    xi0, init_info = initial_parameters(cfg, param, u0, k_end, M, U_ref, seed, cfg.train.init)
    return dict(env=env, checkpoint=checkpoint, env_eval=env_eval, cp_eval=cp_eval,
                target_shape=target_shape, U_ref=U_ref, target_rec=target_rec, u0=u0, k_end=k_end,
                k=k, M=M, param=param, xi0=xi0, init_info=init_info, nodes=T.free_nodes(env_eval),
                ws=workspace_bounds(cfg, k))


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
              "in_workspace": r.get("ws_violation", 0.0) <= 1e-9,
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
    name = f"{args.target}_{tr.param}_{gm}_{optimizer}{variant}_s{args.seed}"
    run = RunLogger(cfg, env=env, root=_ROOT, tag=name,
                    enabled=bool((cfg.get("output") or {}).get("run_log", True)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run.artifact_dir) if getattr(run, "artifact_dir", None) else out
    run_dir.mkdir(parents=True, exist_ok=True)
    ocfg = opt_settings(cfg, optimizer)

    print(f"Task: 3D shape formation, target {args.target} -- {S['target_rec']['description']}")
    print(f"Trajectory: {tr.param} (P = {S['param'].P}), init {S['init_info']['init']}"
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
        "param": tr.param, "grad_method": gm, "optimizer": optimizer, "optimizer_settings": ocfg,
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
    out = Path(args.out)
    cfg = experiment_config(args.config)
    names = check_names(args.targets, target_paths(cfg))
    print(f"mkdir -p {out}/logs")
    for t in names:
        for gm in args.grad_methods:
            for o in args.optimizers:
                for s in args.seeds:
                    name = f"{t}_{args.param}_{gm}_{o}{'_nopegs' if args.train_without_pegs else ''}_s{s}"
                    print(f"python scripts/train_shape.py train --config {args.config} --out {out} "
                          f"--target {t} --param {args.param} --grad-method {gm} --optimizer {o} --seed {s}"
                          + (" --train-without-pegs" if args.train_without_pegs else "")
                          + f" > {out}/logs/{name}.log 2>&1")


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
        for r in [r for r in runs if r["target_name"] == t]:
            z = np.load(out / f"{r['_name']}.npz")
            ok = z["eval_ok"]
            y = 1e3 * np.minimum.accumulate(np.where(ok, z["eval_rms"], np.inf))
            c = COLORS.get((r["grad_method"], r["optimizer"]), INK2)
            ls = "--" if r["trained_without_pegs"] else "-"
            axes[i][0].plot(z["eval_rollouts"], y, color=c, lw=1.0, ls=ls, alpha=0.8)
            axes[i][1].plot(z["eval_time"], y, color=c, lw=1.0, ls=ls, alpha=0.8)
        for ax, xl in zip(axes[i], ("rollouts", "wall time [s]")):
            ax.set_yscale("log")
            ax.axhline(5.0, color=INK2, lw=0.8, ls=":")
            ax.set_xlabel(xl)
            ax.set_ylabel("best shape RMS so far [mm]")
            ax.set_title(t, loc="left", fontsize=9)
    handles = [plt.Line2D([], [], color=c, label=f"{g}, {o}") for (g, o), c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "curves.png", dpi=140)
    plt.close(fig)
    L += ["", "![curves](curves.png)", ""]
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nreport -> {out / 'report.md'}")


# ---------------------------------------------------------------- main

def main():
    cmds = ("train", "targets", "check", "jobs", "report", "figure")
    argv = sys.argv[1:]
    cmd = argv.pop(0) if argv and argv[0] in cmds else "train"
    ap = argparse.ArgumentParser(description=f"3D shape formation against pegs ({cmd})")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", required=cmd != "figure", help="experiment directory (targets/ lives in it)")
    if cmd in ("targets", "jobs"):
        ap.add_argument("--targets", nargs="+", default=None)
    if cmd == "jobs":
        ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        ap.add_argument("--optimizers", nargs="+", choices=T.OPTIMIZERS, default=list(T.OPTIMIZERS))
        ap.add_argument("--grad-methods", nargs="+", choices=GRAD_METHODS, default=list(GRAD_METHODS))
        ap.add_argument("--param", choices=T.PARAMETERIZATIONS, default="net")
        ap.add_argument("--train-without-pegs", action="store_true")
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
    if cmd == "check":
        ap.add_argument("--dirs", type=int, default=4)
        ap.add_argument("--h", type=float, default=1e-5, help="central-difference step in xi")
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
    {"targets": cmd_targets, "check": cmd_check, "jobs": cmd_jobs, "figure": cmd_figure,
     "report": lambda a: report(a.out, a.config), "train": cmd_train}[cmd](args)


if __name__ == "__main__":
    main()
