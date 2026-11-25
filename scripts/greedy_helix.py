"""Greedy shape control of the 3D rod into a one-turn helix (no neural control).

The left end is clamped at the origin with tangent +x and never moves; the
right gripper (x, y, z, theta, phi, yaw) is driven by the greedy controller of
:mod:`greedy_control`: at every step a Levenberg-Marquardt direction from the
implicit shape sensitivity, a few trial step lengths taken as real
quasi-static steps, and the trial with the lowest shape error kept.

Target
------
A helix of ``task.helix.turns`` turns over the whole rod, leaving the clamp
along its tangent.  With ``h = pitch_ratio`` (the cosine of the angle between
tangent and axis) the axis is ``a = (h, sqrt(1 - h^2), 0)`` -- horizontal --
and for a rod of length L::

    radius = L sqrt(1 - h^2) / (2 pi turns),      rise per turn = h L / turns

After one full turn the tangent is back to +x, so the gripper ends where the
helix ends, ``h L a``, with the start's tangent; the turn itself has to come
from the path (the tangent sweeping the cone about ``a``) and the yaw.

Schedules (``greedy.schedule`` or ``--schedule``): ``direct`` aims at the
full helix from the first step; ``homotopy`` aims at the partial helix of
``lam`` turns (same pitch angle), ``lam`` advancing from 0 to 1 whenever the
rod is within ``greedy.advance_tol`` of it.

Usage
-----
    python scripts/greedy_helix.py run --config configs/task_helix_3d.yaml --out experiments/greedy_helix
    python scripts/greedy_helix.py run --config configs/task_helix_3d.yaml --out DIR --schedule direct
    python scripts/greedy_helix.py run --config configs/task_helix_3d.yaml --out DIR --hand left --reg-yaw 1e-6
    python scripts/greedy_helix.py report --out DIR

Each run writes ``<name>.json`` (summary), ``<name>.npz`` (every accepted
step's control and state), ``<name>_shape`` (initial, intermediate and final
rod against the helix), ``<name>_error`` (rms error per step) and a gif.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from blas_threads import limit_blas_threads
limit_blas_threads()

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from config import load_config
from env_factory import (build_env, reset_env, settle_initial, env_dim, env_twist,
                         initial_control, end_control_dim, material_summary)
from greedy_control import greedy_shape_control, SCHEDULES
from rodframes import holonomy
from utils import maybe_save_trajectory_gif, paper_scene
import tro_style as ts

DEFAULT_CONFIG = _ROOT / "configs" / "task_helix_3d.yaml"
HANDS = ("right", "left")


# ---------------------------------------------------------------- target

def helix_shape(nv, L, h, turns=1.0, hand="right"):
    """Nodes ``(nv, 3)`` of a unit-speed helical arc of ``turns`` turns.

    Starts at the origin with tangent +x; axis ``a = (h, s, 0)``,
    ``s = sqrt(1 - h^2)``.  The tangent is ``h a + s (cos p e1 + sin p e2)``
    with ``(e1, e2, a)`` a right-handed frame, ``e2 = +z`` and ``p = pi`` at
    the start: ``p`` increasing along the rod is a right-handed helix (it
    first dips below the clamp), decreasing a left-handed one (rises first).
    ``turns = 0`` is the straight rod along +x.
    """
    if hand not in HANDS:
        raise ValueError(f"handedness must be one of {HANDS}, got {hand!r}")
    s = np.sqrt(1.0 - h * h)
    a, e1, e2 = np.array([h, s, 0.0]), np.array([-s, h, 0.0]), np.array([0.0, 0.0, 1.0])
    S = np.linspace(0.0, L, nv)
    w = (1.0 if hand == "right" else -1.0) * 2.0 * np.pi * turns / L
    if abs(w) < 1e-12:
        return S[:, None] * np.array([1.0, 0.0, 0.0])
    p = np.pi + w * S
    c = (np.sin(p) - np.sin(np.pi)) / w            # int_0^S cos p
    d = (np.cos(np.pi) - np.cos(p)) / w            # int_0^S sin p
    return (h * S)[:, None] * a + s * (c[:, None] * e1 + d[:, None] * e2)


def helix_params(cfg, args):
    hx = dict(cfg.task.helix)
    if args.pitch is not None:
        hx["pitch_ratio"] = args.pitch
    if args.hand is not None:
        hx["handedness"] = args.hand
    if args.turns is not None:
        hx["turns"] = args.turns
    h = float(hx["pitch_ratio"])
    if not 0.0 < h < 1.0:
        raise SystemExit(f"task.helix.pitch_ratio must be in (0, 1), got {h}")
    return h, float(hx["turns"]), str(hx["handedness"])


# ---------------------------------------------------------------- stable twist

def stable_yaw_scan(env, target, yaws):
    """Equilibria at the target's end pose, solved from the target itself, per right-gripper yaw.

    The rod is put exactly on ``target`` and the right gripper at its end
    pose (tangent of the last edge) with each yaw in ``yaws``; one Newton
    solve from there.  Returns ``[(yaw, rms to target, stable, total twist)]``
    for the solves that converged.  A multi-turn helix is a stable
    equilibrium only within a window of twist (with the helix's chirality),
    which this finds; the environment's state is clobbered.
    """
    from newton import is_stable
    X = np.asarray(target, dtype=float)
    t = (X[-1] - X[-2]) / np.linalg.norm(X[-1] - X[-2])
    th, ph = np.arctan2(t[1], t[0]), np.arcsin(np.clip(t[2], -1.0, 1.0))
    rows = []
    for yaw in yaws:
        u = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, *X[-1], th, ph, yaw])
        q = np.zeros(env.n_dof)
        q[:env.n_pos] = X.ravel()
        q[env.n_pos:] = [0.0, yaw]
        env.q, env.u, env.u_prev = q, np.zeros_like(q), u.copy()
        q1, flag, J, _ = env._solve(q, env._control_node_from_pose(u))
        if flag < 0:
            continue
        rms = float(np.sqrt(np.mean(np.sum((env.positions(q1) - X) ** 2, axis=1))))
        rows.append((float(yaw), rms, bool(is_stable(J, env.free_index)),
                     float(env.energy.total_twist(q1))))
    return rows


def pick_stable_yaw(rows, slack=1.2):
    """The smallest |yaw| among the stable solutions within ``slack`` x the best stable rms."""
    st = [r for r in rows if r[2]]
    if not st:
        return None
    best = min(r[1] for r in st)
    return min((r for r in st if r[1] <= slack * best), key=lambda r: abs(r[0]))


# ---------------------------------------------------------------- run

def run_name(schedule, h, turns, hand, reg_yaw, seed=0):
    name = f"{schedule}_h{h:g}_{hand}"
    if turns != 1.0:
        name += f"_t{turns:g}"
    if reg_yaw:
        name += f"_ryaw{reg_yaw:g}"
    if seed:
        name += f"_s{seed}"
    return name


def cmd_run(args):
    cfg = load_config(args.config)
    if env_dim(cfg) != 3:
        raise SystemExit("greedy_helix needs env.dim: 3")
    twist = env_twist(cfg)
    g = dict(cfg.greedy)
    if args.schedule is not None:
        g["schedule"] = args.schedule
    if args.max_steps is not None:
        g["max_steps"] = args.max_steps
    if args.time_limit is not None:
        g["time_limit"] = args.time_limit
    if args.nv is not None:
        cfg["rod"]["nv"] = args.nv
    k = end_control_dim(3, twist)
    if args.seed:
        # A distinct start for the same target: the right gripper's start
        # pose perturbed by 1 cm / 0.02 rad (sd), yaw untouched; seed 0 = as configured.
        key = "u_right_twist" if twist else "u_right_3d"
        rng = np.random.default_rng(args.seed)
        u_r = np.array(cfg["start"][key], dtype=float)
        u_r[:3] += rng.normal(0.0, 0.01, 3)
        u_r[3:5] += rng.normal(0.0, 0.02, 2)
        cfg["start"][key] = u_r.tolist()
    reg = np.zeros(k) if g.get("reg") is None else np.array(g["reg"], dtype=float)
    if args.reg_yaw is not None:
        if not twist:
            raise SystemExit("--reg-yaw needs rod.twist: true")
        reg[5] = args.reg_yaw
    g["reg"] = reg.tolist()
    h, turns, hand = helix_params(cfg, args)

    yaw_scan = None
    if args.yaw_target is not None:
        if not twist:
            raise SystemExit("--yaw-target needs rod.twist: true")
        if args.yaw_target == "auto":
            scan_env = build_env(cfg)
            reset_env(scan_env, cfg)
            tgt = helix_shape(scan_env.nv, scan_env.rod_length, h, turns, hand)
            lo, hi, n = args.yaw_scan
            yaws = np.linspace(lo, hi, int(n))
            rows = stable_yaw_scan(scan_env, tgt, yaws)
            pick = pick_stable_yaw(rows)
            st = [r for r in rows if r[2]]
            print(f"Yaw scan at the target's end pose: {len(rows)} of {len(yaws)} converged, "
                  f"{len(st)} stable" + (f"; stable near-helix rms {1e3 * min(r[1] for r in st):.1f} mm"
                                         if st else ""))
            if pick is None:
                raise SystemExit("no stable equilibrium at the target's end pose for any scanned yaw")
            yaw_target = pick[0]
            yaw_scan = {"rows": rows, "pick": pick}
            print(f"  -> yaw target {yaw_target:+.2f} rad (rms {1e3 * pick[1]:.1f} mm, "
                  f"total twist {pick[3]:+.2f} rad)")
        else:
            yaw_target = float(args.yaw_target)
        g["ref_final"] = [None] * 5 + [yaw_target]
    if args.retreats is not None:
        g["max_retreats"] = args.retreats

    env = build_env(cfg)
    reset_env(env, cfg)
    info0 = settle_initial(env, cfg)
    if info0 is None:
        raise SystemExit("greedy_helix needs newton.initial_equilibrium: true (it starts from "
                         "an equilibrium and its Jacobian)")
    u0 = initial_control(cfg)
    q_init = env.q.copy()
    L, nv = env.rod_length, env.nv
    target_fn = lambda lam: helix_shape(nv, L, h, lam * turns, hand)   # noqa: E731
    target = target_fn(1.0)
    H_target = holonomy(target)
    cols = np.arange(k, 2 * k)                                # the right gripper
    angle_cols = cols[3:]                                     # theta, phi (, yaw)

    name = args.name or run_name(g["schedule"], h, turns, hand, reg[5] if twist else 0.0, args.seed)
    if args.yaw_target is not None and not args.name:
        name += "_yawff"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = np.sqrt(1 - h * h)
    print(f"Task: greedy shape control into a helix  ({name})")
    print(f"Rod: {material_summary(cfg)}, nv={nv}, twist {'on' if twist else 'off'}, "
          f"workspace {'on' if cfg.workspace.enabled else 'off'}")
    print(f"Helix: {turns:g} turn(s), {hand}-handed, pitch ratio {h:g} -> radius "
          f"{1e3 * L * s / (2 * np.pi * turns):.1f} mm, rise {h * L:.3f} m, end at "
          f"{np.round(target[-1], 3)}; Bishop holonomy {H_target:+.3f} rad")
    if abs(H_target) > 0.85 * np.pi:
        print(f"WARNING: the target's holonomy {H_target:+.2f} is close to the wrap at +-pi of the "
              f"twist model (src/rodframes.py); take pitch_ratio > 0.5 with some margin")
    print(f"Greedy: schedule {g['schedule']}, step cap {g['step_cap']:g}, alphas {g['alphas']}, "
          f"reg {np.round(reg, 9).tolist()}, max {g['max_steps']} steps")
    print(f"Start: rms to helix {1e3 * np.sqrt(np.mean(np.sum((env.positions() - target) ** 2, 1))):.1f} mm")

    env.n_newton_failures = env.n_unstable = env.n_snaps = 0
    res = greedy_shape_control(env, u0, info0["J"], target_fn, cols, angle_cols, opts=g)
    X = env.positions()
    err = np.linalg.norm(X - target, axis=1)
    du = np.diff(res["u"][:, cols], axis=0)
    summary = {
        "name": name, "config": str(args.config), "schedule": g["schedule"], "greedy": g,
        "helix": {"pitch_ratio": h, "turns": turns, "handedness": hand,
                  "radius": L * s / (2 * np.pi * turns), "holonomy": H_target},
        "nv": int(nv), "twist": twist, "seed": int(args.seed), "u0": u0.tolist(),
        "stop": res["stop"], "steps": res["steps"], "iterations": res["iterations"],
        "lam_final": res["lam_final"], "time": res["time"],
        "rms_start": float(res["rms_final"][0]), "rms_final": float(res["rms_final"][-1]),
        "max_err_final": float(err.max()),
        "best_rms": float(res["rms_final"].min()), "best_step": int(res["rms_final"].argmin()),
        "u_final": res["u"][-1].tolist(),
        "gripper_path_length": float(np.linalg.norm(du[:, :3], axis=1).sum()),
        "yaw_final": float(res["u"][-1][cols[5]]) if twist else None,
        "twist_final": float(res["twist"][-1]) if twist else None,
        "holonomy_final": float(res["holonomy"][-1]) if twist else None,
        "twist_energy_final": float(0.5 * env.energy.K * res["twist"][-1] ** 2) if twist else None,
        "elastic_energy_final": float(env.energy.energy(env.q)),
        "snaps": int(np.sum(res["snap"])), "snap_trials": int(res["snap_trials"]),
        "retreats": int(res["retreats"]), "yaw_scan": yaw_scan,
        "rejected": res["rejected"],
        "newton_failures": int(env.n_newton_failures), "unstable_steps": int(env.n_unstable),
    }
    json.dump(summary, open(out / f"{name}.json", "w"), indent=1, default=float)
    np.savez_compressed(out / f"{name}.npz", target=target, q_init=q_init,
                        **{key: res[key] for key in ("step", "lam", "u", "q", "objective",
                                                     "rms_target", "rms_final", "holonomy",
                                                     "twist", "snap", "substeps", "alpha", "cap",
                                                     "trials")})

    # figures
    n = len(res["q"])
    snaps = [res["q"][i] for i in (n // 4, n // 2, 3 * n // 4)] if n >= 8 else []
    paper_scene(out / f"{name}_shape", env, rod_init=q_init, rod_final=res["q"][-1], rod_snaps=snaps,
                rod_labels=("Initial rod", "Final shape"),
                curves=[(target, "reference", "Target helix")], frames_q=res["q"][-1],
                title=f"{name}: rms {1e3 * summary['rms_final']:.1f} mm", workspace="off")
    plot_error(res, out / f"{name}_error", name)
    plot_axis_view(target, X, h, out / f"{name}_axis", "down the helix axis")
    maybe_save_trajectory_gif(cfg, list(res["q"]), env, None, save_path=str(out),
                              filename=f"{name}.gif", title=f"greedy helix {name}",
                              gripper_node=nv - 1, target_shape=target)
    print(f"[{name}] stop: {res['stop']} after {res['steps']} steps "
          f"({res['iterations']} iterations, lam {res['lam_final']:.2f}, {res['time']:.0f}s)")
    print(f"[{name}] rms to helix {1e3 * summary['rms_start']:.1f} -> {1e3 * summary['rms_final']:.1f} mm "
          f"(max node {1e3 * summary['max_err_final']:.1f} mm), gripper path {summary['gripper_path_length']:.2f} m"
          + (f", yaw {summary['yaw_final']:+.2f} rad, total twist {summary['twist_final']:+.2f} rad "
             f"({summary['twist_energy_final']:.2f} of {summary['elastic_energy_final']:.2f} J elastic)"
             if twist else ""))
    print(f"[{name}] snaps {summary['snaps']} kept / {summary['snap_trials']} trials, rejected trials {res['rejected']}, "
          f"newton failures {summary['newton_failures']}, unstable {summary['unstable_steps']}")
    print(f"[{name}] -> {out}")


def axis_frame(h):
    """``(a, e1, e2)`` of :func:`helix_shape`: the helix axis and the plane across it."""
    s = np.sqrt(1.0 - h * h)
    return np.array([h, s, 0.0]), np.array([-s, h, 0.0]), np.array([0.0, 0.0, 1.0])


def plot_axis_view(target, final, h, stem, title=None):
    """The rod seen down the helix axis: one full turn is a closed circle."""
    _, e1, e2 = axis_frame(h)
    proj = lambda X: (np.asarray(X) @ e1, np.asarray(X) @ e2)   # noqa: E731
    with ts.style():
        fig, ax = plt.subplots(figsize=(0.5 * ts.COL_W + 0.3, 0.5 * ts.COL_W + 0.4))
        ax.plot(*proj(target), **ts.ROLE["reference"], label="Target helix")
        ax.plot(*proj(final), **ts.ROLE["achieved"], label="Final shape")
        ax.plot(*proj(final[-1:]), "o", color=ts.ROLE["achieved"]["color"], markersize=3)
        ax.plot(0.0, 0.0, "s", color="#52514e", markersize=3)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$e_1$ (m)")
        ax.set_ylabel(r"$z$ (m)")
        if title:
            ax.set_title(title, fontsize=ts.FS_TICK + 1)
        ax.legend(fontsize=ts.FS_TICK - 1, frameon=False, loc="best")
        ts.style_2d(ax)
        ts.save(fig, stem)
        plt.close(fig)


def plot_error(res, stem, name):
    """rms to the final helix and to the current target per step, with lam."""
    with ts.style():
        fig, ax = plt.subplots(figsize=(ts.COL_W, 1.7))
        ax.plot(res["step"], 1e3 * res["rms_final"], **ts.ROLE["achieved"], label="to helix")
        ax.plot(res["step"], 1e3 * res["rms_target"], **ts.ROLE["reference"], label="to current target")
        ax.set_xlabel("Greedy step")
        ax.set_ylabel("RMS error (mm)")
        ax.set_yscale("log")
        ax.set_title(name)
        ax2 = ax.twinx()
        ax2.plot(res["step"], res["lam"], color="#9a9893", linewidth=0.8, linestyle=":")
        ax2.set_ylabel(r"$\lambda$")
        ax2.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=ts.FS_TICK, frameon=False)
        ts.style_2d(ax)
        ts.save(fig, stem, width=ts.COL_W)
        plt.close(fig)


# ---------------------------------------------------------------- report

def cmd_report(args):
    out = Path(args.out)
    runs = [json.load(open(p)) for p in sorted(out.glob("*.json"))]
    if not runs:
        raise SystemExit(f"no runs in {out}")
    head = (f"{'run':32s} {'start':>7s} {'final':>7s} {'max':>7s} {'steps':>6s} {'lam':>5s} "
            f"{'yaw':>7s} {'twist':>7s} {'E_tw':>6s} {'snaps':>5s} {'time':>6s}  stop")
    print(head)
    print("-" * len(head))
    for r in runs:
        tw = r.get("twist")
        print(f"{r['name']:32s} {1e3 * r['rms_start']:7.1f} {1e3 * r['rms_final']:7.1f} "
              f"{1e3 * r['max_err_final']:7.1f} {r['steps']:6d} {r['lam_final']:5.2f} "
              + (f"{r['yaw_final']:+7.2f} {r['twist_final']:+7.2f} {r['twist_energy_final']:6.2f} "
                 if tw else f"{'':7s} {'':7s} {'':6s} ")
              + f"{r['snaps']:5d} {r['time']:6.0f}  {r['stop']}")
    print("(rms / max node error in mm; yaw, total twist in rad; twist energy in J; time in s)")
    with ts.style():
        fig, ax = plt.subplots(figsize=(ts.COL_W, 2.0))
        for r in runs:
            d = np.load(out / f"{r['name']}.npz")
            ax.plot(d["step"], 1e3 * d["rms_final"], linewidth=1.0, label=r["name"])
        ax.set_xlabel("Greedy step")
        ax.set_ylabel("RMS error to helix (mm)")
        ax.set_yscale("log")
        ax.legend(fontsize=ts.FS_TICK - 1, frameon=False)
        ts.style_2d(ax)
        ts.save(fig, out / "report_error", width=ts.COL_W)
        plt.close(fig)
    for r in runs:                                # axis views, re-rendered from the data
        d = np.load(out / f"{r['name']}.npz")
        pos = lambda q: np.asarray(q)[:3 * r["nv"]].reshape(-1, 3)   # noqa: E731
        plot_axis_view(d["target"], pos(d["q"][-1]), r["helix"]["pitch_ratio"],
                       out / f"{r['name']}_axis", f"{r['name']}: {1e3 * r['rms_final']:.1f} mm")
    print(f"figures -> {out / 'report_error'}.pdf/.svg/.png, <run>_axis per run")


def main():
    ap = argparse.ArgumentParser(description="greedy shape control into a one-turn helix")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one greedy run")
    r.add_argument("--config", default=str(DEFAULT_CONFIG))
    r.add_argument("--out", required=True)
    r.add_argument("--schedule", choices=SCHEDULES, default=None, help="default: greedy.schedule")
    r.add_argument("--pitch", type=float, default=None, help="default: task.helix.pitch_ratio")
    r.add_argument("--hand", choices=HANDS, default=None, help="default: task.helix.handedness")
    r.add_argument("--turns", type=float, default=None, help="default: task.helix.turns")
    r.add_argument("--reg-yaw", type=float, default=None,
                   help="penalty on the gripper yaw [m^2/rad^2]; default: greedy.reg[5]")
    r.add_argument("--max-steps", type=int, default=None, help="default: greedy.max_steps")
    r.add_argument("--time-limit", type=float, default=None, help="[s] default: greedy.time_limit")
    r.add_argument("--nv", type=int, default=None, help="default: rod.nv")
    r.add_argument("--yaw-target", default=None,
                   help="final yaw of the yaw reference (rad), or 'auto': scan for the yaw that "
                        "makes the target a stable equilibrium; the reference ramps with lam")
    r.add_argument("--retreats", type=int, default=None, help="default: greedy.max_retreats")
    r.add_argument("--yaw-scan", type=float, nargs=3, metavar=("MIN", "MAX", "N"),
                   default=(-16 * np.pi, 16 * np.pi, 65),
                   help="yaws [rad] scanned by --yaw-target auto (default -16pi 16pi 65)")
    r.add_argument("--seed", type=int, default=0,
                   help="perturb the right gripper's start pose (0: as configured)")
    r.add_argument("--name", default=None)
    p = sub.add_parser("report", help="table and error curves of the runs in --out")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
