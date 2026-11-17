"""Task 4: 3D shape formation -- every free node of the rod reaches a target shape.

The controller is the adjoint neural control of :mod:`neural_control`, the
same one point-to-point uses; only the loss differs.  The terminal loss is
the mean over the non-clamped nodes of the squared distance to the target
shape (the gripper yaws do not enter it)::

    L = 1/n_free * sum_i |x_i(1) - x_i^*|^2,      i over the free nodes

The targets are reachable by construction: each is the final shape of a
scripted gripper path (``TARGET_PATHS``) rolled out with the full quasi-static
physics (equilibrium at u0 first, continuation, stability check, snap rule),
so a policy that reproduces that path reaches it exactly.  The paths bend the
rod into space with some compression and yaw; ``S3`` compresses the rod into
a downward arch and then rotates the gripper's tangent downward, which flips
the arch through a snap-through, so its reference path snaps on the way.

The experiment compares the two optimizers of ``train.optimizer`` on the same
random initialization: ``adam`` (fixed learning rate, every step accepted)
and ``linesearch`` (the paper's Algorithm 1 -- the Adam direction with a
backtracking step length on full rollouts; a trial is rejected on a Newton
failure or an unstable state, and a snap-through is admitted when it lowers
the loss, this being a terminal objective).  It runs on
configs/example_3d_twist.yaml with the workspace off and the ``state``
gradient, no obstacles.

Usage
-----
    python scripts/train_shape.py targets --out DIR          # build S1, S2, S3
    python scripts/train_shape.py --target S1 --optimizer adam       --seed 0 --out DIR
    python scripts/train_shape.py --target S1 --optimizer linesearch --seed 0 --out DIR
    python scripts/train_shape.py jobs --out DIR             # the 18 commands of the comparison
    python scripts/train_shape.py report --out DIR           # table, curves, shape figures
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from blas_threads import limit_blas_threads
limit_blas_threads()

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim

from config import load_config
from env_factory import (
    build_env, reset_env, settle_initial, env_dim, env_twist, control_dim, end_control_dim,
    initial_control, material_summary,
)
from neural_control import (
    BoundaryVelocityNet, resolve_device, rollout, adjoint_gradient, surrogate_loss,
    apply_update, GRAD_METHODS, OPTIMIZERS, LineSearch, trial_rule, flat_gradient,
    get_parameters, set_parameters,
)
from runlog import RunLogger
from utils import maybe_save_trajectory_gif, draw_obstacles, draw_gripper_frames, gripper_frame_legend

DEFAULT_CONFIG = _ROOT / "configs" / "example_3d_twist.yaml"


# ---------------------------------------------------------------- targets

def _ramp(lam, a, b):
    """0 before ``a``, 1 after ``b``, linear in between."""
    return float(np.clip((lam - a) / (b - a), 0.0, 1.0))


#: The scripted right-gripper paths the targets are the end shapes of: each
#: maps the continuation parameter lambda in (0, 1] to the increment of
#: (x, y, z, theta, phi, yaw) over the initial pose.  All three bend the rod
#: into space (compression plus an offset and a yaw).  S3 first buckles the
#: rod into the downward arch gravity selects and then turns the gripper's
#: tangent downward, which has no stable continuation on that arch: the rod
#: snaps to the upward one, so the reference path contains a snap-through.
TARGET_PATHS = {
    "S1": ("bend: compress 0.25 m, offset +0.15 m in y, +0.05 m in z, yaw +1.5 rad",
           lambda l: np.array([-0.25 * l, 0.15 * l, 0.05 * l, 0.0, 0.0, 1.5 * l])),
    "S2": ("lift: compress 0.20 m, offset -0.12 m in y, +0.12 m in z, tangent turned "
           "(theta +0.4, phi -0.3 rad), yaw -1.2 rad",
           lambda l: np.array([-0.20 * l, -0.12 * l, 0.12 * l, 0.4 * l, -0.3 * l, -1.2 * l])),
    "S3": ("flip: compress 0.30 m (first half), then tangent down to phi = -0.6 rad with "
           "+0.08 m in y and yaw +0.8 rad (second half) -- the arch snaps through",
           lambda l: np.array([-0.30 * _ramp(l, 0.0, 0.5), 0.08 * _ramp(l, 0.5, 1.0), 0.0, 0.0,
                               -0.6 * _ramp(l, 0.5, 1.0), 0.8 * _ramp(l, 0.5, 1.0)])),
}


def experiment_config(path):
    """The experiment's config: the twist example, workspace off, state gradient."""
    cfg = load_config(path)
    cfg["workspace"]["enabled"] = False
    cfg["train"]["grad_method"] = "state"
    return cfg


def free_nodes(env):
    """Indices of the nodes whose three position DOFs are all free."""
    free = set(env.free_index.tolist())
    return np.array([i for i in range(env.nv) if all(3 * i + c in free for c in range(3))])


def shape_loss_terms(env, q_list, target_shape, nodes):
    """``(L, dL/dq_free per step, final shape)`` for the terminal shape loss."""
    X = env.positions(q_list[-1])
    d = X[nodes] - target_shape[nodes]
    L = float(np.mean(np.sum(d * d, axis=1)))
    full = np.zeros(env.n_dof)
    for i, row in zip(nodes, d):
        full[3 * i:3 * i + 3] = 2.0 / len(nodes) * row
    a_free = np.zeros((len(q_list), len(env.free_index)), dtype=float)
    a_free[-1] = full[env.free_index]
    return L, a_free, X


def shape_rms(env, X, target_shape, nodes):
    d = X[nodes] - target_shape[nodes]
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def build_target(cfg, env, checkpoint, u0, name):
    """Roll the scripted path out and record its end shape and what happened on the way."""
    desc, path = TARGET_PATHS[name]
    N = env.Nsteps
    env.set_state(checkpoint)
    U, shapes, steps = [], [], []
    t0 = time.time()
    for k in range(1, N + 1):
        u = u0.copy()
        u[6:] += path(k / N)
        _, _, info = env.step(u)
        if not info["converged"]:
            raise RuntimeError(f"target {name}: Newton failed at step {k} ({info['reason']})")
        if info["stable"] is False:
            raise RuntimeError(f"target {name}: unstable equilibrium at step {k}")
        U.append(u)
        shapes.append(env.positions())
        steps.append({"step": k, "substeps": info["substeps"], "snap": bool(info["snap"]),
                      "iterations": info["newton_iterations"],
                      "snaps": [{k2: float(v) if isinstance(v, (float, np.floating)) else v
                                 for k2, v in s.items()} for s in info["snaps"]]})
    shape = shapes[-1]
    snap_steps = [s["step"] for s in steps if s["snap"]]
    rec = {"name": name, "description": desc, "N": N, "nv": int(env.nv),
           "snap_steps": snap_steps, "max_substeps": max(s["substeps"] for s in steps),
           "newton_iterations": int(sum(s["iterations"] for s in steps)),
           "rollout_time": time.time() - t0,
           "u_final": U[-1].tolist(), "mid_final": shape[env.mid_node].tolist(),
           "extent": {"x": [float(shape[:, 0].min()), float(shape[:, 0].max())],
                      "y": [float(shape[:, 1].min()), float(shape[:, 1].max())],
                      "z": [float(shape[:, 2].min()), float(shape[:, 2].max())]},
           "steps": steps}
    return rec, {"shape": shape, "U_ref": np.array(U), "shapes": np.array(shapes)}


def cmd_targets(args):
    cfg = experiment_config(args.config)
    env = build_env(cfg)
    reset_env(env, cfg)
    settle_initial(env, cfg)
    checkpoint = env.get_state()
    u0 = initial_control(cfg)
    out = Path(args.out) / "targets"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Rod: {material_summary(cfg)}\nHorizon: {env.Nsteps} steps, nv = {env.nv}")
    for name in (args.targets or TARGET_PATHS):
        rec, arrs = build_target(cfg, env, checkpoint, u0, name)
        np.savez_compressed(out / f"{name}.npz", **arrs)
        json.dump(rec, open(out / f"{name}.json", "w"), indent=1, default=float)
        print(f"[{name}] {rec['description']}\n       mid {np.round(rec['mid_final'], 3).tolist()}, "
              f"y in {np.round(rec['extent']['y'], 3).tolist()}, z in {np.round(rec['extent']['z'], 3).tolist()}; "
              f"snaps at steps {rec['snap_steps'] or 'none'}, max substeps {rec['max_substeps']}, "
              f"{rec['newton_iterations']} Newton iterations, {rec['rollout_time']:.0f}s", flush=True)
    plot_targets(out, list(args.targets or TARGET_PATHS))


def plot_targets(out, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5.2 * len(names), 4.8))
    for i, name in enumerate(names):
        z = np.load(out / f"{name}.npz")
        rec = json.load(open(out / f"{name}.json"))
        ax = fig.add_subplot(1, len(names), i + 1, projection="3d")
        S = z["shapes"]
        for k in range(0, len(S), max(1, len(S) // 8)):
            ax.plot(S[k, :, 0], S[k, :, 1], S[k, :, 2], color=INK2, lw=0.6, alpha=0.35)
        ax.plot(z["shape"][:, 0], z["shape"][:, 1], z["shape"][:, 2], color=INK, lw=2.2)
        ax.plot([0], [0], [0], "s", color=INK2, ms=5)
        _axes3d(ax, f"{name}" + (f" (snap at step {rec['snap_steps'][0]})" if rec["snap_steps"] else ""))
    fig.tight_layout()
    fig.savefig(out / "targets.png", dpi=140)
    plt.close(fig)


def load_target(out, name):
    z = np.load(Path(out) / "targets" / f"{name}.npz")
    rec = json.load(open(Path(out) / "targets" / f"{name}.json"))
    return z["shape"], rec


# ---------------------------------------------------------------- training

def train_shape(env, checkpoint, net, u0, target_shape, nodes, cfg, epochs, device, lr,
                run=None, log_every=10, tag=""):
    """Both optimizers on the terminal shape loss; returns ``(hist, final)``.

    ``adam``: the loop of train_point_to_point.py (Adam at ``lr``, gradient
    clipping, every step accepted; a rollout with a Newton failure ends the
    run, since no update could follow).  ``linesearch``: Algorithm 1 through
    :class:`neural_control.LineSearch`, the point-to-point rule (snap
    admitted).  ``hist`` has one entry per epoch: the loss of the rollout at
    the weights before that epoch's update, the wall time, the cumulative
    rollouts, and the line search's tau / rejections / snaps.
    """
    optimizer = cfg.train.optimizer
    N = env.Nsteps
    dlam = 1.0 / N
    n_end = net.out_dim
    gm = cfg.train.grad_method
    hist = {k: [] for k in ("loss", "rms", "time", "rollouts", "tau", "rejected",
                            "rejected_reasons", "snaps", "fallback", "newton_failures",
                            "unstable", "max_substeps", "u_paths")}
    hist["stop"] = None
    t_start = time.time()
    n_rollouts = [0]

    def run_policy():
        env.set_state(checkpoint)
        infos = []
        n_rollouts[0] += 1
        u_list, r_list, q_list, J_list, conv = rollout(env, net, u0, N, dlam, device,
                                                       project=None, infos=infos)
        return u_list, r_list, q_list, J_list, conv, infos

    def record(ep, L, X, infos, u_list, rec=None, fallback=False):
        hist["loss"].append(L)
        hist["rms"].append(shape_rms(env, X, target_shape, nodes))
        hist["time"].append(time.time() - t_start)
        hist["rollouts"].append(n_rollouts[0])
        hist["tau"].append(None if rec is None else rec["tau"])
        hist["rejected"].append(0 if rec is None else rec["n_rejected"])
        hist["rejected_reasons"].append([] if rec is None else [r["reason"] for r in rec["rejected"]])
        hist["snaps"].append(int(sum(1 for i in infos if i.get("snap"))))
        hist["fallback"].append(bool(fallback))
        hist["newton_failures"].append(int(env.n_newton_failures))
        hist["unstable"].append(int(env.n_unstable))
        hist["max_substeps"].append(int(max(i["substeps"] for i in infos)))
        hist["u_paths"].append(np.stack([u.detach().cpu().numpy() for u in u_list]))
        if run is not None:
            run.log(epoch=ep, loss=L, error=hist["rms"][-1], rollouts=n_rollouts[0],
                    tau=hist["tau"][-1], rejected=hist["rejected"][-1], snaps=hist["snaps"][-1],
                    fallback=bool(fallback), newton_failures=env.n_newton_failures,
                    unstable_steps=env.n_unstable, max_substeps=hist["max_substeps"][-1],
                    converged=True, steps=len(u_list),
                    u_final=u_list[-1].detach().cpu().numpy())
        if ep % log_every == 0 or ep == epochs - 1 or (rec is not None and rec["tau"] is None):
            tau = "" if rec is None else f"  tau={'-' if rec['tau'] is None else f'{rec['tau']:g}'}  rejected={rec['n_rejected']}"
            print(f"[{tag} ep {ep:03d}] rms={hist['rms'][-1] * 1e3:7.2f} mm{tau}  "
                  f"rollouts={n_rollouts[0]}  snaps={hist['snaps'][-1]}  "
                  f"t={hist['time'][-1]:7.1f}s", flush=True)

    if optimizer == "adam":
        opt = optim.Adam(net.parameters(), lr=lr)
        for ep in range(epochs):
            opt.zero_grad(set_to_none=True)
            u_list, r_list, q_list, J_list, conv, infos = run_policy()
            if not conv.all():
                hist["stop"] = {"epoch": ep, "reason": f"Newton failed at step {len(u_list)}; "
                                                       "no update follows"}
                print(f"[{tag} ep {ep:03d}] {hist['stop']['reason']}", flush=True)
                if run is not None:
                    run.log(epoch=ep, converged=False, steps=len(u_list),
                            newton_failures=env.n_newton_failures, note=hist["stop"]["reason"])
                break
            L, a_free, X = shape_loss_terms(env, q_list, target_shape, nodes)
            g_seq = adjoint_gradient(env, u_list, J_list, a_free, cfg.train.jac_reg, grad_method=gm)
            apply_update(net, opt, surrogate_loss(u_list, r_list, g_seq, dlam, device, grad_method=gm))
            record(ep, L, X, infos, u_list)
        # the policy after the last update
        u_list, _, q_list, _, conv, infos = run_policy()
        final = {"valid": bool(conv.all()), "unstable_steps": int(sum(1 for i in infos if i.get("stable") is False)),
                 "snaps": int(sum(1 for i in infos if i.get("snap"))), "q_list": q_list,
                 "q": q_list[-1]}
        if conv.all():
            L, _, X = shape_loss_terms(env, q_list, target_shape, nodes)
            final.update(loss=L, rms=shape_rms(env, X, target_shape, nodes), shape=X)
        final["U"] = np.stack([u.detach().cpu().numpy() for u in u_list])
        return hist, final, time.time() - t_start

    # ---- linesearch
    ls = LineSearch(net, lr, c=float(cfg.train.linesearch.c),
                    tau_min=float(cfg.train.linesearch.tau_min))

    def evaluate(xi):
        set_parameters(net, xi)
        u_list, r_list, q_list, J_list, conv, infos = run_policy()
        ok, reason = trial_rule(infos, conv, u_list, n_end, None, snap_ok=True)
        tr = {"ok": ok, "reason": reason, "loss": np.inf, "u_list": u_list, "r_list": r_list,
              "q_list": q_list, "J_list": J_list, "conv": conv, "infos": infos}
        if conv.all():
            L, a_free, X = shape_loss_terms(env, q_list, target_shape, nodes)
            tr.update(loss=L, a_free=a_free, shape=X)
        return tr

    def gradient(tr):
        g_seq = adjoint_gradient(env, tr["u_list"], tr["J_list"], tr["a_free"], cfg.train.jac_reg,
                                 grad_method=gm)
        return flat_gradient(net, surrogate_loss(tr["u_list"], tr["r_list"], g_seq, dlam, device,
                                                 grad_method=gm))

    xi = get_parameters(net)
    cur = evaluate(xi)
    if not cur["ok"]:
        hist["stop"] = {"epoch": 0, "reason": f"initial rollout rejected: {cur['reason']}"}
        print(f"[{tag}] the initial rollout is rejected ({cur['reason']}): Algorithm 1 needs a "
              "feasible start, no update is made", flush=True)
    else:
        g = gradient(cur)
        for ep in range(epochs):
            d, fallback = ls.direction(g)
            acc, rec = ls.search(xi, g, cur["loss"], d, evaluate)
            record(ep, cur["loss"], cur["shape"], cur["infos"], cur["u_list"], rec, fallback)
            if acc is None:
                hist["stop"] = {"epoch": ep, "reason": rec["stop"]}
                print(f"[{tag} ep {ep:03d}] line search stopped: {rec['stop']}", flush=True)
                break
            xi = get_parameters(net)
            cur = acc
            g = gradient(cur)
    final = {"valid": bool(cur["conv"].all()),
             "unstable_steps": int(sum(1 for i in cur["infos"] if i.get("stable") is False)),
             "snaps": int(sum(1 for i in cur["infos"] if i.get("snap"))),
             "U": np.stack([u.detach().cpu().numpy() for u in cur["u_list"]]),
             "q_list": cur["q_list"], "q": cur["q_list"][-1]}
    if cur["conv"].all():
        final.update(loss=cur["loss"], rms=shape_rms(env, cur["shape"], target_shape, nodes),
                     shape=cur["shape"])
    return hist, final, time.time() - t_start


def cmd_train(args):
    cfg = experiment_config(args.config)
    if args.optimizer is not None:
        cfg["train"]["optimizer"] = args.optimizer
    optimizer = cfg.train.optimizer
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dim, twist = env_dim(cfg), env_twist(cfg)
    k = end_control_dim(dim, twist)
    env = build_env(cfg)
    reset_env(env, cfg)
    settle_initial(env, cfg)
    checkpoint = env.get_state()
    target_shape, target_rec = load_target(args.out, args.target)
    if target_rec["nv"] != env.nv or target_rec["N"] != env.Nsteps:
        raise SystemExit(f"target {args.target} was built for nv={target_rec['nv']}, "
                         f"N={target_rec['N']}; this config has nv={env.nv}, N={env.Nsteps}")
    nodes = free_nodes(env)
    N = env.Nsteps
    device = resolve_device(cfg.train.device if args.device is None else args.device)
    u0_np = initial_control(cfg)
    u0 = torch.tensor(u0_np, dtype=torch.float32, device=device)
    # The same seed gives both optimizers the same initial policy: nothing
    # consumes the generator between the seed and this line.
    net = BoundaryVelocityNet(hidden=cfg.train.hidden, out_dim=k).to(device)
    lr = cfg.train.lr if args.lr is None else args.lr
    epochs = cfg.train.epochs if args.epochs is None else args.epochs
    name = f"{args.target}_{optimizer}_s{args.seed}"

    run = RunLogger(cfg, env=env, root=_ROOT, tag=name,
                    enabled=bool((cfg.get("output") or {}).get("run_log", True)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Task: 3D shape formation  (adjoint neural control)")
    print(f"Optimizer: {optimizer}  ({'--optimizer' if args.optimizer else 'train.optimizer'})"
          + (f", Armijo c={float(cfg.train.linesearch.c):g}, tau_min={float(cfg.train.linesearch.tau_min):g}"
             if optimizer == "linesearch" else ""))
    print(f"Adjoint gradient: {cfg.train.grad_method}")
    print(f"Environment: {type(env).__name__}  (dim={dim}, twist={twist}, "
          f"control={control_dim(dim, twist)}D), workspace off, no obstacles")
    print(f"Rod material: {material_summary(cfg)}")
    print(f"Target: {args.target} -- {target_rec['description']}"
          + (f"; its reference path snaps at step {target_rec['snap_steps']}" if target_rec["snap_steps"] else ""))
    print(f"Loss: mean over the {len(nodes)} free nodes of |x_i - x_i*|^2")
    print(f"Horizon: {N} steps, single horizon, {epochs} epochs, lr {lr:g}, seed {args.seed}, device {device}")
    nw = cfg.newton
    print(f"Rollout physics: initial equilibrium {'on' if nw.get('initial_equilibrium') else 'off'}, "
          f"continuation {'on (max ' + str(nw.get('max_substeps')) + ' substeps)' if nw.get('continuation') else 'off'}, "
          f"stability check {'on' if nw.get('stability_check') else 'off'}, "
          f"snap rule {'on' if nw.get('snap_rule') else 'off'}")

    env.n_newton_failures = env.n_unstable = env.n_snaps = 0
    hist, final, t_train = train_shape(env, checkpoint, net, u0, target_shape, nodes, cfg, epochs,
                                       device, lr, run=run, tag=name)

    summary = {"task": "shape", "target_name": args.target, "optimizer": optimizer,
               "grad_method": cfg.train.grad_method, "seed": args.seed, "epochs": epochs,
               "epochs_recorded": len(hist["loss"]), "lr": float(lr), "nv": int(env.nv), "N": int(N),
               "n_free_nodes": int(len(nodes)), "train_time": t_train,
               "stop": hist["stop"],
               "final_valid": final["valid"], "final_rms": final.get("rms"), "final_loss": final.get("loss"),
               "best_rms": float(np.nanmin(hist["rms"])) if hist["rms"] else None,
               "best_epoch": int(np.nanargmin(hist["rms"])) if hist["rms"] else None,
               "rollouts_total": int(hist["rollouts"][-1]) if hist["rollouts"] else 0,
               "rejected_total": int(sum(hist["rejected"])),
               "snaps_total": int(env.n_snaps), "snaps_final": final["snaps"],
               "newton_failures_total": int(env.n_newton_failures),
               "unstable_steps_total": int(env.n_unstable), "unstable_steps_final": final["unstable_steps"],
               "target": {k2: target_rec[k2] for k2 in ("description", "snap_steps", "max_substeps")},
               "status": "done"}
    for key in ("loss", "rms", "time", "rollouts", "tau", "rejected", "rejected_reasons", "snaps",
                "fallback", "newton_failures", "unstable", "max_substeps"):
        summary[f"epoch_{key}"] = hist[key]
    arrays = {"target_shape": target_shape, "final_U": final["U"],
              "final_q": np.asarray(final["q"]),
              "epoch_u_paths": np.array(hist["u_paths"], dtype=np.float32)}
    if "shape" in final:
        arrays["final_shape"] = final["shape"]
    np.savez_compressed(out / f"{name}.npz", **arrays)
    json.dump(summary, open(out / f"{name}.json", "w"), indent=1, default=float)

    if "shape" in final:
        plot_shape_result(out / f"{name}.png", target_shape, final["q"], env,
                          f"{name}: rms {final['rms'] * 1e3:.1f} mm")
    # The final rollout, frame by frame, next to the job's png (output.save_gif).
    maybe_save_trajectory_gif(cfg, final["q_list"], env, None, save_path=str(out),
                              filename=f"{name}.gif", title=f"shape formation {name}",
                              gripper_node=env.nv - 1, target_shape=target_shape)
    run.finish(task="shape", target_name=args.target, optimizer=optimizer,
               grad_method=cfg.train.grad_method, final_loss=final.get("loss"),
               final_error=final.get("rms"), best_loss=float(np.nanmin(hist["loss"])) if hist["loss"] else None,
               losses=[float(l) for l in hist["loss"]], epochs_recorded=len(hist["loss"]),
               rollouts_total=summary["rollouts_total"], rejected_total=summary["rejected_total"],
               snaps_total=env.n_snaps, newton_failures_total=env.n_newton_failures,
               unstable_steps_total=env.n_unstable, stop=hist["stop"], device=str(device))
    print(f"[{name}] done: final rms {('%.2f mm' % (final['rms'] * 1e3)) if 'rms' in final else 'n/a'}, "
          f"{summary['rollouts_total']} rollouts, {summary['rejected_total']} rejected, "
          f"{env.n_snaps} snaps, {env.n_unstable} unstable steps, train {t_train:.0f}s", flush=True)


# ---------------------------------------------------------------- figures

#: Fixed categorical slots (dataviz reference palette, slots 1-3, as in
#: train_point_to_point.py): the optimizer keeps its colour in every figure.
COLORS = {"adam": "#2a78d6", "linesearch": "#1baf7a", "reference": "#eb6834"}
LABELS = {"adam": "Adam (fixed lr)", "linesearch": "line search (Algorithm 1)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    from matplotlib.ticker import FuncFormatter, NullFormatter
    if ax.get_yscale() == "log":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def _axes3d(ax, title):
    ax.set_xlabel("x [m]", color=INK2, fontsize=8)
    ax.set_ylabel("y [m]", color=INK2, fontsize=8)
    ax.set_zlabel("z [m]", color=INK2, fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(-0.45, 0.45)
    ax.tick_params(labelsize=7, colors=INK2)
    try:
        ax.set_box_aspect((1.1, 0.8, 0.9))
    except AttributeError:
        pass
    ax.set_title(title, color=INK, fontsize=9)


def plot_shape_result(path, target_shape, q, env, title):
    """Final rod (from the full DOF vector ``q``) against the target, with the
    obstacles and the gripper frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shape = env.positions(q)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(target_shape[:, 0], target_shape[:, 1], target_shape[:, 2], color=INK, lw=2.4, label="target")
    ax.plot(shape[:, 0], shape[:, 1], shape[:, 2], color=COLORS["adam"], lw=1.6, label="final rollout")
    draw_obstacles(ax, env)
    draw_gripper_frames(ax, env, q, 0.1 * env.rod_length)
    gripper_frame_legend(ax, 3)
    _axes3d(ax, title)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- report

def load_runs(out):
    runs = []
    for js in sorted(Path(out).glob("*.json")):
        try:
            s = json.load(open(js))
        except (ValueError, OSError):
            continue
        if not (isinstance(s, dict) and s.get("task") == "shape" and s.get("status") == "done"):
            continue
        npz = js.with_suffix(".npz")
        s["_arrays"] = dict(np.load(npz)) if npz.exists() else {}
        s["_name"] = js.stem
        runs.append(s)
    return runs


def _fmt(v, scale=1.0, nd=1):
    return "-" if v is None or not np.isfinite(v) else f"{v * scale:.{nd}f}"


def _first_below(x, thr):
    for i, v in enumerate(x):
        if v is not None and np.isfinite(v) and v <= thr:
            return i
    return None


def report(out, config=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    runs = load_runs(out)
    if not runs:
        raise SystemExit(f"no shape jobs in {out}")
    # The rod the jobs ran on: the final shapes' gripper frames and the
    # obstacles are drawn from it.
    env = build_env(experiment_config(config))
    targets = sorted({r["target_name"] for r in runs})
    opts = [o for o in ("adam", "linesearch") if any(r["optimizer"] == o for r in runs)]
    N, nv = runs[0]["N"], runs[0]["nv"]
    L = ["# 3D shape formation: Adam versus the line search of Algorithm 1", "",
         f"Rod of {nv} nodes with twist (configs/example_3d_twist.yaml), {N} control steps, workspace "
         "off, no obstacles, `state` gradient. Loss = mean over the free nodes of |x_i - x_i*|^2; "
         "'rms' is its square root. Both optimizers start from the same random policy per seed "
         "(same initialization, learning rate = Adam's). 'epoch k' = the rollout before the k-th "
         "update; 'final' = the policy after the last accepted update. 'rollouts' counts every "
         "rollout, rejected line-search trials included; 'time' is the training wall time. "
         "'unstable steps' counts rollout steps that ended on an unstable equilibrium (the "
         "stability check and snap rule should keep it at 0).", ""]
    L += ["## Targets", ""]
    for t in targets:
        rec = json.load(open(out / "targets" / f"{t}.json"))
        L.append(f"- **{t}**: {rec['description']}. Reference rollout: "
                 + (f"snap-through at step {rec['snap_steps']}" if rec["snap_steps"] else "no snap-through")
                 + f", largest subdivision {rec['max_substeps']} substeps.")
    L += ["", "![targets](targets/targets.png)", ""]

    L += ["## Summary", "",
          "| target | optimizer | seed | start rms [mm] | best rms [mm] (epoch) | final rms [mm] | "
          "epochs | rollouts | rejected trials | snaps (accepted rollouts / all) | Newton failures | "
          "unstable steps | epochs to 10 mm | rollouts to 10 mm | time [s] | stopped |",
          "|" + "---|" * 16]
    rows = []
    for t in targets:
        for o in opts:
            for r in sorted([r for r in runs if r["target_name"] == t and r["optimizer"] == o],
                            key=lambda r: r["seed"]):
                rms = r["epoch_rms"]
                e10 = _first_below(rms, 0.010)
                r10 = r["epoch_rollouts"][e10] if e10 is not None else None
                stop = r.get("stop")
                row = {"target": t, "optimizer": o, "seed": r["seed"],
                       "start_rms": rms[0] if rms else None, "best_rms": r["best_rms"],
                       "best_epoch": r["best_epoch"], "final_rms": r["final_rms"],
                       "epochs": r["epochs_recorded"], "rollouts": r["rollouts_total"],
                       "rejected": r["rejected_total"],
                       "snaps_accepted": int(sum(r["epoch_snaps"])), "snaps_all": r["snaps_total"],
                       "newton_failures": r["newton_failures_total"],
                       "unstable_steps": r["unstable_steps_total"],
                       "epochs_to_10mm": e10, "rollouts_to_10mm": r10,
                       "time": r["train_time"], "stop": (stop or {}).get("reason")}
                rows.append(row)
                L.append(f"| {t} | {LABELS[o]} | {r['seed']} | {_fmt(row['start_rms'], 1e3)} | "
                         f"{_fmt(row['best_rms'], 1e3)} ({row['best_epoch']}) | {_fmt(row['final_rms'], 1e3)} | "
                         f"{row['epochs']} | {row['rollouts']} | {row['rejected']} | "
                         f"{row['snaps_accepted']} / {row['snaps_all']} | {row['newton_failures']} | "
                         f"{row['unstable_steps']} | {e10 if e10 is not None else '-'} | "
                         f"{r10 if r10 is not None else '-'} | {row['time']:.0f} | "
                         f"{'' if not stop else 'ep ' + str(stop['epoch']) + ': ' + stop['reason']} |")
    L.append("")
    L += ["## Per optimizer (median over targets and seeds)", "",
          "| optimizer | runs | final rms [mm] | best rms [mm] | rollouts | rejected trials | "
          "snaps (all) | Newton failures | unstable steps | time [s] |", "|" + "---|" * 10]
    for o in opts:
        R = [x for x in rows if x["optimizer"] == o]
        med = lambda key, sc=1.0: _fmt(float(np.nanmedian([np.nan if x[key] is None else x[key] for x in R])), sc)
        L.append(f"| {LABELS[o]} | {len(R)} | {med('final_rms', 1e3)} | {med('best_rms', 1e3)} | "
                 f"{med('rollouts')} | {med('rejected')} | {sum(x['snaps_all'] for x in R)} | "
                 f"{sum(x['newton_failures'] for x in R)} | {sum(x['unstable_steps'] for x in R)} | {med('time')} |")
    L.append("")

    # ---- curves: loss against iterations, rollouts and wall time, one row per target
    fig, axes = plt.subplots(len(targets), 3, figsize=(14, 3.6 * len(targets)), squeeze=False)
    for i, t in enumerate(targets):
        for ax, (xkey, xlabel) in zip(axes[i], (("epoch", "iteration (update)"),
                                                 ("epoch_rollouts", "rollouts"),
                                                 ("epoch_time", "wall time [s]"))):
            for r in [r for r in runs if r["target_name"] == t]:
                y = np.array(r["epoch_rms"], float) * 1e3
                x = np.arange(len(y)) if xkey == "epoch" else np.array(r[xkey], float)
                ax.plot(x, y, color=COLORS[r["optimizer"]], lw=1.3, alpha=0.85,
                        ls="-" if r["seed"] == 0 else ("--" if r["seed"] == 1 else ":"))
            ax.set_yscale("log")
            ax.set_xlabel(xlabel, color=INK2, fontsize=8)
            ax.set_ylabel("rms shape error [mm]", color=INK2, fontsize=8)
            _style(ax)
        axes[i][0].set_title(f"{t}", color=INK, fontsize=9, loc="left")
    handles = [plt.Line2D([], [], color=COLORS[o], lw=1.6, label=LABELS[o]) for o in opts]
    handles += [plt.Line2D([], [], color=INK2, lw=1.2, ls=s, label=f"seed {k}")
                for k, s in enumerate(("-", "--", ":"))]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "curves.png", dpi=140)
    plt.close(fig)
    L += ["## Convergence", "", "Shape error of the rollout before each update, against the number "
          "of updates, the number of rollouts (rejected trials included) and the wall time. Line "
          "style = seed.", "", "![curves](curves.png)", ""]

    # ---- final shape versus target, per target: best seed of each optimizer
    fig = plt.figure(figsize=(5.4 * len(targets), 5.0))
    for i, t in enumerate(targets):
        ax = fig.add_subplot(1, len(targets), i + 1, projection="3d")
        tgt = None
        for o in opts:
            R = [r for r in runs if r["target_name"] == t and r["optimizer"] == o
                 and "final_shape" in r["_arrays"] and r["final_rms"] is not None]
            if not R:
                continue
            best = min(R, key=lambda r: r["final_rms"])
            tgt = best["_arrays"]["target_shape"]
            S = best["_arrays"]["final_shape"]
            ax.plot(S[:, 0], S[:, 1], S[:, 2], color=COLORS[o], lw=1.6,
                    label=f"{LABELS[o]}, seed {best['seed']}: {best['final_rms'] * 1e3:.1f} mm")
            if "final_q" in best["_arrays"]:
                draw_gripper_frames(ax, env, best["_arrays"]["final_q"], 0.1 * env.rod_length,
                                    linewidth=1.4)
            for r in R:
                if r is not best:
                    S2 = r["_arrays"]["final_shape"]
                    ax.plot(S2[:, 0], S2[:, 1], S2[:, 2], color=COLORS[o], lw=0.8, alpha=0.35)
        if tgt is not None:
            ax.plot(tgt[:, 0], tgt[:, 1], tgt[:, 2], color=INK, lw=2.4, label="target")
        draw_obstacles(ax, env)
        ax.plot([0], [0], [0], "s", color=INK2, ms=5)
        gripper_frame_legend(ax, 3)
        _axes3d(ax, f"{t}: final shape (best seed solid, others faint) vs target")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "shapes.png", dpi=140)
    plt.close(fig)
    L += ["## Final shapes", "", "![shapes](shapes.png)", ""]

    (out / "report.md").write_text("\n".join(L))
    json.dump(rows, open(out / "report_rows.json", "w"), indent=1, default=float)
    print("\n".join(L[:len(L)]))
    print(f"\nreport -> {out / 'report.md'}")


# ---------------------------------------------------------------- jobs

def cmd_jobs(args):
    out = Path(args.out)
    for t in (args.targets or list(TARGET_PATHS)):
        for o in ("adam", "linesearch"):
            for s in args.seeds:
                print(f"python scripts/train_shape.py --config {args.config} --target {t} --optimizer {o} "
                      f"--seed {s} --out {out}"
                      + (f" --epochs {args.epochs}" if args.epochs else "")
                      + f" > {out}/logs/{t}_{o}_s{s}.log 2>&1")


# ---------------------------------------------------------------- main

def main():
    cmds = ("train", "targets", "jobs", "report")
    argv = sys.argv[1:]
    cmd = argv.pop(0) if argv and argv[0] in cmds else "train"

    ap = argparse.ArgumentParser(description=f"3D shape formation by neural control ({cmd})")
    ap.add_argument("--out", required=True, help="experiment directory (targets/ lives in it)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="YAML config over configs/default.yaml (workspace is switched off and "
                         "the state gradient selected regardless)")
    if cmd in ("targets", "jobs"):
        ap.add_argument("--targets", nargs="+", choices=sorted(TARGET_PATHS), default=None)
    if cmd == "jobs":
        ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        ap.add_argument("--epochs", type=int, default=None)
    if cmd == "train":
        ap.add_argument("--target", choices=sorted(TARGET_PATHS), required=True)
        ap.add_argument("--optimizer", choices=OPTIMIZERS, default=None,
                        help="adam or linesearch; default: train.optimizer")
        ap.add_argument("--seed", type=int, default=0, help="policy init seed")
        ap.add_argument("--epochs", type=int, default=None, help="default: train.epochs")
        ap.add_argument("--lr", type=float, default=None, help="default: train.lr")
        ap.add_argument("--device", default=None, help="default: train.device")
    args = ap.parse_args(argv)

    if cmd == "targets":
        cmd_targets(args)
    elif cmd == "jobs":
        cmd_jobs(args)
    elif cmd == "report":
        report(args.out, args.config)
    else:
        cmd_train(args)


if __name__ == "__main__":
    main()
