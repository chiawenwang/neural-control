"""Provisional tile rollouts from what the code can do today.

The final panel (b) tasks (3D contact) are not implemented yet. Until they
are, ``tiles.py --provisional`` builds stand-ins into
``data/rollouts/<task>_provisional/`` in the standard format
(:mod:`paper_figures.common.rollout_io`):

shape_formation  2D, both ends clamped, two ``CircleObstacle`` pegs, scripted
                 gripper path (smoothstep in x, y, theta).
insertion        the 90-degree pipe task, ``configs/task_90deg_pipe.yaml``,
                 replaying a scripted push (tilt, then feed). ``train_pipe.py``
                 saves no trajectory, and its last logged run (60 epochs) took
                 27 min and did not reach the target, so it is not rerun here.
routing          the 3D twist point-to-point setup, ``configs/example_3d_twist.yaml``,
                 replaying a scripted gripper motion with a full turn of yaw.
                 ``train_point_to_point.py`` saved no twist trajectory, and its
                 last logged 200-epoch run took 78 min.

Every step goes through the repo's own environment (``env_factory.build_env``
or ``SimulatorEnv_2D``) and is checked for stability (``stable_env_step``).
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from paper_figures.common import rollout_io
from paper_figures.common import sim_adapters as sa
from paper_figures.common.paths import REPO_ROOT, ROLLOUTS_DIR

from config import load_config                      # noqa: E402  (src/, via paper_figures)
from env_factory import build_env, initial_control, reset_env   # noqa: E402
from rodframes import gripper_director              # noqa: E402
from SimulatorEnv_2D import SimulatorEnv_2D         # noqa: E402
from SimulatorEnv_3D import tangent_from_angles     # noqa: E402

VERSION = "provisional-v1"


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def settings_hash(name, settings):
    blob = json.dumps({"v": VERSION, "name": name, "s": settings}, sort_keys=True, default=float)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _cached(task_dir, h):
    f = task_dir / "scene.json"
    return f.exists() and json.loads(f.read_text()).get("settings_hash") == h


def _run(env, controls, label):
    """Step ``env`` through ``controls`` with stability kicks; returns per-step records."""
    t0 = time.time()
    qs, kicks, eigs, its = [], 0, [], []
    for k, u in enumerate(controls):
        info, n_k, w0 = sa.stable_env_step(env, u)
        if not info["converged"]:
            raise RuntimeError(f"{label}: step {k} did not converge ({info['newton']['reason']})")
        qs.append(info["q"].copy())
        kicks += n_k
        eigs.append(w0)
        its.append(int(info["newton"]["iterations"]))
    print(f"  {label}: {len(controls)} steps in {time.time() - t0:.1f} s, Newton iterations "
          f"median {np.median(its):.0f} / max {max(its)}, stability kicks {kicks}, "
          f"min eig over the path {min(eigs):.3e}")
    return qs, {"seconds": time.time() - t0, "kicks": kicks, "min_eig": float(min(eigs)),
                "newton_median": float(np.median(its)), "newton_max": int(max(its))}


def _edge_circle_contact(nodes, circles, tol):
    """(N-1,) bool: segment within ``tol`` of a circle's boundary (or inside)."""
    a, b = nodes[:-1], nodes[1:]
    mask = np.zeros(len(a), dtype=bool)
    for c, r in circles:
        c = np.asarray(c, dtype=float)
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", c - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
        d = np.linalg.norm(a + t[:, None] * ab - c, axis=1)
        mask |= d <= r + tol
    return mask


# ------------------------------------------------------------ shape formation

def build_shape_formation(pcfg, tol, force=False):
    task_dir = ROLLOUTS_DIR / "shape_formation_provisional"
    h = settings_hash("shape_formation", pcfg)
    if not force and _cached(task_dir, h):
        print(f"  shape_formation_provisional: cached ({h})")
        return task_dir
    mat = sa.material_params(load_config(None).rod.material)
    mat.pop("poisson")
    env = SimulatorEnv_2D(nv=pcfg["nv"], rod_length=1.0, gravity=tuple(pcfg["gravity"]),
                          grasp_mode="both", **mat)
    circles = [((x, y), r) for x, y, r in pcfg["obstacles"]]
    for (c, r) in circles:
        env.add_obstacle(sa.make_circle(c, r, pcfg["k_contact"]))
    env.reset()
    g0, g1 = np.array(pcfg["gripper_start"]), np.array(pcfg["gripper_end"])
    M = int(pcfg["steps"])
    lam = np.arange(1, M + 1) / M
    poses = g0 + smoothstep(lam)[:, None] * (g1 - g0)
    controls = [np.concatenate([[0.0, 0.0, 0.0], p]) for p in poses]
    qs, stats = _run(env, controls, "shape_formation_provisional")
    nodes = np.array([q.reshape(-1, 2) for q in qs])
    contact = np.array([_edge_circle_contact(n, circles, tol) for n in nodes])
    scene = {
        "dim": 2, "units": "m", "title": "Shape formation", "provisional": True,
        "source": ("scripted gripper path (smoothstep) on SimulatorEnv_2D, soft_strip, both "
                   "ends clamped, two CircleObstacle pegs; paper_figures/fig01/provisional.py"),
        "obstacles": [{"type": "circle", "center": list(c), "radius": r} for c, r in circles],
        "fixed_end": {"position": [0.0, 0.0], "tangent": [1.0, 0.0]},
        "gripper_end": "last",
        "stats": stats, "settings_hash": h,
    }
    rollout_io.write_rollout(task_dir, lam=lam, nodes=nodes, gripper_pose=poses,
                             scene=scene, contact_mask=contact)
    return task_dir


# ------------------------------------------------------------------ insertion

def build_insertion(pcfg, tol, force=False):
    task_dir = ROLLOUTS_DIR / "insertion_provisional"
    cfg_path = REPO_ROOT / pcfg["config"]
    h = settings_hash("insertion", {**pcfg, "config_text": cfg_path.read_text()})
    if not force and _cached(task_dir, h):
        print(f"  insertion_provisional: cached ({h})")
        return task_dir
    cfg = load_config(cfg_path)
    env = build_env(cfg)
    reset_env(env, cfg)
    pipe = env.pipe
    u0 = initial_control(cfg)
    M = int(pcfg["steps"])
    lam = np.arange(1, M + 1) / M
    x0, y0, th0 = pcfg["gripper_start"]
    y1, th1 = pcfg["tilt"]
    f = pcfg["tilt_fraction"]
    tilt = smoothstep(lam / f)
    feed = smoothstep((lam - f) / (1.0 - f))
    poses = np.stack([x0 + feed * (pcfg["feed_to_x"] - x0),
                      y0 + tilt * (y1 - y0), th0 + tilt * (th1 - th0)], axis=1)
    controls = [np.concatenate([u0[:3], p]) for p in poses]
    qs, stats = _run(env, controls, "insertion_provisional")
    nodes = np.array([q.reshape(-1, 2) for q in qs])
    m = 8
    contact = []
    for n in nodes:
        P = (1 - np.arange(m) / m)[None, :, None] * n[:-1, None] + (np.arange(m) / m)[None, :, None] * n[1:, None]
        g = pipe.compute_signed_clearance(P.reshape(-1, 2), smooth=False).reshape(len(n) - 1, m)
        contact.append((g < tol).any(axis=1))
    tip_s = [pipe.arclength(n[0])[0] for n in nodes]
    stats.update({"tip_arclength_start": float(tip_s[0]), "tip_arclength_end": float(tip_s[-1]),
                  "corner_arclength": float(pipe.L1),
                  "tip_end": nodes[-1, 0].tolist(), "target": list(cfg.task.target_xy),
                  "max_penetration": float(max(pipe.detect_dlo_pipe_contact(q)["max_penetration"]
                                               for q in qs))})
    print(f"    tip arc length {tip_s[0]:.3f} -> {tip_s[-1]:.3f} m (corner at {pipe.L1:.2f} m), "
          f"tip ends at {np.round(nodes[-1, 0], 3).tolist()}, target {list(cfg.task.target_xy)}")
    p = cfg.pipe
    scene = {
        "dim": 2, "units": "m", "title": "Insertion", "provisional": True,
        "source": (f"scripted push (tilt to y={y1}, theta={th1}, then feed to x={pcfg['feed_to_x']}) "
                   f"replayed on {pcfg['config']} (pipe, contact, stability check); "
                   "paper_figures/fig01/provisional.py"),
        "obstacles": [{"type": "pipe", "wall": pcfg["pipe_wall"],
                       **{k: p[k] for k in ("entry_position", "entry_direction", "exit_direction",
                                            "inner_radius", "bend_radius", "first_segment_length",
                                            "second_segment_length", "geometry_type")}}],
        "target": list(cfg.task.target_xy),
        "fixed_end": None,
        "gripper_end": "last",
        "stats": stats, "settings_hash": h,
    }
    rollout_io.write_rollout(task_dir, lam=lam, nodes=nodes, gripper_pose=poses,
                             scene=scene, contact_mask=np.array(contact))
    return task_dir


# -------------------------------------------------------------------- routing

def gripper_quaternion(theta, phi, yaw):
    """Gripper frame (x = tangent, y = jaw direction = yaw-rotated gripper_director)."""
    t = tangent_from_angles(theta, phi)
    g = gripper_director(t)
    d = np.cos(yaw) * g + np.sin(yaw) * np.cross(t, g)
    return rollout_io.quat_from_matrix(np.column_stack([t, d, np.cross(t, d)]))


def build_routing(pcfg, tol, force=False):
    task_dir = ROLLOUTS_DIR / "routing_provisional"
    cfg_path = REPO_ROOT / pcfg["config"]
    h = settings_hash("routing", {**pcfg, "config_text": cfg_path.read_text()})
    if not force and _cached(task_dir, h):
        print(f"  routing_provisional: cached ({h})")
        return task_dir
    cfg = load_config(cfg_path)
    env = build_env(cfg)
    reset_env(env, cfg)
    u0 = initial_control(cfg)                     # 12D: left and right (x y z th ph yaw)
    M = int(pcfg["steps"])
    lam = np.arange(1, M + 1) / M
    r0, r1 = u0[6:], np.array(pcfg["gripper_end"], dtype=float)
    right = r0 + smoothstep(lam)[:, None] * (r1 - r0)
    controls = [np.concatenate([u0[:6], r]) for r in right]
    qs, stats = _run(env, controls, "routing_provisional")
    nodes = np.array([env.positions(q) for q in qs])
    poses = np.array([np.concatenate([r[:3], gripper_quaternion(r[3], r[4], r[5])]) for r in right])
    stats["twist_end"] = float(env.energy.total_twist(qs[-1]))
    scene = {
        "dim": 3, "units": "m", "title": "Routing", "provisional": True,
        "source": (f"scripted gripper motion to {pcfg['gripper_end']} (x y z theta phi yaw) "
                   f"replayed on {pcfg['config']} (3D, torsion, gravity -z); no pegs yet; "
                   "paper_figures/fig01/provisional.py"),
        "obstacles": [],
        "fixed_end": {"position": u0[:3].tolist(),
                      "tangent": tangent_from_angles(u0[3], u0[4]).tolist()},
        "gripper_end": "last",
        "camera": pcfg["camera"],
        "stats": stats, "settings_hash": h,
    }
    rollout_io.write_rollout(task_dir, lam=lam, nodes=nodes, gripper_pose=poses, scene=scene)
    return task_dir


BUILDERS = {"shape_formation": build_shape_formation, "insertion": build_insertion,
            "routing": build_routing}
