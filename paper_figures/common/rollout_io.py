"""The standard rollout format read by the task tiles.

Experiment scripts write one folder per task::

    paper_figures/data/rollouts/<task>/
        rollout.npz
            lam           (M,)          homotopy progress in [0, 1], one per stored step
            nodes         (M, N, dim)   node positions [m], dim = 2 or 3
            gripper_pose  (M, 3)        2D: x, y, theta [m, m, rad]
                          (M, 7)        3D: x, y, z [m], unit quaternion qw, qx, qy, qz
                                        (gripper frame: x = rod tangent into the gripper,
                                        y = jaw direction)
            contact_mask  (M, N-1)      optional, bool: edge k (nodes k, k+1) touches an obstacle
        scene.json
            {
              "dim": 2 | 3,
              "units": "m",
              "title": "Insertion",                 # optional
              "source": "free text: how it was produced",
              "provisional": false,
              "obstacles": [ ... see below ... ],
              "fixed_end": {"position": [...], "tangent": [...]} | null,
              "gripper_end": "last" | "first",       # which node the gripper holds
              "camera": {"elev": 20, "azim": -60, "zoom": 1.0}       # 3D
                        {"xlim": [x0, x1], "ylim": [y0, y1]}        # 2D, optional
            }

Obstacle entries (all lengths in m)::

    {"type": "circle",   "center": [x, y], "radius": r}                   2D
    {"type": "pipe",     LPipe2D parameters (entry_position, entry_direction,
                         exit_direction, inner_radius, bend_radius,
                         first_segment_length, second_segment_length,
                         geometry_type), "wall": drawn wall thickness}       2D
    {"type": "sphere",   "center": [x, y, z], "radius": r}                3D
    {"type": "cylinder", "p0": [x, y, z], "p1": [x, y, z], "radius": r}   3D (e.g. a peg)
    {"type": "capsule",  "p0": [...], "p1": [...], "radius": r}           3D
    {"type": "plane",    "point": [...], "normal": [...], "size": [a, b]} 3D, drawn as a patch
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OBSTACLE_TYPES = {"circle", "sphere", "cylinder", "capsule", "plane", "pipe"}


def write_rollout(task_dir, *, lam, nodes, gripper_pose, scene, contact_mask=None):
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"lam": np.asarray(lam, dtype=float),
              "nodes": np.asarray(nodes, dtype=float),
              "gripper_pose": np.asarray(gripper_pose, dtype=float)}
    if contact_mask is not None:
        arrays["contact_mask"] = np.asarray(contact_mask, dtype=bool)
    problems = validate(arrays, scene)
    if problems:
        raise ValueError(f"{task_dir}: invalid rollout: {problems}")
    np.savez_compressed(task_dir / "rollout.npz", **arrays)
    (task_dir / "scene.json").write_text(json.dumps(scene, indent=1, default=float) + "\n")


def read_rollout(task_dir):
    task_dir = Path(task_dir)
    with np.load(task_dir / "rollout.npz") as z:
        r = {k: z[k] for k in z.files}
    scene = json.loads((task_dir / "scene.json").read_text())
    problems = validate(r, scene)
    if problems:
        raise ValueError(f"{task_dir}: invalid rollout: {problems}")
    return r, scene


def validate(r, scene):
    out = []
    for key in ("lam", "nodes", "gripper_pose"):
        if key not in r:
            out.append(f"missing {key}")
    if out:
        return out
    lam, nodes, pose = r["lam"], r["nodes"], r["gripper_pose"]
    M = len(lam)
    if nodes.ndim != 3 or nodes.shape[0] != M or nodes.shape[2] not in (2, 3):
        out.append(f"nodes shape {nodes.shape}, expected ({M}, N, 2|3)")
    dim = nodes.shape[2] if nodes.ndim == 3 else None
    if dim is not None and scene.get("dim") != dim:
        out.append(f"scene dim {scene.get('dim')} != nodes dim {dim}")
    want = 3 if dim == 2 else 7
    if pose.shape != (M, want):
        out.append(f"gripper_pose shape {pose.shape}, expected ({M}, {want})")
    if "contact_mask" in r and r["contact_mask"].shape != (M, nodes.shape[1] - 1):
        out.append(f"contact_mask shape {r['contact_mask'].shape}")
    if np.any(np.diff(lam) < 0) or lam.min() < 0 or lam.max() > 1:
        out.append("lam must be nondecreasing in [0, 1]")
    for ob in scene.get("obstacles", []):
        if ob.get("type") not in OBSTACLE_TYPES:
            out.append(f"unknown obstacle type {ob.get('type')}")
    return out


# ----------------------------------------------------------------- rotations

def quat_from_matrix(R):
    """Unit quaternion (w, x, y, z) of a rotation matrix."""
    R = np.asarray(R, dtype=float)
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def matrix_from_quat(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
