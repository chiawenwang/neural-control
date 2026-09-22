# -*- coding: utf-8 -*-
"""Dimension-aware construction of the simulator, controls, and task targets.

The training scripts (``scripts/train_*.py``) are written once and work with
either environment; everything that differs between 2D and 3D is funnelled
through this module, selected by ``env.dim`` in the YAML config.

=========================  ==================  ==================
                           dim = 2             dim = 3
=========================  ==================  ==================
environment                SimulatorEnv_2D     SimulatorEnv_3D
DOFs per node              2                   3
control vector ``u``       6                   10
  per end                  (x, y, theta)       (x, y, z, theta, phi)
policy output (right end)  3                   5
task target                (x, y)              (x, y, z)
obstacles                  CircleObstacle      SphereObstacle
=========================  ==================  ==================
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- dimensions

def env_dim(cfg) -> int:
    """Spatial dimension requested by the config (``env.dim``, default 2)."""
    dim = int((cfg.get("env") or {}).get("dim", 2))
    if dim not in (2, 3):
        raise ValueError(f"env.dim must be 2 or 3, got {dim}")
    return dim


def control_dim(dim: int) -> int:
    """Length of the full control vector u (both ends)."""
    return 6 if dim == 2 else 10


def end_control_dim(dim: int) -> int:
    """Number of control DOFs for a single end — what the policy net outputs."""
    return 3 if dim == 2 else 5


def node_index(dim: int, node: int) -> int:
    """First DOF index of ``node`` in the flat DOF vector q."""
    return dim * node


def node_position(q, dim: int, node: int) -> np.ndarray:
    """Position of ``node`` as a ``dim``-vector."""
    a = dim * node
    return np.asarray(q[a:a + dim], dtype=float)


# ------------------------------------------------------------- env building

def build_env(cfg, **overrides):
    """Instantiate the environment selected by ``env.dim``.

    Everything under ``rod:`` is forwarded; ``overrides`` wins over the config
    (used e.g. by tests that want a smaller rod).
    """
    dim = env_dim(cfg)

    kwargs = dict(
        nv=cfg.rod.nv,
        dt=cfg.rod.dt,
        rod_length=cfg.rod.rod_length,
        total_time=cfg.rod.total_time,
        R_outer=cfg.rod.R_outer,
        r_inner=cfg.rod.r_inner,
        E_al=cfg.rod.E_al,
        rho_al=cfg.rod.rho_al,
        max_newton_iter=cfg.rod.max_newton_iter,
        energy_cfg=cfg.get("energy"),
        newton_cfg=cfg.get("newton"),
        warn_on_newton_failure=bool((cfg.get("newton") or {}).get("verbose", False)),
        save_history=False,
    )

    if dim == 2:
        from SimulatorEnv_2D import SimulatorEnv_2D as EnvCls
    else:
        from SimulatorEnv_3D import SimulatorEnv_3D as EnvCls
        gravity = cfg.rod.get("gravity")
        if gravity is not None:
            kwargs["gravity"] = tuple(float(g) for g in gravity)

    kwargs.update(overrides)
    return EnvCls(**kwargs)


def reset_env(env, cfg):
    """Reset the environment using the ``start:`` block of the config."""
    if env_dim(cfg) == 2:
        return env.reset(init_x=cfg.start.init_x,
                         init_y=cfg.start.init_y,
                         init_theta=cfg.start.init_theta)
    return env.reset(init_x=cfg.start.init_x,
                     init_y=cfg.start.init_y,
                     init_z=cfg.start.get("init_z", 0.0),
                     init_theta=cfg.start.init_theta,
                     init_phi=cfg.start.get("init_phi", 0.0))


# --------------------------------------------------------- controls / tasks

def initial_control(cfg) -> np.ndarray:
    """The initial absolute control u0 (both ends concatenated).

    2D reads ``start.u_left`` / ``start.u_right`` (3 entries each);
    3D reads ``start.u_left_3d`` / ``start.u_right_3d`` (5 entries each).
    """
    dim = env_dim(cfg)
    if dim == 2:
        left, right = cfg.start.u_left, cfg.start.u_right
        keys = ("start.u_left", "start.u_right")
    else:
        left, right = cfg.start.u_left_3d, cfg.start.u_right_3d
        keys = ("start.u_left_3d", "start.u_right_3d")

    n = end_control_dim(dim)
    for key, vec in zip(keys, (left, right)):
        if len(vec) != n:
            raise ValueError(f"{key} must have {n} entries for dim={dim}, got {len(vec)}")

    return np.array(list(left) + list(right), dtype=float)


def task_target(cfg) -> np.ndarray:
    """Point-to-point target for the mid node: ``task.target_xy`` / ``task.target_xyz``."""
    dim = env_dim(cfg)
    key = "target_xy" if dim == 2 else "target_xyz"
    target = np.array(cfg.task[key], dtype=float)
    if target.size != dim:
        raise ValueError(f"task.{key} must have {dim} entries, got {target.size}")
    return target


def make_target_trajectory(N, rod_length, traj_type="circle", dim=2):
    """The N-step target trajectory for the mid node, shape (N, dim).

    The 2D curves are unchanged.  In 3D, ``circle`` and ``figure8`` are the
    same planar curves lifted into the x-y plane (z = 0), so a 3D run can be
    compared directly against the 2D one; ``helix`` and ``figure8_3d`` are
    genuinely three-dimensional.
    """
    lams = np.linspace(0, 1, N)
    targets = np.zeros((N, dim))

    if traj_type == "circle":
        cx, cy, r = rod_length / 2, 0.25, 0.15
        theta = 2 * np.pi * lams
        targets[:, 0] = cx + r * np.cos(theta)
        targets[:, 1] = cy + r * np.sin(theta)
    elif traj_type == "figure8":
        cx, cy, r = rod_length / 2, 0.0, 0.20
        theta = 2 * np.pi * lams
        targets[:, 0] = cx + r * np.sin(theta)
        targets[:, 1] = cy + r * np.sin(2 * theta) / 2
    elif traj_type == "line":
        targets[:, 0] = rod_length / 2
        targets[:, 1] = -0.20 + 0.40 * lams
    elif traj_type == "helix":
        if dim != 3:
            raise ValueError("trajectory_type 'helix' requires env.dim: 3")
        cx, cy, r = rod_length / 2, 0.0, 0.15
        theta = 4 * np.pi * lams
        targets[:, 0] = cx + r * np.cos(theta)
        targets[:, 1] = cy + r * np.sin(theta)
        targets[:, 2] = -0.15 + 0.30 * lams
    elif traj_type == "figure8_3d":
        if dim != 3:
            raise ValueError("trajectory_type 'figure8_3d' requires env.dim: 3")
        cx, r = rod_length / 2, 0.20
        theta = 2 * np.pi * lams
        targets[:, 0] = cx + r * np.sin(theta)
        targets[:, 1] = r * np.sin(2 * theta) / 2
        targets[:, 2] = 0.15 * np.sin(2 * theta)
    else:
        raise ValueError(f"Unknown traj_type: {traj_type}")

    return targets


# ------------------------------------------------------------------- losses

def node_tracking_loss(q, dim, node, target):
    """Squared distance from ``node`` to ``target``. Returns (loss, position)."""
    p = node_position(q, dim, node)
    d = p - np.asarray(target, dtype=float)
    return float(d @ d), p


def dloss_dq_free_node(q, free_index, dim, node, target):
    """dL/dq_free for ``L = |p_node - target|^2``, scattered onto the free DOFs."""
    p = node_position(q, dim, node)
    a_free = np.zeros(len(free_index), dtype=float)
    for c in range(dim):
        idx = dim * node + c
        pos = np.where(free_index == idx)[0]
        if len(pos) == 1:
            a_free[pos[0]] = 2.0 * (p[c] - target[c])
    return a_free
