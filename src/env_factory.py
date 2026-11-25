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


def env_twist(cfg) -> bool:
    """Whether the rod carries torsion (``rod.twist``, default false)."""
    twist = bool((cfg.get("rod") or {}).get("twist", False))
    if twist and env_dim(cfg) != 3:
        raise ValueError("rod.twist: true requires env.dim: 3")
    return twist


def end_control_dim(dim: int, twist: bool = False) -> int:
    """Control DOFs for a single rod end — what the policy net outputs.

    ================  ==========================================
    2D                3   (x, y, theta)
    3D                5   (x, y, z, theta, phi)
    3D + twist        6   (x, y, z, theta, phi, yaw)
    ================  ==========================================

    ``theta``/``phi`` aim the tangent; ``yaw`` is the spin about it, the third
    rotational freedom, and it only exists when the rod can actually twist.
    """
    if dim == 2:
        return 3
    return 6 if twist else 5


def control_dim(dim: int, twist: bool = False) -> int:
    """Length of the full control vector u (both ends)."""
    return 2 * end_control_dim(dim, twist)


def node_index(dim: int, node: int) -> int:
    """First DOF index of ``node`` in the flat DOF vector q."""
    return dim * node


def node_position(q, dim: int, node: int) -> np.ndarray:
    """Position of ``node`` as a ``dim``-vector.

    Positions occupy the front of ``q``, so this is correct with or without
    the gripper yaws appended behind them.
    """
    a = dim * node
    return np.asarray(q[a:a + dim], dtype=float)


# --------------------------------------------------------------- materials

#: The four fields that define what the rod is made of and how thick it is.
#: Everything else the solver needs (I, A, EI, EA, nodal mass) is derived.
MATERIAL_FIELDS = ("R_outer", "r_inner", "E_al", "rho_al")

#: Further material properties with a sensible fallback, so that a preset (or
#: ``material: custom``) may leave them out.  ``poisson`` is what turns the
#: bending stiffness into the torsional one: for the circular cross-section
#: ``J = 2I`` and ``G = E / (2(1+nu))``, so ``GJ = EI / (1 + nu)``.  It is
#: therefore part of what the rod is *made of*, not a solver setting, and
#: belongs next to E in a preset -- aluminium, nylon and steel genuinely differ
#: (~0.33, ~0.40, ~0.30).  Only the 3D environment with ``rod.twist`` uses it.
MATERIAL_DEFAULTS = {"poisson": 0.3}

#: Everything a material may specify: the required fields plus the optional ones.
MATERIAL_ALL = MATERIAL_FIELDS + tuple(MATERIAL_DEFAULTS)


def resolve_material(cfg) -> dict:
    """Resolve ``rod.material`` against the ``materials:`` presets.

    Precedence: a field given explicitly under ``rod:`` wins over the preset,
    so a config can pick a material and still tweak one dimension of it.
    ``material: custom`` (or null/absent) skips the presets, in which case all
    four required fields must be given under ``rod:``.  Optional fields
    (:data:`MATERIAL_DEFAULTS`) fall back to their defaults if nobody sets them.
    """
    rod = cfg.get("rod") or {}
    presets = cfg.get("materials") or {}
    name = rod.get("material")

    params = {}
    if name is not None and name != "custom":
        if name not in presets:
            known = ", ".join(sorted(presets)) or "(none defined)"
            raise ValueError(
                f"rod.material = {name!r} is not defined under `materials:`. "
                f"Known materials: {known}. Use `custom` to set "
                f"{'/'.join(MATERIAL_FIELDS)} directly under `rod:`."
            )
        preset = presets[name] or {}
        unknown = sorted(set(preset) - set(MATERIAL_ALL))
        if unknown:
            raise ValueError(
                f"material {name!r} has unsupported field(s) {unknown}; "
                f"a material may only set {list(MATERIAL_ALL)}."
            )
        params.update(preset)

    # Explicit rod-level values override the preset.
    params.update({k: rod[k] for k in MATERIAL_ALL if k in rod})

    missing = [k for k in MATERIAL_FIELDS if k not in params]
    if missing:
        where = "rod.material" if name in (None, "custom") else f"material {name!r}"
        raise ValueError(f"{where} leaves {missing} undefined; set them under `rod:`.")

    for key, default in MATERIAL_DEFAULTS.items():
        params.setdefault(key, default)
    return {k: float(params[k]) for k in MATERIAL_ALL}


def material_summary(cfg) -> str:
    """One-line human-readable description of the resolved rod material."""
    mat = resolve_material(cfg)
    rod = cfg.get("rod") or {}
    name = rod.get("material") or "custom"
    overridden = [k for k in MATERIAL_ALL if k in rod]

    I_sec = np.pi * (mat["R_outer"] ** 4 - mat["r_inner"] ** 4) / 4
    A_sec = np.pi * (mat["R_outer"] ** 2 - mat["r_inner"] ** 2)
    EI, EA = mat["E_al"] * I_sec, mat["E_al"] * A_sec

    using_preset = name != "custom"
    suffix = (f" (overriding {', '.join(overridden)})"
              if overridden and using_preset else "")
    line = (f"{name}{suffix}: R_outer={mat['R_outer']:g} m, "
            f"r_inner={mat['r_inner']:g} m, E={mat['E_al']:.3g} Pa, "
            f"rho={mat['rho_al']:g} kg/m^3, nu={mat['poisson']:g}"
            f"  ->  EI={EI:.3g} N*m^2, EA={EA:.3g} N")
    if env_twist(cfg):
        line += f", GJ={EI / (1.0 + mat['poisson']):.3g} N*m^2"
    return line


# ------------------------------------------------------------- env building

def build_env(cfg, **overrides):
    """Instantiate the environment selected by ``env.dim``.

    Everything under ``rod:`` is forwarded, with the cross-section/material
    fields resolved from ``rod.material`` + the ``materials:`` presets (see
    :func:`resolve_material`); ``overrides`` wins over the config (used e.g. by
    tests that want a smaller rod).
    """
    dim = env_dim(cfg)

    material = resolve_material(cfg)
    # Only the 3D environment has a use for the Poisson ratio (it sets GJ);
    # the 2D one would reject the keyword.
    poisson = material.pop("poisson")

    kwargs = dict(
        nv=cfg.rod.nv,
        dt=cfg.rod.dt,
        rod_length=cfg.rod.rod_length,
        total_time=cfg.rod.total_time,
        **material,
        max_newton_iter=cfg.rod.max_newton_iter,
        energy_cfg=cfg.get("energy"),
        newton_cfg=cfg.get("newton"),
        warn_on_newton_failure=bool((cfg.get("newton") or {}).get("verbose", False)),
        save_history=False,
    )

    grasp_mode = str((cfg.get("env") or {}).get("grasp_mode", "both"))
    kwargs["grasp_mode"] = grasp_mode
    if dim == 2:
        from SimulatorEnv_2D import SimulatorEnv_2D as EnvCls
        gravity = cfg.rod.get("gravity_2d")
        if gravity is not None:
            kwargs["gravity"] = tuple(float(g) for g in gravity)
    else:
        if grasp_mode == "right" and env_twist(cfg):
            raise ValueError("env.grasp_mode: right in 3D needs rod.twist: false (a free end has no yaw)")
        from SimulatorEnv_3D import SimulatorEnv_3D as EnvCls
        gravity = cfg.rod.get("gravity")
        if gravity is not None:
            kwargs["gravity"] = tuple(float(g) for g in gravity)
        kwargs["twist"] = env_twist(cfg)
        kwargs["poisson"] = poisson

    kwargs.update(overrides)
    env = EnvCls(**kwargs)
    attach_pipe(env, cfg)
    attach_peg(env, cfg)
    env.workspace_box = workspace_box(cfg, dim)
    return env


def workspace_box(cfg, dim):
    """``(lo, hi)`` position corners of the gripper workspace, or None when it is off.

    Only what the plots need (the figures and GIFs draw it); the controller
    reads the full bounds, angles included, from ``neural_control.workspace_bounds``.
    """
    ws = cfg.get("workspace") or {}
    if not ws.get("enabled", False):
        return None
    sfx = "_3d" if dim == 3 else ""
    return (np.array(ws[f"position_min{sfx}"], dtype=float),
            np.array(ws[f"position_max{sfx}"], dtype=float))


def build_pipe(cfg, env):
    """The ``pipe:`` geometry with the ``contact:`` parameters, for ``env``'s rod."""
    from contact import LPipe2D, LPipe3D
    p = cfg.pipe
    c = cfg.get("contact") or {}
    if env.dim == 3:
        # The sharp corner in space: 3-vectors for the entry position and the
        # two directions (the same keys as in 2D, one entry longer).
        return LPipe3D(
            entry_position=p.entry_position,
            entry_direction=p.entry_direction,
            exit_direction=p.exit_direction,
            inner_radius=p.inner_radius,
            bend_radius=p.bend_radius,
            first_segment_length=p.first_segment_length,
            second_segment_length=p.second_segment_length,
            rod_radius=env.R_outer,
            nv=env.nv,
            deltaL=env.deltaL,
            k_contact=c.get("k_contact", 1.0e7),
            smoothing=c.get("smoothing", 1.0e-4),
            corner_smoothing=c.get("corner_smoothing", 5.0e-4),
            samples_per_edge=c.get("samples_per_edge", 4),
            geometry_type=p.get("geometry_type", "l_corner_3d"),
        )
    return LPipe2D(
        entry_position=p.entry_position,
        entry_direction=p.entry_direction,
        exit_direction=p.exit_direction,
        inner_radius=p.inner_radius,
        bend_radius=p.bend_radius,
        first_segment_length=p.first_segment_length,
        second_segment_length=p.second_segment_length,
        rod_radius=env.R_outer,
        nv=env.nv,
        deltaL=env.deltaL,
        k_contact=c.get("k_contact", 1.0e7),
        smoothing=c.get("smoothing", 1.0e-4),
        corner_smoothing=c.get("corner_smoothing", 5.0e-4),
        samples_per_edge=c.get("samples_per_edge", 4),
        geometry_type=p.get("geometry_type", "l_bend_2d"),
    )


def attach_pipe(env, cfg):
    """Build ``pipe:`` into ``env`` and, with ``contact.enabled``, make it act.

    The pipe is kept on ``env.pipe`` either way (diagnostics, drawing); only
    when contact is on is it added to ``env.obstacles``, which is what the
    equilibrium solve sees.  Fails loudly on settings that would otherwise be
    silently ignored: friction (not implemented yet), contact with nothing to
    touch, and a starting rod that is already inside a wall.
    """
    fr = cfg.get("friction") or {}
    if fr.get("enabled", False):
        raise NotImplementedError(
            "friction.enabled: true -- friction is not implemented yet (only "
            "frictionless normal contact, Version 1). Set friction.enabled: false.")
    pipe_on = bool((cfg.get("pipe") or {}).get("enabled", False))
    contact_on = bool((cfg.get("contact") or {}).get("enabled", False))
    if contact_on and not pipe_on:
        raise ValueError("contact.enabled: true but there is no geometry to touch "
                         "(pipe.enabled: false)")
    if not pipe_on:
        return None

    pipe = build_pipe(cfg, env)
    env.pipe = pipe
    if contact_on:
        env.add_obstacle(pipe)

    q0 = env.nodes.reshape(-1)        # positions only; the pipe reads just those
    pen = pipe.detect_dlo_pipe_contact(q0)["max_penetration"]
    if pen > 0.0:
        raise ValueError(
            f"The initial (straight) rod penetrates the pipe wall by {pen:.3g} m. "
            "Move pipe.entry_position / entry_direction so the inserted part of "
            "the rod lies inside the first segment.")
    return pipe


_PEG_KEYS = ("center_xz", "center_y", "radius", "half_length_y", "k_contact")
#: Contact-model keys, set once in the ``peg:`` block for every peg.
_PEG_MODEL_DEFAULTS = {"model": "smooth", "smoothing": 1.0e-4, "samples_per_edge": 4}


def attach_peg(env, cfg):
    """Add the ``peg:`` cylinders (axis along y) to ``env.obstacles``; 3D only.

    The ``peg:`` block is the first peg; each ``peg.extra`` entry is another,
    with the block's radius, half length and stiffness unless it sets its own.
    Returns the pegs (empty when ``peg.enabled`` is false).  A rod that starts
    inside a peg is rejected rather than left to be pushed out.

    ``peg.model``: ``smooth`` (default) is the paper's contact model
    (contact.PegObstacle: edge quadrature, squared-softplus penalty with
    smoothing length ``peg.smoothing``, clearance of the rod surface);
    ``node`` is the legacy node-based quadratic penalty (PegObstacleNode),
    kept only to reproduce runs made before 2026-11-24.
    """
    p = cfg.get("peg") or {}
    if not p.get("enabled", False):
        return []
    if env_dim(cfg) != 3:
        raise ValueError("peg.enabled: true needs env.dim: 3 (the peg is a cylinder along y)")
    from contact import PegObstacle, PegObstacleNode
    opt = {k: p.get(k, v) for k, v in _PEG_MODEL_DEFAULTS.items()}
    if opt["model"] not in ("smooth", "node"):
        raise ValueError(f"peg.model must be smooth or node, got {opt['model']!r}")
    spacing = cfg.rod.rod_length / (cfg.rod.nv - 1)
    pegs = []
    for i, extra in enumerate([{}] + list(p.get("extra") or [])):
        bad = sorted(set(extra) - set(_PEG_KEYS))
        if bad:
            raise ValueError(f"peg.extra[{i - 1}] sets {bad}; a peg takes {list(_PEG_KEYS)}")
        q = {key: extra.get(key, p[key]) for key in _PEG_KEYS}
        cx, cz = (float(v) for v in q["center_xz"])
        name = "peg" if i == 0 else f"peg.extra[{i - 1}]"
        if opt["model"] == "node":
            peg = PegObstacleNode(cx, float(q["center_y"]), cz, float(q["radius"]),
                                  float(q["half_length_y"]), k_contact=float(q["k_contact"]))
            if peg.radius <= spacing:
                raise ValueError(f"{name} radius {peg.radius} m is not above the node spacing "
                                 f"{spacing:.4g} m: the rod could slip between two nodes")
            inside = any(peg.node_energy(*x) > 0.0 for x in env.nodes.reshape(-1, 3))
        else:
            peg = PegObstacle(cx, float(q["center_y"]), cz, float(q["radius"]),
                              float(q["half_length_y"]), rod_radius=env.R_outer,
                              nv=env.nv, deltaL=env.deltaL, k_contact=float(q["k_contact"]),
                              smoothing=float(opt["smoothing"]),
                              samples_per_edge=int(opt["samples_per_edge"]))
            inside = peg.report(env.nodes.reshape(-1))["max_penetration"] > 0.0
        if inside:
            raise ValueError(f"the initial rod penetrates {name}; move its center_xz / center_y")
        env.add_obstacle(peg)
        pegs.append(peg)
    return pegs


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
                     init_phi=cfg.start.get("init_phi", 0.0),
                     init_yaw=cfg.start.get("init_yaw", 0.0))


def settle_initial(env, cfg):
    """Solve the equilibrium at the initial control before the horizon starts.

    With ``newton.initial_equilibrium`` the rod is brought to equilibrium at
    ``u0`` (by continuation from the pose that clamps the undeformed rod, see
    SimulatorEnv_*.settle) and that state is what the rollout checkpoint
    holds, so the first control step leaves from an equilibrium rather than
    from the straight rod.  The equilibrium has to converge and, when the
    stability check is on, to be stable: an unstable start would put the
    whole rollout on a branch the rod cannot be on.  Returns the step info,
    or None when the option is off (the checkpoint is then the reset state,
    as before).
    """
    if not bool((cfg.get("newton") or {}).get("initial_equilibrium", False)):
        return None
    info = env.settle(initial_control(cfg))
    if not info["converged"]:
        raise RuntimeError("no equilibrium at the initial control u0: "
                           f"{info['reason'] or info['newton'].get('reason', '')}")
    if info["stable"] is False:
        raise RuntimeError("the equilibrium at the initial control u0 is unstable "
                           "(J_ff is not positive definite); the rollout cannot start there")
    return info


# --------------------------------------------------------- controls / tasks

def initial_control(cfg) -> np.ndarray:
    """The initial absolute control u0 (both ends concatenated).

    Which keys are read follows ``env.dim`` and ``rod.twist``:

    ====================  ===========================================  =======
    2D                    ``start.u_left``      / ``u_right``          3 each
    3D                    ``start.u_left_3d``   / ``u_right_3d``       5 each
    3D + twist            ``start.u_left_twist``/ ``u_right_twist``    6 each
    ====================  ===========================================  =======
    """
    dim = env_dim(cfg)
    twist = env_twist(cfg)
    if dim == 2:
        left, right = cfg.start.u_left, cfg.start.u_right
        keys = ("start.u_left", "start.u_right")
    elif twist:
        left, right = cfg.start.u_left_twist, cfg.start.u_right_twist
        keys = ("start.u_left_twist", "start.u_right_twist")
    else:
        left, right = cfg.start.u_left_3d, cfg.start.u_right_3d
        keys = ("start.u_left_3d", "start.u_right_3d")

    n = end_control_dim(dim, twist)
    for key, vec in zip(keys, (left, right)):
        if len(vec) != n:
            raise ValueError(
                f"{key} must have {n} entries for dim={dim}"
                f"{' with twist' if twist else ''}, got {len(vec)}")

    return np.array(list(left) + list(right), dtype=float)


def initial_midpoint(env, cfg) -> np.ndarray:
    """Where the controlled node sits at the *start* of a rollout.

    Not the undeformed rod's midpoint: the rod sags as soon as the first
    equilibrium is solved, so a trajectory anchored at ``env.nodes[mid]`` would
    already be wrong on step 0.  This takes one step with the initial control,
    reads the midpoint, and restores the environment afterwards.
    """
    dim = env_dim(cfg)
    state = env.get_state()
    try:
        _, _, info = env.step(initial_control(cfg))
        return node_position(info["q"], dim, env.mid_node)
    finally:
        env.set_state(state)


def task_target(cfg) -> np.ndarray:
    """Point-to-point target for the mid node: ``task.target_xy`` / ``task.target_xyz``."""
    dim = env_dim(cfg)
    key = "target_xy" if dim == 2 else "target_xyz"
    target = np.array(cfg.task[key], dtype=float)
    if target.size != dim:
        raise ValueError(f"task.{key} must have {dim} entries, got {target.size}")
    return target


def make_target_trajectory(N, rod_length, traj_type="circle", dim=2, up_axis=None,
                           start=None, end=None, sine=None):
    """The N-step target trajectory for the mid node, shape (N, dim).

    ``segment`` is the default and the one to reach for: a straight line from
    where the mid node actually starts (``start``, see :func:`initial_midpoint`)
    to the point-to-point target (``end``, ``task.target_*``).  Because it
    begins at the rod's own configuration the error at step 0 is ~zero, so the
    loss measures how well the policy *follows* rather than being dominated by
    a gap the rod was never given a chance to close.  It also makes tracking
    and point-to-point directly comparable: same endpoint, same target, one
    asks only for the destination and the other for the whole path to it.

    The fixed curves (``circle``, ``figure8``, ``line``) ignore ``start``:
    they sit wherever they are defined, so the rod generally has to travel to
    reach step 0 and the first part of the run is spent closing that gap.

    The planar curves (``circle``, ``figure8``, ``line``) are laid out in the
    *vertical* plane -- the one containing gravity -- because that is what they
    were designed for: a circle whose centre sits above the rod is a lift
    against gravity, not a horizontal sweep.  Which plane that is depends on
    the convention: gravity runs along -y in 2D and along -z in 3D, so the
    curves occupy x-y in 2D and x-z in 3D.  ``up_axis`` overrides the choice
    for a config that tilts gravity somewhere else.

    Reference paths that start where the rod is -- straight lines, sine waves
    and circular arcs anchored at the mid node -- live in the named presets of
    scripts/train_tracking.py, which are the ones its RRT comparison verifies.
    """
    lams = np.linspace(0, 1, N)
    targets = np.zeros((N, dim))
    # Index of the axis gravity acts along; the planar curves stand up in it.
    up = (1 if dim == 2 else 2) if up_axis is None else int(up_axis)

    if traj_type == "segment":
        if start is None or end is None:
            raise ValueError(
                "trajectory_type 'segment' needs both ends: pass start="
                "initial_midpoint(env, cfg) and end=task_target(cfg).")
        p0 = np.asarray(start, dtype=float).reshape(-1)
        p1 = np.asarray(end, dtype=float).reshape(-1)
        if p0.size != dim or p1.size != dim:
            raise ValueError(f"segment endpoints must be {dim}D, got "
                             f"{p0.size} and {p1.size}")
        targets[:] = p0 + np.outer(lams, p1 - p0)
    elif traj_type == "sine_xz":
        # 3D: a sine in the vertical x-z plane from `start` (the ready position
        # of the mid node), drifting by (dx, dy) and bowing x by `bow` -- the keys of
        # task.tracking.sine_xz.  Step k targets t = (k+1)/N, so the last step
        # lands at t = 1, as in train_tracking.reference_path.
        if dim != 3 or start is None or sine is None:
            raise ValueError("trajectory_type 'sine_xz' needs dim 3, start= and sine=")
        p0 = np.asarray(start, dtype=float).reshape(-1)
        t = np.arange(1, N + 1) / N
        targets[:, 0] = p0[0] + float(sine["dx"]) * t
        targets[:, 1] = p0[1] + float(sine["dy"]) * t
        targets[:, 2] = p0[2] + float(sine["amp"]) * np.sin(2 * np.pi * float(sine["periods"]) * t)
        # `bow`: let x follow the surface a bowed rod's mid node lives on,
        # x ~ x0 - bow * (y^2 + z^2) (about 1.43 for the 1 m soft_strip), so a
        # path that moves in y or z moves *towards* the fixed end instead of
        # stretching the rod; `dx` is on top of that.
        bow = float(sine.get("bow", 0.0))
        if bow:
            targets[:, 0] -= bow * ((targets[:, 1] ** 2 + targets[:, 2] ** 2)
                                    - (p0[1] ** 2 + p0[2] ** 2))
    elif traj_type == "circle":
        cx, cu, r = rod_length / 2, 0.25, 0.15
        theta = 2 * np.pi * lams
        targets[:, 0] = cx + r * np.cos(theta)
        targets[:, up] = cu + r * np.sin(theta)
    elif traj_type == "figure8":
        cx, cu, r = rod_length / 2, 0.0, 0.20
        theta = 2 * np.pi * lams
        targets[:, 0] = cx + r * np.sin(theta)
        targets[:, up] = cu + r * np.sin(2 * theta) / 2
    elif traj_type == "line":
        targets[:, 0] = rod_length / 2
        targets[:, up] = -0.20 + 0.40 * lams
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
