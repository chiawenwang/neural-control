# -*- coding: utf-8 -*-
"""3D elastic-rod simulation environment.

This is the 3D counterpart of :mod:`SimulatorEnv_2D`.  The API is identical
(``reset`` / ``step`` / ``get_state`` / ``set_state`` / ``render``) so the
training scripts can drive either one; only the dimensions change:

=====================  ==========================  ==========================
                       SimulatorEnv_2D             SimulatorEnv_3D
=====================  ==========================  ==========================
DOF vector ``q``       ``2*nv``  [x0,y0,x1,y1,…]   ``3*nv``  [x0,y0,z0,…]
control ``u``          6   [x,y,th] per end        10  [x,y,z,th,ph] per end
boundary DOFs ``q_b``  8   (2 nodes per end)       12  (2 nodes per end)
elastic energy         ``der``                     ``der3d``
obstacles              ``CircleObstacle``          ``SphereObstacle``
=====================  ==========================  ==========================

End-effector pose
-----------------
Each end is clamped: its position *and* its tangent direction are prescribed.
In 2D the tangent needed one angle; in 3D it needs two, so each end's control
is ``(x, y, z, theta, phi)``::

    d(theta, phi) = ( cos(phi) * cos(theta),
                      cos(phi) * sin(theta),
                      sin(phi) )

``theta`` is the azimuth in the x-y plane and ``phi`` the elevation out of it,
so ``phi = 0`` reproduces the 2D convention ``d = (cos th, sin th)`` exactly.
The two clamped nodes at each end are then::

    node_0     = (x_l, y_l, z_l)
    node_1     = node_0 + deltaL * d(th_l, ph_l)
    node_{n-2} = node_{n-1} - deltaL * d(th_r, ph_r)
    node_{n-1} = (x_r, y_r, z_r)

Gravity
-------
``gravity`` is a full 3-vector and defaults to ``(0, -9.81, 0)`` — i.e. y is
"down", matching the 2D environment — so a 3D run confined to z = 0 with
``phi = 0`` reproduces the 2D run node for node.  Set it to
``(0, 0, -9.81)`` for the more usual z-up convention.
"""
import numpy as np
import matplotlib.pyplot as plt

from contact import get_contact_forces
from energy import build_energy
from rodframes import RodFrames
from newton import solve_equilibrium

############################


def tangent_from_angles(theta, phi):
    """Unit tangent for azimuth ``theta`` and elevation ``phi``."""
    cp = np.cos(phi)
    return np.array([cp * np.cos(theta), cp * np.sin(theta), np.sin(phi)], dtype=float)


def objfun_3d(q_old, u_old, dt, tol, maximum_iter,
              m, mMat, energy, W, deltaL, free_index,
              control_node=None,
              use_inertia=False,
              obstacles=None,
              newton_cfg=None,
              fixed_index=None,
              n_pos=None):
    """Single-step implicit (quasi-static) equilibrium solve.

    Thin wrapper around :func:`newton.solve_equilibrium`: it only applies the
    clamped boundary DOFs and hands the rest to the shared globalized Newton
    solver. ``newton_cfg`` is the ``newton:`` block of the YAML config.
    """
    if control_node is None:
        raise ValueError("Control node position must be provided.")

    q_start = q_old.copy()
    # `control_node` is listed in exactly `free_index`'s complement order, so a
    # single scatter handles both the twist-free layout (12 clamped position
    # DOFs) and the twisting one (12 positions + the two end twist angles).
    q_start[fixed_index] = control_node

    cfg = dict(newton_cfg or {})
    q_new, flag, J, info = solve_equilibrium(
        q_start,
        energy=energy,
        W=W,
        free_index=free_index,
        dim=3,
        tol=tol,
        obstacles=obstacles or [],
        n_pos=n_pos,
        max_iter=maximum_iter,
        q_old=q_old,
        u_old=u_old,
        dt=dt,
        m=m,
        mMat=mMat,
        use_inertia=use_inertia,
        line_search=cfg.get("line_search", True),
        max_backtrack=cfg.get("max_backtrack", 12),
        nonmonotone_memory=cfg.get("nonmonotone_memory", 8),
        reg=cfg.get("reg", 0.0),
    )
    return q_new, flag, J, info


############################
# SimulatorEnv_3D
############################

class SimulatorEnv_3D:
    """
    A standardized 3D elastic-rod simulation environment:
      - Control input: (x, y, z, theta, phi) for each end, given externally.
      - State: internal rod node positions q, velocity u, current time ctime, etc.
      - Interface: reset(), step(control), render()
    """

    #: Spatial dimension — lets generic code branch without isinstance checks.
    dim = 3

    def __init__(self,
                 nv=51,
                 dt=0.01,
                 rod_length=1.0,
                 total_time=10.0,
                 R_outer=0.013,
                 r_inner=0.011,
                 E_al=70e9,
                 rho_al=2700,
                 max_newton_iter=1000,
                 save_history=False,
                 gravity=(0.0, -9.81, 0.0),
                 twist=False,
                 poisson=0.3,
                 energy_cfg=None,
                 energy=None,
                 newton_cfg=None,
                 warn_on_newton_failure=False):
        self.nv = nv
        self.dt = dt
        self.rod_length = rod_length
        self.total_time = total_time
        self.R_outer = R_outer
        self.r_inner = r_inner
        self.E_al = E_al
        self.rho_al = rho_al
        self.max_newton_iter = max_newton_iter
        self.save_history = save_history
        self.gravity = np.asarray(gravity, dtype=float)
        #: Torsion on/off.  With twist the DOF vector grows by one angle per
        #: edge and each gripper gains a yaw input (see the module docstring).
        self.twist = bool(twist)
        self.poisson = float(poisson)
        # Newton solver options (the `newton:` block of the YAML config) and
        # a running count of steps whose equilibrium solve did not converge.
        self.newton_cfg = dict(newton_cfg or {})
        self.warn_on_newton_failure = warn_on_newton_failure
        self.n_newton_failures = 0

        # Cross-section and material parameters
        self.I_sec = np.pi * (R_outer ** 4 - r_inner ** 4) / 4
        self.A_sec = np.pi * (R_outer ** 2 - r_inner ** 2)
        self.EI = E_al * self.I_sec
        self.EA = E_al * self.A_sec

        # Spatial discretization
        self.deltaL = rod_length / (nv - 1)

        # Elastic energy model. Defaults to the built-in 3D DER; `energy_cfg`
        # is the `energy:` block of the YAML config, and `energy` lets a caller
        # inject a RodEnergy instance directly (overrides energy_cfg).
        # Material frames, only needed when the rod can twist.  They are state:
        # carried from step to step by parallel transport, like the warm start.
        self.frames = RodFrames(nv) if self.twist else None

        self.energy = energy if energy is not None else build_energy(
            energy_cfg, EI=self.EI, EA=self.EA, deltaL=self.deltaL, nv=nv, dim=3,
            twist=self.twist, frames=self.frames, poisson=self.poisson)

        # Node coordinates (initially a straight rod along +x)
        self.nodes = np.zeros((nv, 3))
        self.nodes[:, 0] = np.arange(nv) * self.deltaL

        # DOF layout: positions first, then (with twist) one angle per edge.
        self.n_pos = 3 * nv
        self.n_twist = (nv - 1) if self.twist else 0
        self.n_dof = self.n_pos + self.n_twist

        # Mass
        m_node = np.pi * (R_outer ** 2 - r_inner ** 2) * rod_length * rho_al / (nv - 1)
        self.m_node = m_node
        self.m = np.full(self.n_dof, m_node)
        self.mMat = np.diag(self.m)

        # Gravity (external load vector W, per DOF); it does no work on twist.
        self.W = np.zeros(self.n_dof)
        self.W[:self.n_pos] = np.tile(self.gravity, nv) * m_node

        # Newton iteration tolerance
        self.tol = self.energy.stiffness_scale / rod_length ** 2 * 1e-3

        # Constrained DOFs: the two nodes at each end are clamped by the
        # control, and with twist so is each end edge's twist angle (the
        # gripper yaw).
        clamped = [np.arange(0, 6), np.arange(3 * nv - 6, 3 * nv)]
        if self.twist:
            clamped.append(np.array([self.n_pos, self.n_pos + nv - 2]))
        self.fixed_index = np.concatenate(clamped)
        self.free_index = np.setdiff1d(np.arange(self.n_dof), self.fixed_index)

        # Mid-node index
        self.mid_node = nv // 2

        # Total number of steps
        self.Nsteps = int(round(total_time / dt))

        self.dlamda = 1 / self.Nsteps

        self.obstacles = []

        # State variables (initialized in reset())
        self.q = None
        self.u = None
        self.ctime = None
        self.current_step = None
        self.lamda = None

        # History log (optional)
        self.history = {
            "error": [],
            "control": [],
            "midx": [],
            "midy": [],
            "midz": [],
        }

    # ---------- obstacle management ----------

    def add_obstacle(self, obstacle):
        """Add a rigid spherical obstacle (see contact.SphereObstacle)."""
        self.obstacles.append(obstacle)

    def remove_obstacle(self, obstacle):
        """Remove a specific obstacle from the simulation."""
        self.obstacles.remove(obstacle)

    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles.clear()

    # ---------- inner tools ----------

    @property
    def control_size(self) -> int:
        """Length of the control vector: 10 without twist, 12 with yaw."""
        return 12 if self.twist else 10

    def _control_node_from_pose(self, control_vec):
        """Map the end-effector control onto the clamped boundary DOFs.

        Without twist (10D control)::

            [x_l, y_l, z_l, th_l, ph_l,  x_r, y_r, z_r, th_r, ph_r]
            -> [node_0, node_1, node_{n-2}, node_{n-1}]              (12)

        With twist each end gains a yaw (12D control)::

            [x_l, y_l, z_l, th_l, ph_l, yaw_l,  x_r, ..., yaw_r]
            -> [node_0, node_1, node_{n-2}, node_{n-1}, theta^0, theta^{n-2}]  (14)

        ``yaw`` is the rotation of the gripper's cross-section about the rod
        tangent, measured from that edge's reference director.  Because the
        reference frame is carried by twist-free parallel transport, holding
        yaw constant means applying no twist, and changing it twists the rod.
        The two angle inputs ``th``/``ph`` fix where the tangent *points*; yaw
        is the third rotational freedom, the spin about it.
        """
        control_vec = np.asarray(control_vec, dtype=float)
        k = 6 if self.twist else 5
        p_l, th_l, ph_l = control_vec[0:3], control_vec[3], control_vec[4]
        p_r, th_r, ph_r = control_vec[k:k + 3], control_vec[k + 3], control_vec[k + 4]

        n0 = p_l
        n1 = p_l + tangent_from_angles(th_l, ph_l) * self.deltaL
        nm2 = p_r - tangent_from_angles(th_r, ph_r) * self.deltaL
        nm1 = p_r
        out = [n0, n1, nm2, nm1]
        if self.twist:
            out.append(np.array([control_vec[5], control_vec[11]]))
        return np.concatenate(out)

    # Alias, so 2D code paths calling the 2D name keep working.
    _control_node_from_xytheta = _control_node_from_pose

    def positions(self, q=None):
        """The node positions as an ``(nv, 3)`` array.

        With twist the DOF vector also carries the per-edge angles, so this is
        the safe way to get coordinates out of ``q`` (plain ``reshape(-1, 3)``
        is only correct when twist is off).
        """
        q = self.q if q is None else q
        return np.asarray(q, dtype=float)[:self.n_pos].reshape(-1, 3)

    def twist_angles(self, q=None):
        """The per-edge twist angles, shape ``(nv - 1,)``; empty without twist."""
        q = self.q if q is None else q
        return np.asarray(q, dtype=float)[self.n_pos:]

    def _get_midpoint(self):
        """Returns the current midpoint coordinates (midx, midy, midz)."""
        a = 3 * self.mid_node
        return self.q[a], self.q[a + 1], self.q[a + 2]

    def _get_observation(self):
        """The observation contains only the midpoint coordinates."""
        midx, midy, midz = self._get_midpoint()
        return {
            "time": self.ctime,
            "midx": midx,
            "midy": midy,
            "midz": midz,
        }

    # ---------- API ----------

    def reset(self,
              init_x=None,
              init_y=None,
              init_z=None,
              init_theta=None,
              init_phi=None,
              init_yaw=None):
        """
        Resets the environment and returns the initial observation.
        The initial control value only affects the control applied on the
        first step; by default the end is at (rod_length, 0, 0, 0, 0).
        """
        if init_x is None:
            init_x = self.rod_length
        if init_y is None:
            init_y = 0.0
        if init_z is None:
            init_z = 0.0
        if init_theta is None:
            init_theta = 0.0
        if init_phi is None:
            init_phi = 0.0
        if init_yaw is None:
            init_yaw = 0.0

        self.q = np.zeros(self.n_dof)
        self.q[:self.n_pos] = self.nodes.reshape(-1)
        if self.twist:
            # A straight rod with a uniform twist angle carries no twist at
            # all: only differences between adjacent edges cost energy.
            self.q[self.n_pos:] = init_yaw
            self.frames.reset(self.nodes)

        self.u = np.zeros_like(self.q)
        self.ctime = 0.0
        self.current_step = 0
        self.lamda = 0.0

        # Clear history
        if self.save_history:
            for k in self.history:
                self.history[k] = []

        # No integration step is taken in reset(); just return the current observation.
        return self._get_observation()

    def step(self, control, use_inertia=False):
        """
        Single simulation step.
          control: 10D vector [x_l, y_l, z_l, th_l, ph_l, x_r, y_r, z_r, th_r, ph_r]
        Returns:
          obs, done, info
        """
        control = np.asarray(control, dtype=float).flatten()
        if control.shape[0] != self.control_size:
            layout = ("[x_l, y_l, z_l, th_l, ph_l, yaw_l, "
                      "x_r, y_r, z_r, th_r, ph_r, yaw_r]" if self.twist else
                      "[x_l, y_l, z_l, th_l, ph_l, x_r, y_r, z_r, th_r, ph_r]")
            raise ValueError(
                f"control must be a {self.control_size}D vector: {layout}")

        control_node = self._control_node_from_pose(control)

        q_new, flag, J, ninfo = objfun_3d(
            q_old=self.q,
            u_old=self.u,
            dt=self.dt,
            tol=self.tol,
            maximum_iter=self.max_newton_iter,
            m=self.m,
            mMat=self.mMat,
            energy=self.energy,
            W=self.W,
            deltaL=self.deltaL,
            free_index=self.free_index,
            control_node=control_node,
            use_inertia=use_inertia,
            obstacles=self.obstacles,
            newton_cfg=self.newton_cfg,
            fixed_index=self.fixed_index,
            n_pos=self.n_pos,
        )

        if flag < 0:
            self.n_newton_failures += 1
            if self.warn_on_newton_failure:
                print(f"[newton] step {self.current_step}: {ninfo['reason']} "
                      f"(residual {ninfo['residual']:.3e} vs tol {self.tol:.3e}, "
                      f"{ninfo['iterations']} iters)")

        # Update velocity and time
        u_new = (q_new - self.q) / self.dt
        self.q = q_new
        if self.twist:
            # Carry the reference directors onto the new configuration, by
            # twist-free parallel transport. This is the frame counterpart of
            # the warm start: it keeps the twist measured along one continuous
            # branch instead of jumping between frame choices.
            self.frames.advance(self.positions(q_new))
        self.u = u_new
        self.ctime += self.dt
        self.current_step += 1
        self.lamda = self.current_step * self.dlamda

        # Observation and status
        obs = self._get_observation()
        done = (self.current_step >= self.Nsteps) or (flag < 0)

        info = {
            "converged": (flag > 0),
            "control": control,
            "q": self.q.copy(),
            "u": self.u.copy(),
            "J": J,
            "free_index": self.free_index.copy(),
            "fixed_index": self.fixed_index.copy(),
            "control_node": control_node.copy(),  # clamped boundary DOFs
            "newton": ninfo,
        }

        # Save history
        if self.save_history:
            self.history["control"].append(control.copy())
            self.history["midx"].append(obs["midx"])
            self.history["midy"].append(obs["midy"])
            self.history["midz"].append(obs["midz"])

        return obs, done, info

    # ---------- RHC state checkpointing ----------

    def get_state(self):
        """Snapshot the current simulator state (used to warm-start RHC segments)."""
        state = {
            "q": self.q.copy(),
            "u": self.u.copy(),
            "ctime": self.ctime,
            "current_step": self.current_step,
            "lamda": self.lamda,
        }
        if self.twist:
            # The material frames are state too -- restoring q without them
            # would silently change what "zero twist" means.
            state["frames"] = self.frames.get_state()
        return state

    def set_state(self, state):
        """Restore a state previously returned by get_state()."""
        self.q = state["q"].copy()
        self.u = state["u"].copy()
        self.ctime = state["ctime"]
        self.current_step = state["current_step"]
        self.lamda = state["lamda"]
        if self.twist and "frames" in state:
            self.frames.set_state(state["frames"])

    def render(self, saveimg=False, show=True,
               xlim=None, ylim=None, zlim=None):
        """Visualizes the current rod shape in a 3D axes."""
        pts = self.positions()
        midx, midy, midz = self._get_midpoint()

        if not getattr(self, "_render_initialized", False):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection="3d")
            self._render_initialized = True
            plt.ion()

        ax = self.ax
        ax.cla()
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'ko-', markersize=3, label="rod")
        ax.plot([midx], [midy], [midz], 'ro', markersize=8, label="midpoint")

        for obs in self.obstacles:
            _plot_sphere(ax, obs.cx, obs.cy, obs.cz, obs.radius)

        ax.set_title(f"t = {self.ctime:.3f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        L = self.rod_length
        ax.set_xlim(*(xlim if xlim else (-0.2, L + 0.3)))
        ax.set_ylim(*(ylim if ylim else (-0.6 * L, 0.6 * L)))
        ax.set_zlim(*(zlim if zlim else (-0.6 * L, 0.6 * L)))
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:      # matplotlib < 3.3
            pass
        ax.legend()

        if saveimg:
            plt.savefig(f'imgs/render3d_{self.ctime:.3f}.png')
            print(f"Saved image: imgs/render3d_{self.ctime:.3f}.png")

        if show:
            plt.pause(0.01)


def _plot_sphere(ax, cx, cy, cz, radius, n=16):
    """Draw a translucent sphere (obstacle) on a 3D axes."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    xs = cx + radius * np.outer(np.cos(u), np.sin(v))
    ys = cy + radius * np.outer(np.sin(u), np.sin(v))
    zs = cz + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='steelblue', alpha=0.4, linewidth=0)


if __name__ == "__main__":
    # =========================================================
    # Demo: sweep the right end out of the x-y plane, so the rod
    # deforms in all three dimensions.
    # =========================================================
    L = 1.0
    T = 2.0

    env = SimulatorEnv_3D(nv=21, dt=0.01, rod_length=L, total_time=T,
                          R_outer=0.003, r_inner=0.0,
                          E_al=2e9, rho_al=1200, save_history=True)
    obs = env.reset()

    done = False
    while not done:
        lam = env.current_step / env.Nsteps
        control = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0,                    # left end: clamped at origin
            L - 0.15 * lam, 0.0, 0.35 * np.sin(np.pi * lam),  # right end position
            0.0, 0.6 * lam,                              # right end tangent angles
        ])
        obs, done, info = env.step(control)

    print("Final midpoint:", env._get_midpoint())
    print("Converged:", info["converged"])
