# -*- coding: utf-8 -*-
"""Sampling-based global planner for the point-to-point task.

An RRT over the right gripper's pose, following the planner of Yu et al.,
"Generalizable whole-body global manipulation of deformable linear objects by
dual-arm robot in 3-D constrained environments" (reference/), cut down to what
this repository's task has: one driven end (the left is held fixed), no robot
arm, no obstacles, and a goal on the mid node only.

What carries over from the paper
--------------------------------
* **Coarse planning model.**  The planner runs on its own environment, built
  from the same config with fewer nodes (``nv=11``: the paper's ``m = 10``
  feature points).  Its paths are therefore only approximately right for the
  fine rod the controller drives -- the modelling error the local stage is
  there to absorb.
* **Every node is a stable equilibrium.**  A node is reached by quasi-static
  stepping from its parent, each sub-step a Newton solve warm-started from the
  previous one.  That is the paper's ``ForwardPred`` (eq. 14): set the new end
  poses, then project onto the stable manifold starting from the old shape,
  which also respects that the rod is under-actuated.  Newton alone does not
  guarantee *stable*: compressing a straight rod it happily stays on the
  straight, unstable branch, so every sub-step goes through
  :func:`stable_step` (the C_stable constraint).
* **Two exploration modes** (Algorithm 6, Appendix C).  With probability
  ``p_ts`` an iteration is *task-space guided*: sample a mid-node position
  (the goal itself with probability ``p_goal``), take the node whose mid node
  is nearest, and move the gripper greedily toward it through the rod's local
  Jacobian.  Otherwise it is *full-space*: sample a gripper pose, take the
  nearest node in pose, and extend toward it.
* **Constrained extend** (Algorithm 5): grow in steps of ``eta`` until the
  target is reached, a step fails (Newton does not converge, or a constraint
  is violated), or the distance stops shrinking.
* **Constraints**: converged, stable equilibrium; no overstretch,
  ``|p_r - p_l| <= L - eps_stretch`` (eq. 26); the gripper inside a workspace:
  a position box and, optionally, limits on its angles (``ang_min/ang_max``).
* **ShortenPath**: random shortcuts, each accepted only if re-simulating the
  whole shortened path from the start still reaches the goal.

The result is a list of control waypoints; :func:`resample_path` turns it into
the ``N`` per-step controls of a rollout.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from implicit_grad_tools import local_dxf_dxb, _boundary_jacobian


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def min_stiffness_eig(env, info):
    """Smallest eigenvalue of J_ff (the free-DOF Hessian) at a solved step,
    relative to the bending stiffness scale ``EI / deltaL^3``.  Negative: the
    equilibrium is unstable (e.g. a compressed rod still straight)."""
    f = env.free_index
    J = info["J"][np.ix_(f, f)]
    return float(np.linalg.eigvalsh(0.5 * (J + J.T))[0]) / (env.EI / env.deltaL ** 3)


def stable_step(env, u, nudge=0.01, tol=-1e-8):
    """``env.step(u)``, but never left on an unstable equilibrium.

    The quasi-static Newton solve converges to *any* equilibrium near its warm
    start.  Compressing a straight rod, that is the straight configuration
    itself -- an exact equilibrium, but an unstable one, which a real rod
    leaves by buckling.  When the solve lands there, the free DOFs are nudged
    along the most negative stiffness mode (both ways, ``nudge`` metres at the
    largest node) and re-solved; the stable result with the lower energy wins.
    This is the paper's projection onto *stable* configurations
    (ProjectStableDLOConfig).  Returns the step's ``info`` with an extra
    ``"stable"`` flag; the env is left in the returned state.
    """
    state0 = env.get_state()
    _, _, info = env.step(u, use_inertia=False)
    if not info["converged"]:
        info["stable"] = False
        return info
    lam = min_stiffness_eig(env, info)
    if lam >= tol:
        info["stable"] = True
        return info
    f = env.free_index
    J = info["J"][np.ix_(f, f)]
    v = np.linalg.eigh(0.5 * (J + J.T))[1][:, 0]
    v = v / np.max(np.abs(v))
    q_bad = info["q"].copy()
    best = None
    for sgn in (1.0, -1.0):
        st = dict(state0)
        q = q_bad.copy()
        q[f] += sgn * nudge * v
        st["q"] = q
        env.set_state(st)
        _, _, inf2 = env.step(u, use_inertia=False)
        if inf2["converged"] and min_stiffness_eig(env, inf2) >= tol:
            E = env.energy.energy(inf2["q"]) - env.W @ inf2["q"]     # elastic + gravity
            if best is None or E < best[0]:
                best = (E, inf2, env.get_state())
    if best is None:
        env.set_state(state0)
        info["stable"] = False
        return info
    env.set_state(best[2])
    best[1]["stable"] = True
    return best[1]


@dataclass
class PlannerParams:
    goal_tol: float = 0.02          # [m] mid node within this of the target: done
    eta_pos: float = 0.05           # [m] max gripper translation per tree edge
    eta_rot: float = 0.25           # [rad] max gripper rotation per tree edge
    sub_pos: float = 0.01           # [m] max translation per Newton solve
    sub_rot: float = 0.05           # [rad] max rotation per Newton solve
    w_rot: float = 0.2              # [m/rad] rotation weight in the pose metric
    p_ts: float = 0.5               # P(task-space guided iteration)   (paper: 0.5)
    p_goal: float = 0.5             # P(goal | task-space sample)
    greedy_steps: int = 20          # max greedy Jacobian moves per task-space iteration
    eps_stretch: float = 0.0        # [m] overstretch margin (paper: 0.01; 0 here because
                                    # the task starts from the straight rod, |p_r - p_l| = L)
    min_end_gap: float = 0.05       # [m] keep the grippers apart
    phi_max: float = 1.3            # [rad] elevation limit, away from the pole
    box_min: tuple = (-0.3, -0.7, -0.6)
    box_max: tuple = (1.1, 0.7, 0.8)
    ang_min: tuple | None = None    # [rad] gripper angle limits, (theta, phi) in 3D; None:
    ang_max: tuple | None = None    # theta free (wraps), phi within +-phi_max
    max_iter: int = 3000
    time_limit: float = 600.0       # [s]
    shortcut_iters: int = 40
    stable_only: bool = True        # every node a stable equilibrium (paper: C_stable)
    seed: int = 0


@dataclass
class Node:
    u: np.ndarray                   # full control vector (both ends)
    state: dict                     # env.get_state() at the equilibrium
    J: np.ndarray                   # equilibrium Jacobian there
    parent: int = -1
    mid: np.ndarray = field(default=None)
    pos: np.ndarray = field(default=None)


class RRTPlanner:
    """RRT for the right gripper on a quasi-static rod (``env``, usually coarse)."""

    def __init__(self, env, u0, target, params: PlannerParams | None = None):
        self.env = env
        self.p = params or PlannerParams()
        self.u0 = np.asarray(u0, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.dim = env.dim
        self.twist = bool(getattr(env, "twist", False))
        self.k = len(self.u0) // 2                    # controls per end
        self.L = env.rod_length
        self.rng = np.random.default_rng(self.p.seed)
        self.nodes: list[Node] = []
        self.n_solves = 0
        self.n_failed = 0
        # mid-node rows of the free DOFs, for the greedy Jacobian step
        self.mid_rows = np.array([int(np.where(env.free_index == self.dim * env.mid_node + c)[0][0])
                                  for c in range(self.dim)])
        # A free azimuth wraps, so the short way round is a different raw
        # value.  Inside angle limits the raw difference is the rotation: the
        # short way round could cross the excluded side.
        self._wrap_az = _wrap if self.p.ang_min is None else (lambda a: a)
        if self.p.ang_min is not None:
            self._ang = slice(self.dim, self.dim + len(self.p.ang_min))

    # ------------------------------------------------------------ geometry
    def _right(self, u):
        return u[self.k:2 * self.k]

    def _pose_delta(self, ua, ub):
        """(translation, rotation) between the right-end poses of two controls."""
        ra, rb = self._right(ua), self._right(ub)
        dp = np.linalg.norm(rb[:self.dim] - ra[:self.dim])
        da = rb[self.dim:] - ra[self.dim:]
        da[0] = self._wrap_az(da[0])
        return dp, float(np.linalg.norm(da))

    def pose_dist(self, ua, ub):
        dp, da = self._pose_delta(ua, ub)
        return dp + self.p.w_rot * da

    def _unwrap_toward(self, u_from, u_to):
        """``u_to`` with its azimuth moved to the branch nearest ``u_from``."""
        u = u_to.copy()
        i = self.k + self.dim
        u[i] = u_from[i] + self._wrap_az(u_to[i] - u_from[i])
        return u

    def _valid(self, u):
        r = self._right(u)
        p_r, p_l = r[:self.dim], u[:self.dim]
        gap = np.linalg.norm(p_r - p_l)
        if gap > self.L - self.p.eps_stretch or gap < self.p.min_end_gap:
            return False
        if np.any(p_r < np.asarray(self.p.box_min)[:self.dim]) or \
           np.any(p_r > np.asarray(self.p.box_max)[:self.dim]):
            return False
        if self.dim == 3 and abs(r[self.dim + 1]) > self.p.phi_max:
            return False
        if self.p.ang_min is not None and (np.any(r[self._ang] < self.p.ang_min) or
                                           np.any(r[self._ang] > self.p.ang_max)):
            return False
        return True

    def _project_valid(self, u):
        """Nearest-ish valid control: clip to the box and the angle limits,
        then pull or push the gripper along the line to the left end into the
        allowed gap.

        A box that contains the left end (the default one) keeps that last
        step inside it: clipping never lengthens the gap, and scaling toward
        the left end never leaves the box.  This is what lets an edge slide
        along the overstretch boundary instead of dying on it -- the start
        itself sits exactly there (straight rod, gap = L).  With a workspace
        box that excludes the left end the result can be invalid, and the
        simulation then rejects the edge.
        """
        u = np.array(u, dtype=float)
        r = u[self.k:2 * self.k]                      # a view: edits land in u
        p_l = u[:self.dim]
        r[:self.dim] = np.clip(r[:self.dim], np.asarray(self.p.box_min)[:self.dim],
                               np.asarray(self.p.box_max)[:self.dim])
        d = r[:self.dim] - p_l
        gap = np.linalg.norm(d)
        g_max = self.L - self.p.eps_stretch - 1e-9
        if gap > g_max:
            r[:self.dim] = p_l + d * (g_max / gap)
        elif 0.0 < gap < self.p.min_end_gap:
            r[:self.dim] = p_l + d * (self.p.min_end_gap / gap)
        if self.dim == 3:
            r[self.dim + 1] = np.clip(r[self.dim + 1], -self.p.phi_max, self.p.phi_max)
        if self.p.ang_min is not None:
            r[self._ang] = np.clip(r[self._ang], self.p.ang_min, self.p.ang_max)
        return u

    # ---------------------------------------------------------- simulation
    def _mid(self, q):
        a = self.dim * self.env.mid_node
        return np.asarray(q[a:a + self.dim], dtype=float)

    def _make_node(self, u, info, parent):
        q = info["q"]
        return Node(u=u.copy(), state=self.env.get_state(), J=info["J"], parent=parent,
                    mid=self._mid(q),
                    pos=np.asarray(q[:self.dim * self.env.nv], dtype=float).reshape(-1, self.dim))

    def _n_sub(self, ua, ub):
        # Raw angle difference, not the wrapped one: simulate() interpolates
        # the raw values, so that is the rotation actually performed.
        ra, rb = self._right(ua), self._right(ub)
        dp = np.linalg.norm(rb[:self.dim] - ra[:self.dim])
        da = np.linalg.norm(rb[self.dim:] - ra[self.dim:])
        return max(1, int(np.ceil(max(dp / self.p.sub_pos, da / self.p.sub_rot))))

    def simulate(self, state, u_from, u_to):
        """Quasi-static move from ``state`` (at ``u_from``) to ``u_to``.

        Returns the last step's ``info`` (env left at the end state), or None
        if a sub-step fails to converge or leaves the valid set.
        """
        self.env.set_state(state)
        n = self._n_sub(u_from, u_to)
        info = None
        for j in range(1, n + 1):
            info = self._step(u_from + (u_to - u_from) * (j / n))
            if info is None:
                return None
        return info

    def _step(self, u):
        """One validated quasi-static solve at ``u``; its info, or None."""
        if not self._valid(u):
            return None
        if self.p.stable_only:
            info = stable_step(self.env, u)
            ok = info["stable"]
        else:
            _, _, info = self.env.step(u, use_inertia=False)
            ok = info["converged"]
        self.n_solves += 1
        if not ok:
            self.n_failed += 1
            return None
        return info

    def trace(self, U):
        """The mid-node path of waypoints ``U`` as the planner simulates it.

        Returns ``(s, mids)``: the pose-metric arc length of every sub-step
        (the parameter :func:`resample_path` spaces the rollout steps by) and
        the mid node there, or None if the path does not replay.
        """
        info = self.simulate(self.start_state, U[0], U[0])
        if info is None:
            return None
        s, mids, s0 = [0.0], [self._mid(info["q"])], 0.0
        for a, b in zip(U[:-1], U[1:]):
            seg = path_length(np.stack([a, b]), self.k, self.dim, self.p.w_rot)[-1]
            n = self._n_sub(a, b)
            for j in range(1, n + 1):
                info = self._step(a + (b - a) * (j / n))
                if info is None:
                    return None
                s.append(s0 + seg * j / n)
                mids.append(self._mid(info["q"]))
            s0 += seg
        return np.array(s), np.array(mids)

    def _steer(self, i_from, u_to):
        """One edge of at most ``eta`` toward ``u_to``; the new node or None."""
        nf = self.nodes[i_from]
        u_to = self._unwrap_toward(nf.u, u_to)
        dp, da = self._pose_delta(nf.u, u_to)
        s = min(1.0, self.p.eta_pos / max(dp, 1e-12), self.p.eta_rot / max(da, 1e-12))
        u_new = self._project_valid(nf.u + s * (u_to - nf.u))
        if self.pose_dist(nf.u, u_new) < 1e-6:
            return None
        info = self.simulate(nf.state, nf.u, u_new)
        if info is None:
            return None
        return self._make_node(u_new, info, i_from)

    def _add(self, node):
        n = len(self.nodes)
        if n == len(self._R):                         # grow the search arrays
            self._R = np.concatenate([self._R, np.zeros_like(self._R)])
            self._M = np.concatenate([self._M, np.zeros_like(self._M)])
        self._R[n] = self._right(node.u)
        self._M[n] = node.mid
        self.nodes.append(node)
        return n

    def _nearest_pose(self, u):
        """Index of the node nearest ``u`` in the (vectorized) pose metric."""
        R = self._R[:len(self.nodes)]
        r = self._right(u)
        dp = np.linalg.norm(R[:, :self.dim] - r[:self.dim], axis=1)
        da = R[:, self.dim:] - r[self.dim:]
        da[:, 0] = self._wrap_az(da[:, 0])
        return int(np.argmin(dp + self.p.w_rot * np.linalg.norm(da, axis=1)))

    def _nearest_mid(self, m, exclude=()):
        d = np.linalg.norm(self._M[:len(self.nodes)] - m, axis=1)
        if exclude:
            d[list(exclude)] = np.inf
        i = int(np.argmin(d))
        return i if np.isfinite(d[i]) else None

    def _goal_reached(self, i):
        return np.linalg.norm(self.nodes[i].mid - self.target) <= self.p.goal_tol

    # ---------------------------------------------------- exploration modes
    def _extend_full(self, u_rand):
        """Algorithm 5: extend the pose-nearest node toward ``u_rand``."""
        i = self._nearest_pose(u_rand)
        d_last = self.pose_dist(self.nodes[i].u, u_rand)
        while True:
            if d_last < 1e-6:
                return i
            new = self._steer(i, u_rand)
            if new is None:
                return i
            d = self.pose_dist(new.u, u_rand)
            if d >= d_last:
                return i
            i, d_last = self._add(new), d
            if self._goal_reached(i):
                return i

    def mid_jacobian(self, node):
        """d(mid node) / d(right-end control) at ``node``, shape (dim, k)."""
        env = self.env
        dxf_dxb = local_dxf_dxb(node.J, env.free_index, env.fixed_index, 1e-9)
        B = _boundary_jacobian(node.u, env.deltaL, self.dim, self.twist,
                               getattr(env, "boundary_rows", None))
        return dxf_dxb[self.mid_rows] @ B[:, self.k:2 * self.k]

    def _extend_task(self, m_rand, toward_goal=False):
        """Task-space guided: greedy Jacobian moves of the mid-nearest node.

        Greedy descent toward the fixed goal is deterministic, so a run that
        stalls short of it (a local minimum of the mid-node distance on that
        equilibrium branch) would stall again from the same node -- and the
        nodes it created lead straight back into the same basin.  The
        goal-nearest node would then be retried forever.  So when a
        goal-directed run fails, its start node and everything it added are
        dropped from the goal-directed nearest-neighbour search
        (``self.dead``); they stay available to every other kind of extension.
        """
        i = self._nearest_mid(m_rand, exclude=self.dead if toward_goal else ())
        if i is None:
            return 0
        first, n_before = i, len(self.nodes)
        # Column scaling: a radian of gripper rotation costs like w_rot metres.
        S = np.ones(self.k)
        S[self.dim:] = 1.0 / self.p.w_rot
        for _ in range(self.p.greedy_steps):
            nd = self.nodes[i]
            e = m_rand - nd.mid
            if np.linalg.norm(e) < 1e-3:
                break
            G = self.mid_jacobian(nd) * S
            du = S * (G.T @ np.linalg.solve(G @ G.T + 1e-4 * np.eye(self.dim), e))
            u_to = nd.u.copy()
            u_to[self.k:2 * self.k] += du
            new = self._steer(i, u_to)
            if new is None or np.linalg.norm(m_rand - new.mid) >= np.linalg.norm(e):
                break
            i = self._add(new)
            if self._goal_reached(i):
                break
        if toward_goal and not self._goal_reached(i):
            self.dead.add(first)
            self.dead.update(range(n_before, len(self.nodes)))
        return i

    def _sample_pose(self):
        while True:
            u = self.u0.copy()
            r = u[self.k:2 * self.k]
            r[:self.dim] = self.rng.uniform(self.p.box_min, self.p.box_max)[:self.dim]
            if self.p.ang_min is not None:
                r[self._ang] = self.rng.uniform(self.p.ang_min, self.p.ang_max)
            else:
                r[self.dim] = self.rng.uniform(-np.pi, np.pi)
                if self.dim == 3:
                    r[self.dim + 1] = self.rng.uniform(-self.p.phi_max, self.p.phi_max)
            if self._valid(u):
                return u

    def _sample_mid(self):
        """(mid-node sample, whether it is the goal)."""
        if self.rng.random() < self.p.p_goal:
            return self.target.copy(), True
        # The mid node is half a rod length along the rod from the fixed end.
        while True:
            m = self.rng.uniform(-0.5 * self.L, 0.5 * self.L, size=self.dim)
            if np.linalg.norm(m - self.u0[:self.dim]) < 0.5 * self.L:
                return m, False

    # ------------------------------------------------------------- planning
    def plan(self, start_state):
        """Grow the tree from ``start_state`` (rod at rest, controls ``u0``).

        Returns a dict with the waypoints (``U``, shape (M, n_u)), the planned
        mid-node path at those waypoints and planning statistics.
        """
        t0 = time.time()
        info = self.simulate(start_state, self.u0, self.u0)      # settle under gravity
        if info is None:
            raise RuntimeError("the start configuration does not converge")
        self.nodes, self.dead = [], set()
        self._R = np.zeros((256, self.k))
        self._M = np.zeros((256, self.dim))
        self._add(self._make_node(self.u0, info, -1))
        self.start_state = start_state

        goal = 0 if self._goal_reached(0) else None
        it = 0
        while goal is None and it < self.p.max_iter and time.time() - t0 < self.p.time_limit:
            it += 1
            if self.rng.random() < self.p.p_ts:
                i = self._extend_task(*self._sample_mid())
            else:
                i = self._extend_full(self._sample_pose())
            if self._goal_reached(i):
                goal = i
        t_search = time.time() - t0
        if goal is None:
            return {"success": False, "iterations": it, "nodes": len(self.nodes),
                    "time": t_search, "n_solves": self.n_solves}

        path = []
        j = goal
        while j >= 0:
            path.append(self.nodes[j].u)
            j = self.nodes[j].parent
        U_raw = np.array(path[::-1])

        t1 = time.time()
        U = self.shorten(U_raw)
        t_short = time.time() - t1
        ok, mids, err = self.check_path(U)
        return {
            "success": True, "U_raw": U_raw, "U": U, "mid_waypoints": mids,
            "planned_error": err, "iterations": it, "nodes": len(self.nodes),
            "time_search": t_search, "time_shorten": t_short,
            "time": t_search + t_short, "n_solves": self.n_solves,
        }

    def _run_path(self, U, state=None, start=0):
        """Simulate waypoints ``U[start:]`` from ``state`` (default: the rest
        state, settling at ``U[0]``).  Returns (states, positions) at each
        simulated waypoint, or None if a step fails."""
        if state is None:
            info = self.simulate(self.start_state, U[0], U[0])
            if info is None:
                return None
            states, pos = [self.env.get_state()], [self._positions(info["q"])]
            start = 0
        else:
            states, pos = [], []
        prev_state = state
        for a in range(start, len(U) - 1):
            info = self.simulate(states[-1] if states else prev_state, U[a], U[a + 1])
            if info is None:
                return None
            states.append(self.env.get_state())
            pos.append(self._positions(info["q"]))
        return states, pos

    def _positions(self, q):
        return np.asarray(q[:self.dim * self.env.nv], dtype=float).reshape(-1, self.dim)

    def check_path(self, U):
        """Re-simulate ``U`` from the start; (ok, mid at each waypoint, final error)."""
        res = self._run_path(U)
        if res is None:
            return False, None, np.inf
        mids = np.array([p[self.env.mid_node] for p in res[1]])
        err = float(np.linalg.norm(mids[-1] - self.target))
        return err <= self.p.goal_tol, mids, err

    def shorten(self, U, match_tol=2e-3, time_limit=180.0):
        """Random shortcuts (ShortenPath), each kept only if the path still works.

        The equilibrium at every waypoint is cached, so a shortcut ``i -> j``
        only re-simulates its own segment: if the rod arrives at waypoint
        ``j`` in the shape it had there before (``match_tol``), everything
        after ``j`` is unchanged; otherwise the rest of the path is
        re-simulated and must still reach the goal.  The result is verified
        once more from the start; on failure the input path is returned.
        """
        U0 = np.array(U)
        U = U0.copy()
        res = self._run_path(U)
        if res is None:
            return U0
        states, pos = res
        t0 = time.time()
        for _ in range(self.p.shortcut_iters):
            if len(U) < 3 or time.time() - t0 > time_limit:
                break
            i, j = sorted(self.rng.choice(len(U), size=2, replace=False))
            if j - i < 2:
                continue
            # Take the short way round from U[i] to U[j]: shift the azimuth of
            # the whole tail by the multiple of 2*pi that does it (the same
            # gripper poses, so the cached states stay valid).
            a = self.k + self.dim
            tail_wp = U[j:].copy()
            tail_wp[:, a] += (U[i, a] + self._wrap_az(U[j, a] - U[i, a])) - U[j, a]
            info = self.simulate(states[i], U[i], tail_wp[0])
            if info is None:
                continue
            p_j = self._positions(info["q"])
            s_j = self.env.get_state()
            if np.max(np.linalg.norm(p_j - pos[j], axis=1)) <= match_tol:
                tail_states, tail_pos = states[j + 1:], pos[j + 1:]
            else:
                rest = self._run_path(tail_wp, state=s_j, start=0)
                if rest is None:
                    continue
                tail_states, tail_pos = rest
                if np.linalg.norm((tail_pos[-1] if tail_pos else p_j)[self.env.mid_node]
                                  - self.target) > self.p.goal_tol:
                    continue
            U = np.concatenate([U[:i + 1], tail_wp])
            states = states[:i + 1] + [s_j] + list(tail_states)
            pos = pos[:i + 1] + [p_j] + list(tail_pos)
        ok, _, _ = self.check_path(U)
        return U if ok else U0


def path_length(U, k, dim, w_rot):
    """Cumulative pose-metric length along waypoints ``U`` (right end).

    Raw angle differences: between waypoints the planner moves linearly in
    the raw control values, so that is the rotation the path performs.
    """
    R = U[:, k:2 * k]
    dp = np.linalg.norm(np.diff(R[:, :dim], axis=0), axis=1)
    da = np.linalg.norm(np.diff(R[:, dim:], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(dp + w_rot * da)])


def resample_path(U, N, k, dim, w_rot=0.2):
    """``N`` controls at constant pose-metric speed along the waypoints.

    Linear in the raw control values between waypoints -- exactly the motion
    the planner simulated and validated (wrapping the azimuth here would turn
    a validated +4 rad rotation into a -2.3 rad one, and the rod onto another
    equilibrium branch).

    Row ``j`` is the control of rollout step ``j``, i.e. at ``lambda = (j+1)/N``
    (``train_adjoint.rollout_segment`` integrates before it steps), so the last
    row is the final waypoint and ``U[0]`` itself (the start) is not repeated.
    """
    U = np.array(U, dtype=float)
    s = path_length(U, k, dim, w_rot)
    if s[-1] <= 0:
        return np.repeat(U[-1:], N, axis=0)
    s_q = s[-1] * np.arange(1, N + 1) / N
    return np.stack([np.interp(s_q, s, U[:, c]) for c in range(U.shape[1])], axis=1)
