# -*- coding: utf-8 -*-
"""
Contact mechanics for the elastic rod simulation (2D and 3D).

Penalty-energy approach:
    E_c = 0.5 * k_contact * max(0, r - dist)^2

where dist is the distance from a rod node to the obstacle center.
E_c → ∞ as penetration grows; E_c = 0 when the node is outside.

LPipe2D (bottom of the file) is the rod--wall counterpart: a smooth penalty on
the signed clearance from a rigid 90-degree pipe, sampled along the edges.
"""
import numpy as np


class CircleObstacle:
    """
    Rigid circular obstacle that interacts with an elastic rod via contact forces.

    When a rod node at (xk, yk) penetrates the circle (dist < radius), a
    repulsive force along the outward normal pushes it back out:

        F_c = k_contact * phi * n        (phi = radius - dist,  n = outward normal)

    The corresponding force Jacobian (needed by the Newton-Raphson solver) is:

        J_c = k_contact * [ (phi/dist)·I  -  (1 + phi/dist)·n⊗n ]
    """

    def __init__(self, cx, cy, radius, k_contact=1e7):
        """
        Parameters
        ----------
        cx, cy      : float – obstacle center
        radius      : float – obstacle radius
        k_contact   : float – penalty stiffness; tune to ~10–100× EI/L^2
        """
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.k_contact = k_contact

    def move_to(self, cx, cy):
        """Translate the obstacle center to (cx, cy)."""
        self.cx = cx
        self.cy = cy

    def node_energy(self, xk, yk):
        """Penalty energy ``E_c = 0.5 * k * max(0, radius - dist)^2`` of one node.

        The energy the force and Jacobian above differentiate; the energy
        descent of the snap rule (src/newton.py: minimize_potential) needs the
        value itself, which is the total potential's contact term.
        """
        phi = self.radius - float(np.hypot(xk - self.cx, yk - self.cy))
        return 0.5 * self.k_contact * phi * phi if phi > 0.0 else 0.0

    def node_contact(self, xk, yk):
        """
        Contact force and force Jacobian for a single rod node at (xk, yk).

        Returns
        -------
        F : ndarray (2,)   – contact force  = -∇ E_c
        J : ndarray (2, 2) – force Jacobian = -Hess E_c
        """
        dx = xk - self.cx
        dy = yk - self.cy
        dist = np.sqrt(dx * dx + dy * dy)

        # Degenerate case: node exactly at center
        if dist < 1e-10:
            return np.array([self.k_contact * self.radius, 0.0]), np.zeros((2, 2))

        phi = self.radius - dist  # > 0 means penetration (inside obstacle)

        if phi <= 0.0:
            return np.zeros(2), np.zeros((2, 2))

        n = np.array([dx, dy]) / dist  # unit outward normal (center → node)

        # Contact force: repels node along outward normal
        F = self.k_contact * phi * n

        # Force Jacobian = -Hess(E_c)
        # Hess(E_c) = k * [ n⊗n·(1 + phi/dist)  −  (phi/dist)·I ]
        pn = phi / dist
        J = self.k_contact * (pn * np.eye(2) - (1.0 + pn) * np.outer(n, n))

        return F, J


class SphereObstacle:
    """Rigid spherical obstacle — the 3D counterpart of :class:`CircleObstacle`.

    Same penalty energy, same force and Jacobian formulas, just with a 3-vector
    center and 3x3 blocks::

        E_c = 0.5 * k_contact * max(0, radius - dist)^2
        F_c = k_contact * phi * n
        J_c = k_contact * [ (phi/dist)·I  -  (1 + phi/dist)·n⊗n ]
    """

    def __init__(self, cx, cy, cz, radius, k_contact=1e7):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.radius = radius
        self.k_contact = k_contact

    @property
    def center(self):
        return np.array([self.cx, self.cy, self.cz], dtype=float)

    def move_to(self, cx, cy, cz):
        """Translate the obstacle center to (cx, cy, cz)."""
        self.cx = cx
        self.cy = cy
        self.cz = cz

    def node_energy(self, xk, yk, zk):
        """``E_c = 0.5 * k * max(0, radius - dist)^2`` of one node (see CircleObstacle)."""
        d = np.array([xk - self.cx, yk - self.cy, zk - self.cz], dtype=float)
        phi = self.radius - float(np.linalg.norm(d))
        return 0.5 * self.k_contact * phi * phi if phi > 0.0 else 0.0

    def node_contact(self, xk, yk, zk):
        """Contact force (3,) and force Jacobian (3, 3) for a single rod node."""
        d = np.array([xk - self.cx, yk - self.cy, zk - self.cz], dtype=float)
        dist = float(np.linalg.norm(d))

        # Degenerate case: node exactly at center
        if dist < 1e-10:
            return np.array([self.k_contact * self.radius, 0.0, 0.0]), np.zeros((3, 3))

        phi = self.radius - dist  # > 0 means penetration (inside obstacle)
        if phi <= 0.0:
            return np.zeros(3), np.zeros((3, 3))

        n = d / dist  # unit outward normal (center -> node)
        F = self.k_contact * phi * n

        pn = phi / dist
        J = self.k_contact * (pn * np.eye(3) - (1.0 + pn) * np.outer(n, n))

        return F, J


class PegObstacleNode:
    """LEGACY node-based peg (``peg.model: node``): kept to reproduce runs made
    before 2026-11-24.  It is NOT the contact model of the paper (Sec. VI-B):
    the penalty is a one-sided quadratic on the nodes only (C1, its Hessian
    jumps at contact onset), and the clearance ignores the rod radius.  Use
    :class:`PegObstacle` (the default).

    Rigid peg: a cylinder parallel to y, 3D only (a capsule at its two ends).

    Seen in the x-z plane it is a disc of ``radius`` at ``(cx, cz)``; along the
    axis it spans ``cy +- half_length``.  A node's distance is taken to the
    axis *segment*, so the penalty is :class:`SphereObstacle`'s with the nearest
    axis point in place of the center::

        d = p - (cx, clip(y, cy-h, cy+h), cz),   E_c = 0.5 k max(0, radius - |d|)^2
        F_c = k phi n,    J_c = J_sphere(d) @ P

    ``P = diag(1, 0, 1)`` while the node is within the axis range (the nearest
    point follows y), the identity outside it.  Contact is node-based, so keep
    ``radius`` above the node spacing or an edge can pass between two nodes.
    """

    def __init__(self, cx, cy, cz, radius, half_length, k_contact=1e7):
        self.cx, self.cy, self.cz = cx, cy, cz
        self.radius = radius
        self.half_length = half_length
        self.k_contact = k_contact

    def _offset(self, xk, yk, zk):
        """``d`` (3,) from the nearest axis point and whether the node is inside the y range."""
        yc = min(max(yk, self.cy - self.half_length), self.cy + self.half_length)
        return np.array([xk - self.cx, yk - yc, zk - self.cz], dtype=float), yc == yk

    def bounds(self):
        """``(lo, hi)`` corners of the box around the peg."""
        c = np.array([self.cx, self.cy, self.cz], dtype=float)
        e = np.array([self.radius, self.radius + self.half_length, self.radius])
        return c - e, c + e

    def node_energy(self, xk, yk, zk):
        d, _ = self._offset(xk, yk, zk)
        phi = self.radius - float(np.linalg.norm(d))
        return 0.5 * self.k_contact * phi * phi if phi > 0.0 else 0.0

    def node_contact(self, xk, yk, zk):
        """Contact force (3,) and force Jacobian (3, 3) for a single rod node."""
        d, inside = self._offset(xk, yk, zk)
        dist = float(np.linalg.norm(d))
        if dist < 1e-10:                       # on the axis: push out along +z
            return np.array([0.0, 0.0, self.k_contact * self.radius]), np.zeros((3, 3))
        phi = self.radius - dist
        if phi <= 0.0:
            return np.zeros(3), np.zeros((3, 3))
        n = d / dist
        pn = phi / dist
        J = self.k_contact * (pn * np.eye(3) - (1.0 + pn) * np.outer(n, n))
        if inside:
            J[:, 1] = 0.0
        return self.k_contact * phi * n, J

    def draw_surface(self, ax, alpha=0.35, color="steelblue"):
        """Translucent cylinder on a 3D axes; returns the artists."""
        th = np.linspace(0.0, 2 * np.pi, 32)
        ys = np.linspace(self.cy - self.half_length, self.cy + self.half_length, 2)
        T, Y = np.meshgrid(th, ys)
        surf = ax.plot_surface(self.cx + self.radius * np.cos(T), Y,
                               self.cz + self.radius * np.sin(T), color=color, alpha=alpha,
                               linewidth=0, shade=True)
        return [surf]


def _edge_samples(nv, deltaL, m):
    """Quadrature points of the edges: ``(edge, fraction, weight)``.

    ``m`` points per edge at fractions 0, 1/m, ..., (m-1)/m, plus the last
    node; trapezoidal rest-length weights, so ``sum(weight) = rod length``.
    """
    frac = np.tile(np.arange(m) / m, nv - 1)
    edge = np.repeat(np.arange(nv - 1), m)
    edge = np.append(edge, nv - 2)
    frac = np.append(frac, 1.0)
    w = np.full(edge.size, deltaL / m)
    w[0] *= 0.5
    w[-1] *= 0.5
    return edge, frac, w


class PegObstacle:
    """Rigid peg, a cylinder with its axis along y (3D) -- the contact model of
    the paper, Sec. VI-B, eqs. (contact:energy)-(contact:peg).

    Edge-based smooth penalty after the implicit contact model (Choi et al.
    2021, Tong et al. 2023): on every edge ``(x_j, x_{j+1})`` the quadrature
    points ``p = (1-s) x_j + s x_{j+1}`` carry rest-length weights ``w``, and::

        E_c = sum_p  w * k_c / 2 * h(d(p))^2,     h(d) = delta * log(1 + exp(-d / delta))
        d(p) = |(I - a a^T)(p - c)| - (radius + rod_radius)

    ``d`` is the signed clearance of the rod *surface* (rod radius included),
    positive outside.  ``k_c`` [N/m^2] is a stiffness per unit rod length,
    ``delta`` [m] the smoothing length; the penalty tends to the one-sided
    quadratic ``k_c/2 * max(0, -d)^2`` as ``delta -> 0``.  E_c is C-infinity
    wherever no sample point is on the axis, so Assumption 1 holds.

    The peg is finite, ``cy +- half_length`` along y; beyond its ends the
    distance is taken to the end point of the axis (a capsule).  That cap is
    only C1 where it meets the cylinder; :meth:`report` flags a contact there
    (``end_contact``), which the experiments must not have.

    The whole-rod interface (``rod_contact``, ``compute_contact_energy``) is
    what :func:`get_contact_forces` / :func:`get_contact_energy` use, as for
    the pipe.
    """

    D = 3

    def __init__(self, cx, cy, cz, radius, half_length, rod_radius, nv, deltaL,
                 k_contact=1.0e7, smoothing=1.0e-4, samples_per_edge=4, active_cutoff=30.0):
        self.cx, self.cy, self.cz = float(cx), float(cy), float(cz)
        self.radius = float(radius)
        self.half_length = float(half_length)
        self.r_rod = float(rod_radius)
        self.r_eff = self.radius + self.r_rod
        self.nv = int(nv)
        self.deltaL = float(deltaL)
        self.k_contact = float(k_contact)
        self.eps = float(smoothing)
        if self.eps <= 0.0:
            raise ValueError(f"peg.smoothing must be > 0 (got {smoothing})")
        self.samples_per_edge = int(samples_per_edge)
        self.active_cutoff = float(active_cutoff)
        self.s_edge, self.s_frac, self.s_w = _edge_samples(self.nv, self.deltaL,
                                                           self.samples_per_edge)

    # ------------------------------------------------------------ geometry

    @property
    def center(self):
        return np.array([self.cx, self.cy, self.cz], dtype=float)

    def sample_points(self, q, m=None):
        """Quadrature points ``(n, 3)``; ``m`` points per edge instead of the model's."""
        X = np.asarray(q[:3 * self.nv], dtype=float).reshape(self.nv, 3)
        if m is None:
            e, f = self.s_edge, self.s_frac
        else:
            e, f, _ = _edge_samples(self.nv, self.deltaL, m)
        return (1.0 - f)[:, None] * X[e] + f[:, None] * X[e + 1]

    def compute_signed_clearance(self, P, derivs=0):
        """``d`` of points ``P`` (n, 3), and with ``derivs`` its gradient / Hessian."""
        P = np.atleast_2d(np.asarray(P, dtype=float))
        n = P.shape[0]
        y = P[:, 1]
        yc = np.clip(y, self.cy - self.half_length, self.cy + self.half_length)
        v = P - np.column_stack([np.full(n, self.cx), yc, np.full(n, self.cz)])
        rho = np.linalg.norm(v, axis=1)
        d = rho - self.r_eff
        if not derivs:
            return d
        ok = rho > 1e-14
        nrm = np.zeros_like(v)
        nrm[ok] = v[ok] / rho[ok, None]
        if derivs == 1:
            return d, nrm
        inside = (y > self.cy - self.half_length) & (y < self.cy + self.half_length)
        proj = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        proj[inside, 1, 1] = 0.0                      # I - a a^T, a = e_y
        H = np.zeros((n, 3, 3))
        H[ok] = (proj[ok] - nrm[ok, :, None] * nrm[ok, None, :]) / rho[ok, None, None]
        return d, nrm, H

    # ------------------------------------------------------------ penalty

    def _penalty(self, d):
        """``h(d)``, ``h'(d)``, ``h''(d)`` of the softplus gap."""
        z = -d / self.eps
        h = self.eps * np.logaddexp(0.0, z)
        sig = 0.5 * (1.0 + np.tanh(0.5 * z))           # (1 + e^{d/delta})^{-1}, overflow-free
        return h, -sig, sig * (1.0 - sig) / self.eps

    def _active(self, q):
        S = self.sample_points(q)
        d = self.compute_signed_clearance(S)
        return S, np.nonzero(d < self.active_cutoff * self.eps)[0]

    def compute_contact_energy(self, q):
        """``E_c(q)`` [J]."""
        h = self._penalty(self.compute_signed_clearance(self.sample_points(q)))[0]
        return 0.5 * self.k_contact * float(self.s_w @ (h * h))

    def _scatter_vec(self, idx, vec):
        out = np.zeros((self.nv, 3))
        e, f = self.s_edge[idx], self.s_frac[idx]
        np.add.at(out, e, (1.0 - f)[:, None] * vec)
        np.add.at(out, e + 1, f[:, None] * vec)
        return out.reshape(-1)

    def contact_gradient_hessian(self, q, want_hess=True):
        """``dE_c/dq`` and ``d2E_c/dq2`` over the ``3 nv`` position DOFs."""
        S, idx = self._active(q)
        n3 = 3 * self.nv
        if idx.size == 0:
            return np.zeros(n3), (np.zeros((n3, n3)) if want_hess else None)
        if want_hess:
            d, Gd, Hd = self.compute_signed_clearance(S[idx], derivs=2)
        else:
            d, Gd = self.compute_signed_clearance(S[idx], derivs=1)
        h, dh, d2h = self._penalty(d)
        kw = self.k_contact * self.s_w[idx]
        a = kw * h * dh                                   # dE/dd  (= w phi')
        grad = self._scatter_vec(idx, a[:, None] * Gd)
        if not want_hess:
            return grad, None
        b = kw * (dh * dh + h * d2h)                      # d2E/dd2 (= w phi'')
        Hs = b[:, None, None] * Gd[:, :, None] * Gd[:, None, :] + a[:, None, None] * Hd
        e, f = self.s_edge[idx], self.s_frac[idx]
        H = np.zeros((n3, n3))
        c3 = np.arange(3)
        for na, ca in ((e, 1.0 - f), (e + 1, f)):
            rows = 3 * na[:, None] + c3[None, :]
            for nb, cb in ((e, 1.0 - f), (e + 1, f)):
                cols = 3 * nb[:, None] + c3[None, :]
                np.add.at(H, (rows[:, :, None], cols[:, None, :]), (ca * cb)[:, None, None] * Hs)
        return grad, H

    def rod_contact(self, q, want_jac=True):
        """Whole-rod hook of :func:`get_contact_forces`: ``(F_c, J_c) = (-grad, -Hess)``."""
        grad, H = self.contact_gradient_hessian(q, want_hess=want_jac)
        return -grad, (None if H is None else -H)

    # ------------------------------------------------------------ diagnostics

    def report(self, q, dense_samples=16, threshold=None):
        """Geometric contact report of the rod ``q`` (positions first).

        ``min_clearance`` / ``max_penetration`` [m] of the rod surface, from
        ``dense_samples`` points per edge (the geometric value, independent
        of the quadrature of the energy); ``edge_clearance`` (nv-1,) the
        smallest clearance per edge; ``touching_edges`` the edges closer than
        ``threshold`` (default 3 delta); ``force`` [N] the total normal force
        of the penalty, ``sum_p w k_c h sigma``; ``end_contact`` whether a
        touching point lies beyond the ends of the axis (on a cap).
        """
        thr = 3.0 * self.eps if threshold is None else float(threshold)
        m = int(dense_samples)
        e, _, _ = _edge_samples(self.nv, self.deltaL, m)
        P = self.sample_points(q, m=m)
        d = self.compute_signed_clearance(P)
        edge_clear = np.full(self.nv - 1, np.inf)
        np.minimum.at(edge_clear, e, d)
        touch = d < thr
        beyond = (P[:, 1] <= self.cy - self.half_length) | (P[:, 1] >= self.cy + self.half_length)
        S = self.sample_points(q)
        ds = self.compute_signed_clearance(S)
        h, dh, _ = self._penalty(ds)
        force = float(np.sum(self.s_w * self.k_contact * h * (-dh)))
        return {"min_clearance": float(d.min()),
                "max_penetration": float(max(0.0, -d.min())),
                "edge_clearance": edge_clear,
                "touching_edges": edge_clear < thr,
                "n_touching": int(touch.sum()),
                "force": force,
                "end_contact": bool(np.any(touch & beyond))}

    def crossed(self, q_prev, q_new):
        """Whether moving the rod from ``q_prev`` to ``q_new`` passes it THROUGH the peg.

        The penalty acts on states only, so one large step (a Newton step, a
        predictor, a descent step) can carry part of the rod across the peg.
        Every quadrature point's straight segment from its old to its new
        position is tested against the solid core (axis distance < radius,
        rod radius not included, within the axis range).  Reliable for
        displacements small against the radius, which the solvers enforce
        with ``newton.max_node_step``.
        """
        a = self.sample_points(q_prev)
        b = self.sample_points(q_new)
        c = np.array([self.cx, self.cz])
        A, B = a[:, [0, 2]] - c, b[:, [0, 2]] - c
        D = B - A
        dd = np.sum(D * D, 1)
        t = np.where(dd > 1e-18, np.clip(-np.sum(A * D, 1) / np.maximum(dd, 1e-300), 0.0, 1.0), 0.0)
        P = A + t[:, None] * D
        y = a[:, 1] + t * (b[:, 1] - a[:, 1])
        return bool(np.any((np.linalg.norm(P, axis=1) < self.radius)
                           & (np.abs(y - self.cy) <= self.half_length)))

    def bounds(self):
        """``(lo, hi)`` corners of the box around the peg."""
        c = self.center
        e = np.array([self.radius, self.radius + self.half_length, self.radius])
        return c - e, c + e

    def draw_surface(self, ax, alpha=0.35, color="steelblue"):
        """Translucent cylinder on a 3D axes; returns the artists."""
        th = np.linspace(0.0, 2 * np.pi, 32)
        ys = np.linspace(self.cy - self.half_length, self.cy + self.half_length, 2)
        T, Y = np.meshgrid(th, ys)
        surf = ax.plot_surface(self.cx + self.radius * np.cos(T), Y,
                               self.cz + self.radius * np.sin(T), color=color, alpha=alpha,
                               linewidth=0, shade=True)
        return [surf]


def any_crossed(obstacles, q_prev, q_new):
    """Whether a step from ``q_prev`` to ``q_new`` passes the rod through any obstacle
    that can tell (``crossed``; the pegs)."""
    return any(o.crossed(q_prev, q_new) for o in obstacles if hasattr(o, "crossed"))


def step_cap_factor(dq_pos, dim, max_node_step):
    """Largest factor in (0, 1] keeping every node's move ``|factor * dq|`` within
    ``max_node_step`` (None or 0: no cap)."""
    if not max_node_step:
        return 1.0
    d = np.asarray(dq_pos, dtype=float).reshape(-1, dim)
    m = float(np.linalg.norm(d, axis=1).max()) if d.size else 0.0
    return 1.0 if m <= max_node_step else float(max_node_step) / m


def get_contact_forces(q, obstacles, dim=2, forces_only=False, n_pos=None):
    """
    Aggregate contact force vector and Jacobian matrix from all obstacles.

    Parameters
    ----------
    q         : ndarray (dim*N,) – DOF vector [x0, y0, (z0), x1, y1, (z1), …]
    obstacles : list of CircleObstacle (dim=2) or SphereObstacle (dim=3), and/or
                whole-rod obstacles such as LPipe2D (anything with ``rod_contact``)
    dim       : int – 2 or 3
    forces_only : bool – return just Fc, skipping the (dim*N, dim*N) Jacobian.
                  Used by the Newton line search, which only needs the residual.
    n_pos     : int or None – how many leading entries of q are positions.
                The twisting rod appends per-edge angles after them; contact
                acts on the nodes only, so everything past n_pos stays zero.

    Returns
    -------
    Fc : ndarray (dim*N,)        – total contact force
    Jc : ndarray (dim*N, dim*N) – total contact force Jacobian (unless forces_only)
    """
    ndof = q.size
    npos = ndof if n_pos is None else int(n_pos)
    N = npos // dim
    Fc = np.zeros(ndof)

    # Obstacles that see the whole rod (edges, not just nodes) expose
    # rod_contact(q) -> (F, J) instead of a per-node node_contact.
    rod_obs = [o for o in obstacles if hasattr(o, "rod_contact")]
    node_obs = [o for o in obstacles if not hasattr(o, "rod_contact")]

    if forces_only:
        for obs in rod_obs:
            Fc[:npos] += obs.rod_contact(q[:npos], want_jac=False)[0]
        for obs in node_obs:
            for k in range(N):
                a = dim * k
                Fc[a:a + dim] += obs.node_contact(*q[a:a + dim])[0]
        return Fc

    Jc = np.zeros((ndof, ndof))
    for obs in rod_obs:
        F_rod, J_rod = obs.rod_contact(q[:npos])
        Fc[:npos] += F_rod
        Jc[:npos, :npos] += J_rod
    for obs in node_obs:
        for k in range(N):
            a = dim * k
            F_node, J_node = obs.node_contact(*q[a:a + dim])
            Fc[a:a + dim] += F_node
            Jc[a:a + dim, a:a + dim] += J_node

    return Fc, Jc


def get_contact_energy(q, obstacles, dim=2, n_pos=None):
    """Total contact energy ``E_c(q)`` of all obstacles, the value whose
    derivatives :func:`get_contact_forces` returns.

    Whole-rod obstacles expose ``compute_contact_energy(q)`` (LPipe2D), node
    obstacles ``node_energy(*p)``.  Used by the total potential the snap rule
    descends (src/newton.py: total_potential).
    """
    q = np.asarray(q, dtype=float)
    npos = q.size if n_pos is None else int(n_pos)
    N = npos // dim
    E = 0.0
    for obs in obstacles:
        if hasattr(obs, "rod_contact"):
            E += float(obs.compute_contact_energy(q[:npos]))
        else:
            for k in range(N):
                a = dim * k
                E += float(obs.node_energy(*q[a:a + dim]))
    return E


# =============================================================================
# DLO -- rigid-wall contact: a 90-degree L-shaped pipe (2D)
# =============================================================================

class LPipe2D:
    """Rigid 90-degree L-shaped pipe in the plane, cut into a solid block.

    Two shapes (``geometry_type``):

    ``l_bend_2d``   a rounded elbow -- the channel follows a centerline with a
                    quarter arc of radius ``bend_radius`` (described below);
    ``l_corner_2d`` a sharp right-angle corner -- two straight rectangular
                    channels of half-width ``inner_radius`` meeting at a
                    square junction, like two drilled bores. The first runs
                    from the entry face to the far (outer) wall of the second;
                    the second starts at the near wall of the first and runs
                    ``second_segment_length`` past the corner, closed at the
                    end. ``bend_radius`` is not used. Its clearance is the
                    union of the two channels and the free space in front of
                    the face, each channel the intersection of its walls::

                        g = smax( smin(lat_A, end_A),
                                  smin(lat_C, start_C, end_C),  face )

                    where every term is a plane wall (distance - rod_radius).
                    The convex inner corner (a max) and the concave outer
                    corner (a min) are rounded by the same C2 blend as the
                    entry lips, over ``corner_smoothing``.

    For ``l_bend_2d`` the pipe is a channel of half-width ``inner_radius``
    around a centerline made of three G1-continuous pieces::

        A  straight, entry_position -> bend start   (length first_segment_length,
                                                      direction entry_direction)
        B  quarter arc of radius bend_radius
        C  straight, bend end -> closed far end     (length second_segment_length,
                                                      direction exit_direction)

    Everything on the far side of the entry face is solid except the channel;
    in front of the face is free space, where the gripper and the not yet
    inserted part of the rod live.  So the region the rod *centerline* may
    occupy is::

        allowed = {tube shrunk by rod_radius}  U  {more than rod_radius in front of the face}

    and its signed clearance (> 0 inside, < 0 penetrating) is

        g(p) = smax( (inner_radius - rod_radius) - dist(p, centerline),
                     -(p - entry_position) . entry_direction - rod_radius ).

    The union is a max, which has a kink at the two entry lips; ``smax`` rounds
    it off over a band of width ``corner_smoothing`` (a C2 quartic blend,
    see compute_signed_clearance), so the Newton solver sees a C1 potential
    there.  Outside that band it *is* the hard max, and inside it the
    clearance is overestimated by at most 3/16 * corner_smoothing.  ``dist`` to a G1 centerline is itself C1 inside the
    tube as long as ``inner_radius < bend_radius``, which is checked.

    Contact energy (smooth normal penalty, cf. the implicit contact model of
    Tong et al. for DER knot tying, with the rod--rod segment distance replaced
    by this rod--wall clearance)::

        E_c(q) = k_contact / 2 * sum_i w_i * h(g_i(q))^2,
        h(g)   = eps * log(1 + exp(-g / eps))          (softplus, eps = smoothing)

    evaluated at ``samples_per_edge`` points per edge (nodes included) with
    trapezoidal length weights ``w_i``, so an edge cannot slip through the
    wall between two nodes that are both inside.  ``F_c = -dE_c/dq`` and its
    exact Jacobian ``-d2E_c/dq2`` go into the equilibrium residual, so contact
    changes the solved state *and* the adjoint gradient, which is built from
    that same Jacobian.

    ``k_contact`` [N/m^2] is a stiffness per unit rod length: a penetration
    ``delta`` produces a line load of ~ ``k_contact * delta``.

    The whole-rod interface (``rod_contact``) is what ``get_contact_forces``
    looks for, so this plugs into the existing solver without touching it.
    """

    #: Pieces of the centerline, in order along it.
    PIECES = ("first_segment", "bend", "second_segment")
    #: Spatial dimension of the points (LPipe3D sets 3).
    D = 2

    def __init__(self, entry_position, entry_direction, exit_direction,
                 inner_radius, bend_radius, first_segment_length,
                 second_segment_length, rod_radius, nv, deltaL,
                 k_contact=1.0e7, smoothing=1.0e-4, corner_smoothing=5.0e-4,
                 samples_per_edge=4, active_cutoff=30.0, geometry_type="l_bend_2d"):
        if geometry_type not in ("l_bend_2d", "l_corner_2d"):
            raise ValueError(f"pipe.geometry_type must be l_bend_2d or l_corner_2d, "
                             f"got {geometry_type!r}")
        self.geometry_type = geometry_type
        self.corner = geometry_type == "l_corner_2d"
        self.P0 = np.asarray(entry_position, dtype=float).reshape(self.D)
        d1 = np.asarray(entry_direction, dtype=float).reshape(self.D)
        d2 = np.asarray(exit_direction, dtype=float).reshape(self.D)
        self.d1 = d1 / np.linalg.norm(d1)
        self.d2 = d2 / np.linalg.norm(d2)
        if abs(self.d1 @ self.d2) > 1e-9:
            raise ValueError("pipe.entry_direction and pipe.exit_direction must be "
                             f"perpendicular for a 90-degree bend (dot = {self.d1 @ self.d2:.3g})")

        self.r_in = float(inner_radius)
        self.Rb = float(bend_radius)
        self.L1 = float(first_segment_length)
        self.L2 = float(second_segment_length)
        self.r_rod = float(rod_radius)
        self.r_eff = self.r_in - self.r_rod
        if self.r_eff <= 0.0:
            raise ValueError(f"pipe.inner_radius ({self.r_in:g}) must exceed the rod "
                             f"radius ({self.r_rod:g}); the rod does not fit in the pipe.")
        if not self.corner and self.r_in >= self.Rb:
            raise ValueError(f"pipe.bend_radius ({self.Rb:g}) must exceed pipe.inner_radius "
                             f"({self.r_in:g}), otherwise the inner wall of the bend "
                             "folds over itself and the clearance is not smooth.")

        self.k_contact = float(k_contact)
        self.eps = float(smoothing)
        self.tau = float(corner_smoothing)
        self.samples_per_edge = int(samples_per_edge)
        self.active_cutoff = float(active_cutoff)

        # Key points of the centerline -- built once, never per call.
        if self.corner:
            # Centerline: straight to the corner point P1, then straight on.
            # The "bend" is the square junction, |s - L1| <= inner_radius.
            self.P1 = self.P0 + self.L1 * self.d1             # corner
            self.P2 = self.P1
            self.P3 = self.P1 + self.L2 * self.d2             # closed far end
            self.s_bend_start = self.L1 - self.r_in
            self.s_bend_end = self.L1 + self.r_in
            self.length = self.L1 + self.L2
            # Tightest turn the rod is forced into, for the force/strain estimates.
            self.turn_radius = self.r_in
        else:
            self.P1 = self.P0 + self.L1 * self.d1             # bend start
            self.C = self.P1 + self.Rb * self.d2              # bend center
            self.P2 = self.P1 + self.Rb * (self.d1 + self.d2) # bend end
            self.P3 = self.P2 + self.L2 * self.d2             # closed far end
            self.s_bend_start = self.L1
            self.s_bend_end = self.L1 + 0.5 * np.pi * self.Rb
            self.length = self.s_bend_end + self.L2
            self.turn_radius = self.Rb

        self.nv = int(nv)
        self.deltaL = float(deltaL)
        self._build_samples(self.samples_per_edge)

    # ------------------------------------------------------------ sampling

    def _build_samples(self, m):
        """Fixed interpolation of the sample points from the nodes."""
        nv = self.nv
        frac = np.tile(np.arange(m) / m, nv - 1)
        edge = np.repeat(np.arange(nv - 1), m)
        edge = np.append(edge, nv - 2)                 # ...plus the last node
        frac = np.append(frac, 1.0)
        w = np.full(edge.size, self.deltaL / m)
        w[0] *= 0.5
        w[-1] *= 0.5                                   # trapezoid: sum(w) = rod length
        self.s_edge, self.s_frac, self.s_w = edge, frac, w

    def sample_points(self, q, m=None):
        """Points along the rod at which contact is evaluated, ``(n_samples, 2)``."""
        X = np.asarray(q[:self.D * self.nv], dtype=float).reshape(self.nv, self.D)
        if m is None:
            e, f = self.s_edge, self.s_frac
        else:
            e = np.append(np.repeat(np.arange(self.nv - 1), m), self.nv - 2)
            f = np.append(np.tile(np.arange(m) / m, self.nv - 1), 1.0)
        return (1.0 - f)[:, None] * X[e] + f[:, None] * X[e + 1]

    # ------------------------------------------------------------ geometry

    def _centerline_distance(self, P, derivs):
        """Distance to the centerline, its gradient and Hessian; plus the piece index."""
        n = P.shape[0]
        D = np.full((n, 3), np.inf)
        G = np.zeros((n, 3, 2))
        H = np.zeros((n, 3, 2, 2))
        I2 = np.eye(2)

        def segment(k, A, u, L, open_start=False):
            t = ((P - A) @ u) / L
            # open_start: the first segment's walls end at the entry face, they
            # do not close round it, so before the face the distance is to the
            # line itself -- smooth across the face, unlike a round cap.
            tc = np.clip(t, -np.inf if open_start else 0.0, 1.0)
            v = P - (A + (tc * L)[:, None] * u)
            d = np.linalg.norm(v, axis=1)
            D[:, k] = d
            if derivs:
                ok = d > 1e-14
                nrm = np.zeros_like(v)
                nrm[ok] = v[ok] / d[ok, None]
                G[:, k] = nrm
                if derivs > 1:
                    # Interior of a straight piece: in 2D, I - n n^T - u u^T = 0.
                    # Past an end the distance is to a point: (I - n n^T) / d.
                    cap = ok & (((t < 0.0) & (not open_start)) | (t > 1.0))
                    H[cap, k] = (I2 - nrm[cap, :, None] * nrm[cap, None, :]) / d[cap, None, None]

        segment(0, self.P0, self.d1, self.L1, open_start=True)
        segment(2, self.P2, self.d2, self.L2)

        w = P - self.C
        rho = np.linalg.norm(w, axis=1)
        in_sector = (w @ -self.d2 >= 0.0) & (w @ self.d1 >= 0.0) & (rho > 1e-14)
        D[in_sector, 1] = np.abs(rho[in_sector] - self.Rb)
        if derivs:
            e = np.zeros_like(w)
            e[in_sector] = w[in_sector] / rho[in_sector, None]
            sgn = np.sign(rho - self.Rb)
            G[:, 1] = sgn[:, None] * e
            if derivs > 1:
                idx = in_sector
                H[idx, 1] = (sgn[idx, None, None]
                             * (I2 - e[idx, :, None] * e[idx, None, :]) / rho[idx, None, None])

        piece = np.argmin(D, axis=1)
        rows = np.arange(n)
        return D[rows, piece], G[rows, piece], H[rows, piece], piece

    def compute_signed_clearance(self, P, derivs=0, smooth=True):
        """Signed clearance g of rod-centerline points ``P`` (n, 2) from the walls.

        ``g > 0`` inside the allowed region, ``g < 0`` penetrating, ``-g`` is the
        penetration depth.  ``derivs`` = 0, 1, 2 also returns the gradient
        (n, 2) and Hessian (n, 2, 2).  ``grad g`` normalized is the contact
        normal: it points away from the wall, the direction the wall pushes
        the rod (``F_c = -dE_c/dg * grad g`` with ``dE_c/dg < 0``).  ``smooth=False`` uses
        the hard max/min at the lips and corners -- the geometric value, for reporting.
        """
        P = np.atleast_2d(np.asarray(P, dtype=float))
        n = P.shape[0]
        tau = self.tau if smooth else 0.0
        face = self._plane(-(P - self.P0) @ self.d1 - self.r_rod, -self.d1, n)

        if self.corner:
            a = (P - self.P0) @ self.d1          # along the first channel
            lA = (P - self.P0) @ self.d2         # across it (+ toward the inner side)
            a2 = (P - self.P1) @ self.d2         # along the second channel
            lC = (P - self.P1) @ self.d1         # across it (+ toward the outer wall)
            sA = np.where(lA >= 0.0, 1.0, -1.0)
            sC = np.where(lC >= 0.0, 1.0, -1.0)
            lat_A = self._plane(self.r_eff - np.abs(lA), -sA[:, None] * self.d2, n)
            end_A = self._plane(self.L1 + self.r_in - self.r_rod - a, -self.d1, n)
            lat_C = self._plane(self.r_eff - np.abs(lC), -sC[:, None] * self.d1, n)
            start_C = self._plane(a2 + self.r_in - self.r_rod, self.d2, n)
            end_C = self._plane(self.L2 - self.r_rod - a2, -self.d2, n)
            ch_A = self._smin2(lat_A, end_A, tau)
            ch_C = self._smin2(self._smin2(lat_C, start_C, tau), end_C, tau)
            g, grad, hess = self._smax2(self._smax2(ch_A, ch_C, tau), face, tau)
        else:
            dist, dG, dH, _ = self._centerline_distance(P, max(derivs, 1))
            g, grad, hess = self._smax2((self.r_eff - dist, -dG, -dH), face, tau)

        if not derivs:
            return g
        if derivs == 1:
            return g, grad
        return g, grad, hess

    def _plane(self, g, grad, n):
        """(value, gradient, Hessian) of a term linear in the point."""
        return g, np.broadcast_to(grad, (n, self.D)).copy(), np.zeros((n, self.D, self.D))

    @staticmethod
    def _smax2(A, B, tau):
        """C2 soft max of two (value, gradient, Hessian) triples, with compact support.

        Exactly max(a, b) unless the two are within ``tau`` of each other, where
            smax = (a + b)/2 + tau * f(x),  x = (a - b)/tau,
            f(x) = 3/16 + 3/8 x^2 - 1/16 x^4,
        which meets max(a, b) with matching first and second derivatives at
        |x| = 1 -- a C1 blend leaves a Hessian jump there that Newton chatters
        on -- and overestimates by at most 3/16 tau, where a = b.  ``tau = 0``
        is the hard max (the geometric value, for reporting).
        """
        a, Ga, Ha = A
        b, Gb, Hb = B
        if tau <= 0.0:
            pick = a >= b
            return (np.where(pick, a, b), np.where(pick[:, None], Ga, Gb),
                    np.where(pick[:, None, None], Ha, Hb))
        x = (a - b) / tau
        band = np.abs(x) < 1.0
        xb = x[band]
        g = np.maximum(a, b)
        g[band] = 0.5 * (a[band] + b[band]) + tau * (3 / 16 + 3 / 8 * xb ** 2 - xb ** 4 / 16)
        # Weight on the first term: 1 / 0 outside the band, a smooth ramp inside.
        w = np.where(x >= 0.0, 1.0, 0.0)
        w[band] = 0.5 + 0.75 * xb - 0.25 * xb ** 3
        G = w[:, None] * Ga + (1.0 - w)[:, None] * Gb
        H = w[:, None, None] * Ha + (1.0 - w)[:, None, None] * Hb
        dd = Ga[band] - Gb[band]
        c = 0.75 * (1.0 - xb ** 2) / tau
        H[band] += c[:, None, None] * dd[:, :, None] * dd[:, None, :]
        return g, G, H

    @classmethod
    def _smin2(cls, A, B, tau):
        """C2 soft min: ``-smax(-a, -b)``; underestimates the clearance (conservative)."""
        g, G, H = cls._smax2((-A[0], -A[1], -A[2]), (-B[0], -B[1], -B[2]), tau)
        return -g, -G, -H

    # ------------------------------------------------------------ contact

    def _penalty(self, g):
        """h(g) = eps*softplus(-g/eps) and its first two g-derivatives."""
        z = -g / self.eps
        h = self.eps * np.logaddexp(0.0, z)
        sig = 0.5 * (1.0 + np.tanh(0.5 * z))        # expit(z), overflow-free
        return h, -sig, sig * (1.0 - sig) / self.eps

    def _active(self, q):
        S = self.sample_points(q)
        g = self.compute_signed_clearance(S)
        return S, np.nonzero(g < self.active_cutoff * self.eps)[0]

    def compute_contact_energy(self, q):
        """E_c(q) [J]."""
        g = self.compute_signed_clearance(self.sample_points(q))
        h = self._penalty(g)[0]
        return 0.5 * self.k_contact * float(self.s_w @ (h * h))

    def _scatter_vec(self, idx, vec):
        out = np.zeros((self.nv, self.D))
        e, f = self.s_edge[idx], self.s_frac[idx]
        np.add.at(out, e, (1.0 - f)[:, None] * vec)
        np.add.at(out, e + 1, f[:, None] * vec)
        return out.reshape(-1)

    def contact_gradient_hessian(self, q, want_hess=True):
        """dE_c/dq and d2E_c/dq2 over the 2*nv position DOFs."""
        S, idx = self._active(q)
        n2 = self.D * self.nv
        if idx.size == 0:
            return np.zeros(n2), (np.zeros((n2, n2)) if want_hess else None)

        if want_hess:
            g, Gg, Hg = self.compute_signed_clearance(S[idx], derivs=2)
        else:
            g, Gg = self.compute_signed_clearance(S[idx], derivs=1)
        h, dh, d2h = self._penalty(g)
        kw = self.k_contact * self.s_w[idx]
        a = kw * h * dh                               # dE/dg
        grad = self._scatter_vec(idx, a[:, None] * Gg)
        if not want_hess:
            return grad, None

        b = kw * (dh * dh + h * d2h)                  # d2E/dg2
        Hs = b[:, None, None] * Gg[:, :, None] * Gg[:, None, :] + a[:, None, None] * Hg
        e, f = self.s_edge[idx], self.s_frac[idx]
        nodes = (e, e + 1)
        coef = (1.0 - f, f)
        H = np.zeros((n2, n2))
        c2 = np.arange(self.D)
        for na, ca in zip(nodes, coef):
            rows = self.D * na[:, None] + c2[None, :]
            for nb, cb in zip(nodes, coef):
                cols = self.D * nb[:, None] + c2[None, :]
                np.add.at(H, (rows[:, :, None], cols[:, None, :]),
                          (ca * cb)[:, None, None] * Hs)
        return grad, H

    def compute_contact_force(self, q):
        """F_c = -dE_c/dq, shape (2*nv,)."""
        return -self.contact_gradient_hessian(q, want_hess=False)[0]

    def compute_contact_jacobian(self, q):
        """dF_c/dq = -d2E_c/dq2, shape (2*nv, 2*nv)."""
        return -self.contact_gradient_hessian(q)[1]

    def rod_contact(self, q, want_jac=True):
        """Whole-rod hook used by :func:`get_contact_forces`: ``(F_c, J_c)``."""
        grad, H = self.contact_gradient_hessian(q, want_hess=want_jac)
        return -grad, (None if H is None else -H)

    def detect_dlo_pipe_contact(self, q, dense_samples=16, threshold=None):
        """Geometric contact report, with the edges sampled densely.

        Uses the hard (unsmoothed) clearance, so ``max_penetration`` is the
        geometric depth, not the potential's.  ``threshold`` (default
        3*smoothing) is the clearance below which a point counts as touching.
        """
        thr = 3.0 * self.eps if threshold is None else float(threshold)
        P = self.sample_points(q, m=dense_samples)
        g, grad = self.compute_signed_clearance(P, derivs=1, smooth=False)
        touching = g < thr
        nrm = grad[touching]
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-14)
        nodes = self.compute_signed_clearance(np.asarray(q[:self.D * self.nv]).reshape(-1, self.D),
                                              smooth=False)
        return {
            "min_clearance": float(g.min()),
            "max_penetration": float(max(0.0, -g.min())),
            "max_node_penetration": float(max(0.0, -nodes.min())),
            "n_contacts": int(touching.sum()),
            "contact_points": P[touching],
            "contact_normals": nrm,
        }

    def compute_contact_loss(self, q, margin=0.0):
        """Optimization penalty on penetration beyond ``margin`` -- NOT the physics.

        ``L = sum_i (w_i / L_rod) * relu(-g_i - margin)^2`` [m^2], on the same
        sample points as the energy.  Returns ``(L, dL/dq)`` over the position
        DOFs.  This only steers the optimizer away from deep contact; the wall
        acts on the rod through ``rod_contact`` regardless.
        """
        S = self.sample_points(q)
        g, Gg = self.compute_signed_clearance(S, derivs=1)
        pen = np.maximum(-g - margin, 0.0)
        wl = self.s_w / (self.deltaL * (self.nv - 1))
        L = float(wl @ (pen * pen))
        idx = np.nonzero(pen > 0.0)[0]
        if idx.size == 0:
            return L, np.zeros(self.D * self.nv)
        dLdg = -2.0 * wl[idx] * pen[idx]
        return L, self._scatter_vec(idx, dLdg[:, None] * Gg[idx])

    # ------------------------------------------------------------ the path

    def centerline_point(self, s):
        """Point on the centerline at arc length ``s`` (clamped to [0, length])."""
        s = float(np.clip(s, 0.0, self.length))
        if self.corner:
            if s <= self.L1:
                return self.P0 + s * self.d1
            return self.P1 + (s - self.L1) * self.d2
        if s <= self.s_bend_start:
            return self.P0 + s * self.d1
        if s <= self.s_bend_end:
            phi = (s - self.s_bend_start) / self.Rb
            return self.C + self.Rb * (np.cos(phi) * -self.d2 + np.sin(phi) * self.d1)
        return self.P2 + (s - self.s_bend_end) * self.d2

    def arclength(self, p):
        """Arc length along the centerline of the closest point to ``p``, and d s/d p.

        Unclamped before the entry (negative) so a tip that has not entered
        yet still has a well-defined, differentiable progress.
        """
        p = np.asarray(p, dtype=float).reshape(1, 2)
        if self.corner:
            # Nearest of the two straight legs (the first one open before the entry).
            p = p[0]
            a = float((p - self.P0) @ self.d1)
            a2 = float((p - self.P1) @ self.d2)
            dA = abs(float((p - self.P0) @ self.d2)) if a <= self.L1 else np.linalg.norm(p - self.P1)
            dC = abs(float((p - self.P1) @ self.d1)) if a2 >= 0.0 else np.linalg.norm(p - self.P1)
            if dA <= dC:
                return a, self.d1.copy()
            return float(self.L1 + a2), self.d2.copy()
        piece = int(self._centerline_distance(p, 0)[3][0])
        p = p[0]
        if piece == 0:
            return float((p - self.P0) @ self.d1), self.d1.copy()
        if piece == 2:
            return float(self.s_bend_end + (p - self.P2) @ self.d2), self.d2.copy()
        w = p - self.C
        rho = float(np.linalg.norm(w))
        phi = np.arctan2(w @ self.d1, w @ -self.d2)
        tang = -np.sin(phi) * -self.d2 + np.cos(phi) * self.d1
        return float(self.s_bend_start + self.Rb * phi), (self.Rb / rho) * tang

    def piece_of(self, p):
        """Name of the centerline piece closest to ``p``."""
        if self.corner:
            s = self.arclength(p)[0]
            return self.PIECES[0 if s < self.s_bend_start else 1 if s <= self.s_bend_end else 2]
        p = np.asarray(p, dtype=float).reshape(1, 2)
        return self.PIECES[int(self._centerline_distance(p, 0)[3][0])]

    # ------------------------------------------------------------ drawing

    def draw(self, ax, color="steelblue", alpha=0.18, n=200):
        """Pipe walls, entry face and channel on a matplotlib axis."""
        if self.corner:
            r, P1, d1, d2 = self.r_in, self.P1, self.d1, self.d2
            wall = np.array([self.P0 + r * d2, P1 - r * d1 + r * d2, P1 - r * d1 + self.L2 * d2,
                             P1 + r * d1 + self.L2 * d2, P1 + r * d1 - r * d2, self.P0 - r * d2])
            ax.fill(wall[:, 0], wall[:, 1], color=color, alpha=alpha, lw=0, zorder=0)
            ax.plot(wall[:, 0], wall[:, 1], color=color, lw=2, zorder=1)
            c = np.array([self.P0, P1, self.P3])
            for sgn in (1, -1):
                a = self.P0 + sgn * r * d2
                b = self.P0 + sgn * 3.5 * r * d2
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=2, zorder=1)
            ax.plot(c[:, 0], c[:, 1], ':', color=color, lw=1, alpha=0.8, zorder=1)
            return
        s = np.linspace(0.0, self.length, n)
        c = np.array([self.centerline_point(v) for v in s])
        t = np.gradient(c, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
        left, right = c + self.r_in * nrm, c - self.r_in * nrm
        # Closed far end: a half circle from the left wall round to the right one.
        cap_a = np.linspace(0.0, np.pi, 40)
        cap = (self.P3 + self.r_in * (np.cos(cap_a)[:, None] * nrm[-1]
                                      + np.sin(cap_a)[:, None] * self.d2))
        wall = np.vstack([left, cap, right[::-1]])
        ax.fill(wall[:, 0], wall[:, 1], color=color, alpha=alpha, lw=0, zorder=0)
        ax.plot(wall[:, 0], wall[:, 1], color=color, lw=2, zorder=1)
        # Entry face of the block, a little either side of the opening.
        lat = np.array([-self.d1[1], self.d1[0]])
        span = 2.5 * self.r_in
        for sgn in (1, -1):
            a = self.P0 + sgn * self.r_in * lat
            b = self.P0 + sgn * (self.r_in + span) * lat
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=2, zorder=1)
        ax.plot(c[:, 0], c[:, 1], ':', color=color, lw=1, alpha=0.8, zorder=1)


# =============================================================================
# DLO -- rigid-wall contact: a 90-degree corner pipe in space (3D)
# =============================================================================

class LPipe3D(LPipe2D):
    """Two cylindrical bores meeting at a right angle, in space.

    The 3D counterpart of :class:`LPipe2D`'s ``l_corner_2d``: bore A of
    radius ``inner_radius`` runs from the entry face at ``entry_position``
    along ``entry_direction`` to the far wall of bore C, bore C runs from the
    near wall of bore A along ``exit_direction`` for ``second_segment_length``
    and is closed at the end.  The channel-wall terms of the 2D corner become
    *radial* distances from the bore axes (``r_eff - rho``), the end planes,
    the entry face and the C2 soft max / min blends are the same, and so are
    the sampling along the edges, the penalty energy and the whole-rod
    ``rod_contact`` hook.  Only the sharp corner is implemented (the rounded
    elbow would need a torus).
    """

    D = 3

    def __init__(self, *args, geometry_type="l_corner_3d", **kwargs):
        if geometry_type != "l_corner_3d":
            raise ValueError("the 3D pipe is the sharp corner only: pipe.geometry_type: l_corner_3d")
        super().__init__(*args, geometry_type="l_corner_2d", **kwargs)
        self.geometry_type = "l_corner_3d"

    # ------------------------------------------------------------ geometry

    def _radial(self, rel, axis, n):
        """``r_eff - rho`` with gradient and Hessian, rho the distance from the axis."""
        a = rel @ axis
        w = rel - a[:, None] * axis[None, :]
        rho = np.linalg.norm(w, axis=1)
        ok = rho > 1e-12
        nrm = np.zeros_like(w)
        nrm[ok] = w[ok] / rho[ok, None]
        G = -nrm
        H = np.zeros((n, 3, 3))
        I3 = np.eye(3)
        P = I3 - np.outer(axis, axis)
        H[ok] = -(P[None] - nrm[ok, :, None] * nrm[ok, None, :]) / rho[ok, None, None]
        return self.r_eff - rho, G, H

    def compute_signed_clearance(self, P, derivs=0, smooth=True):
        P = np.atleast_2d(np.asarray(P, dtype=float))
        n = P.shape[0]
        tau = self.tau if smooth else 0.0
        rel = P - self.P0
        a = rel @ self.d1
        rel2 = P - self.P1
        a2 = rel2 @ self.d2
        face = self._plane(-a - self.r_rod, -self.d1, n)
        lat_A = self._radial(rel, self.d1, n)
        end_A = self._plane(self.L1 + self.r_in - self.r_rod - a, -self.d1, n)
        lat_C = self._radial(rel2, self.d2, n)
        start_C = self._plane(a2 + self.r_in - self.r_rod, self.d2, n)
        end_C = self._plane(self.L2 - self.r_rod - a2, -self.d2, n)
        ch_A = self._smin2(lat_A, end_A, tau)
        ch_C = self._smin2(self._smin2(lat_C, start_C, tau), end_C, tau)
        g, grad, hess = self._smax2(self._smax2(ch_A, ch_C, tau), face, tau)
        if not derivs:
            return g
        if derivs == 1:
            return g, grad
        return g, grad, hess

    def arclength(self, p):
        p = np.asarray(p, dtype=float).reshape(3)
        a = float((p - self.P0) @ self.d1)
        a2 = float((p - self.P1) @ self.d2)
        rel, rel2 = p - self.P0, p - self.P1
        rho_A = float(np.linalg.norm(rel - a * self.d1))
        rho_C = float(np.linalg.norm(rel2 - a2 * self.d2))
        dA = rho_A if a <= self.L1 else float(np.linalg.norm(rel2))
        dC = rho_C if a2 >= 0.0 else float(np.linalg.norm(rel2))
        if dA <= dC:
            return a, self.d1.copy()
        return float(self.L1 + a2), self.d2.copy()

    def piece_of(self, p):
        s = self.arclength(p)[0]
        return self.PIECES[0 if s < self.s_bend_start else 1 if s <= self.s_bend_end else 2]

    def far_wall_point(self, s2):
        """Point on the centerline-reachable far (outer) wall of bore C, ``s2``
        along the exit direction from the corner: where a rod pushed round
        the corner ends up, since it cannot leave that wall again."""
        return self.P1 + self.r_eff * self.d1 + float(s2) * self.d2

    # ------------------------------------------------------------ drawing

    def draw(self, ax, color="steelblue", alpha=0.15, n=40):
        """The two bores as translucent cylinders, plus the centerline."""
        def basis(axis):
            a = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            e1 = np.cross(axis, a); e1 /= np.linalg.norm(e1)
            return e1, np.cross(axis, e1)

        th = np.linspace(0, 2 * np.pi, n)
        for start, axis, length in ((self.P0, self.d1, self.L1 + self.r_in),
                                    (self.P1 - self.r_in * self.d2, self.d2, self.L2 + self.r_in)):
            e1, e2 = basis(axis)
            s = np.linspace(0.0, length, 2)
            ring = self.r_in * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
            X = start[None, None, :] + s[None, :, None] * axis[None, None, :] + ring[:, None, :]
            ax.plot_surface(X[..., 0], X[..., 1], X[..., 2], color=color, alpha=alpha,
                            linewidth=0, shade=False)
            for k in (0, 1):
                ax.plot(X[:, k, 0], X[:, k, 1], X[:, k, 2], color=color, lw=1, alpha=0.6)
        c = np.array([self.P0, self.P1, self.P3])
        ax.plot(c[:, 0], c[:, 1], c[:, 2], ':', color=color, lw=1, alpha=0.8)
