# -*- coding: utf-8 -*-
"""
Contact mechanics for the elastic rod simulation (2D and 3D).

Penalty-energy approach:
    E_c = 0.5 * k_contact * max(0, r - dist)^2

where dist is the distance from a rod node to the obstacle center.
E_c → ∞ as penetration grows; E_c = 0 when the node is outside.
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


def get_contact_forces(q, obstacles, dim=2, forces_only=False):
    """
    Aggregate contact force vector and Jacobian matrix from all obstacles.

    Parameters
    ----------
    q         : ndarray (dim*N,) – DOF vector [x0, y0, (z0), x1, y1, (z1), …]
    obstacles : list of CircleObstacle (dim=2) or SphereObstacle (dim=3)
    dim       : int – 2 or 3
    forces_only : bool – return just Fc, skipping the (dim*N, dim*N) Jacobian.
                  Used by the Newton line search, which only needs the residual.

    Returns
    -------
    Fc : ndarray (dim*N,)        – total contact force
    Jc : ndarray (dim*N, dim*N) – total contact force Jacobian (unless forces_only)
    """
    ndof = q.size
    N = ndof // dim
    Fc = np.zeros(ndof)

    if forces_only:
        for obs in obstacles:
            for k in range(N):
                a = dim * k
                Fc[a:a + dim] += obs.node_contact(*q[a:a + dim])[0]
        return Fc

    Jc = np.zeros((ndof, ndof))
    for obs in obstacles:
        for k in range(N):
            a = dim * k
            F_node, J_node = obs.node_contact(*q[a:a + dim])
            Fc[a:a + dim] += F_node
            Jc[a:a + dim, a:a + dim] += J_node

    return Fc, Jc
