# -*- coding: utf-8 -*-
"""Material frames and the reference twist, for the 3D rod with torsion.

Background
----------
A rod that can twist needs more than node positions: each edge carries a
*material frame* describing how its cross-section is oriented about the edge
tangent.  Following the Discrete Elastic Rods formulation (Bergou et al. 2008;
the same construction DEFORM uses), the frame is split into

* a **reference director** ``u^i`` per edge, perpendicular to that edge's
  tangent, carried along by *parallel transport* -- a rotation that adds no
  twist, so the reference frame itself is twist-free by construction; and
* a scalar **twist angle** ``theta^i`` per edge, the rotation of the material
  frame away from the reference director about the tangent.

``theta`` is a genuine degree of freedom, solved for alongside the positions.
The reference directors are *state*: they are stored between steps and carried
to the new configuration by parallel transport, exactly like the warm start
that keeps the position solve on one equilibrium branch.

The reference twist
-------------------
Parallel transport along the rod is not integrable: carrying ``u^{i-1}`` from
edge ``i-1`` onto edge ``i`` does not in general land on ``u^i``.  The angle
between them is the **reference twist** ``psi_i``, and the material twist at
interior node ``i`` is

    m_i = theta^i - theta^{i-1} + psi_i

``psi_i`` depends on the *positions*, which is precisely what couples twisting
to bending: this is the discrete form of the Calugareanu-White-Fuller relation
(link = twist + writhe).  Without it, yaw at the gripper would be a dead input
-- the rod would twist but its centreline would not move, and the gradient of
any position-based loss with respect to yaw would be exactly zero.

Derivatives
-----------
``psi_grad_hess`` gives the exact first and second derivatives of ``psi_i``
with respect to the three nodes of its stencil.  The gradient has the
classical holonomy part plus a re-adaptation part (see that function's
docstring); both have the same ``kb / 2L`` shape.  Everything here is checked
against finite differences of a direct parallel-transport evaluation of
``psi`` by ``scripts/check_twist.py``.

One subtlety, and it matters: differentiating the ``kb/(2L)`` expression once
more does **not** by itself give the Hessian -- the naive Jacobian of that
expression carries a spurious antisymmetric part and is off by more than 100%.
The true Hessian is its **symmetric part**.  ``psi_hess`` returns that, and
``scripts/check_twist.py`` verifies it against a second-difference Hessian of
``psi``, which is symmetric by construction.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------- primitives

def hat(v):
    """Skew-symmetric matrix with ``hat(v) @ w == cross(v, w)``."""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def parallel_transport(u, t_from, t_to):
    """Carry ``u`` from the frame adapted to ``t_from`` onto ``t_to``.

    This is the rotation about ``t_from x t_to`` taking one tangent to the
    other; it is the unique rotation doing so that adds no twist about the
    tangent, which is what makes the reference frame twist-free.
    """
    b = np.cross(t_from, t_to)
    nb = np.linalg.norm(b)
    if nb < _EPS:                      # tangents (anti)parallel: nothing to do
        return np.asarray(u, dtype=float).copy()
    b = b / nb
    # Re-orthogonalize against both tangents; cheap insurance against drift.
    b = b - np.dot(b, t_from) * t_from
    b = b / max(np.linalg.norm(b), _EPS)
    b = b - np.dot(b, t_to) * t_to
    b = b / max(np.linalg.norm(b), _EPS)
    n_from = np.cross(t_from, b)
    n_to = np.cross(t_to, b)
    return np.dot(u, t_from) * t_to + np.dot(u, n_from) * n_to + np.dot(u, b) * b


def signed_angle(u, v, n):
    """Angle from ``u`` to ``v``, signed by the reference normal ``n``."""
    w = np.cross(u, v)
    angle = np.arctan2(np.linalg.norm(w), float(np.dot(u, v)))
    return -angle if float(np.dot(n, w)) < 0.0 else angle


def tangents(pos):
    """Unit tangents of every edge of the polyline ``pos`` of shape (nv, 3)."""
    e = pos[1:] - pos[:-1]
    n = np.linalg.norm(e, axis=1)
    return e / np.maximum(n, _EPS)[:, None]


def curvature_binormal(te, tf):
    """``kb = 2 te x tf / (1 + te . tf)`` -- the discrete curvature binormal."""
    return 2.0 * np.cross(te, tf) / (1.0 + float(np.dot(te, tf)))


# -------------------------------------------------------- reference frame state

class RodFrames:
    """The reference directors of every edge, and the tangents they belong to.

    ``directors[i]`` is a unit vector perpendicular to ``stored_tangents[i]``.
    Both are state carried between control steps: :meth:`advance` parallel
    transports the directors onto a new configuration, and :meth:`adapt`
    produces directors for an arbitrary configuration *without* mutating the
    state, which is what the energy evaluation needs during a Newton solve.
    """

    def __init__(self, nv, seed_director=(0.0, 0.0, 1.0)):
        self.nv = int(nv)
        self.n_edges = self.nv - 1
        self._seed = np.asarray(seed_director, dtype=float)
        self.directors = None
        self.stored_tangents = None

    # ---------- lifecycle ----------

    def reset(self, pos):
        """Initialise the frame on the configuration ``pos`` (nv, 3).

        The first edge's director is the seed vector projected perpendicular to
        its tangent; every later edge is obtained by parallel transport *along
        the rod*, which makes the initial reference twist exactly zero.
        """
        t = tangents(np.asarray(pos, dtype=float))
        u = np.zeros((self.n_edges, 3))

        seed = self._seed
        if abs(float(np.dot(seed, t[0]))) > 1.0 - 1e-6:   # seed parallel to t0
            seed = np.array([0.0, 1.0, 0.0])
            if abs(float(np.dot(seed, t[0]))) > 1.0 - 1e-6:
                seed = np.array([1.0, 0.0, 0.0])
        u0 = seed - float(np.dot(seed, t[0])) * t[0]
        u[0] = u0 / max(np.linalg.norm(u0), _EPS)
        for i in range(1, self.n_edges):
            u[i] = parallel_transport(u[i - 1], t[i - 1], t[i])

        self.directors = u
        self.stored_tangents = t.copy()

    def advance(self, pos):
        """Carry the stored directors onto the configuration ``pos`` and keep them.

        Called once per committed control step -- the frame analogue of the
        warm start that keeps the position solve on one equilibrium branch.
        """
        self.directors = self.adapt(pos)
        self.stored_tangents = tangents(np.asarray(pos, dtype=float))

    def adapt(self, pos):
        """Stored directors parallel transported onto ``pos``; state untouched."""
        t = tangents(np.asarray(pos, dtype=float))
        u = np.empty_like(self.directors)
        for i in range(self.n_edges):
            u[i] = parallel_transport(self.directors[i], self.stored_tangents[i], t[i])
        return u

    # ---------- checkpointing (RHC segments rewind the simulator) ----------

    def get_state(self):
        return {"directors": self.directors.copy(),
                "stored_tangents": self.stored_tangents.copy()}

    def set_state(self, state):
        self.directors = state["directors"].copy()
        self.stored_tangents = state["stored_tangents"].copy()

    # ---------- the reference twist ----------

    def reference_twist(self, pos):
        """``psi`` at every interior node, shape (nv - 2,)."""
        pos = np.asarray(pos, dtype=float)
        t = tangents(pos)
        u = self.adapt(pos)
        psi = np.zeros(self.nv - 2)
        for i in range(1, self.n_edges):
            carried = parallel_transport(u[i - 1], t[i - 1], t[i])
            psi[i - 1] = signed_angle(carried, u[i], t[i])
        return psi


# ------------------------------------------------- derivatives of psi (exact)

def _kb_and_jac(a, edge, n_edge):
    """``kb(a, t)`` for a FIXED first argument ``a``, plus ``d(kb)/d(edge)``.

    ``t = edge / n_edge``.  Used for the re-adaptation legs of the transport.
    """
    t = edge / n_edge
    chi = 1.0 + float(np.dot(a, t))
    if chi < _EPS:
        return np.zeros(3), np.zeros((3, 3))
    kb = 2.0 * np.cross(a, t) / chi
    P = (np.eye(3) - np.outer(t, t)) / n_edge          # dt/d(edge)
    dkb = (2.0 / chi) * (hat(a) @ P) - (1.0 / chi) * np.outer(kb, P @ a)
    return kb, dkb


def psi_grad_hess(p0, p1, p2, tbar_e, tbar_f):
    """``(grad(9,), hess(9, 9))`` of psi -- both exact.

    ``psi`` is measured between directors carried from the *stored* tangents
    ``tbar_e``/``tbar_f`` onto the current ones, so its derivative has two
    parts:

    * the classical **holonomy** of the along-the-rod leg ``t_e -> t_f``::

          dpsi/de += +kb(t_e, t_f) / (2 |e|)
          dpsi/df += +kb(t_e, t_f) / (2 |f|)

    * the **re-adaptation** of each stored director, ``tbar -> t``, which has
      exactly the same form on its own leg::

          dpsi/de += +kb(tbar_e, t_e) / (2 |e|)
          dpsi/df += -kb(tbar_f, t_f) / (2 |f|)

      This term vanishes identically when the stored tangents equal the
      current ones, which is why it is easy to miss: it is invisible at the
      start of a step and grows as the Newton iterate moves away.  Dropping it
      costs ~50% of the gradient by the time the solve has travelled a
      realistic distance.

    Both parts are differentiated once more for the Hessian.  As with the
    bending stencil, the raw chain-rule matrix carries a spurious
    antisymmetric piece; the true Hessian is its symmetric part.
    """
    e, f = p1 - p0, p2 - p1
    ne, nf = np.linalg.norm(e), np.linalg.norm(f)
    if ne < _EPS or nf < _EPS:
        return np.zeros(9), np.zeros((9, 9))

    te, tf = e / ne, f / nf
    chi = 1.0 + float(np.dot(te, tf))
    if chi < _EPS:                      # 180 degree fold: genuinely singular
        return np.zeros(9), np.zeros((9, 9))
    kb = 2.0 * np.cross(te, tf) / chi

    I3 = np.eye(3)
    Pe = (I3 - np.outer(te, te)) / ne       # dte/de
    Pf = (I3 - np.outer(tf, tf)) / nf       # dtf/df

    # --- holonomy leg t_e -> t_f ---
    dkb_de = -(2.0 / chi) * (hat(tf) @ Pe) - (1.0 / chi) * np.outer(kb, Pe @ tf)
    dkb_df = (2.0 / chi) * (hat(te) @ Pf) - (1.0 / chi) * np.outer(kb, Pf @ te)

    A = kb / (2.0 * ne)
    B = kb / (2.0 * nf)
    dA_de = dkb_de / (2.0 * ne) - np.outer(kb, te) / (2.0 * ne ** 2)
    dA_df = dkb_df / (2.0 * ne)
    dB_de = dkb_de / (2.0 * nf)
    dB_df = dkb_df / (2.0 * nf) - np.outer(kb, tf) / (2.0 * nf ** 2)

    # --- re-adaptation legs tbar -> t (each touches only its own edge) ---
    kbe, dkbe = _kb_and_jac(np.asarray(tbar_e, dtype=float), e, ne)
    kbf, dkbf = _kb_and_jac(np.asarray(tbar_f, dtype=float), f, nf)

    A = A + kbe / (2.0 * ne)
    B = B - kbf / (2.0 * nf)
    dA_de = dA_de + dkbe / (2.0 * ne) - np.outer(kbe, te) / (2.0 * ne ** 2)
    dB_df = dB_df - (dkbf / (2.0 * nf) - np.outer(kbf, tf) / (2.0 * nf ** 2))

    M = np.zeros((6, 6))
    M[0:3, 0:3] = dA_de
    M[0:3, 3:6] = dA_df
    M[3:6, 0:3] = dB_de
    M[3:6, 3:6] = dB_df

    Bm = np.zeros((6, 9))
    Bm[0:3, 0:3] = -I3
    Bm[0:3, 3:6] = I3
    Bm[3:6, 3:6] = -I3
    Bm[3:6, 6:9] = I3

    g = Bm.T @ np.concatenate([A, B])
    H = Bm.T @ M @ Bm
    return g, 0.5 * (H + H.T)


def psi_grad(p0, p1, p2, tbar_e, tbar_f):
    """``d(psi)/d(p0, p1, p2)``, shape (9,)."""
    return psi_grad_hess(p0, p1, p2, tbar_e, tbar_f)[0]
