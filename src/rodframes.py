# -*- coding: utf-8 -*-
"""Reference and material frames of the 3D rod with torsion.

The frames follow the Discrete Elastic Rods implementation of MAE 263F
(github.com/StructuresComp/mae-263F, ``src/der/der_functions.py``), i.e. the
time-parallel reference frames of Bergou et al., *Discrete Viscous Threads*:

* **Reference frame.**  Every edge ``i`` carries its own adapted frame
  ``(a1^i, a2^i, t^i)``.  At reset it is built by parallel transport *along the
  rod* (:func:`space_parallel`), seeded on the first edge with the gripper's
  yaw reference :func:`gripper_director`.  After that it is carried *in time*:
  when the centreline moves from the last accepted state to a new one, each
  edge's ``a1`` is parallel transported from its old tangent to its new one
  (:func:`time_parallel`).  The frame is therefore state, owned by the energy
  (``energy.DERElasticEnergy3DTwist``), which keeps the accepted state and
  rebuilds the frame from it at every evaluation.
* **Reference twist.**  Neighbouring reference frames are not parallel
  transports of each other any more; the mismatch at interior node ``i`` is
  the reference twist ``refTwist_i`` (:func:`reference_twist`), added to the
  local twist ``theta^i - theta^{i-1}``.  It is resolved against the previous
  value so it is continuous across +-pi.
* **Material frame.**  Edge ``i``'s material frame is its reference frame
  rotated by the edge's own angle ``theta^i`` about the tangent
  (:func:`material_directors`).  The angles are degrees of freedom, one per
  edge, behind the positions in ``q``.

Where the gripper's yaw is measured from
----------------------------------------
A yaw command is an angle of the wrist, measured from a frame that depends on
the gripper's own orientation: :func:`gripper_director` is that reference
direction for an end tangent, the vector perpendicular to the tangent in the
vertical plane through it (``+z`` for the straight rest pose).  It seeds the
reference frame at reset, so ``theta^0 = yaw`` there.

Bishop holonomy (diagnostic only)
---------------------------------
:func:`bishop_directors`, :func:`holonomy` and their derivatives below belong
to the previous quasistatic-twist model, where the holonomy of the Bishop
frame entered the twisting energy.  The energy no longer uses them; they stay
until the callers that log the holonomy are cleaned up.  ``holonomy`` is still
a well-defined geometric quantity of the centreline.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12

#: Axis the gripper's yaw reference is built from; also the chart axis of the
#: (azimuth, elevation) control angles, and so singular for an end edge along
#: it.  z-up, matching SimulatorEnv_3D's gravity convention.
_UP = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------- primitives

def _cross3(a, b):
    """``np.cross`` of two 3-vectors, written out.

    The same products in the same order as numpy's, so the same bits, without
    numpy's axis handling -- which made ``np.cross`` about half the cost of a
    rollout (it is called per node and per Newton iteration).  1-D 3-vectors
    only; batched callers keep ``np.cross``.
    """
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def hat(v):
    """Skew-symmetric matrix with ``hat(v) @ w == cross(v, w)``."""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def parallel_transport(u, t_from, t_to):
    """Carry ``u`` from the frame adapted to ``t_from`` onto ``t_to``.

    This is the rotation about ``t_from x t_to`` taking one tangent to the
    other; it is the unique rotation doing so that adds no twist about the
    tangent, which is what makes the Bishop frame twist-free.
    """
    b = _cross3(t_from, t_to)
    nb = np.linalg.norm(b)
    if nb < _EPS:                      # tangents (anti)parallel: nothing to do
        return np.asarray(u, dtype=float).copy()
    b = b / nb
    # Re-orthogonalize against both tangents; cheap insurance against drift.
    b = b - np.dot(b, t_from) * t_from
    b = b / max(np.linalg.norm(b), _EPS)
    b = b - np.dot(b, t_to) * t_to
    b = b / max(np.linalg.norm(b), _EPS)
    n_from = _cross3(t_from, b)
    n_to = _cross3(t_to, b)
    return np.dot(u, t_from) * t_to + np.dot(u, n_from) * n_to + np.dot(u, b) * b


def signed_angle(u, v, n):
    """Angle from ``u`` to ``v``, signed by the reference normal ``n``."""
    w = _cross3(u, v)
    angle = np.arctan2(np.linalg.norm(w), float(np.dot(u, v)))
    return -angle if float(np.dot(n, w)) < 0.0 else angle


def tangents(pos):
    """Unit tangents of every edge of the polyline ``pos`` of shape (nv, 3)."""
    e = pos[1:] - pos[:-1]
    n = np.linalg.norm(e, axis=1)
    return e / np.maximum(n, _EPS)[:, None]


def curvature_binormal(te, tf):
    """``kb = 2 te x tf / (1 + te . tf)`` -- the discrete curvature binormal."""
    return 2.0 * _cross3(te, tf) / (1.0 + float(np.dot(te, tf)))


# ------------------------------------------------- the gripper's yaw reference

def gripper_director(t):
    """Unit vector perpendicular to ``t``, the zero of that end's yaw.

    ``t x normalize(z x t)``: the direction perpendicular to the tangent in the
    vertical plane through it -- "up, as far as the tangent allows".  For the
    straight rest pose (``t = +x``) it is ``+z``.  Singular for a vertical
    tangent, where the control's own elevation angle is singular too.
    """
    t = np.asarray(t, dtype=float)
    a = _cross3(_UP, t)
    na = np.linalg.norm(a)
    if na < _EPS:
        raise ValueError("gripper_director: tangent is vertical, the yaw "
                         "reference (and the control's elevation angle) is "
                         "singular there")
    return _cross3(t, a / na)


def director_spin(t):
    """``omega`` with ``d(angle of gripper_director)/d t = omega . dt``.

    The angle is measured about ``t`` and against parallel transport, so this
    is how much the yaw reference itself spins when the gripper only tilts.
    In the azimuth/elevation chart of the controls it is exactly
    ``sin(phi) dtheta``; in the components of ``t``::

        omega = t_z * (-t_y, t_x, 0) / (t_x^2 + t_y^2)
    """
    t = np.asarray(t, dtype=float)
    s = t[0] * t[0] + t[1] * t[1]
    if s < _EPS:
        raise ValueError("director_spin: tangent is vertical (see "
                         "gripper_director)")
    return t[2] * np.array([-t[1], t[0], 0.0]) / s


def director_spin_jac(t):
    """``d(omega)/dt``, shape (3, 3), for :func:`director_spin`."""
    t = np.asarray(t, dtype=float)
    s = t[0] * t[0] + t[1] * t[1]
    if s < _EPS:
        raise ValueError("director_spin_jac: tangent is vertical")
    c = np.array([-t[1], t[0], 0.0])
    J = np.empty((3, 3))
    J[:, 0] = t[2] * (np.array([0.0, 1.0, 0.0]) / s - 2.0 * t[0] * c / s ** 2)
    J[:, 1] = t[2] * (np.array([-1.0, 0.0, 0.0]) / s - 2.0 * t[1] * c / s ** 2)
    J[:, 2] = c / s
    return J


def _spin_grad_jac(p_a, p_b):
    """Gradient and Jacobian of the yaw reference's angle on the edge a->b.

    Returns ``(g (6,), J (6, 6))`` over the stacked nodes ``(p_a, p_b)`` for
    the scalar whose variation is ``director_spin(t) . dt``.  That scalar is
    not a function on its own (the spin is a non-exact one-form -- there is no
    frame field on the sphere that is parallel everywhere); only the whole of
    ``H`` is, which is why ``J`` here need not be symmetric while the assembled
    Hessian of ``H`` is.
    """
    e = np.asarray(p_b, dtype=float) - np.asarray(p_a, dtype=float)
    n = np.linalg.norm(e)
    if n < _EPS:
        return np.zeros(6), np.zeros((6, 6))
    t = e / n
    w = director_spin(t)
    P = np.eye(3) - np.outer(t, t)
    # d/de of (omega(t(e)) / |e|), with dt/de = P / |e| and d(1/|e|)/de = -t/|e|^2
    Jg = (director_spin_jac(t) @ P - np.outer(w, t)) / n ** 2

    g = np.concatenate([-w / n, w / n])
    J = np.block([[Jg, -Jg], [-Jg, Jg]])
    return g, J


# ------------------------------------------------------- the reference frame

def rotate_axis_angle(v, z, theta):
    """``v`` rotated by ``theta`` about the unit axis ``z`` (Rodrigues)."""
    if theta == 0:
        return v
    c, s = np.cos(theta), np.sin(theta)
    return c * v + s * _cross3(z, v) + float(np.dot(z, v)) * (1.0 - c) * z


def space_parallel(a1_first, pos):
    """``(a1, a2)`` of every edge, by parallel transport along the rod.

    ``a1`` on the first edge is ``a1_first``; every later edge gets the
    previous one transported across the node between them.  Used once, at
    reset, to start the reference frame.
    """
    t = tangents(np.asarray(pos, dtype=float))
    a1 = np.empty_like(t)
    a2 = np.empty_like(t)
    u = np.asarray(a1_first, dtype=float)
    a1[0] = u
    a2[0] = _cross3(t[0], u)
    for c in range(1, len(t)):
        u = parallel_transport(u, t[c - 1], t[c])
        u = u - np.dot(u, t[c]) * t[c]
        u = u / np.linalg.norm(u)
        a1[c] = u
        a2[c] = _cross3(t[c], u)
    return a1, a2


def time_parallel(a1_old, pos_old, pos):
    """``(a1, a2)`` carried in time from the centreline ``pos_old`` to ``pos``.

    Each edge's ``a1`` is parallel transported from that edge's old tangent to
    its new one -- no information passes between edges.
    """
    t_old = tangents(np.asarray(pos_old, dtype=float))
    t = tangents(np.asarray(pos, dtype=float))
    a1 = np.empty_like(t)
    a2 = np.empty_like(t)
    for c in range(len(t)):
        u = parallel_transport(a1_old[c], t_old[c], t[c])
        u = u - np.dot(u, t[c]) * t[c]
        u = u / np.linalg.norm(u)
        a1[c] = u
        a2[c] = _cross3(t[c], u)
    return a1, a2


def reference_twist(a1, t, guess):
    """Reference twist at every node, shape ``(nv,)``; the end entries are 0.

    At interior node ``c`` it is the angle from ``a1^{c-1}`` transported across
    the node to ``a1^c``, about ``t^c``, resolved on the 2 pi branch nearest
    ``guess[c]``.  Returns a new array: ``guess`` is left untouched.
    """
    ref = np.array(guess, dtype=float, copy=True)
    for c in range(1, len(t)):
        ut = parallel_transport(a1[c - 1], t[c - 1], t[c])
        ut = rotate_axis_angle(ut, t[c], ref[c])
        ref[c] = ref[c] + signed_angle(ut, a1[c], t[c])
    return ref


def material_directors(a1, a2, theta):
    """``(m1, m2)``: the reference frame rotated by ``theta`` about each tangent."""
    c = np.cos(theta)[:, None]
    s = np.sin(theta)[:, None]
    return c * a1 + s * a2, -s * a1 + c * a2


# ------------------------------------------------------------- the Bishop frame

def bishop_directors(pos):
    """The Bishop director of every edge, shape ``(nv - 1, 3)``.

    Seeded with :func:`gripper_director` of the first edge -- so the left
    gripper's yaw is the first edge's material angle outright -- and parallel
    transported along the rod.  A pure function of ``pos``: nothing is carried
    between steps.
    """
    pos = np.asarray(pos, dtype=float)
    t = tangents(pos)
    u = np.empty_like(t)
    u[0] = gripper_director(t[0])
    for i in range(1, len(t)):
        u[i] = parallel_transport(u[i - 1], t[i - 1], t[i])
    return u


def holonomy(pos):
    """``H``: the angle the transport chain loses against the far yaw reference.

    The Bishop director of the last edge, against that edge's
    :func:`gripper_director`, about its tangent.  Zero for any planar
    configuration in the vertical plane, which is why a flat rod's twist is
    just the difference of the two yaws.
    """
    # Inlined rather than calling bishop_directors: this runs once per energy
    # gradient and once per Hessian, and only the last director is needed.
    pos = np.asarray(pos, dtype=float)
    t = tangents(pos)
    u = gripper_director(t[0])
    for i in range(1, len(t)):
        u = parallel_transport(u, t[i - 1], t[i])
    return signed_angle(gripper_director(t[-1]), u, t[-1])


def holonomy_grad_hess(pos, need_hess=True):
    """``(H, grad (3 nv,), hess (3 nv, 3 nv))`` -- all exact.

    ``hess`` is ``None`` when ``need_hess`` is False; the Newton line search
    asks for the gradient far more often than for the Hessian.
    """
    pos = np.asarray(pos, dtype=float)
    nv = pos.shape[0]
    g = np.zeros(3 * nv)
    H = np.zeros((3 * nv, 3 * nv)) if need_hess else None

    # The transport chain: one holonomy stencil per interior node.
    for i in range(1, nv - 1):
        gp, Hp = psi_grad_hess(pos[i - 1], pos[i], pos[i + 1], need_hess=need_hess)
        sl = slice(3 * (i - 1), 3 * (i + 2))
        g[sl] += gp
        if need_hess:
            H[sl, sl] += Hp

    # The yaw reference spins with each end edge: + at the seeded end, - at
    # the end the chain is compared against.
    for sl, sign in ((slice(0, 6), 1.0), (slice(3 * nv - 6, 3 * nv), -1.0)):
        gs, Js = _spin_grad_jac(pos[sl.start // 3], pos[sl.start // 3 + 1])
        g[sl] += sign * gs
        if need_hess:
            H[sl, sl] += sign * Js

    # Each piece above is a one-form, not the gradient of a function of its own,
    # so the Jacobian assembled from them carries a spurious antisymmetric part.
    # Only their sum is exact, and its symmetric part is the Hessian -- verified
    # against second differences by scripts/check_twist.py.
    if need_hess:
        H = 0.5 * (H + H.T)

    return holonomy(pos), g, H


# ------------------------------------------- derivatives of one holonomy stencil

def psi_grad_hess(p0, p1, p2, need_hess=True):
    """``(grad (9,), hess (9, 9))`` of one node's transport holonomy.

    The classical result (Bergou et al. 2008, eq. 9): carrying a director
    across the node between edges ``e = p1 - p0`` and ``f = p2 - p1`` rotates
    it about the tangent at the rate::

        dpsi/de = kb(t_e, t_f) / (2 |e|)
        dpsi/df = kb(t_e, t_f) / (2 |f|)

    with ``kb`` the curvature binormal.  Summed over the interior nodes this is
    the gradient of the chain's holonomy (eq. 10); on its own it is a one-form,
    not the gradient of anything (see :func:`_spin_grad_jac`).

    The expression is differentiated once more for the Hessian.  As with the
    bending stencil, the raw chain-rule matrix carries a spurious
    antisymmetric piece; the true second derivative is its symmetric part.
    """
    e, f = np.asarray(p1, float) - np.asarray(p0, float), \
           np.asarray(p2, float) - np.asarray(p1, float)
    ne, nf = np.linalg.norm(e), np.linalg.norm(f)
    zero_h = np.zeros((9, 9)) if need_hess else None
    if ne < _EPS or nf < _EPS:
        return np.zeros(9), zero_h

    te, tf = e / ne, f / nf
    chi = 1.0 + float(np.dot(te, tf))
    if chi < _EPS:                      # 180 degree fold: genuinely singular
        return np.zeros(9), zero_h
    # -kb, so that scattering onto the nodes below lands on the paper's
    # +kb/(2|e^{i-1}|) and -kb/(2|e^i|); the sign is the orientation
    # convention of signed_angle, and check_twist.py pins it down.
    kb = -2.0 * _cross3(te, tf) / chi

    A = kb / (2.0 * ne)
    B = kb / (2.0 * nf)

    Bm = np.zeros((6, 9))
    I3 = np.eye(3)
    Bm[0:3, 0:3] = -I3
    Bm[0:3, 3:6] = I3
    Bm[3:6, 3:6] = -I3
    Bm[3:6, 6:9] = I3

    g = Bm.T @ np.concatenate([A, B])
    if not need_hess:
        return g, None

    Pe = (I3 - np.outer(te, te)) / ne       # dte/de
    Pf = (I3 - np.outer(tf, tf)) / nf       # dtf/df
    dkb_de = (2.0 / chi) * (hat(tf) @ Pe) - (1.0 / chi) * np.outer(kb, Pe @ tf)
    dkb_df = -(2.0 / chi) * (hat(te) @ Pf) - (1.0 / chi) * np.outer(kb, Pf @ te)

    M = np.zeros((6, 6))
    M[0:3, 0:3] = dkb_de / (2.0 * ne) - np.outer(kb, te) / (2.0 * ne ** 2)
    M[0:3, 3:6] = dkb_df / (2.0 * ne)
    M[3:6, 0:3] = dkb_de / (2.0 * nf)
    M[3:6, 3:6] = dkb_df / (2.0 * nf) - np.outer(kb, tf) / (2.0 * nf ** 2)

    Hs = Bm.T @ M @ Bm
    return g, 0.5 * (Hs + Hs.T)


# ------------------------------------------------------- material frames (view)

def material_frames(pos, yaw_l, yaw_r):
    """``(m1, m2)`` of every edge, shapes ``(nv - 1, 3)`` -- for drawing only.

    The quasistatic material frame of sec. 5: the Bishop frame rotated by an
    angle that grows uniformly along the rod, from ``yaw_l`` on the first edge
    to ``yaw_l + Phi`` on the last.
    """
    pos = np.asarray(pos, dtype=float)
    t = tangents(pos)
    u = bishop_directors(pos)
    n_e = len(t)
    phi_total = float(yaw_r) - float(yaw_l) - holonomy(pos)
    frac = np.arange(n_e) / max(n_e - 1, 1)
    theta = float(yaw_l) + phi_total * frac

    v = np.cross(t, u)
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    return c * u + s * v, -s * u + c * v
