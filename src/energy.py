# -*- coding: utf-8 -*-
"""Pluggable elastic-energy models for the rod simulator.

The simulator solves a quasi-static equilibrium every step (see ``objfun`` in
SimulatorEnv_2D.py).  The *only* place the material model enters that solve is
the elastic term::

    F_elastic, J_elastic = energy.force_and_jacobian(q)

Everything else (Newton loop, contact, gravity, boundary conditions, the
adjoint gradient) is independent of which energy is used.  This module defines
the interface for that one call.

Adding your own energy
----------------------
1. Subclass :class:`RodEnergy`, implement ``gradient`` and ``hessian``.
2. Register it with ``@register_energy("my_name")``.
3. In your YAML config set ``energy.custom: true`` and ``energy.type: my_name``.

Sign convention (IMPORTANT)
---------------------------
``gradient``/``hessian`` return the derivatives of the ENERGY::

    gradient(q) = dE/dq        hessian(q) = d2E/dq2

``force_and_jacobian`` negates them, because the Newton solver works with
forces::

    F = -dE/dq                 J = -d2E/dq2

Getting this backwards makes the Newton solve diverge or converge to a
maximum.  Run :meth:`RodEnergy.check_gradients` on any new subclass.
"""
from __future__ import annotations

import numpy as np


############################
# Base class
############################

class RodEnergy:
    """Common interface for every rod elastic-energy model."""

    #: Human-readable name, used in log output.
    name = "base"

    #: Whether the model carries twist DOFs (and so needs material frames).
    #: Twist-free models leave the DOF vector as pure positions; a twisting
    #: model also sets ``n_twist``, how many angles it appends to them.
    has_twist = False

    #: Spatial dimension the model is written for (2 or 3).  The DOF vector q
    #: is laid out node-major, i.e. ``q = [x0, y0, (z0), x1, y1, (z1), ...]``
    #: with ``dim`` entries per node.  Set it to ``None`` if the model works in
    #: any dimension; ``build_energy`` then accepts it for both 2D and 3D and
    #: passes the environment's ``dim`` to the constructor.
    dim = 2

    # ---------- to be implemented by subclasses ----------

    def energy(self, q: np.ndarray) -> float:
        """Total elastic energy E(q).  Optional: only used by check_gradients."""
        raise NotImplementedError(f"{type(self).__name__}.energy() not implemented")

    def gradient(self, q: np.ndarray) -> np.ndarray:
        """dE/dq, shape (dim*nv,)."""
        raise NotImplementedError(f"{type(self).__name__}.gradient() not implemented")

    def hessian(self, q: np.ndarray) -> np.ndarray:
        """d2E/dq2, shape (dim*nv, dim*nv).

        Must be the EXACT second derivative, not a quasi-Newton approximation:
        this matrix is reused by the adjoint gradient (implicit_grad_tools.py).
        An approximate Hessian still lets Newton converge but silently corrupts
        the training gradient.
        """
        raise NotImplementedError(f"{type(self).__name__}.hessian() not implemented")

    @property
    def stiffness_scale(self) -> float:
        """Characteristic stiffness, used only to size the Newton tolerance
        (``tol = stiffness_scale / L**2 * 1e-3``)."""
        raise NotImplementedError(
            f"{type(self).__name__}.stiffness_scale not implemented")

    # ---------- provided by the base class ----------

    def force_and_jacobian(self, q: np.ndarray):
        """The single method ``objfun`` calls.  Returns ``(F, J) = (-dE/dq, -d2E/dq2)``."""
        return -self.gradient(q), -self.hessian(q)

    def force(self, q: np.ndarray) -> np.ndarray:
        """``-dE/dq`` alone.

        The Newton line search (src/newton.py) evaluates the residual many
        times per iteration but never needs the Hessian, which dominates the
        assembly cost.  Override this when the gradient is genuinely cheaper
        than ``force_and_jacobian``.
        """
        return -self.gradient(q)

    def check_gradients(self, q: np.ndarray, eps: float = 1e-6, verbose: bool = True):
        """Finite-difference self-check.  Run this before wiring a new subclass in.

        Verifies ``J == dF/dq`` (always), and ``F == -dE/dq`` when ``energy()``
        is implemented.  Returns a dict of relative errors.
        """
        q = np.asarray(q, dtype=float)
        F0, J0 = self.force_and_jacobian(q)

        J_fd = np.zeros_like(J0)
        for i in range(q.size):
            dq = np.zeros_like(q)
            dq[i] = eps
            F_plus, _ = self.force_and_jacobian(q + dq)
            F_minus, _ = self.force_and_jacobian(q - dq)
            J_fd[:, i] = (F_plus - F_minus) / (2.0 * eps)

        scale = max(np.abs(J0).max(), 1e-30)
        errors = {"hessian_rel_err": float(np.abs(J0 - J_fd).max() / scale)}

        try:
            g_fd = np.zeros_like(q)
            for i in range(q.size):
                dq = np.zeros_like(q)
                dq[i] = eps
                g_fd[i] = (self.energy(q + dq) - self.energy(q - dq)) / (2.0 * eps)
            gscale = max(np.abs(F0).max(), 1e-30)
            errors["gradient_rel_err"] = float(np.abs(-F0 - g_fd).max() / gscale)
        except NotImplementedError:
            errors["gradient_rel_err"] = None

        if verbose:
            print(f"[check_gradients] {self.name}: {errors}")
        return errors


############################
# Registry
############################

ENERGY_REGISTRY = {}


def register_energy(name: str):
    """Class decorator: make an energy selectable via ``energy.type: <name>``."""
    def _wrap(cls):
        ENERGY_REGISTRY[name] = cls
        cls.name = name
        return cls
    return _wrap


############################
# Built-in: the existing DER model
############################

@register_energy("der")
class DERElasticEnergy(RodEnergy):
    """The repository's original Discrete Elastic Rod energy (stretching + bending).

    Wraps the existing ``getFs``/``getFb`` verbatim — no formula is rewritten.
    Those two already return ``(-dE/dq, -d2E/dq2)``, so this class overrides
    ``force_and_jacobian`` directly instead of going through
    ``gradient``/``hessian`` (which would need pointless sign flips).
    """

    def __init__(self, EI: float, EA: float, deltaL: float, **_ignored):
        self.EI = EI
        self.EA = EA
        self.deltaL = deltaL

    def force_and_jacobian(self, q):
        # Imported lazily: SimulatorEnv_2D imports this module at load time.
        from SimulatorEnv_2D import getFs, getFb
        Fs, Js = getFs(q, self.EA, self.deltaL)
        Fb, Jb = getFb(q, self.EI, self.deltaL)
        return Fs + Fb, Js + Jb

    def energy(self, q):
        """The same stretching and bending energies getFs/getFb differentiate.

        Reported as a diagnostic (how hard a rollout is working the rod); the
        solve itself only ever needs the force and its Jacobian above.
        """
        x = np.asarray(q, dtype=float).reshape(-1, 2)
        e = np.diff(x, axis=0)
        ne = np.linalg.norm(e, axis=1)
        E = 0.5 * (self.EA / self.deltaL) * float(np.sum((ne - self.deltaL) ** 2))
        t = e / np.maximum(ne, 1e-30)[:, None]
        c = np.clip(np.sum(t[:-1] * t[1:], axis=1), -1.0 + 1e-14, 1.0)
        # |kb|^2 = 4 (1 - c) / (1 + c) for the curvature binormal at each node
        E += (self.EI / (2.0 * self.deltaL)) * float(np.sum(4.0 * (1.0 - c) / (1.0 + c)))
        return float(E)

    def gradient(self, q):
        F, _ = self.force_and_jacobian(q)
        return -F

    def hessian(self, q):
        _, J = self.force_and_jacobian(q)
        return -J

    def force(self, q):
        # getFs/getFb build force and Jacobian together, so there is no
        # cheaper force-only path here; just avoid the double sign flip.
        return self.force_and_jacobian(q)[0]

    @property
    def stiffness_scale(self):
        return self.EI


############################
# Example custom energy (template)
############################

@register_energy("linear_spring")
class LinearSpringEnergy(RodEnergy):
    """Minimal demo: a chain of linear springs, no bending stiffness.

    Not physically interesting — it exists so the ``energy.custom`` switch can
    be exercised end to end, and as a copy-paste template showing the shapes
    and sign convention a real custom energy must follow.

        E = sum_k  0.5 * k_spring * (|x_{k+1} - x_k| - deltaL)^2

    Dimension-generic: ``dim = None`` at class level marks it usable from both
    the 2D and the 3D environment, which pass their own ``dim`` in.
    """

    dim = None

    def __init__(self, deltaL: float, k_spring: float = 1.0e4, dim: int = 2, **_ignored):
        self.deltaL = deltaL
        self.k_spring = k_spring
        self.dim = int(dim)

    def energy(self, q):
        x = np.asarray(q, dtype=float).reshape(-1, self.dim)
        d = x[1:] - x[:-1]
        length = np.linalg.norm(d, axis=1)
        return float(0.5 * self.k_spring * np.sum((length - self.deltaL) ** 2))

    def gradient(self, q):
        x = np.asarray(q, dtype=float).reshape(-1, self.dim)
        g = np.zeros_like(x)
        d = x[1:] - x[:-1]
        length = np.linalg.norm(d, axis=1)
        length = np.where(length < 1e-12, 1e-12, length)
        coeff = self.k_spring * (length - self.deltaL) / length
        contrib = coeff[:, None] * d          # dE/d(x_{k+1}) for each edge
        g[1:] += contrib
        g[:-1] -= contrib
        return g.reshape(-1)

    def hessian(self, q):
        d_ = self.dim
        x = np.asarray(q, dtype=float).reshape(-1, d_)
        n = x.shape[0]
        H = np.zeros((d_ * n, d_ * n))
        eye = np.eye(d_)
        for k in range(n - 1):
            d = x[k + 1] - x[k]
            length = max(float(np.linalg.norm(d)), 1e-12)
            u = d / length
            # d2E/d(x_{k+1})2 for one edge
            block = self.k_spring * (
                np.outer(u, u) + (1.0 - self.deltaL / length) * (eye - np.outer(u, u))
            )
            a, b = d_ * k, d_ * (k + 1)
            H[b:b + d_, b:b + d_] += block
            H[a:a + d_, a:a + d_] += block
            H[a:a + d_, b:b + d_] -= block
            H[b:b + d_, a:a + d_] -= block
        return H

    @property
    def stiffness_scale(self):
        return self.k_spring * self.deltaL ** 2


############################
# Built-in: 3D Discrete Elastic Rod (naturally straight, twist-free)
############################

@register_energy("der3d")
class DERElasticEnergy3D(RodEnergy):
    """Discrete Elastic Rod energy in 3D, for a naturally straight rod.

    DOF layout is node-major: ``q = [x0, y0, z0, x1, y1, z1, ...]``.

    Stretching (one linear spring per edge, identical to the 2D model)::

        E_s = sum_k  0.5 * (EA / dL) * (|e_k| - dL)^2

    Bending (isotropic, zero natural curvature, no twist DOF)::

        E_b = sum_k (EI / (2 * dL)) * |(kb)_k|^2

    where ``kb = 2 * te x tf / (1 + te . tf)`` is the curvature binormal of the
    two edges meeting at node k.  Because the rod is naturally straight and
    bending is isotropic, ``|kb|^2`` collapses to a function of the single
    scalar ``c = te . tf``::

        |kb|^2 = 4 * (1 - c) / (1 + c)

    which is what makes the exact gradient *and* exact Hessian cheap to write
    down (the adjoint gradient in implicit_grad_tools.py needs the exact
    Hessian, not an approximation).  Restricted to the z = 0 plane this energy
    reduces term-by-term to the 2D ``der`` model, so a 3D run with all z DOFs
    pinned at 0 reproduces the 2D simulator.

    Twisting would require a material frame and parallel transport; it is
    deliberately not modelled here (see :class:`DERElasticEnergy3DTwist`).
    """

    dim = 3

    def __init__(self, EI: float, EA: float, deltaL: float, **_ignored):
        self.EI = EI
        self.EA = EA
        self.deltaL = deltaL
        # Bending prefactor: EI / (2 * l_k), with Voronoi length l_k = deltaL.
        self._kb_coeff = EI / (2.0 * deltaL)
        # Stretching spring constant: EA / dL  (E = 0.5 * k * (|e| - dL)^2).
        self._ks = EA / deltaL

    # ---------- per-stencil kernels ----------

    def _stretch_stencil(self, p0, p1):
        """(E, grad(6,), hess(6,6)) for one stretching edge."""
        d = p1 - p0
        length = float(np.linalg.norm(d))
        if length < 1e-12:
            return 0.0, np.zeros(6), np.zeros((6, 6))
        t = d / length
        dl = length - self.deltaL

        E = 0.5 * self._ks * dl * dl
        gd = self._ks * dl * t                       # dE/d(edge vector)
        # d2E/d(edge)2 = k * [ t t^T + (dl/length) * (I - t t^T) ]
        P = np.eye(3) - np.outer(t, t)
        Hd = self._ks * (np.outer(t, t) + (dl / length) * P)

        g = np.concatenate([-gd, gd])
        H = np.zeros((6, 6))
        H[0:3, 0:3] = Hd
        H[3:6, 3:6] = Hd
        H[0:3, 3:6] = -Hd
        H[3:6, 0:3] = -Hd
        return E, g, H

    def _bend_stencil(self, p0, p1, p2):
        """(E, grad(9,), hess(9,9)) for one bending stencil (nodes p0-p1-p2)."""
        ee = p1 - p0
        ef = p2 - p1
        ne = float(np.linalg.norm(ee))
        nf = float(np.linalg.norm(ef))
        if ne < 1e-12 or nf < 1e-12:
            return 0.0, np.zeros(9), np.zeros((9, 9))

        te = ee / ne
        tf = ef / nf
        c = float(np.dot(te, tf))
        # chi -> 0 is a 180 deg fold; the energy is genuinely singular there.
        c = min(1.0, max(-1.0 + 1e-8, c))
        chi = 1.0 + c

        # f(c) = |kb|^2 = 4 (1 - c) / (1 + c)
        f = 4.0 * (1.0 - c) / chi
        fp = -8.0 / chi ** 2
        fpp = 16.0 / chi ** 3

        I3 = np.eye(3)
        Pe = I3 - np.outer(te, te)
        Pf = I3 - np.outer(tf, tf)

        # dc/d(ee), dc/d(ef)
        ge = (tf - c * te) / ne
        gf = (te - c * tf) / nf

        # d2c / d(ee,ef)2
        Hee = -(np.outer(te, ge) + np.outer(ge, te)) / ne - c * Pe / ne ** 2
        Hff = -(np.outer(tf, gf) + np.outer(gf, tf)) / nf - c * Pf / nf ** 2
        Hef = (Pf / nf - np.outer(te, gf)) / ne

        M = np.zeros((6, 6))
        M[0:3, 0:3] = Hee
        M[0:3, 3:6] = Hef
        M[3:6, 0:3] = Hef.T
        M[3:6, 3:6] = Hff

        # d(ee, ef) / d(p0, p1, p2)
        B = np.zeros((6, 9))
        B[0:3, 0:3] = -I3
        B[0:3, 3:6] = I3
        B[3:6, 3:6] = -I3
        B[3:6, 6:9] = I3

        gc = B.T @ np.concatenate([ge, gf])          # dc/d(nodes), (9,)
        Hc = B.T @ M @ B                              # d2c/d(nodes)2, (9,9)

        E = self._kb_coeff * f
        g = self._kb_coeff * fp * gc
        H = self._kb_coeff * (fpp * np.outer(gc, gc) + fp * Hc)
        return E, g, H

    # ---------- RodEnergy interface ----------

    def energy(self, q):
        x = np.asarray(q, dtype=float).reshape(-1, 3)
        n = x.shape[0]
        E = 0.0
        for k in range(n - 1):
            E += self._stretch_stencil(x[k], x[k + 1])[0]
        for k in range(1, n - 1):
            E += self._bend_stencil(x[k - 1], x[k], x[k + 1])[0]
        return float(E)

    def gradient(self, q):
        # Assembled independently of hessian(), so RodEnergy.force() (used by
        # the Newton line search) is genuinely cheap for this model.
        x = np.asarray(q, dtype=float).reshape(-1, 3)
        n = x.shape[0]
        g = np.zeros(3 * n)
        for k in range(n - 1):
            _, gs, _ = self._stretch_stencil(x[k], x[k + 1])
            g[3 * k:3 * k + 6] += gs
        for k in range(1, n - 1):
            _, gb, _ = self._bend_stencil(x[k - 1], x[k], x[k + 1])
            g[3 * k - 3:3 * k + 6] += gb
        return g

    def hessian(self, q):
        x = np.asarray(q, dtype=float).reshape(-1, 3)
        n = x.shape[0]
        H = np.zeros((3 * n, 3 * n))
        for k in range(n - 1):
            _, _, Hs = self._stretch_stencil(x[k], x[k + 1])
            a = 3 * k
            H[a:a + 6, a:a + 6] += Hs
        for k in range(1, n - 1):
            _, _, Hb = self._bend_stencil(x[k - 1], x[k], x[k + 1])
            a = 3 * k - 3
            H[a:a + 9, a:a + 9] += Hb
        return H

    @property
    def stiffness_scale(self):
        return self.EI


############################
# Built-in: 3D DER with torsion (reference frames carried in time)
############################
#
# The three stencils below are ported line by line from the MAE 263F appendix
# (github.com/StructuresComp/mae-263F, appendix/gradEs_hessEs.py,
# gradEb_hessEb.py, gradEt_hessEt.py at commit 7d4a4b1; M. Khalid Jawed,
# UCLA), including its Panetta et al. 2019 correction of the bending and
# twisting Hessians.  Each returns the gradient and Hessian of the ENERGY of one
# stencil over its local DOFs; the bending/twisting ones are ordered
# ``[node0 (3), theta_e, node1 (3), theta_f, node2 (3)]`` as in 263F.  The only
# change is ``need_hess``, which returns before the Hessian is built.

def _stretch_grad_hess(node0, node1, l_k, EA, need_hess=True):
    """``(dE/dq (6,), d2E/dq2 (6, 6))`` of one edge; 263F ``gradEs_hessEs``."""
    edge = node1 - node0
    edgeLen = np.linalg.norm(edge)
    tangent = edge / edgeLen
    epsX = edgeLen / l_k - 1
    dF_unit = EA * tangent * epsX
    dF = np.zeros(6)
    dF[0:3] = -dF_unit
    dF[3:6] = dF_unit
    if not need_hess:
        return dF, None

    M = EA * ((1 / l_k - 1 / edgeLen) * np.eye(3)
              + 1 / edgeLen * np.outer(edge, edge) / edgeLen ** 2)
    dJ = np.zeros((6, 6))
    dJ[0:3, 0:3] = M
    dJ[3:6, 3:6] = M
    dJ[0:3, 3:6] = -M
    dJ[3:6, 0:3] = -M
    return dF, dJ


def _curvature(node0, node1, node2, m1e, m2e, m1f, m2f):
    """``(kappa1, kappa2)`` at the turning node; 263F ``computekappa``."""
    t0 = (node1 - node0) / np.linalg.norm(node1 - node0)
    t1 = (node2 - node1) / np.linalg.norm(node2 - node1)
    kb = 2.0 * np.cross(t0, t1) / (1.0 + np.dot(t0, t1))
    return np.array([0.5 * np.dot(kb, m2e + m2f), -0.5 * np.dot(kb, m1e + m1f)])


def _bend_grad_hess(node0, node1, node2, m1e, m2e, m1f, m2f, kappaBar, l_k,
                    EI1, EI2, need_hess=True):
    """``(dE/dq (11,), d2E/dq2 (11, 11))`` of one bending stencil; 263F
    ``gradEb_hessEb`` (Panetta 2019 correction)."""
    from rodframes import hat as crossMat

    gradKappa = np.zeros((11, 2))

    ee = node1 - node0
    ef = node2 - node1
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)
    te = ee / norm_e
    tf = ef / norm_f

    # Curvature binormal
    kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))
    chi = 1.0 + np.dot(te, tf)
    tilde_t = (te + tf) / chi
    tilde_d1 = (m1e + m1f) / chi
    tilde_d2 = (m2e + m2f) / chi

    # Curvatures
    kappa1 = 0.5 * np.dot(kb, m2e + m2f)
    kappa2 = -0.5 * np.dot(kb, m1e + m1f)

    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))
    Dkappa2De = 1.0 / norm_e * (-kappa2 * tilde_t - np.cross(tf, tilde_d1))
    Dkappa2Df = 1.0 / norm_f * (-kappa2 * tilde_t + np.cross(te, tilde_d1))
    gradKappa[0:3, 0] = -Dkappa1De
    gradKappa[4:7, 0] = Dkappa1De - Dkappa1Df
    gradKappa[8:11, 0] = Dkappa1Df
    gradKappa[0:3, 1] = -Dkappa2De
    gradKappa[4:7, 1] = Dkappa2De - Dkappa2Df
    gradKappa[8:11, 1] = Dkappa2Df
    gradKappa[3, 0] = -0.5 * np.dot(kb, m1e)
    gradKappa[7, 0] = -0.5 * np.dot(kb, m1f)
    gradKappa[3, 1] = -0.5 * np.dot(kb, m2e)
    gradKappa[7, 1] = -0.5 * np.dot(kb, m2f)

    # Gradient of Eb
    dkappaVector = np.array([kappa1, kappa2]) - kappaBar
    gradKappa_1 = gradKappa[:, 0]
    gradKappa_2 = gradKappa[:, 1]
    dE_dKappa1 = EI1 / l_k * dkappaVector[0]
    dE_dKappa2 = EI2 / l_k * dkappaVector[1]
    dF = dE_dKappa1 * gradKappa_1 + dE_dKappa2 * gradKappa_2
    if not need_hess:
        return dF, None

    # Hessian of the two curvatures
    DDkappa1 = np.zeros((11, 11))
    DDkappa2 = np.zeros((11, 11))

    norm2_e = norm_e ** 2
    norm2_f = norm_f ** 2

    tt_o_tt = np.outer(tilde_t, tilde_t)
    tmp = np.cross(tf, tilde_d2)
    tf_c_d2t_o_tt = np.outer(tmp, tilde_t)
    tt_o_tf_c_d2t = np.transpose(tf_c_d2t_o_tt)
    kb_o_d2e = np.outer(kb, m2e)
    te_o_te = np.outer(te, te)
    Id3 = np.eye(3)

    D2kappa1De2 = (1.0 / norm2_e * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tt_o_tf_c_d2t)
                   - kappa1 / (chi * norm2_e) * (Id3 - te_o_te)
                   + 1.0 / (2.0 * norm2_e) * kb_o_d2e)

    tmp = np.cross(te, tilde_d2)
    te_c_d2t_o_tt = np.outer(tmp, tilde_t)
    tt_o_te_c_d2t = np.transpose(te_c_d2t_o_tt)
    kb_o_d2f = np.outer(kb, m2f)
    tf_o_tf = np.outer(tf, tf)

    D2kappa1Df2 = (1.0 / norm2_f * (2 * kappa1 * tt_o_tt + te_c_d2t_o_tt + tt_o_te_c_d2t)
                   - kappa1 / (chi * norm2_f) * (Id3 - tf_o_tf)
                   + 1.0 / (2.0 * norm2_f) * kb_o_d2f)

    te_o_tf = np.outer(te, tf)
    D2kappa1DfDe = (-kappa1 / (chi * norm_e * norm_f) * (Id3 + te_o_tf)
                    + 1.0 / (norm_e * norm_f) * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt
                                                 + tt_o_te_c_d2t - crossMat(tilde_d2)))
    D2kappa1DeDf = np.transpose(D2kappa1DfDe)

    tmp = np.cross(tf, tilde_d1)
    tf_c_d1t_o_tt = np.outer(tmp, tilde_t)
    tt_o_tf_c_d1t = np.transpose(tf_c_d1t_o_tt)
    kb_o_d1e = np.outer(kb, m1e)

    D2kappa2De2 = (1.0 / norm2_e * (2.0 * kappa2 * tt_o_tt + tf_c_d1t_o_tt + tt_o_tf_c_d1t)
                   - kappa2 / (chi * norm2_e) * (Id3 - te_o_te)
                   - 1.0 / (2.0 * norm2_e) * kb_o_d1e)

    tmp = np.cross(te, tilde_d1)
    te_c_d1t_o_tt = np.outer(tmp, tilde_t)
    tt_o_te_c_d1t = np.transpose(te_c_d1t_o_tt)
    kb_o_d1f = np.outer(kb, m1f)

    D2kappa2Df2 = (1.0 / norm2_f * (2 * kappa2 * tt_o_tt - te_c_d1t_o_tt - tt_o_te_c_d1t)
                   - kappa2 / (chi * norm2_f) * (Id3 - tf_o_tf)
                   - 1.0 / (2.0 * norm2_f) * kb_o_d1f)

    D2kappa2DfDe = (-kappa2 / (chi * norm_e * norm_f) * (Id3 + te_o_tf)
                    + 1.0 / (norm_e * norm_f) * (2 * kappa2 * tt_o_tt + tf_c_d1t_o_tt
                                                 - tt_o_te_c_d1t + crossMat(tilde_d1)))
    D2kappa2DeDf = np.transpose(D2kappa2DfDe)

    D2kappa1Dthetae2 = -0.5 * np.dot(kb, m2e)
    D2kappa1Dthetaf2 = -0.5 * np.dot(kb, m2f)
    D2kappa2Dthetae2 = 0.5 * np.dot(kb, m1e)
    D2kappa2Dthetaf2 = 0.5 * np.dot(kb, m1f)

    D2kappa1DeDthetae = 1.0 / norm_e * (0.5 * np.dot(kb, m1e) * tilde_t - 1.0 / chi * np.cross(tf, m1e))
    D2kappa1DeDthetaf = 1.0 / norm_e * (0.5 * np.dot(kb, m1f) * tilde_t - 1.0 / chi * np.cross(tf, m1f))
    D2kappa1DfDthetae = 1.0 / norm_f * (0.5 * np.dot(kb, m1e) * tilde_t + 1.0 / chi * np.cross(te, m1e))
    D2kappa1DfDthetaf = 1.0 / norm_f * (0.5 * np.dot(kb, m1f) * tilde_t + 1.0 / chi * np.cross(te, m1f))
    D2kappa2DeDthetae = 1.0 / norm_e * (0.5 * np.dot(kb, m2e) * tilde_t - 1.0 / chi * np.cross(tf, m2e))
    D2kappa2DeDthetaf = 1.0 / norm_e * (0.5 * np.dot(kb, m2f) * tilde_t - 1.0 / chi * np.cross(tf, m2f))
    D2kappa2DfDthetae = 1.0 / norm_f * (0.5 * np.dot(kb, m2e) * tilde_t + 1.0 / chi * np.cross(te, m2e))
    D2kappa2DfDthetaf = 1.0 / norm_f * (0.5 * np.dot(kb, m2f) * tilde_t + 1.0 / chi * np.cross(te, m2f))

    for DD, De2, DfDe, DeDf, Df2, Dte2, Dtf2, DeDte, DfDte, DeDtf, DfDtf in (
            (DDkappa1, D2kappa1De2, D2kappa1DfDe, D2kappa1DeDf, D2kappa1Df2,
             D2kappa1Dthetae2, D2kappa1Dthetaf2, D2kappa1DeDthetae, D2kappa1DfDthetae,
             D2kappa1DeDthetaf, D2kappa1DfDthetaf),
            (DDkappa2, D2kappa2De2, D2kappa2DfDe, D2kappa2DeDf, D2kappa2Df2,
             D2kappa2Dthetae2, D2kappa2Dthetaf2, D2kappa2DeDthetae, D2kappa2DfDthetae,
             D2kappa2DeDthetaf, D2kappa2DfDthetaf)):
        # Curvature terms
        DD[0:3, 0:3] = De2
        DD[0:3, 4:7] = -De2 + DfDe
        DD[0:3, 8:11] = -DfDe
        DD[4:7, 0:3] = -De2 + DeDf
        DD[4:7, 4:7] = De2 - DeDf - DfDe + Df2
        DD[4:7, 8:11] = DfDe - Df2
        DD[8:11, 0:3] = -DeDf
        DD[8:11, 4:7] = DeDf - Df2
        DD[8:11, 8:11] = Df2
        # Twist terms
        DD[3, 3] = Dte2
        DD[7, 7] = Dtf2
        # Curvature-twist coupled terms
        DD[0:3, 3] = -DeDte
        DD[4:7, 3] = DeDte - DfDte
        DD[8:11, 3] = DfDte
        DD[3, 0:3] = DD[0:3, 3]
        DD[3, 4:7] = DD[4:7, 3]
        DD[3, 8:11] = DD[8:11, 3]
        DD[0:3, 7] = -DeDtf
        DD[4:7, 7] = DeDtf - DfDtf
        DD[8:11, 7] = DfDtf
        DD[7, 0:3] = DD[0:3, 7]
        DD[7, 4:7] = DD[4:7, 7]
        DD[7, 8:11] = DD[8:11, 7]

    # Hessian of Eb
    d2E_dKappa11 = EI1 / l_k
    d2E_dKappa22 = EI2 / l_k
    dJ = (dE_dKappa1 * DDkappa1 + dE_dKappa2 * DDkappa2
          + d2E_dKappa11 * np.outer(gradKappa_1, gradKappa_1)
          + d2E_dKappa22 * np.outer(gradKappa_2, gradKappa_2))
    return dF, dJ


def _twist_grad_hess(node0, node1, node2, theta_e, theta_f, refTwist, twistBar,
                     l_k, GJ, need_hess=True):
    """``(dE/dq (11,), d2E/dq2 (11, 11))`` of one twisting stencil; 263F
    ``gradEt_hessEt_panetta``."""
    from rodframes import hat as crossMat

    gradTwist = np.zeros(11)
    ee = node1 - node0
    ef = node2 - node1
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)
    norm2_e = norm_e ** 2
    norm2_f = norm_f ** 2
    te = ee / norm_e
    tf = ef / norm_f

    # Curvature binormal
    kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))

    # Gradient of twist wrt DOFs
    gradTwist[0:3] = -0.5 / norm_e * kb
    gradTwist[8:11] = 0.5 / norm_f * kb
    gradTwist[4:7] = -(gradTwist[0:3] + gradTwist[8:11])
    gradTwist[3] = -1
    gradTwist[7] = 1

    integratedTwist = theta_f - theta_e + refTwist - twistBar
    dE_dTau = GJ / l_k * integratedTwist
    dF = dE_dTau * gradTwist
    if not need_hess:
        return dF, None

    chi = 1.0 + np.dot(te, tf)
    tilde_t = (te + tf) / chi

    # Hessian of twist wrt DOFs (Panetta 2019).  D2mDfDe is d2m / de^i de^{i-1}.
    D2mDe2 = -0.5 / norm2_e * (np.outer(kb, (te + tilde_t)) + 2.0 / chi * crossMat(tf))
    D2mDf2 = -0.5 / norm2_f * (np.outer(kb, (tf + tilde_t)) - 2.0 / chi * crossMat(te))
    D2mDfDe = 0.5 / (norm_e * norm_f) * (2.0 / chi * crossMat(te) - np.outer(kb, tilde_t))
    D2mDeDf = 0.5 / (norm_e * norm_f) * (-2.0 / chi * crossMat(tf) - np.outer(kb, tilde_t))

    DDtwist = np.zeros((11, 11))
    DDtwist[0:3, 0:3] = D2mDe2
    DDtwist[0:3, 4:7] = -D2mDe2 + D2mDfDe
    DDtwist[4:7, 0:3] = -D2mDe2 + D2mDeDf
    DDtwist[4:7, 4:7] = D2mDe2 - (D2mDeDf + D2mDfDe) + D2mDf2
    DDtwist[0:3, 8:11] = -D2mDfDe
    DDtwist[8:11, 0:3] = -D2mDeDf
    DDtwist[8:11, 4:7] = D2mDeDf - D2mDf2
    DDtwist[4:7, 8:11] = D2mDfDe - D2mDf2
    DDtwist[8:11, 8:11] = D2mDf2

    d2E_dTau2 = GJ / l_k
    dJ = dE_dTau * DDtwist + d2E_dTau2 * np.outer(gradTwist, gradTwist)
    return dF, dJ


@register_energy("der3d_twist")
class DERElasticEnergy3DTwist(RodEnergy):
    """3D Discrete Elastic Rod with torsion -- the MAE 263F model.

    Stretching, bending and twisting exactly as in 263F ``getFs``, ``getFb`` and
    ``getFt``, with one material angle per edge and reference frames carried in
    time (see :mod:`rodframes`).  The DOF vector is 263F's, permuted so the
    positions come first and every ``q[:n_pos]`` in the codebase still means
    the centreline::

        q = [ x0 y0 z0 ... x_{nv-1} y_{nv-1} z_{nv-1} | theta^0 ... theta^{ne-1} ]
            \\________________ 3 * nv positions _______/ \\___ ne edge angles ___/

    Energy, summed over edges ``c`` and interior nodes ``i``::

        E_s = sum_c (EA l_c / 2) (|e^c| / l_c - 1)^2
        E_b = sum_i (EI / 2 lv_i) |kappa_i - kappaBar_i|^2
              kappa_i = ( kb.(m2^{i-1} + m2^i) / 2,  -kb.(m1^{i-1} + m1^i) / 2 )
        E_t = sum_i (GJ / 2 lv_i) (theta^i - theta^{i-1} + refTwist_i - twistBar_i)^2

    with ``l_c`` the rest edge lengths, ``lv_i`` the rest Voronoi lengths and
    ``kappaBar`` the curvature of the rest shape, all set by :meth:`reset_frame`.

    Frame state
    -----------
    The material frame depends on the reference frame, which is carried from
    the last accepted state, so the energy is a function of ``q`` *and* that
    state: ``(q_ref, a1_ref, refTwist_ref)``.  Every evaluation transports
    ``a1_ref`` in time from ``q_ref`` to ``q`` and resolves the reference twist
    against ``refTwist_ref`` (as 263F's ``objfun`` does at every Newton
    iterate), without changing the state.  :meth:`commit` makes ``q`` the new
    accepted state; it does not change the energy at ``q``.

    Hessian
    -------
    :meth:`hessian` is 263F's (Panetta-corrected) matrix, which is not
    symmetric.  The gradient formulas treat the reference frame as moving with
    ``q`` by parallel transport from ``q`` itself, which is exact only at
    ``q = q_ref``; there the symmetric part of the matrix is the true second
    derivative of the energy (checked by ``scripts/check_twist.py``).
    Symmetrising it is left to the callers.
    """

    dim = 3
    has_twist = True

    def __init__(self, EI: float, EA: float, deltaL: float, nv: int,
                 GJ: float = None, poisson: float = 0.3, **_ignored):
        self.EI = EI
        self.EA = EA
        self.deltaL = deltaL
        self.nv = int(nv)
        self.ne = self.nv - 1
        #: Twist DOFs behind the positions in ``q``: one material angle per edge.
        self.n_twist = self.ne
        # Circular cross-section: J = 2I and G = E / (2(1+nu)), so GJ = EI/(1+nu).
        self.GJ = float(EI / (1.0 + poisson)) if GJ is None else float(GJ)

        # Global DOF indices of each bending/twisting stencil, in 263F's local
        # order [node0, theta_e, node1, theta_f, node2], for interior node c.
        np_ = 3 * self.nv
        self._idx_bt = [np.array([3 * c - 3, 3 * c - 2, 3 * c - 1, np_ + c - 1,
                                  3 * c, 3 * c + 1, 3 * c + 2, np_ + c,
                                  3 * c + 3, 3 * c + 4, 3 * c + 5])
                        for c in range(1, self.ne)]

        # Rest geometry and frame state: set by reset_frame().
        self.refLen = self.voronoiRefLen = self.kappaBar = self.twistBar = None
        self.q_ref = self.a1_ref = self.refTwist_ref = None

    # ---------- layout helpers ----------

    @property
    def n_pos(self) -> int:
        """Number of position DOFs; the edge angles follow them in ``q``."""
        return 3 * self.nv

    @property
    def n_dof(self) -> int:
        return 3 * self.nv + self.n_twist

    def split(self, q):
        """``q -> (positions (nv, 3), theta (ne,))``."""
        q = np.asarray(q, dtype=float)
        return q[:self.n_pos].reshape(-1, 3), q[self.n_pos:]

    # ---------- frame state ----------

    def reset_frame(self, rest_nodes):
        """Start from the rest shape ``rest_nodes`` (nv, 3), with every theta = 0.

        Sets the rest edge and Voronoi lengths and ``kappaBar`` from it, and the
        reference frame by parallel transport along the rod, seeded with the
        left gripper's yaw reference.  The reference twist is then zero.
        """
        from rodframes import gripper_director, space_parallel, tangents
        pos = np.asarray(rest_nodes, dtype=float).reshape(-1, 3).copy()
        if pos.shape[0] != self.nv:
            raise ValueError(f"reset_frame: {pos.shape[0]} nodes, expected {self.nv}")

        self.refLen = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        self.voronoiRefLen = np.empty(self.nv)
        self.voronoiRefLen[0] = 0.5 * self.refLen[0]
        self.voronoiRefLen[-1] = 0.5 * self.refLen[-1]
        self.voronoiRefLen[1:-1] = 0.5 * (self.refLen[:-1] + self.refLen[1:])

        a1, a2 = space_parallel(gripper_director(tangents(pos)[0]), pos)
        # theta = 0, so the material frame is the reference frame.
        self.kappaBar = np.zeros((self.nv, 2))
        for c in range(1, self.ne):
            self.kappaBar[c] = _curvature(pos[c - 1], pos[c], pos[c + 1],
                                          a1[c - 1], a2[c - 1], a1[c], a2[c])
        self.twistBar = np.zeros(self.nv)

        self.q_ref = pos
        self.a1_ref = a1
        self.refTwist_ref = np.zeros(self.nv)

    def frames(self, q):
        """``(a1, a2, refTwist, m1, m2)`` at ``q``, carried from the accepted state."""
        from rodframes import time_parallel, reference_twist, material_directors, tangents
        if self.q_ref is None:
            raise RuntimeError("DERElasticEnergy3DTwist: call reset_frame(rest_nodes) first")
        pos, theta = self.split(q)
        a1, a2 = time_parallel(self.a1_ref, self.q_ref, pos)
        ref = reference_twist(a1, tangents(pos), self.refTwist_ref)
        m1, m2 = material_directors(a1, a2, theta)
        return a1, a2, ref, m1, m2

    def commit(self, q):
        """Make ``q`` the accepted state the frames are carried from."""
        a1, _, ref, _, _ = self.frames(q)
        self.q_ref = self.split(q)[0].copy()
        self.a1_ref = a1
        self.refTwist_ref = ref

    def get_frame_state(self):
        return {"q_ref": self.q_ref.copy(), "a1_ref": self.a1_ref.copy(),
                "refTwist_ref": self.refTwist_ref.copy()}

    def set_frame_state(self, state):
        self.q_ref = np.array(state["q_ref"], dtype=float)
        self.a1_ref = np.array(state["a1_ref"], dtype=float)
        self.refTwist_ref = np.array(state["refTwist_ref"], dtype=float)

    # ---------- diagnostics ----------

    def total_twist(self, q):
        """The rod's total twist, the sum of the integrated twists, in radians."""
        _, theta = self.split(q)
        ref = self.frames(q)[2]
        return float(theta[-1] - theta[0] + np.sum(ref[1:-1] - self.twistBar[1:-1]))

    def holonomy(self, q):
        """Bishop holonomy of the centreline -- geometry only, not in the energy."""
        from rodframes import holonomy
        return float(holonomy(self.split(q)[0]))

    # ---------- RodEnergy interface ----------

    def energy(self, q):
        pos, theta = self.split(q)
        _, _, ref, m1, m2 = self.frames(q)
        E = 0.0
        for c in range(self.ne):
            l_k = self.refLen[c]
            E += 0.5 * self.EA * l_k * (np.linalg.norm(pos[c + 1] - pos[c]) / l_k - 1.0) ** 2
        for c in range(1, self.ne):
            l_k = self.voronoiRefLen[c]
            dk = _curvature(pos[c - 1], pos[c], pos[c + 1],
                            m1[c - 1], m2[c - 1], m1[c], m2[c]) - self.kappaBar[c]
            E += 0.5 * self.EI / l_k * float(dk @ dk)
            tau = theta[c] - theta[c - 1] + ref[c] - self.twistBar[c]
            E += 0.5 * self.GJ / l_k * tau ** 2
        return float(E)

    def _assemble(self, q, need_hess):
        """``(dE/dq, d2E/dq2 or None)``, assembled stencil by stencil as in 263F."""
        pos, theta = self.split(q)
        _, _, ref, m1, m2 = self.frames(q)
        nd = self.n_dof
        g = np.zeros(nd)
        H = np.zeros((nd, nd)) if need_hess else None

        for c in range(self.ne):
            dF, dJ = _stretch_grad_hess(pos[c], pos[c + 1], self.refLen[c], self.EA,
                                        need_hess)
            sl = slice(3 * c, 3 * c + 6)
            g[sl] += dF
            if need_hess:
                H[sl, sl] += dJ

        for c in range(1, self.ne):
            idx = self._idx_bt[c - 1]
            l_k = self.voronoiRefLen[c]
            dFb, dJb = _bend_grad_hess(pos[c - 1], pos[c], pos[c + 1],
                                       m1[c - 1], m2[c - 1], m1[c], m2[c],
                                       self.kappaBar[c], l_k, self.EI, self.EI, need_hess)
            dFt, dJt = _twist_grad_hess(pos[c - 1], pos[c], pos[c + 1],
                                        theta[c - 1], theta[c], ref[c], self.twistBar[c],
                                        l_k, self.GJ, need_hess)
            g[idx] += dFb + dFt
            if need_hess:
                H[np.ix_(idx, idx)] += dJb + dJt
        return g, H

    def gradient(self, q):
        return self._assemble(q, need_hess=False)[0]

    def hessian(self, q):
        return self._assemble(q, need_hess=True)[1]

    def force_and_jacobian(self, q):
        g, H = self._assemble(q, need_hess=True)
        return -g, -H

    @property
    def stiffness_scale(self):
        return self.EI


############################
# Factory
############################

def build_energy(energy_cfg, *, EI: float, EA: float, deltaL: float, nv: int,
                 dim: int = 2, twist: bool = False,
                 poisson: float = 0.3) -> RodEnergy:
    """Build the energy model selected by the ``energy:`` block of the config.

    ``energy_cfg`` may be None or a dict like::

        energy:
          custom: false          # false -> the built-in model for this rod
          type: null             # only consulted when custom is true;
                                 # null means "the built-in one"
          params: {}             # extra kwargs forwarded to the subclass

    Which model is built in depends on the *rod*, not on ``type``:

    ====================  ===============
    2D                    ``der``
    3D                    ``der3d``
    3D + ``rod.twist``    ``der3d_twist``
    ====================  ===============

    With ``custom: false`` (the default) ``type`` is never read at all, so
    leaving it ``null`` is honest: naming a model there would suggest it is in
    use when it is not.  With ``custom: true`` it selects a registered class,
    ``null`` still meaning the built-in default, and the choice is checked
    against ``dim`` and ``twist`` so a mismatched model is rejected rather than
    silently misused.
    """
    cfg = dict(energy_cfg or {})
    if twist:
        if dim != 3:
            raise ValueError("rod.twist requires env.dim: 3")
        default_cls, default_name = DERElasticEnergy3DTwist, "der3d_twist"
    elif dim == 2:
        default_cls, default_name = DERElasticEnergy, "der"
    else:
        default_cls, default_name = DERElasticEnergy3D, "der3d"

    if not cfg.get("custom", False):
        return default_cls(EI=EI, EA=EA, deltaL=deltaL, nv=nv,
                           poisson=poisson)

    # `or` rather than a dict default: an explicit `type: null` should mean
    # "the built-in default", not None.
    type_name = cfg.get("type") or default_name
    if type_name not in ENERGY_REGISTRY:
        raise KeyError(
            f"Unknown energy type {type_name!r}. "
            f"Registered: {sorted(ENERGY_REGISTRY)}. "
            f"Use @register_energy('{type_name}') on your subclass."
        )

    cls = ENERGY_REGISTRY[type_name]
    if bool(getattr(cls, "has_twist", False)) != bool(twist):
        want = "a twist-capable" if twist else "a twist-free"
        raise ValueError(
            f"energy.type={type_name!r} is not {want} model, but rod.twist="
            f"{twist}. Twist-capable: "
            f"{sorted(k for k, v in ENERGY_REGISTRY.items() if getattr(v, 'has_twist', False))}."
        )
    cls_dim = getattr(cls, "dim", dim)
    if cls_dim is not None and cls_dim != dim:
        raise ValueError(
            f"energy.type={type_name!r} is written for dim={cls.dim}, but the "
            f"environment is {dim}D. Pick a {dim}D energy (registered: "
            f"{sorted(k for k, v in ENERGY_REGISTRY.items() if getattr(v, 'dim', 2) in (dim, None))})."
        )

    params = dict(cfg.get("params") or {})
    return cls(EI=EI, EA=EA, deltaL=deltaL, nv=nv, dim=dim,
               poisson=poisson, **params)
