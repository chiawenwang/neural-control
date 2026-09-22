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

    A twist DOF per edge would require material frames and parallel transport;
    it is deliberately not modelled here.
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
# Factory
############################

def build_energy(energy_cfg, *, EI: float, EA: float, deltaL: float, nv: int,
                 dim: int = 2) -> RodEnergy:
    """Build the energy model selected by the ``energy:`` block of the config.

    ``energy_cfg`` may be None or a dict like::

        energy:
          custom: false          # false -> the built-in DER for this `dim`
          type: der              # only consulted when custom is true
          params: {}             # extra kwargs forwarded to the subclass

    ``dim`` selects the default (``der`` in 2D, ``der3d`` in 3D) and is checked
    against the registered class, so a 2D energy cannot be used with the 3D
    environment.

    Defaults to :class:`DERElasticEnergy`, so existing configs keep working.
    """
    cfg = dict(energy_cfg or {})
    default_cls = DERElasticEnergy if dim == 2 else DERElasticEnergy3D
    if not cfg.get("custom", False):
        return default_cls(EI=EI, EA=EA, deltaL=deltaL)

    type_name = cfg.get("type", "der" if dim == 2 else "der3d")
    if type_name not in ENERGY_REGISTRY:
        raise KeyError(
            f"Unknown energy type {type_name!r}. "
            f"Registered: {sorted(ENERGY_REGISTRY)}. "
            f"Use @register_energy('{type_name}') on your subclass."
        )

    cls = ENERGY_REGISTRY[type_name]
    cls_dim = getattr(cls, "dim", dim)
    if cls_dim is not None and cls_dim != dim:
        raise ValueError(
            f"energy.type={type_name!r} is written for dim={cls.dim}, but the "
            f"environment is {dim}D. Pick a {dim}D energy (registered: "
            f"{sorted(k for k, v in ENERGY_REGISTRY.items() if getattr(v, 'dim', 2) in (dim, None))})."
        )

    params = dict(cfg.get("params") or {})
    return cls(EI=EI, EA=EA, deltaL=deltaL, nv=nv, dim=dim, **params)
