"""Thin adapters over ``src/``: build environments, constrained solves, energies.

The figures never change the simulator. Everything goes through the public
pieces of ``src/``: ``SimulatorEnv_2D`` / ``SimulatorEnv_3D`` for the rod,
:func:`newton.solve_equilibrium` for every equilibrium, and the env's own
energy model. What this module adds:

* **constrained solves**: pinning one more DOF (the midpoint's y) is just a
  smaller ``free_index`` for ``solve_equilibrium``; the residual left on the
  pinned DOF is the reaction ``dE/ds``.
* **stability**: Newton returns *any* root, saddles included (see the
  quasi-static-branch caveat in the README). Every accepted state is checked
  for a positive definite Hessian over the DOFs it was solved for and, if it
  is not, kicked along the unstable mode and re-solved (the same remedy as
  ``SimulatorEnv_2D._stabilize``).
* **a fallback across a snap**: past a fold the warm start has no nearby
  root and plain Newton can stall (it wanders on the indefinite residual).
  Then the energy itself is minimized from the warm start: Newton steps with
  the Hessian shifted to be positive definite and an Armijo backtracking on
  the total energy. Every step lowers the energy, so it ends in a valley
  rather than on a saddle, and the Newton step along the soft mode crosses
  the flat "ghost" of the fold quickly. The result is handed back to
  ``solve_equilibrium``, which confirms it (and returns the Jacobian).
* **energies**: elastic (the env's model), contact (the penalty energy whose
  force ``contact.CircleObstacle`` implements; checked against it by finite
  differences) and gravity (``-W . q``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from paper_figures.common.paths import use_src

use_src()

from config import load_config                          # noqa: E402  (src/)
from contact import CircleObstacle, get_contact_forces  # noqa: E402
from env_factory import resolve_material                # noqa: E402
from newton import solve_equilibrium                    # noqa: E402
from SimulatorEnv_2D import SimulatorEnv_2D             # noqa: E402


# ---------------------------------------------------------------- materials

def material_params(name: str) -> dict:
    """A ``materials:`` preset of configs/default.yaml, resolved by src/env_factory."""
    base = load_config(None)
    return resolve_material({"rod": {"material": name},
                             "materials": base.get("materials")})


# ------------------------------------------------------------------ contact

def circle_contact_energy(q, obstacles, dim=2) -> float:
    """``sum_nodes 0.5 k max(0, r - |x - c|)^2`` -- the energy whose negative
    gradient ``CircleObstacle.node_contact`` returns."""
    x = np.asarray(q, dtype=float)[: dim * (len(q) // dim)].reshape(-1, dim)
    E = 0.0
    for ob in obstacles:
        c = np.array([ob.cx, ob.cy] if dim == 2 else [ob.cx, ob.cy, ob.cz])
        pen = np.maximum(0.0, ob.radius - np.linalg.norm(x - c, axis=1))
        E += 0.5 * ob.k_contact * float(pen @ pen)
    return E


def check_contact_energy(obstacles, q, h=1e-7) -> float:
    """Max relative error of ``-dE_c/dq`` (central differences) against the
    forces ``src/contact.py`` applies. Needs some penetrating nodes in ``q``."""
    q = np.asarray(q, dtype=float)
    F = get_contact_forces(q, obstacles, dim=2, forces_only=True)
    g = np.zeros_like(q)
    for i in range(q.size):
        e = np.zeros_like(q)
        e[i] = h
        g[i] = (circle_contact_energy(q + e, obstacles)
                - circle_contact_energy(q - e, obstacles)) / (2 * h)
    scale = max(float(np.abs(F).max()), 1e-30)
    return float(np.abs(-g - F).max() / scale)


# -------------------------------------------------------------------- solves

@dataclass
class Solve:
    """One accepted (or failed) equilibrium, with what it cost."""
    q: np.ndarray
    ok: bool
    J: np.ndarray                  # Hessian of the total potential (solve_equilibrium's J)
    iterations: int                # Newton iterations over every attempt
    min_eig: float                 # lowest eigenvalue over the solved-for DOFs
    attempts: list = field(default_factory=list)   # (kind, iterations, converged)

    @property
    def stable(self) -> bool:
        return self.ok and self.min_eig >= 0.0


def lowest_mode(J, free):
    """Lowest eigenpair of the Hessian block ``J[free, free]``, sign-fixed so the
    largest component is positive (deterministic kicks)."""
    H = J[np.ix_(free, free)]
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    v = V[:, 0]
    v = v * np.sign(v[np.argmax(np.abs(v))])
    return float(w[0]), v, float(np.abs(np.diag(H)).max())


class PlanarArch:
    """The clamped-clamped strip of Fig. 1(a) on ``SimulatorEnv_2D``.

    Both ends clamped (``grasp_mode="both"``) at a distance ``(1 - eps) L``,
    the left one fixed and horizontal, the right one rotated by ``theta_r``::

        control = (x_l, y_l, th_l, x_r, y_r, th_r) = (0, 0, 0, (1 - eps) L, 0, theta_r)

    ``s`` is the midpoint node's y. Passing ``s`` to the solves pins that DOF
    (it leaves ``free_index``), which is how the constrained landscape
    ``E(s; lambda)`` is computed.
    """

    def __init__(self, *, material, nv, length, eps, gravity, max_iter,
                 obstacles=(), kick_amp=1e-3, descent_max_iter=500):
        mat = material_params(material)
        mat.pop("poisson")
        self.env = SimulatorEnv_2D(nv=nv, rod_length=length, gravity=tuple(gravity),
                                   grasp_mode="both", max_newton_iter=max_iter, **mat)
        env = self.env
        for ob in obstacles:
            env.add_obstacle(ob)
        self.obstacles = env.obstacles
        self.nv, self.L, self.eps = int(nv), float(length), float(eps)
        self.ell = (1.0 - self.eps) * self.L
        self.mid = env.mid_node
        self.pin = 2 * self.mid + 1
        self.free = env.free_index
        self.free_pinned = self.free[self.free != self.pin]
        self.pin_pos = int(np.where(self.free == self.pin)[0][0])
        self.E_unit = env.EI / self.L          # energy scale  EI / L
        self.F_unit = env.EI / self.L ** 2     # force scale, so dE/ds is per (EI/L)/L
        self.tol = env.tol
        self.max_iter = int(max_iter)
        self.kick_amp = float(kick_amp)
        self.descent_max_iter = int(descent_max_iter)

    # ---------------------------------------------------------------- basics

    def control(self, theta_r):
        return np.array([0.0, 0.0, 0.0, self.ell, 0.0, float(theta_r)])

    def apply_boundary(self, q, theta_r, s=None):
        q = np.array(q, dtype=float, copy=True)
        cn = self.env._control_node_from_xytheta(self.control(theta_r))
        q[self.env.fixed_index] = cn[self.env.boundary_rows]
        if s is not None:
            q[self.pin] = float(s)
        return q

    def free_for(self, s):
        return self.free if s is None else self.free_pinned

    def nodes(self, q):
        return np.asarray(q, dtype=float).reshape(-1, 2)

    def s_of(self, q) -> float:
        return float(q[self.pin])

    def initial_guess(self, sign=+1, n_fine=4001):
        """Clamped first-mode arch ``y = A sin^2(pi x / ell)`` with ``A`` chosen so
        its arc length is ``L``, nodes at equal arc length; ``sign`` = +1 up."""
        xs = np.linspace(0.0, self.ell, n_fine)
        prof = np.sin(np.pi * xs / self.ell) ** 2

        def arclen(A):
            return float(np.sum(np.hypot(np.diff(xs), np.diff(A * prof))))

        lo, hi = 0.0, self.L
        for _ in range(80):
            A = 0.5 * (lo + hi)
            lo, hi = (A, hi) if arclen(A) < self.L else (lo, A)
        y = sign * A * prof
        sarc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(y)))])
        st = np.linspace(0.0, sarc[-1], self.nv)
        q = np.empty(2 * self.nv)
        q[0::2] = np.interp(st, sarc, xs)
        q[1::2] = np.interp(st, sarc, y)
        return q, float(A)

    # --------------------------------------------------------------- energies

    def elastic_energy(self, q) -> float:
        return float(self.env.energy.energy(q))

    def contact_energy(self, q) -> float:
        return circle_contact_energy(q, self.obstacles) if self.obstacles else 0.0

    def total_energy(self, q) -> float:
        """Elastic + contact + gravity potential ``-W . q``, in J."""
        return (self.elastic_energy(q) + self.contact_energy(q)
                - float(self.env.W @ np.asarray(q, dtype=float)))

    def residual(self, q):
        """``f = dE_total/dq`` on every DOF (the reaction on the clamped and
        pinned ones, zero on the solved-for ones at equilibrium)."""
        F_el = self.env.energy.force(q)
        Fc = (get_contact_forces(q, self.obstacles, dim=2, forces_only=True)
              if self.obstacles else 0.0)
        return -F_el - Fc - self.env.W

    def schur_curvature(self, J) -> float:
        """``d2E/ds2`` of the constrained landscape at an unconstrained
        equilibrium: ``1 / [H_ff^{-1}]_ss`` (it vanishes at a fold whose mode
        moves the midpoint)."""
        H = J[np.ix_(self.free, self.free)]
        e = np.zeros(len(self.free))
        e[self.pin_pos] = 1.0
        return float(1.0 / np.linalg.solve(H, e)[self.pin_pos])

    # ----------------------------------------------------------------- solves

    def newton(self, q0, theta_r, s=None, max_iter=None):
        """One call of ``solve_equilibrium`` from ``q0`` with the boundary applied."""
        q = self.apply_boundary(q0, theta_r, s)
        q, flag, J, info = solve_equilibrium(
            q, energy=self.env.energy, W=self.env.W, free_index=self.free_for(s),
            dim=2, tol=self.tol, obstacles=self.obstacles,
            max_iter=int(max_iter or self.max_iter))
        return q, flag > 0, J, int(info["iterations"])

    def gradient_hessian(self, q):
        """``(dE_total/dq, d2E_total/dq2)`` over every DOF."""
        F_el, J_el = self.env.energy.force_and_jacobian(q)
        if self.obstacles:
            Fc, Jc = get_contact_forces(q, self.obstacles, dim=2)
        else:
            Fc, Jc = 0.0, 0.0
        return -F_el - Fc - self.env.W, -J_el - Jc

    def descend(self, q0, theta_r, s=None):
        """Minimize the total energy from ``q0`` (shifted-Hessian Newton, Armijo
        on E), then confirm with ``solve_equilibrium``. Returns
        ``(q, ok, J, iterations)``."""
        q = self.apply_boundary(q0, theta_r, s)
        free = self.free_for(s)
        E = self.total_energy(q)
        its = 0
        for its in range(1, self.descent_max_iter + 1):
            g, H = self.gradient_hessian(q)
            gf = g[free]
            if np.linalg.norm(gf) < self.tol:
                break
            Hf = 0.5 * (H[np.ix_(free, free)] + H[np.ix_(free, free)].T)
            w_min = float(np.linalg.eigvalsh(Hf)[0])
            floor = 1e-8 * float(np.abs(np.diag(Hf)).max())
            mu = 0.0 if w_min > floor else 10.0 * floor - w_min
            d = -np.linalg.solve(Hf + mu * np.eye(len(free)), gf)
            slope = float(gf @ d)
            alpha = 1.0
            for _ in range(40):
                q_try = q.copy()
                q_try[free] += alpha * d
                E_try = self.total_energy(q_try)
                if E_try <= E + 1e-4 * alpha * slope:
                    break
                alpha *= 0.5
            else:
                break                      # no descent possible: stationary to precision
            q, E = q_try, E_try
        q, ok, J, n = self.newton(q, theta_r, s)
        return q, ok, J, its + n

    def solve(self, q0, theta_r, s=None, *, max_iter=None, kicks=6,
              fallback=True) -> Solve:
        """Equilibrium near ``q0`` that is *stable* over the solved-for DOFs.

        Warm-started Newton first; if it fails, energy descent from the same
        warm start (:meth:`descend`); then up to ``kicks`` nudges along an
        unstable mode, each followed by a new solve.
        """
        free = self.free_for(s)
        attempts = []
        q, ok, J, n = self.newton(q0, theta_r, s, max_iter)
        attempts.append(("newton", n, ok))
        if not ok and fallback:
            q, ok, J, n = self.descend(q0, theta_r, s)
            attempts.append(("descent", n, ok))
        if not ok:
            return Solve(q, False, J, sum(a[1] for a in attempts), float("nan"), attempts)
        w0, v, scale = lowest_mode(J, free)
        for _ in range(kicks):
            if w0 >= -1e-9 * scale:
                break
            q_try = q.copy()
            q_try[free] += self.kick_amp * v / np.abs(v).max()
            # downhill from the kicked state; plain Newton tends to fall back
            # onto the saddle it was kicked off
            if fallback:
                q2, ok2, J2, n = self.descend(q_try, theta_r, s)
                attempts.append(("kick+descent", n, ok2))
            else:
                q2, ok2, J2, n = self.newton(q_try, theta_r, s, max_iter)
                attempts.append(("kick", n, ok2))
            if not ok2:
                break
            q, J = q2, J2
            w0, v, scale = lowest_mode(J, free)
        return Solve(q, True, J, sum(a[1] for a in attempts), w0, attempts)


def make_circle(center, radius, k_contact) -> CircleObstacle:
    return CircleObstacle(cx=float(center[0]), cy=float(center[1]),
                          radius=float(radius), k_contact=float(k_contact))


# ------------------------------------------------------- generic env stepping

def stable_env_step(env, control, *, kicks=6, amp=1e-3):
    """``env.step(control)`` followed by stability kicks (any 2D/3D env).

    ``SimulatorEnv_3D`` has no stability check, and a compressed rod can sit on
    an unstable straight branch (see the README). Returns ``(info, n_kicks,
    min_eig)``; ``env.q`` holds the accepted state.
    """
    _, _, info = env.step(control)
    if not info["converged"]:
        return info, 0, float("nan")
    J, free = info["J"], env.free_index
    w0, v, scale = lowest_mode(J, free)
    n_kicks = 0
    for _ in range(kicks):
        if w0 >= -1e-9 * scale:
            break
        q_try = env.q.copy()
        q_try[free] += amp * v / np.abs(v).max()
        q_new, flag, J_new, _ = solve_equilibrium(
            q_try, energy=env.energy, W=env.W, free_index=free, dim=env.dim,
            tol=env.tol, obstacles=env.obstacles, n_pos=getattr(env, "n_pos", None),
            max_iter=env.max_newton_iter)
        if flag < 0:
            break
        env.q, J = q_new, J_new
        n_kicks += 1
        w0, v, scale = lowest_mode(J, free)
    info["q"] = env.q.copy()
    info["J"] = J
    return info, n_kicks, w0
