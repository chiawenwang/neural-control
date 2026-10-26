# -*- coding: utf-8 -*-
"""Newton solver for statics and implicit-Euler dynamics of a DER rod.

Both problems are written as minimising one scalar objective over the free
DOFs z = [x_free, θ_free]:

  statics   Π(z) = E_elastic(z) + E_gravity(z)
  dynamics  Π(z) = 1/(2h²) (x - x̃)ᵀ M (x - x̃) + E_elastic(z) + E_gravity(z)
                   x̃ = x_n + h v_n         (backward Euler, incremental potential)

θ has no mass, so in the dynamic objective it is simply minimised at every
step: that is the quasistatic material frame of §5, ∂E/∂θ^j = 0 (Eq. 4), and
Algorithm 1 lines 6/11 collapse into one Newton solve.  The paper's Newton +
line search for θ (§5.1) is the θ-block of this same solve.
"""
from __future__ import annotations

import torch
from torch.func import grad, hessian


def newton(obj, z0, tol=1e-9, max_iter=50, verbose=False):
    """Damped Newton with backtracking line search on obj.

    Converged when the step is tiny (|Δz|∞ < tol) - the force residual is
    reported as `res`.  If the Hessian is not positive definite (e.g. at a
    buckling point) a multiple of the identity is added until it is.
    """
    g_fn, H_fn = grad(obj), hessian(obj)
    z = z0.clone()
    f = obj(z)
    for it in range(max_iter):
        g, H = g_fn(z), H_fn(z)
        mu, I = 0.0, torch.eye(len(z))
        scale = H.diagonal().abs().max().item()
        while True:
            L, info = torch.linalg.cholesky_ex(H + mu * I)
            if info == 0:
                break
            mu = max(1e-10 * scale, 10 * mu)
        dz = -torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)
        a = 1.0
        while a > 1e-8:
            f_new = obj(z + a * dz)
            if f_new <= f + 1e-4 * a * (g @ dz):
                break
            a *= 0.5
        z = z + a * dz
        f = f_new
        step = (a * dz).abs().max().item()
        if verbose:
            print(f"    newton {it:2d}  Π={f.item(): .6e}  |g|={g.norm().item():.2e}  step={step:.2e}  α={a:.3g}")
        if step < tol:
            return z, dict(iters=it + 1, res=g_fn(z).norm().item(), converged=True)
    return z, dict(iters=max_iter, res=g_fn(z).norm().item(), converged=False)


def static_solve(rod, z0, tol=1e-9, max_iter=50, verbose=False):
    obj = lambda z: rod.elastic_energy(z) + rod.gravity_energy(z)
    z, info = newton(obj, z0, tol, max_iter, verbose)
    rod.track_holonomy(z)
    return z, info


def dynamic_step(rod, z, v, dt, damping=0.0, tol=1e-9, max_iter=50):
    """One backward-Euler step.  v is the velocity of the free positions.

    `damping` c gives a mass-proportional viscous force -c M v (Rayleigh α-damping),
    i.e. an extra ½ c/h (x - x_n)ᵀ M (x - x_n) in Π.
    """
    m = rod.mass.repeat_interleave(3)[rod.x_free]
    nx = rod.n_xfree
    x_n = z[:nx]
    x_tilde = x_n + dt * v

    def obj(zz):
        dx = zz[:nx] - x_tilde
        inertia = 0.5 / dt ** 2 * (m * dx * dx).sum()
        visc = 0.5 * damping / dt * (m * (zz[:nx] - x_n) ** 2).sum()
        return inertia + visc + rod.elastic_energy(zz) + rod.gravity_energy(zz)

    z_new, info = newton(obj, z, tol, max_iter)
    rod.track_holonomy(z_new)
    v_new = (z_new[:nx] - x_n) / dt
    return z_new, v_new, info
