# -*- coding: utf-8 -*-
"""Minimal Discrete Elastic Rods (DER), following Bergou et al. 2008.

    M. Bergou, M. Wardetzky, S. Robinson, B. Audoly, E. Grinspun,
    "Discrete Elastic Rods", ACM TOG 27(3), 2008.
    (reference/Discrete elastic rods.pdf)

Only the elastic model lives here: geometry, frames, the three energies.
Every function is written as a direct transcription of a formula in the paper;
the section / equation number is given in each docstring.  Forces and
stiffness are NOT hand-derived: torch autograd differentiates the energy, which
automatically produces the holonomy terms of §6 (Eq. 9-10), because the Bishop
frame below is built by explicit parallel transport and is therefore a
differentiable function of the centerline.

Notation (paper §4.2, "Primal vs. dual"):
    vertices  x_0 ... x_{n+1}          (N = n+2 nodes, lower index)
    edges     e^0 ... e^n,  e^j = x_{j+1} - x_j   (n+1 edges, upper index)
    interior vertices i = 1..n carry curvature (κb)_i
    per edge: tangent t^j, Bishop frame (u^j, v^j), angle θ^j,
              material frame (m1^j, m2^j)
Barred quantities (x̄, ē, l̄_i, ω̄) are the undeformed configuration.
"""
from __future__ import annotations

import math

import torch

torch.set_default_dtype(torch.float64)


############################
# Geometry (§4.2)
############################

def cross(a, b):
    return torch.linalg.cross(a, b, dim=-1)


def dot(a, b):
    return (a * b).sum(-1)


def edges(x):
    """e^j = x_{j+1} - x_j  (§4.2 'Discrete framed curves')."""
    return x[1:] - x[:-1]


def curvature_binormal(e):
    """(κb)_i = 2 e^{i-1} × e^i / (|e^{i-1}||e^i| + e^{i-1}·e^i)      Eq. (1)

    Magnitude is 2 tan(φ_i/2), φ_i = turning angle at vertex i.
    Returns shape (n, 3) for the n interior vertices i = 1..n.
    """
    e0, e1 = e[:-1], e[1:]
    n0, n1 = e0.norm(dim=-1), e1.norm(dim=-1)
    return 2.0 * cross(e0, e1) / (n0 * n1 + dot(e0, e1)).unsqueeze(-1)


def parallel_transport(u, t1, t2):
    """Rotate u by P, the rotation about t1×t2 that maps t1 to t2.   §4.2.2

        P t1 = t2,   P (t1 × t2) = t1 × t2

    Rodrigues' formula written without normalising the axis, so it stays
    smooth (and autograd-safe) when t1 = t2, where P = Id.
    """
    b = cross(t1, t2)
    c = dot(t1, t2)
    return u * c + cross(b, u) + b * dot(b, u) / (1.0 + c)


def bishop_frame(t, u0):
    """Space-parallel-transport u0 along the rod:  u^j = P_j u^{j-1}.   §4.2.2

    v^j = t^j × u^j.  Returns u, v of shape (n+1, 3).
    """
    us = [u0]
    for j in range(1, t.shape[0]):
        us.append(parallel_transport(us[-1], t[j - 1], t[j]))
    u = torch.stack(us)
    return u, cross(t, u)


def material_frame(u, v, theta):
    """m1 =  cos θ u + sin θ v,   m2 = -sin θ u + cos θ v.   §4.2.2"""
    c, s = torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1)
    return c * u + s * v, -s * u + c * v


def material_curvatures(kb, m1, m2):
    """ω_i^j = ( (κb)_i · m2^j , -(κb)_i · m1^j ),  j ∈ {i-1, i}.   Eq. (2)

    Returns (w_prev, w_next), each (n, 2): ω_i^{i-1} and ω_i^{i}.
    """
    w_prev = torch.stack([dot(kb, m2[:-1]), -dot(kb, m1[:-1])], dim=-1)
    w_next = torch.stack([dot(kb, m2[1:]), -dot(kb, m1[1:])], dim=-1)
    return w_prev, w_next


def signed_angle(a, b, axis):
    """Angle that rotates a onto b about axis (all ⟂ axis)."""
    return torch.atan2(dot(cross(a, b), axis), dot(a, b))


############################
# Energies
############################

def stretch_energy(e, e_bar_len, ks):
    """E_stretch = ½ Σ_j k_s (|e^j|/|ē^j| - 1)² |ē^j|,   k_s = EA.

    The paper treats the rod as inextensible (§4.1, §8) and notes "it is
    straightforward to drop this assumption by also including a stretching
    term" - this is that term (axial strain energy, integrated per edge).
    """
    eps = e.norm(dim=-1) / e_bar_len - 1.0
    return 0.5 * ks * (eps ** 2 * e_bar_len).sum()


def bend_energy(w_prev, w_next, w_bar_prev, w_bar_next, l_bar, B):
    """E_bend = Σ_i 1/(2 l̄_i) Σ_{j=i-1}^{i} (ω_i^j - ω̄_i^j)ᵀ B (ω_i^j - ω̄_i^j)   Eq. (3)

    l̄_i = |ē^{i-1}| + |ē^i| (twice the Voronoi length of vertex i).
    For B = α Id, ω̄ = 0 this reduces to Σ α (κb)_i² / l̄_i  (§4.2.1).
    """
    dp, dn = w_prev - w_bar_prev, w_next - w_bar_next
    quad = dot(dp @ B, dp) + dot(dn @ B, dn)
    return (quad / (2.0 * l_bar)).sum()


def twist_energy(theta, ref_twist_bar, l_bar, beta):
    """E_twist = Σ_i β (m_i - m̄_i)² / l̄_i,   m_i = θ^i - θ^{i-1}.   §4.2.2

    m_i is the integrated discrete twist at interior vertex i.  m̄_i = 0 for
    the rest state, since the undeformed material frame is chosen to be the
    Bishop frame (§4.1.1).  β = GJ.
    """
    m = theta[1:] - theta[:-1]
    return (beta * (m - ref_twist_bar) ** 2 / l_bar).sum()


############################
# Cross-section -> stiffness
############################

def section_stiffness(youngs, poisson, a, b):
    """Elliptic cross-section with semi-axes a (along m1) and b (along m2).

    ω = (κb·m2, -κb·m1): ω_1 bends about the m1 axis, ω_2 about m2, so
    B = diag(E I_1, E I_2) with I_1 = π a b³/4 (about m1), I_2 = π a³ b/4.
    Circle (a = b = r): α = E π r⁴/4,  β = G π r⁴/2,  k_s = E π r².
    """
    G = youngs / (2.0 * (1.0 + poisson))
    A = math.pi * a * b
    I1 = math.pi * a * b ** 3 / 4.0
    I2 = math.pi * a ** 3 * b / 4.0
    J = math.pi * a ** 3 * b ** 3 / (a ** 2 + b ** 2)
    return dict(A=A, ks=youngs * A, B=torch.diag(torch.tensor([youngs * I1, youngs * I2])),
                beta=G * J, EI=youngs * math.sqrt(I1 * I2), G=G)


############################
# Rod
############################

class Rod:
    """A DER rod: rest state, boundary conditions and the energy E(x, θ).

    DOFs  q = [x_0 ... x_{n+1} (3 each), θ^0 ... θ^n].
    Boundary conditions remove DOFs; the solver only ever sees the free
    vector z, mapped back with  x = x_bc + S_x z_x,  θ = θ_bc + S_θ z_θ.

    Material-frame boundary conditions (§5 'Boundary conditions'):
      * clamped start: x_0, x_1 fixed, θ^0 = 0.  Fixes t^0, so u^0 = ū^0.
      * clamped end  : x_{n+1}, x_n fixed and the *lab-frame* material
        direction m1^n is prescribed (rotated by `end_twist` about t^n).
        θ^n is then not free: since u^n follows from parallel transport along
        the whole (deformed) centerline, θ^n = end_twist - δ, where δ is the
        angle the Bishop vector at the end edge has drifted from its rest
        value - exactly the holonomy Ψ_n of §6 (Eq. 10).  Its x-gradient is
        what produces the twist-bend coupling forces of §7.1.
    """

    def __init__(self, cfg):
        rc, bc, ec = cfg["rod"], cfg["bc"], cfg["energy"]
        self.N = N = int(rc["n_nodes"])
        L = float(rc["length"])
        self.L = L

        # --- rest centerline: straight along x, or an arc of curvature κ̄ in x-y
        s = torch.linspace(0.0, L, N)
        k0 = float(rc.get("rest_curvature", 0.0))
        if abs(k0) < 1e-12:
            x_bar = torch.stack([s, torch.zeros(N), torch.zeros(N)], -1)
        else:
            x_bar = torch.stack([torch.sin(k0 * s) / k0, (1 - torch.cos(k0 * s)) / k0,
                                 torch.zeros(N)], -1)
        self.x_bar = x_bar

        # --- material
        sec = section_stiffness(float(rc["youngs"]), float(rc["poisson"]),
                                float(rc["radius_a"]), float(rc["radius_b"]))
        self.ks = sec["ks"] * float(ec.get("stretch_scale", 1.0)) * bool(ec.get("stretch", True))
        self.B = sec["B"] * float(ec.get("bend_scale", 1.0)) * bool(ec.get("bend", True))
        self.beta = sec["beta"] * float(ec.get("twist_scale", 1.0)) * bool(ec.get("twist", True))
        self.EI, self.EA, self.GJ, self.A = sec["EI"], sec["ks"], sec["beta"], sec["A"]

        # --- rest quantities (barred, precomputed - Algorithm 1 line 1)
        e_bar = edges(x_bar)
        self.e_bar_len = e_bar.norm(dim=-1)
        self.l_bar = self.e_bar_len[:-1] + self.e_bar_len[1:]
        self.t_bar = e_bar / self.e_bar_len.unsqueeze(-1)
        u0 = torch.tensor([0.0, 0.0, 1.0])            # ⟂ t̄^0 for any x-y rest curve
        self.u0_bar = u0 - dot(u0, self.t_bar[0]) * self.t_bar[0]
        self.u0_bar = self.u0_bar / self.u0_bar.norm()
        self.u_bar, v_bar = bishop_frame(self.t_bar, self.u0_bar)
        # rest material frame = Bishop frame (θ̄ = 0)  ->  ω̄ from Eq. (2), m̄ = 0
        kb_bar = curvature_binormal(e_bar)
        self.w_bar_prev, self.w_bar_next = material_curvatures(kb_bar, self.u_bar, v_bar)
        self.ref_twist_bar = torch.zeros(N - 2)

        # --- lumped mass: ρ A × Voronoi length of each vertex
        rho = float(rc["density"])
        vor = torch.zeros(N)
        vor[:-1] += 0.5 * self.e_bar_len
        vor[1:] += 0.5 * self.e_bar_len
        self.mass = rho * self.A * vor
        self.gravity = torch.tensor([float(g) for g in cfg["gravity"]])

        # --- boundary conditions
        self.clamp_start = bc["start"] == "clamp"
        self.end = bc["end"]                          # clamp | pin | free
        self.end_twist = float(bc.get("end_twist", 0.0))
        self.end_shift = torch.tensor([float(v) for v in bc.get("end_shift", [0, 0, 0])])
        fixed_nodes = []
        if self.clamp_start:
            fixed_nodes += [0, 1]
        if self.end == "clamp":
            fixed_nodes += [N - 2, N - 1]
        elif self.end == "pin":
            fixed_nodes += [N - 1]
        self.fixed_nodes = sorted(set(fixed_nodes))
        self.theta_fixed = ([0] if self.clamp_start else []) + ([N - 2] if self.end == "clamp" else [])

        self.x_free = [i for i in range(3 * N) if i // 3 not in self.fixed_nodes]
        self.th_free = [j for j in range(N - 1) if j not in self.theta_fixed]
        self.S_x = torch.eye(3 * N)[:, self.x_free]
        self.S_th = torch.eye(N - 1)[:, self.th_free]
        self.n_xfree = len(self.x_free)
        self.set_bc_targets()
        # end holonomy δ is accumulated step by step so it can wind past ±π:
        # δ = δ_acc + angle(u_ref -> u^n), u_ref = u^n of the last converged state
        self._delta_acc = 0.0
        self._u_end_ref = self.u_bar[-1].clone()

    # ---------- boundary values ----------
    def set_bc_targets(self, end_twist=None, end_shift=None):
        """(Re)set prescribed values; call between load steps."""
        if end_twist is not None:
            self.end_twist = float(end_twist)
        if end_shift is not None:
            self.end_shift = torch.as_tensor(end_shift, dtype=torch.float64)
        x_bc = self.x_bar.clone()
        if self.end in ("clamp", "pin"):
            x_bc[-1] += self.end_shift
            if self.end == "clamp":
                x_bc[-2] += self.end_shift
        self.x_bc = x_bc.reshape(-1) @ (torch.eye(3 * self.N) - self.S_x @ self.S_x.T)

    def apply_end_twist(self, z, end_twist):
        """Set the clamped-end rotation and return a predictor for z.

        The increment Δ is spread over the free θ^j in proportion to arc length
        (the uniform-twist solution of Eq. (5)), instead of landing entirely on
        the last vertex as one large local twist.
        """
        d = float(end_twist) - self.end_twist
        self.set_bc_targets(end_twist=end_twist)
        w = torch.cat([torch.zeros(1), torch.cumsum(self.l_bar, 0)]) / self.l_bar.sum()
        return z + torch.cat([torch.zeros(self.n_xfree), d * w[self.th_free]])

    def unpack(self, z):
        x = (self.x_bc + self.S_x @ z[: self.n_xfree]).reshape(self.N, 3)
        theta = self.S_th @ z[self.n_xfree:]
        return x, theta

    def pack(self, x, theta):
        return torch.cat([x.reshape(-1)[self.x_free], theta[self.th_free]])

    def z_rest(self):
        return self.pack(self.x_bar, torch.zeros(self.N - 1))

    # ---------- frames ----------
    def frames(self, x, theta):
        """Bishop + material frames for configuration (x, θ).

        u^0: clamped start -> ū^0.  Free start -> ū^0 parallel-transported in
        *time* from t̄^0 to t^0 (§4.2.2 'Bishop frame update').
        θ^n of a clamped end is filled in here (see class docstring).
        """
        e = edges(x)
        t = e / e.norm(dim=-1, keepdim=True)
        u0 = self.u0_bar if self.clamp_start else parallel_transport(self.u0_bar, self.t_bar[0], t[0])
        u, v = bishop_frame(t, u0)
        if self.end == "clamp":
            delta = self._delta_acc + signed_angle(self._u_end_ref, u[-1], t[-1])
            theta = torch.cat([theta[:-1], (self.end_twist - delta).reshape(1)])
        m1, m2 = material_frame(u, v, theta)
        return e, t, u, v, theta, m1, m2

    def track_holonomy(self, z):
        """Advance the reference for δ to the converged state z (call after each solve)."""
        if self.end != "clamp":
            return
        with torch.no_grad():
            x, _ = self.unpack(z)
            e = edges(x)
            t = e / e.norm(dim=-1, keepdim=True)
            u0 = self.u0_bar if self.clamp_start else parallel_transport(self.u0_bar, self.t_bar[0], t[0])
            u, _ = bishop_frame(t, u0)
            self._delta_acc += float(signed_angle(self._u_end_ref, u[-1], t[-1]))
            self._u_end_ref = u[-1].clone()

    # ---------- energies ----------
    def energy_terms(self, x, theta):
        e, t, u, v, theta, m1, m2 = self.frames(x, theta)
        kb = curvature_binormal(e)
        w_prev, w_next = material_curvatures(kb, m1, m2)
        return {
            "stretch": stretch_energy(e, self.e_bar_len, self.ks),
            "bend": bend_energy(w_prev, w_next, self.w_bar_prev, self.w_bar_next, self.l_bar, self.B),
            "twist": twist_energy(theta, self.ref_twist_bar, self.l_bar, self.beta),
        }

    def elastic_energy(self, z):
        x, theta = self.unpack(z)
        return sum(self.energy_terms(x, theta).values())

    def gravity_energy(self, z):
        x, _ = self.unpack(z)
        return -(self.mass * (x @ self.gravity)).sum()

    def full_theta(self, z):
        """θ including the BC-determined θ^n (for plotting / diagnostics)."""
        x, theta = self.unpack(z)
        return self.frames(x, theta)[4]
