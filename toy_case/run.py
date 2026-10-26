# -*- coding: utf-8 -*-
"""Toy DER experiments.  Usage:

    python run.py <experiment> [--set key.sub=value ...] [--config config.yaml]

experiments: gradcheck | cantilever | twist | buckle | swing | all
Figures go to toy_case/outputs/.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import time
import warnings

import torch
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import der  # noqa: E402
from der import Rod  # noqa: E402
from solver import dynamic_step, static_solve  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")


############################
# Config
############################

def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def set_key(cfg, dotted, raw):
    *path, last = dotted.split(".")
    d = cfg
    for p in path:
        d = d.setdefault(p, {})
    d[last] = yaml.safe_load(raw)


def load_cfg(path, exp, overrides):
    with open(path) as f:
        full = yaml.safe_load(f)
    cfg = deep_merge(full["base"], full["experiments"].get(exp, {}))
    for o in overrides:
        k, v = o.split("=", 1)
        set_key(cfg, k, v)
    return cfg


def solve(rod, z, cfg):
    s = cfg["solver"]
    z, info = static_solve(rod, z, tol=float(s["tol"]), max_iter=int(s["max_iter"]),
                           verbose=bool(s["verbose"]))
    if not info["converged"]:
        print(f"  [warn] Newton did not converge: {info}")
    return z, info


def print_rod(rod):
    B = rod.B.diagonal().tolist()
    print(f"  N={rod.N} nodes, L={rod.L} m | k_s=EA={rod.ks:.4g} N, "
          f"B=diag({B[0]:.4g}, {B[1]:.4g}) N·m², β=GJ={rod.beta:.4g} N·m²")


def axes3d(fig, pos=111):
    ax = fig.add_subplot(pos, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return ax


def equal_3d(ax, pts):
    c = (pts.max(0) + pts.min(0)) / 2
    r = (pts.max(0) - pts.min(0)).max() / 2 + 1e-9
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def draw_frames(ax, x, m1, every=3, scale=0.05):
    mid = 0.5 * (x[1:] + x[:-1])
    for j in range(0, len(mid), every):
        p, d = mid[j], m1[j] * scale
        ax.plot([p[0], p[0] + d[0]], [p[1], p[1] + d[1]], [p[2], p[2] + d[2]], "r-", lw=1)


############################
# Experiments
############################

def exp_gradcheck(cfg):
    """1) autograd force vs central differences on the full BC'd energy.
       2) autograd ∂E/∂θ vs paper Eq. (7).
       3) autograd gradient of the Bishop-frame rotation vs paper Eq. (9)-(10)."""
    torch.manual_seed(0)
    rod = Rod(cfg)
    print_rod(rod)
    z = rod.z_rest() + 0.02 * torch.randn(len(rod.z_rest()))
    f = rod.elastic_energy
    g = torch.func.grad(f)(z)
    h = 1e-6
    fd = torch.stack([(f(z + h * ei) - f(z - h * ei)) / (2 * h) for ei in torch.eye(len(z))])
    print(f"  [1] |∇E_autograd - ∇E_FD| / |∇E| = {(g - fd).norm() / g.norm():.2e}")

    # [2] Eq. (7): ∂E/∂θ^j = Σ_{i∈{j,j+1}} (1/l̄_i)(ω_i^j)ᵀ J B (ω_i^j - ω̄_i^j) + 2β(m_j/l̄_j - m_{j+1}/l̄_{j+1})
    x, _ = rod.unpack(z)
    theta = 0.3 * torch.randn(rod.N - 1)
    e = der.edges(x)
    t = e / e.norm(dim=-1, keepdim=True)
    u, v = der.bishop_frame(t, rod.u0_bar)
    kb = der.curvature_binormal(e)

    def E_theta(th):
        m1, m2 = der.material_frame(u, v, th)
        wp, wn = der.material_curvatures(kb, m1, m2)
        return (der.bend_energy(wp, wn, rod.w_bar_prev, rod.w_bar_next, rod.l_bar, rod.B)
                + der.twist_energy(th, rod.ref_twist_bar, rod.l_bar, rod.beta))

    g_auto = torch.func.grad(E_theta)(theta)
    m1, m2 = der.material_frame(u, v, theta)
    wp, wn = der.material_curvatures(kb, m1, m2)
    J = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    JB = J @ rod.B
    dW_next = ((wn @ JB.T) * (wn - rod.w_bar_next)).sum(-1) / rod.l_bar   # vertex i, edge i   (ω_i^i)
    dW_prev = ((wp @ JB.T) * (wp - rod.w_bar_prev)).sum(-1) / rod.l_bar   # vertex i, edge i-1 (ω_i^{i-1})
    mt = (theta[1:] - theta[:-1]) / rod.l_bar
    g_paper = torch.zeros(rod.N - 1)
    g_paper[:-1] += dW_prev - 2 * rod.beta * mt        # edge j = i-1: vertex i is to its right
    g_paper[1:] += dW_next + 2 * rod.beta * mt         # edge j = i  : vertex i is to its left
    print(f"  [2] Eq.(7)  |∂E/∂θ autograd - paper| / |·| = {(g_auto - g_paper).norm() / g_auto.norm():.2e}")

    # [3] Eq. (9)-(10): rotation Ψ_j of the Bishop frame at edge j under a centerline variation,
    # measured against time-parallel transport of the old frame.  Start is left free (u^0 is
    # time-transported, as in the paper's diagram).
    j = rod.N - 2
    x0 = x.detach()
    e0 = der.edges(x0)
    t0 = e0 / e0.norm(dim=-1, keepdim=True)
    u_old, _ = der.bishop_frame(t0, rod.u0_bar - der.dot(rod.u0_bar, t0[0]) * t0[0])
    u_old = u_old / u_old.norm(dim=-1, keepdim=True)

    def psi(xf):
        xx = xf.reshape(-1, 3)
        ee = der.edges(xx)
        tt = ee / ee.norm(dim=-1, keepdim=True)
        uu, _ = der.bishop_frame(tt, der.parallel_transport(u_old[0], t0[0], tt[0]))
        ref = der.parallel_transport(u_old[j], t0[j], tt[j])
        return der.signed_angle(ref, uu[j], tt[j])

    g_auto = torch.func.grad(psi)(x0.reshape(-1)).reshape(-1, 3)
    kb0 = der.curvature_binormal(e0)
    n_e = e0.norm(dim=-1)
    g_paper = torch.zeros_like(x0)
    for i in range(1, j + 1):                 # Ψ_j = Σ_{i=1}^{j} ψ_i,  Eq. (10)
        gm = kb0[i - 1] / (2 * n_e[i - 1])    # ∇_{i-1} ψ_i
        gp = -kb0[i - 1] / (2 * n_e[i])       # ∇_{i+1} ψ_i
        g_paper[i - 1] += gm
        g_paper[i + 1] += gp
        g_paper[i] += -(gm + gp)
    rel = lambda a, b: ((a - b).norm() / a.norm()).item()
    sgn = "+" if rel(g_auto, g_paper) < rel(g_auto, -g_paper) else "-"
    print(f"  [3] Eq.(9)-(10)  |∇Ψ autograd - ({sgn}) paper| / |·| = "
          f"{min(rel(g_auto, g_paper), rel(g_auto, -g_paper)):.2e}   "
          f"(sign '{sgn}': u^j rotates by {sgn}Ψ_j about t^j)")


def exp_cantilever(cfg):
    """Clamped-free rod sagging under its weight vs. Euler-Bernoulli δ = q L_eff⁴ / (8 EI).

    DER clamps the whole first edge e^0, so the beam is effectively built in at
    the midpoint of e^0: L_eff = L - |ē^0|/2.  The O(1/N) gap to the nominal-L
    formula is that, not discretisation error of the energy.
    """
    rows, shapes = [], []
    for N in cfg["mesh"]:
        c = copy.deepcopy(cfg)
        c["rod"]["n_nodes"] = N
        rod = Rod(c)
        t0 = time.time()
        z, info = solve(rod, rod.z_rest(), c)
        x, _ = rod.unpack(z)
        q = float(rod.mass.sum() / rod.L) * abs(float(rod.gravity[2]))
        L_eff = rod.L - float(rod.e_bar_len[0]) / 2
        d_sim = -float(x[-1, 2])
        d_nom = q * rod.L ** 4 / (8 * rod.EI)
        d_eff = q * L_eff ** 4 / (8 * rod.EI)
        rows.append((N, d_sim, d_nom, d_eff, info["iters"], time.time() - t0))
        shapes.append((N, x.detach().numpy()))
    print_rod(rod)
    print(f"  {'N':>4} {'δ_sim':>11} {'δ_EB(L)':>11} {'δ_EB(L_eff)':>12} {'sim/EB(L_eff)':>14} {'iters':>5} {'time':>6}")
    for N, ds, dn, de, it, tm in rows:
        print(f"  {N:4d} {ds:11.5e} {dn:11.5e} {de:12.5e} {ds / de:14.5f} {it:5d} {tm:5.1f}s")
    print(f"  δ/L = {rows[-1][1] / cfg['rod']['length']:.3f}  (EB is linear theory: expect agreement only for δ/L << 1)")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for N, xs in shapes:
        ax.plot(xs[:, 0], xs[:, 2], ".-", ms=3, label=f"N={N}")
    EI, L = rod.EI, rod.L
    xx = torch.linspace(0, L, 100)
    ax.plot(xx, -q * xx ** 2 * (6 * L ** 2 - 4 * L * xx + xx ** 2) / (24 * EI), "k--", lw=1, label="Euler-Bernoulli")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.legend()
    ax.set_title("Cantilever under gravity")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cantilever.png"), dpi=130)


def exp_twist(cfg):
    """Straight isotropic rod, both ends clamped, end rotated by Φ, no gravity.

    Paper Eq. (5): twist is uniform, m_i / l̄_i = Φ / (2L) with 2L = Σ l̄_i.
    Paper Eq. (6): E = β Φ² / (2L) (bending stays zero, the rod stays straight).
    """
    phi = float(cfg["phi"])
    rod = Rod(cfg)
    print_rod(rod)
    z = rod.z_rest()
    n_load = max(1, int(math.ceil(abs(phi) / 0.5)))        # load in ≤ 0.5 rad steps
    for k in range(1, n_load + 1):
        z, info = solve(rod, rod.apply_end_twist(z, phi * k / n_load), cfg)
    x, th = rod.unpack(z)
    terms = {k: float(v) for k, v in rod.energy_terms(x, th).items()}
    theta = rod.full_theta(z).detach()
    m_over_l = (theta[1:] - theta[:-1]) / rod.l_bar
    twoL = float(rod.l_bar.sum())
    print(f"  Φ = {phi:.4f} rad,  θ^n - θ^0 = {float(theta[-1] - theta[0]):.6f}")
    print(f"  m_i/l̄_i : mean {float(m_over_l.mean()):.6e}, spread {float(m_over_l.max() - m_over_l.min()):.2e}"
          f"   Eq.(5) Φ/(2L) = {phi / twoL:.6e}")
    print(f"  E_twist = {terms['twist']:.6e}   Eq.(6) βΦ²/(2L) = {rod.beta * phi ** 2 / twoL:.6e}")
    simple = cfg["rod"].get("rest_curvature", 0.0) == 0 and cfg["rod"]["radius_a"] == cfg["rod"]["radius_b"]
    print(f"  E_bend = {terms['bend']:.2e}, E_stretch = {terms['stretch']:.2e}"
          + ("  (should be ~0)" if simple else "  (curved/anisotropic rod: Eq. 5-6 do not apply, cf. Fig. 5)"))
    print(f"  end torque  ∂E/∂Φ = βΦ/L = {rod.beta * phi / (twoL / 2):.6e} N·m")

    fig = plt.figure(figsize=(10, 3.5))
    ax = axes3d(fig, 121)
    xs = x.detach().numpy()
    _, _, _, _, _, m1, _ = rod.frames(x, th)
    ax.plot(*xs.T, "k.-")
    draw_frames(ax, xs, m1.detach().numpy(), every=2)
    equal_3d(ax, xs)
    ax.set_title(f"material frame m1 (Φ={phi:.2f})")
    ax2 = fig.add_subplot(122)
    ax2.plot(theta.numpy(), "o-")
    ax2.set_xlabel("edge j")
    ax2.set_ylabel("θ^j [rad]")
    ax2.set_title("θ is linear in j  (uniform twist, Eq. 5)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "twist.png"), dpi=130)


def straight_branch_onset(rod, phi_max, tol=1e-4):
    """Bisect for the end rotation at which the straight, uniformly twisted rod
    stops being a minimum (smallest eigenvalue of ∂²E/∂z² crosses 0).

    That state is an exact equilibrium for every Φ (Eq. 5), so no solve is
    needed - only the Hessian.  Valid when end_shift is purely axial.
    """
    stretch = 1.0 + float(rod.end_shift[0]) / rod.L
    x = rod.x_bar * torch.tensor([stretch, 1.0, 1.0])
    w = torch.cat([torch.zeros(1), torch.cumsum(rod.l_bar, 0)]) / rod.l_bar.sum()

    def lam(phi):
        rod.set_bc_targets(end_twist=phi)
        z = rod.pack(x, phi * w)
        return float(torch.linalg.eigvalsh(torch.func.hessian(rod.elastic_energy)(z))[0])

    if lam(phi_max) > 0:
        return None
    lo, hi = 0.0, phi_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if lam(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def exp_buckle(cfg):
    """Twist buckling of a clamped-clamped rod, then twist -> writhe.

    (a) Linear stability of the straight, uniformly twisted rod: Φ_c vs. Greenhill.
    (b) Phase 1: twist the ends to phi_max at fixed end distance (past Φ_c the
        rod only bulges slightly - a fixed end distance plus stiff k_s blocks coiling).
        Phase 2: bring the ends together by `slack`; twist energy converts into
        bending (writhe) and the rod coils into a 3D loop.  This exchange is
        driven by the holonomy term (§6): θ^n of the clamped end depends on x.
    """
    rod = Rod(cfg)
    print_rod(rod)
    phi_c = straight_branch_onset(Rod(cfg), float(cfg["phi_max"]))
    if phi_c is None:
        print("  straight rod stays stable for Φ ≤ phi_max")
    else:
        print(f"  straight rod loses stability at Φ_c = {phi_c:.4f} rad  ->  M·L/EI = βΦ_c/EI = "
              f"{rod.beta * phi_c / rod.EI:.4f}   (Greenhill, clamped ends: 2.861π = {2.861 * math.pi:.4f})")
    z = rod.z_rest()
    nx = rod.n_xfree
    s = rod.x_bar[:, 0] / rod.L
    bump = torch.stack([0 * s, torch.sin(math.pi * s), 0.3 * torch.sin(2 * math.pi * s)], -1).reshape(-1)
    kick = torch.zeros_like(z)
    kick[:nx] = float(cfg["perturb"]) * bump[rod.x_free]

    phi_max, slack = float(cfg["phi_max"]), torch.tensor([float(v) for v in cfg["slack"]])
    n1, n2 = int(cfg["phi_steps"]), int(cfg["slack_steps"])
    loads = [(phi_max * k / n1, torch.zeros(3)) for k in range(1, n1 + 1)] + \
            [(phi_max, slack * k / n2) for k in range(1, n2 + 1)]
    hist, snaps = [], []
    for k, (phi, shift) in enumerate(loads):
        z = rod.apply_end_twist(z, phi)
        rod.set_bc_targets(end_shift=shift)
        z, info = solve(rod, z + kick, cfg)
        x, th = rod.unpack(z)
        terms = {kk: float(v) for kk, v in rod.energy_terms(x, th).items()}
        hist.append((terms["twist"], terms["bend"], terms["stretch"]))
        if k == n1 - 1 or (k >= n1 and (k - n1 + 1) % max(1, n2 // 4) == 0):
            snaps.append((f"Φ={phi:.1f}, slack={-float(shift[0]):.2f}", x.detach().numpy(),
                          rod.frames(x, th)[5].detach().numpy()))
        if k % 5 == 4 or k == len(loads) - 1:
            print(f"  [{'twist' if k < n1 else 'slack'}] Φ={phi:6.2f} shift={float(shift[0]):+.3f}  "
                  f"E_twist={terms['twist']:.3e}  E_bend={terms['bend']:.3e}  "
                  f"E_stretch={terms['stretch']:.2e}  iters={info['iters']}")
    h = torch.tensor(hist)

    fig = plt.figure(figsize=(12, 4.8))
    ax = axes3d(fig, 121)
    allpts = torch.cat([torch.as_tensor(sn[1]) for sn in snaps]).numpy()
    for label, xs, _ in snaps:
        ax.plot(*xs.T, ".-", ms=2, label=label)
    draw_frames(ax, snaps[-1][1], snaps[-1][2], every=1, scale=0.03)
    equal_3d(ax, allpts)
    ax.legend(fontsize=7)
    ax.set_title("twist -> writhe (red: m1 on final shape)")
    ax2 = fig.add_subplot(122)
    step = torch.arange(1, len(loads) + 1)
    for i, name in enumerate(["E_twist", "E_bend", "E_stretch"]):
        ax2.plot(step, h[:, i], label=name)
    ax2.axvline(n1 + 0.5, color="k", ls=":", lw=1)
    if phi_c is not None:
        ax2.axvline(phi_c / phi_max * n1, color="gray", ls="--", lw=1, label="Φ_c")
    ax2.set_xlabel(f"load step  (1-{n1}: twist to Φ={phi_max:g},  {n1 + 1}-{n1 + n2}: slack)")
    ax2.set_ylabel("energy [J]")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "buckle.png"), dpi=130)


def exp_swing(cfg):
    """Horizontal cantilever released from rest under gravity (backward Euler)."""
    rod = Rod(cfg)
    print_rod(rod)
    d = cfg["dynamics"]
    dt, steps, c = float(d["dt"]), int(d["steps"]), float(d["damping"])
    z = rod.z_rest()
    v = torch.zeros(rod.n_xfree)
    m = rod.mass.repeat_interleave(3)[rod.x_free]
    log, snaps = [], []
    t0 = time.time()
    for k in range(steps + 1):
        if k:
            z, v, info = dynamic_step(rod, z, v, dt, c, tol=float(cfg["solver"]["tol"]),
                                      max_iter=int(cfg["solver"]["max_iter"]))
        x, th = rod.unpack(z)
        terms = {kk: float(val) for kk, val in rod.energy_terms(x, th).items()}
        KE = float(0.5 * (m * v * v).sum())
        PE = float(rod.gravity_energy(z))
        log.append((k * dt, KE, PE, terms["stretch"], terms["bend"], terms["twist"], float(x[-1, 2])))
        if k % max(1, steps // 8) == 0:
            snaps.append((k * dt, x.detach().numpy()))
    print(f"  {steps} steps of dt={dt} in {time.time() - t0:.1f}s")
    h = torch.tensor(log)
    tot = h[:, 1:6].sum(1)
    print(f"  tip z: min {float(h[:, 6].min()):.4f} m, final {float(h[-1, 6]):.4f} m")
    print(f"  total energy: start {float(tot[0]):.4e}, end {float(tot[-1]):.4e} J "
          f"(backward Euler is numerically dissipative - smaller dt -> less loss)")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for t, xs in snaps:
        axs[0].plot(xs[:, 0], xs[:, 2], ".-", ms=2, label=f"t={t:.2f}")
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("z [m]")
    axs[0].legend(fontsize=7)
    axs[0].set_title("swing snapshots")
    for i, name in enumerate(["kinetic", "gravity", "stretch", "bend", "twist"]):
        axs[1].plot(h[:, 0], h[:, 1 + i], label=name)
    axs[1].plot(h[:, 0], tot, "k--", label="total")
    axs[1].set_xlabel("t [s]")
    axs[1].set_ylabel("energy [J]")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "swing.png"), dpi=130)


EXPERIMENTS = dict(gradcheck=exp_gradcheck, cantilever=exp_cantilever, twist=exp_twist,
                   buckle=exp_buckle, swing=exp_swing)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=list(EXPERIMENTS) + ["all"])
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config entries, e.g. rod.n_nodes=61 bc.end_twist=2.0")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for name in names:
        print(f"=== {name} ===")
        EXPERIMENTS[name](load_cfg(args.config, name, args.set))


if __name__ == "__main__":
    main()
