"""Finite-difference check of the smooth peg contact (contact.PegObstacle).

Run after touching src/contact.py::

    python scripts/check_peg_contact.py

1. dE_c/dq against central differences of E_c, and d2E_c/dq2 against central
   differences of dE_c/dq, on rods placed across a peg: penetrating, at the
   onset (clearance ~ delta) and with a point on the cap beyond the axis end.
   Relative errors are printed and asserted below 1e-5.
2. The Hessian is symmetric.
3. The force/energy profile against the analytic phi of the paper: a single
   point at clearance d has w*phi_delta(d) energy.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from contact import PegObstacle  # noqa: E402


def rod_through(peg, nv, L, offset, tilt=0.15, y0=0.0, rng=None):
    """A slightly bent rod passing the peg at clearance ``offset`` from its surface."""
    s = np.linspace(-0.5 * L, 0.5 * L, nv)
    X = np.zeros((nv, 3))
    X[:, 0] = peg.cx + s
    X[:, 1] = y0 + tilt * s
    # arc that touches the peg from above at s = 0
    X[:, 2] = peg.cz + peg.r_eff + offset + 0.6 * s ** 2
    if rng is not None:
        X += 1e-3 * rng.standard_normal(X.shape)
    return X.reshape(-1)


def check(peg, q, h=1e-7):
    g, H = peg.contact_gradient_hessian(q)
    n = q.size
    g_fd = np.zeros(n)
    H_fd = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        g_fd[i] = (peg.compute_contact_energy(q + e) - peg.compute_contact_energy(q - e)) / (2 * h)
        H_fd[:, i] = (peg.contact_gradient_hessian(q + e, want_hess=False)[0]
                      - peg.contact_gradient_hessian(q - e, want_hess=False)[0]) / (2 * h)
    eg = np.linalg.norm(g - g_fd) / max(np.linalg.norm(g), 1e-300)
    eH = np.linalg.norm(H - H_fd) / max(np.linalg.norm(H), 1e-300)
    asym = np.linalg.norm(H - H.T) / max(np.linalg.norm(H), 1e-300)
    return eg, eH, asym, np.linalg.norm(g)


def main():
    rng = np.random.default_rng(0)
    nv, L = 11, 0.3
    dL = L / (nv - 1)
    worst = 0.0
    for delta in (1e-4, 5e-4):
        peg = PegObstacle(0.1, 0.0, 0.2, 0.04, 0.25, rod_radius=0.003, nv=nv, deltaL=dL,
                          k_contact=1e7, smoothing=delta, samples_per_edge=4)
        cases = {"penetrating (-0.5 mm)": rod_through(peg, nv, L, -5e-4, rng=rng),
                 "onset (+delta)": rod_through(peg, nv, L, delta, rng=rng),
                 "near cap (y past the end)": rod_through(peg, nv, L, -2e-4, y0=0.26, tilt=0.0, rng=rng)}
        for name, q in cases.items():
            eg, eH, asym, gn = check(peg, q, h=1e-7 if delta < 3e-4 else 1e-6)
            rep = peg.report(q)
            print(f"delta={delta:.0e}  {name:28s} |g|={gn:9.3e}  grad err {eg:.2e}  "
                  f"Hess err {eH:.2e}  asym {asym:.1e}  pen {rep['max_penetration'] * 1e3:.3f} mm  "
                  f"force {rep['force']:.3f} N  end_contact {rep['end_contact']}")
            if gn > 0:
                worst = max(worst, eg, eH)
    # analytic profile of one isolated point
    peg = PegObstacle(0.0, 0.0, 0.0, 0.04, 0.25, rod_radius=0.003, nv=2, deltaL=1.0,
                      k_contact=1e7, smoothing=1e-4, samples_per_edge=1)
    for d in (-1e-3, 0.0, 2e-4):
        z = peg.r_eff + d
        q = np.array([0.0, 0.0, z, 5.0, 0.0, z])        # node 0 at clearance d; node 1 far away
        E = peg.compute_contact_energy(q)
        phi = 0.5 * 1e7 * (1e-4 * np.log1p(np.exp(-d / 1e-4))) ** 2
        print(f"single point d={d * 1e3:+.1f} mm: E={E:.6e}  w*phi={0.5 * phi:.6e}")
        assert abs(E - 0.5 * phi) <= 1e-12 + 1e-10 * abs(phi)
    print(f"largest relative FD error: {worst:.2e}")
    assert worst < 1e-5, "peg contact derivatives disagree with finite differences"
    print("OK")


if __name__ == "__main__":
    main()
