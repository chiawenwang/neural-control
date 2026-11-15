"""Validate the two adjoint gradients that train.grad_method selects.

Run this after touching src/implicit_grad_tools.py or the rollout / update in
src/neural_control.py::

    python scripts/check_grad_method.py
    python scripts/check_grad_method.py --refine     # also the step-refinement sweep

For each of 2D, 3D and 3D + twist (the example configs' soft strip, nv = 21,
N = 8 steps, a small float64 policy), with a terminal loss (point-to-point)
and a loss on every step (tracking):

  1. state is the exact gradient: theta.grad against central differences of
     the rollout loss along random unit directions d of the weights, as
     |g . d - fd| / |g| (relative to the gradient, so a direction nearly
     orthogonal to it does not inflate the number).  The Newton tolerance is
     tightened so the solver's slop sits below what is measured.
  2. proxy is the formula it claims to be: theta.grad against
         dL/dr_j = dlam * B_j^T S_j^T (a_j + ... + a_{N-1})
     assembled by hand, with S_j = -J_ff^{-1} J_fb built explicitly (inverse,
     not solve) and contracted with the policy's Jacobians d r_j / d theta.
  3. under state a terminal loss costs one adjoint solve; proxy needs N.
  4. how far proxy is from the finite differences -- reported, not asserted:
     that is the frozen-tangent error, a property of the method.
  5. (--refine) proxy against state as N grows from 25 to 800 with the same
     policy: whether the frozen-tangent error goes away.  Reported only, with
     the final mid node and the number of unstable steps (J_ff not positive
     definite): the step size can change which equilibrium the rollout
     reaches, and rows on different branches are not comparable.

The workspace is off: its projection is straight-through, so the training
gradient deliberately ignores the clamp that finite differences would see.
The adjoint's regularization is off too: ``train.jac_reg`` (1e-6) shifts S by
about jac_reg / lambda_min(J_ff), ~1e-7 relative on this rod, which would hide
everything below it.  How large that shift is gets printed alongside.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from blas_threads import limit_blas_threads
limit_blas_threads()

import argparse

import numpy as np
import torch

import implicit_grad_tools
from config import load_config
from env_factory import (build_env, reset_env, env_dim, env_twist, end_control_dim,
                         initial_control, task_target, dloss_dq_free_node, node_position)
from neural_control import BoundaryVelocityNet, rollout, adjoint_gradient, surrogate_loss

TOL_FD = 1e-6        # state vs central differences, relative (typically ~1e-9)
TOL_FORMULA = 1e-10  # proxy vs its own formula assembled by hand, relative
# The default Newton tolerance (EI / L^2 * 1e-3) leaves each equilibrium
# uncertain by more than a 1e-5 weight perturbation moves it; tighten it so
# the comparison reports the gradient's error, not the solver's.  1e-6 is
# below this soft strip's residual floor: Newton then never converges.
TOL_FACTOR = 1e-5
H_FD = 1e-5          # step along a unit direction of the weights
JAC_REG = 0.0        # see the module docstring
N_STEPS, NV, HIDDEN = 8, 21, 8
CASES = {"2d": (None, 2), "3d": ("configs/example_3d.yaml", 3),
         "3d_twist": ("configs/example_3d_twist.yaml", 3)}
LOSSES = ("terminal", "tracking")
CPU = torch.device("cpu")

results = []


def report(name, err, tol):
    ok = err < tol
    results.append(ok)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:58s} {err:.3e}  (tol {tol:g})")


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:58s} {detail}")


def setup(case, n_steps=N_STEPS):
    path, dim = CASES[case]
    cfg = load_config(None if path is None else _ROOT / path)
    cfg["env"]["dim"] = dim
    cfg["rod"]["nv"] = NV
    cfg["rod"]["total_time"] = n_steps * cfg["rod"]["dt"]
    cfg["workspace"]["enabled"] = False
    env = build_env(cfg)
    reset_env(env, cfg)
    env.tol *= TOL_FACTOR
    k = end_control_dim(env_dim(cfg), env_twist(cfg))
    torch.manual_seed(0)
    net = BoundaryVelocityNet(hidden=HIDDEN, out_dim=k).double()
    u0 = torch.tensor(initial_control(cfg), dtype=torch.float64)
    return cfg, env, env.get_state(), net, u0


def loss_and_sources(env, checkpoint, q_list, target, kind):
    """``(L, dL/dq_free per step)``: ``terminal`` is ``|mid_N - target|^2``;
    ``tracking`` the mean over the steps of ``|mid_k - ref_k|^2`` along the
    straight line from the mid node's start to the target."""
    dim, N = env.dim, len(q_list)
    a = np.zeros((N, len(env.free_index)))
    if kind == "terminal":
        terms = [(N - 1, 1.0, target)]
    else:
        mid0 = node_position(checkpoint["q"], dim, env.mid_node)
        terms = [(i, 1.0 / N, mid0 + (i + 1) / N * (target - mid0)) for i in range(N)]
    L = 0.0
    for i, w, ref in terms:
        d = node_position(q_list[i], dim, env.mid_node) - ref
        L += w * float(d @ d)
        a[i] += w * dloss_dq_free_node(q_list[i], env.free_index, dim, env.mid_node, ref)
    return L, a


def run(env, checkpoint, net, u0, n_steps):
    env.set_state(checkpoint)
    u_list, r_list, q_list, J_list, conv = rollout(env, net, u0, n_steps, 1.0 / n_steps, CPU)
    if not conv.all():
        raise RuntimeError("a Newton solve failed; the check needs a converged rollout")
    return u_list, r_list, q_list, J_list


def theta_grad(env, net, u_list, r_list, J_list, a, dlam, jac_reg, grad_method):
    g = adjoint_gradient(env, u_list, J_list, a, jac_reg, grad_method=grad_method)
    net.zero_grad(set_to_none=True)
    surrogate_loss(u_list, r_list, g, dlam, CPU, grad_method=grad_method).backward(
        retain_graph=True)
    return torch.cat([p.grad.reshape(-1) for p in net.parameters()]).numpy().copy()


def proxy_by_hand(env, net, u_list, J_list, a, dlam, jac_reg):
    """theta.grad of the proxy formula, from explicit S_j and the policy's Jacobians."""
    free, fixed = env.free_index, env.fixed_index
    params = list(net.parameters())
    N, n_left = len(u_list), net.out_dim
    A = np.cumsum(a[::-1], axis=0)[::-1]                  # A_j = a_j + ... + a_{N-1}
    total = np.zeros(sum(p.numel() for p in params))
    for j in range(N):
        J = J_list[j]
        S = -np.linalg.inv(J[np.ix_(free, free)] + jac_reg * np.eye(len(free))) \
            @ J[np.ix_(free, fixed)]
        B = implicit_grad_tools._boundary_jacobian(
            u_list[j].detach().numpy(), env.deltaL, env.dim, getattr(env, "twist", False),
            getattr(env, "boundary_rows", None))
        g = B.T @ (S.T @ A[j])                            # dL/dr_j / dlam, all n_u entries
        y = net(torch.tensor([[j / N]], dtype=torch.float64)).squeeze(0)
        for c in range(y.numel()):                        # right end only: the left is held
            dy = torch.autograd.grad(y[c], params, retain_graph=True)
            total += dlam * g[n_left + c] * torch.cat([d.reshape(-1) for d in dy]).numpy()
    return total


def loss_at(env, checkpoint, net, u0, theta, target, kind):
    torch.nn.utils.vector_to_parameters(torch.as_tensor(theta), net.parameters())
    with torch.no_grad():
        _, _, q_list, _ = run(env, checkpoint, net, u0, N_STEPS)
    return loss_and_sources(env, checkpoint, q_list, target, kind)[0]


def count_solves(fn):
    """Call ``fn`` and count the per-step equilibrium solves it makes."""
    orig, n = implicit_grad_tools.local_dxf_dxb, [0]

    def counted(*a, **kw):
        n[0] += 1
        return orig(*a, **kw)
    implicit_grad_tools.local_dxf_dxb = counted
    try:
        fn()
    finally:
        implicit_grad_tools.local_dxf_dxb = orig
    return n[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refine", action="store_true",
                    help="also sweep N = 25 ... 800 (proxy vs state; a few minutes)")
    ap.add_argument("--directions", type=int, default=3,
                    help="random weight directions per finite-difference check")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    dlam = 1.0 / N_STEPS

    for case in CASES:
        cfg, env, checkpoint, net, u0 = setup(case)
        target, train_reg = task_target(cfg), float(cfg.train.jac_reg)
        u_list, r_list, q_list, J_list = run(env, checkpoint, net, u0, N_STEPS)
        theta0 = torch.nn.utils.parameters_to_vector(net.parameters()).detach().numpy().copy()
        print(f"\n{case}: nv={env.nv}, N={N_STEPS}, {theta0.size} weights, "
              f"Newton tol {env.tol:.1e}")

        # Every gradient first, while the weights are the ones the graph was built with.
        grads = {}
        for kind in LOSSES:
            _, a = loss_and_sources(env, checkpoint, q_list, target, kind)
            grads[kind] = (
                theta_grad(env, net, u_list, r_list, J_list, a, dlam, JAC_REG, "state"),
                theta_grad(env, net, u_list, r_list, J_list, a, dlam, JAC_REG, "proxy"),
                proxy_by_hand(env, net, u_list, J_list, a, dlam, JAC_REG),
                theta_grad(env, net, u_list, r_list, J_list, a, dlam, train_reg, "state"))
        _, a_T = loss_and_sources(env, checkpoint, q_list, target, "terminal")
        n_solves = {m: count_solves(lambda: adjoint_gradient(env, u_list, J_list, a_T, JAC_REG,
                                                             grad_method=m))
                    for m in ("state", "proxy")}

        for kind in LOSSES:
            g_state, g_proxy, hand, g_reg = grads[kind]
            # 1. state vs central differences along random unit directions
            errs_s, errs_p, errs_r = [], [], []
            for _ in range(args.directions):
                d = rng.standard_normal(theta0.size)
                d /= np.linalg.norm(d)
                fd = (loss_at(env, checkpoint, net, u0, theta0 + H_FD * d, target, kind)
                      - loss_at(env, checkpoint, net, u0, theta0 - H_FD * d, target, kind)) \
                    / (2 * H_FD)
                scale = np.linalg.norm(g_state)
                errs_s.append(abs(g_state @ d - fd) / scale)
                errs_p.append(abs(g_proxy @ d - fd) / scale)
                errs_r.append(abs(g_reg @ d - fd) / scale)
            report(f"{kind:8s} state vs finite differences (worst of {args.directions})",
                   max(errs_s), TOL_FD)
            # 2. proxy vs its formula assembled by hand
            report(f"{kind:8s} proxy vs B_j^T S_j^T (a_j+...+a_N), explicit S_j",
                   float(np.linalg.norm(g_proxy - hand) / np.linalg.norm(hand)), TOL_FORMULA)
            # 4. the frozen-tangent error, and the regularization's, for the record
            print(f"         {kind:8s} |proxy - state| / |state| "
                  f"{np.linalg.norm(g_proxy - g_state) / np.linalg.norm(g_state):.3f}"
                  f"  (proxy vs fd {max(errs_p):.2e});  state with jac_reg={train_reg:g} "
                  f"vs fd {max(errs_r):.1e}")
        # 3. solves per backward pass
        check("terminal loss: one solve under state, N under proxy",
              n_solves["state"] == 1 and n_solves["proxy"] == N_STEPS,
              f"state {n_solves['state']}, proxy {n_solves['proxy']}")

    if args.refine:
        print("\n5. step refinement: |proxy - state| / |state| for the same policy, terminal "
              "loss\n   (state is the exact gradient at every N; compare rows that end on "
              "the same branch)")
        for case in CASES:
            for n in (25, 50, 100, 200, 400, 800):
                cfg, env, checkpoint, net, u0 = setup(case, n)
                env.tol /= TOL_FACTOR                     # no finite differences here
                u_list, r_list, q_list, J_list = run(env, checkpoint, net, u0, n)
                _, a = loss_and_sources(env, checkpoint, q_list, task_target(cfg), "terminal")
                gs = theta_grad(env, net, u_list, r_list, J_list, a, 1.0 / n, JAC_REG, "state")
                gp = theta_grad(env, net, u_list, r_list, J_list, a, 1.0 / n, JAC_REG, "proxy")
                free = env.free_index
                unstable = sum(np.linalg.eigvalsh(J[np.ix_(free, free)]
                                                  + J[np.ix_(free, free)].T).min() < 0
                               for J in J_list)
                mid = node_position(q_list[-1], env.dim, env.mid_node)
                print(f"   {case:9s} N={n:4d}  {np.linalg.norm(gp - gs) / np.linalg.norm(gs):.4f}"
                      f"   final mid {np.array2string(mid, precision=3)}"
                      f"   unstable steps {unstable}/{n}")

    print("\n" + "=" * 72)
    print(f"{sum(results)}/{len(results)} checks passed"
          f"{'' if all(results) else '  <-- SOMETHING IS WRONG'}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
