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
  6. the state gradient across the quasi-static stepping of
     src/continuation.py (3D + twist, terminal loss): a scripted gripper path
     plus a small policy perturbation, on two rollouts -- one whose large
     compression steps make the stepping subdivide (substeps, no snap), and
     one that turns the end tangent of a buckled arch downward so the arch
     snaps through.  The snap has to happen at the same step for the policy
     and both of its finite-difference perturbations, and the state gradient
     has to match the central differences as in check 1: on one branch the
     end state is a function of the current boundary, and after the snap the
     rollout is on the next branch, where that holds again.

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
from env_factory import (build_env, reset_env, settle_initial, env_dim, env_twist,
                         end_control_dim, initial_control, task_target, dloss_dq_free_node,
                         node_position)
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


# ------------------------------------------------- 6. continuation and snap

#: Scripted right-gripper paths (increments of x, y, z, theta, phi, yaw over
#: the start pose) the policy perturbs.  ``pre`` is run first, in 20 stable
#: steps, to reach the start state.
SCRIPTED = {
    # The globalized Newton solve takes even a 0.1 m compression step in one
    # go, so the step is made unsolvable in one go instead: with the iteration
    # cap at 40 the first step (buckling onset, ~110 iterations) fails and
    # the stepping subdivides it; the substeps converge well within the cap.
    "substeps": {"pre": None, "N": 3, "expect": "substeps", "max_iter": 40,
                 "path": lambda l: np.array([-0.3 * l, 0, 0, 0, 0, 0]),
                 "what": "compress 0.3 m in 3 steps from the settled rod, Newton capped at 40 iterations"},
    "snap": {"pre": np.array([-0.3, 0, 0, 0, 0, 0]), "N": 8, "expect": "snap", "max_iter": None,
             "path": lambda l: np.array([0, 0, 0, 0, -1.5 * l, 0]),
             "what": "buckled arch (x_r = 0.7), end tangent turned down to phi = -1.5 rad"},
}
POLICY_SCALE = 0.1      # the policy's rate is scaled down so the script sets the path


def setup_scripted(case, n_steps):
    cfg = load_config(_ROOT / CASES["3d_twist"][0])
    cfg["rod"]["nv"] = NV
    cfg["rod"]["total_time"] = n_steps * cfg["rod"]["dt"]
    cfg["workspace"]["enabled"] = False
    env = build_env(cfg)
    reset_env(env, cfg)
    env.tol *= TOL_FACTOR
    settle_initial(env, cfg)
    if SCRIPTED[case]["max_iter"] is not None:
        env.max_newton_iter = SCRIPTED[case]["max_iter"]
    u0 = initial_control(cfg)
    if SCRIPTED[case]["pre"] is not None:
        for j in range(1, 21):
            u = u0.copy()
            u[6:] += SCRIPTED[case]["pre"] * j / 20
            _, _, info = env.step(u)
            if not (info["converged"] and info["stable"]):
                raise RuntimeError(f"{case}: the pre-motion failed at step {j}")
        u0 = u
    k = end_control_dim(3, True)
    torch.manual_seed(0)
    net = BoundaryVelocityNet(hidden=HIDDEN, out_dim=k).double()
    return cfg, env, env.get_state(), net, torch.tensor(u0, dtype=torch.float64)


def scripted_rollout(env, checkpoint, net, u0, case):
    """``u_k = u0 + path(k/N) + POLICY_SCALE * dlam * sum_{i<=k} net(i/N)`` on the right end."""
    N = SCRIPTED[case]["N"]
    dlam = 1.0 / N
    env.set_state(checkpoint)
    u_list, r_list, q_list, J_list, infos = [], [], [], [], []
    acc = torch.zeros(12, dtype=torch.float64)
    for k in range(N):
        lam = torch.tensor([[k / N]], dtype=torch.float64)
        rate = torch.cat([torch.zeros(6, dtype=torch.float64), net(lam).squeeze(0)])
        acc = acc + POLICY_SCALE * dlam * rate
        base = u0.clone()
        base[6:] += torch.tensor(SCRIPTED[case]["path"]((k + 1) / N), dtype=torch.float64)
        u = base + acc
        _, _, info = env.step(u.detach().numpy())
        if not info["converged"]:
            raise RuntimeError(f"{case}: Newton failed at step {k + 1} ({info['reason']})")
        u_list.append(u)
        r_list.append(rate)
        q_list.append(info["q"])
        J_list.append(info["J"])
        infos.append(info)
    return u_list, r_list, q_list, J_list, infos


def scripted_loss(env, checkpoint, net, u0, case, theta, target):
    torch.nn.utils.vector_to_parameters(torch.as_tensor(theta), net.parameters())
    with torch.no_grad():
        _, _, q_list, _, infos = scripted_rollout(env, checkpoint, net, u0, case)
    L = node_tracking_loss_terminal(q_list[-1], env, target)
    return L, [i["snap"] for i in infos], [i["substeps"] for i in infos]


def node_tracking_loss_terminal(q, env, target):
    d = node_position(q, 3, env.mid_node) - target
    return float(d @ d)


def check_scripted(case, rng, directions):
    spec = SCRIPTED[case]
    cfg, env, checkpoint, net, u0 = setup_scripted(case, spec["N"])
    target = np.array([0.4, 0.05, 0.1])
    u_list, r_list, q_list, J_list, infos = scripted_rollout(env, checkpoint, net, u0, case)
    N = spec["N"]
    a = np.zeros((N, len(env.free_index)))
    a[-1] = dloss_dq_free_node(q_list[-1], env.free_index, 3, env.mid_node, target)
    g = theta_grad(env, net, u_list, r_list, J_list, a, 1.0 / N, JAC_REG, "state")
    theta0 = torch.nn.utils.parameters_to_vector(net.parameters()).detach().numpy().copy()
    subs = [i["substeps"] for i in infos]
    snaps = [k + 1 for k, i in enumerate(infos) if i["snap"]]
    unstable = sum(1 for i in infos if i["stable"] is False)
    print(f"\n  {case}: {spec['what']}; N={N}, policy scale {POLICY_SCALE}")
    print(f"       substeps per step {subs}, snap at steps {snaps or 'none'}, unstable steps {unstable},"
          f" final mid {np.array2string(node_position(q_list[-1], 3, env.mid_node), precision=3)}")
    for i in infos:
        for sn in i["snaps"]:
            print(f"       snap ({sn['kind']}): energy drop {sn['energy_drop']:+.4e} J, "
                  f"max node displacement {sn['max_displacement']:.3f} m"
                  + (f", at substep {sn['substep']}/{sn['of']}" if sn["substep"] else ""))
    if spec["expect"] == "substeps":
        check(f"{case}: the stepping subdivided at least one step, without a snap",
              max(subs) > 1 and not snaps, f"substeps {subs}")
    else:
        check(f"{case}: a snap-through occurred", bool(snaps), f"steps {snaps}")
    check(f"{case}: every step ended on a stable equilibrium", unstable == 0)
    errs, same = [], True
    for _ in range(directions):
        d = rng.standard_normal(theta0.size)
        d /= np.linalg.norm(d)
        Lp, sp, _ = scripted_loss(env, checkpoint, net, u0, case, theta0 + H_FD * d, target)
        Lm, sm, _ = scripted_loss(env, checkpoint, net, u0, case, theta0 - H_FD * d, target)
        same &= (sp == sm == [i["snap"] for i in infos])
        fd = (Lp - Lm) / (2 * H_FD)
        errs.append(abs(g @ d - fd) / np.linalg.norm(g))
    if spec["expect"] == "snap":
        check(f"{case}: the snap is at the same step for theta and theta +- h d", same)
    report(f"{case}: state vs finite differences (worst of {directions})", max(errs), TOL_FD)


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

    print("\n6. the state gradient across subdivided steps and a snap-through (3D + twist)")
    for case in SCRIPTED:
        check_scripted(case, rng, args.directions)

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
