# -*- coding: utf-8 -*-
"""The adjoint neural control itself: controller, rollout, and the update.

All three tasks -- point-to-point (``scripts/train_point_to_point.py``),
path tracking (``scripts/train_tracking.py``) and pipe insertion
(``scripts/train_pipe.py``) -- are the same method.  A small network maps the
continuation parameter ``lambda`` to the gripper's control *rate*; the rate is
integrated into a command, the command is executed by the equilibrium
simulator, and the loss is pushed back through the equilibrium Jacobians by
the multi-step adjoint (:mod:`implicit_grad_tools`).  What differs between the
tasks is only the loss, so that is all their scripts hold; everything else
lives here, in one copy.

The one subtlety is the shape of the update.  The adjoint gives its per-step
vectors as plain numbers, so the loss is handed to autograd as a surrogate
whose gradient with respect to the network's weights is the method's
(:func:`surrogate_loss`).  ``train.grad_method`` picks the method, and with it
what the vectors are attached to:

    state   (exact, default)   sum_k <u_k, dL/du_k>   on the executed commands,
                               still attached to the graph that integrated
                               them from the rates
    proxy   (frozen tangent)   sum_j dlam <r_j, q_j>  on the rates themselves

(:func:`implicit_grad_tools.multistep_adjoint_gradient` derives both).  Terms
that are functions of the command alone -- control effort, the workspace
penalty -- are added to that surrogate directly and need no adjoint.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from implicit_grad_tools import GRAD_METHODS, multistep_adjoint_gradient

try:                                   # the audit is optional: see compute_audit.py
    from compute_audit import ComputeAudit
    AUDIT = ComputeAudit(enabled=False)
except Exception:                      # pragma: no cover - audit is a dev tool
    AUDIT = None


def set_audit(audit):
    """Install a live :class:`ComputeAudit` for the rollout to annotate."""
    global AUDIT
    AUDIT = audit
    return audit


def _audit_step(k):
    if AUDIT is not None:
        AUDIT.set_context(step=k)


# --------------------------------------------------------------- controller

class BoundaryVelocityNet(nn.Module):
    """Maps the continuation parameter lambda to the right end's control rate.

    ``out_dim`` is 3 in 2D ((x, y, theta) rate), 5 in 3D
    ((x, y, z, theta, phi) rate) and 6 in 3D with twist (plus the gripper's
    yaw rate); the left end is held fixed.
    """

    def __init__(self, hidden=64, out_dim=3):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, lam):
        return self.mlp(lam)


def reinit_net_(net: nn.Module):
    """Reinitialize network weights (used at the start of every RHC segment)."""
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
            nn.init.zeros_(m.bias)
    net.apply(_init)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def segment_lengths(N: int, num_segments: int):
    """Split N steps into num_segments chunks, as evenly as possible."""
    base, rem = divmod(N, num_segments)
    return [base + 1 if i < rem else base for i in range(num_segments)]


# ------------------------------------------------------------------ rollout

def rollout(env, net, u_start, T, dlam, device, project=None, stop_on_failure=True):
    """Roll out ``T`` steps of the policy from the env's current state.

    ``lambda`` is reset to [0, 1] over this segment, matching the original
    adjoint_RHC_*.py convention, and the command is the integral of the rate::

        u_k = u_start + dlam * sum_{i <= k} net(i / T)

    ``project`` (optional) maps each accumulated command to the one actually
    executed, e.g. onto a robot workspace; None -> executed as integrated.

    ``stop_on_failure`` decides what a step whose Newton solve fails means.
    True (point-to-point, pipe): the rollout ends there, since the states that
    follow would be built on a configuration that is not an equilibrium.
    False (tracking): the rollout continues and the caller masks that step out
    of the loss -- the later steps still carry usable per-step terms.

    Returns ``(u_list, r_list, q_list, J_list, converged)``: the commands and
    the rates as torch tensors still attached to the graph (the ``state``
    gradient goes on the commands, the ``proxy`` one on the rates), the
    configurations, the equilibrium Jacobians, and a per-step boolean mask
    (``converged.all()`` is the "did the whole rollout survive" answer).
    """
    u_list, r_list, q_list, J_list = [], [], [], []
    converged = np.zeros(T, dtype=bool)

    u = u_start.clone()
    n_left = net.out_dim          # left end is held fixed -> zero rate
    dtype = next(net.parameters()).dtype     # float32 in training, float64 in the checks
    for k in range(T):
        _audit_step(k)
        lam = torch.tensor([[k / T]], dtype=dtype, device=device)
        u_dot_full = torch.cat([torch.zeros(n_left, dtype=dtype, device=device),
                                net(lam).squeeze(0)])
        u = u + dlam * u_dot_full
        if project is not None:
            u = project(u)
        u_list.append(u)
        r_list.append(u_dot_full)

        _, _, info = env.step(u.detach().cpu().numpy(), use_inertia=False)
        q_list.append(info["q"])
        J_list.append(info["J"])
        converged[k] = info["converged"]
        if stop_on_failure and not info["converged"]:
            break

    return u_list, r_list, q_list, J_list, converged


def execute_controls(env, checkpoint, U, stable=False, J_out=None):
    """Open loop: step the env through the given absolute commands.

    ``stable=True`` steps with :func:`dlo_rrt.stable_step` (the planner's own
    model of the rod, which refuses an unstable equilibrium); the default is
    the plain ``env.step`` the controller trains on.  ``J_out`` (a list)
    collects the equilibrium Jacobians.
    """
    env.set_state(checkpoint)
    q_list, ok = [], True
    for u in U:
        if stable:
            from dlo_rrt import stable_step
            info = stable_step(env, u)
            conv = info["stable"]
        else:
            _, _, info = env.step(u, use_inertia=False)
            conv = info["converged"]
        q_list.append(info["q"])
        if J_out is not None:
            J_out.append(info["J"])
        if not conv:
            ok = False
            break
    return q_list, ok


# ------------------------------------------------------------------- update

def adjoint_gradient(env, u_list, J_list, a_free_per_step, jac_reg=0.0, *, grad_method):
    """The adjoint's per-step vectors, from the per-step ``dL/dq_free``.

    ``grad_method`` is ``train.grad_method``: ``state`` gives ``dL/du_k`` for
    every executed command, ``proxy`` the frozen-tangent ``q_j`` for every
    rate (see multistep_adjoint_gradient).  :func:`surrogate_loss` puts each
    where it belongs, so pass it the same ``grad_method``.
    """
    u_seq = np.stack([u.detach().cpu().numpy() for u in u_list], axis=0)
    return multistep_adjoint_gradient(
        u_seq, J_list, env.free_index, env.fixed_index, env.deltaL,
        a_free_per_step, jac_reg=jac_reg, dim=env.dim,
        twist=getattr(env, "twist", False),
        boundary_rows=getattr(env, "boundary_rows", None),
        grad_method=grad_method,
    )


def surrogate_loss(u_list, r_list, g_seq, dlam, device, *, grad_method):
    """The scalar autograd differentiates for the adjoint part of the loss.

    Its value is not the loss; its gradient with respect to the policy's
    weights is the loss's, as ``grad_method`` computes it.  What the vectors
    of :func:`adjoint_gradient` are attached to is the second half of the
    method::

        state   sum_k <u_k, p_k>         p_k = dL/du_k, on the executed commands
        proxy   sum_j dlam <r_j, q_j>    on the rates -- on u_j, autograd would
                                         sum the future losses a second time
    """
    s = torch.zeros((), device=device)
    if grad_method == "state":
        for u_k, p_k in zip(u_list, g_seq):
            s = s + torch.dot(u_k, torch.tensor(p_k, dtype=u_k.dtype, device=device))
    elif grad_method == "proxy":
        for r_j, q_j in zip(r_list, g_seq):
            s = s + dlam * torch.dot(r_j, torch.tensor(q_j, dtype=r_j.dtype, device=device))
    else:
        raise ValueError(f"grad_method must be one of {GRAD_METHODS}, got {grad_method!r}")
    return s


def apply_update(net, opt, surrogate, clip=1.0):
    """Backward, gradient clipping, one optimizer step."""
    surrogate.backward()
    torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], clip)
    opt.step()


# ---------------------------------------------------------------- workspace

def workspace_bounds(cfg, dim, k):
    """``(lo, hi)`` over the right gripper's ``k`` controls, or None when off.

    3D reads the ``*_3d`` keys (x, y, z, theta, phi), 2D the plain ones
    (x, y, theta); a control neither lists (the twist yaw) is unbounded.
    """
    ws = cfg.get("workspace") or {}
    if not ws.get("enabled", False):
        return None
    sfx = "_3d" if dim == 3 else ""
    ang = np.array(ws[f"orientation_limits{sfx}"], dtype=float).reshape(-1, 2)
    lo, hi = np.full(k, -np.inf), np.full(k, np.inf)
    lo[:dim], hi[:dim] = ws[f"position_min{sfx}"], ws[f"position_max{sfx}"]
    lo[dim:dim + len(ang)], hi[dim:dim + len(ang)] = ang[:, 0], ang[:, 1]
    return lo, hi


def workspace_projection(bounds, u0, k):
    """Clamp the right gripper's command onto the workspace before executing it.

    Straight-through: the forward value is clamped, the backward pass is the
    identity, so the task gradient is not cut off at the boundary --
    :func:`workspace_penalty`, which acts on the *unprojected* command, is what
    pulls the policy back in.
    """
    if bounds is None:
        return None
    lo = torch.full_like(u0, -float("inf"))
    hi = torch.full_like(u0, float("inf"))
    lo[k:] = torch.tensor(bounds[0], dtype=u0.dtype)
    hi[k:] = torch.tensor(bounds[1], dtype=u0.dtype)
    return lambda u: u + (torch.clamp(u, lo, hi) - u).detach()


def workspace_penalty(net, u0, u_list, k, bounds, weight, device, dlam):
    """``weight * mean_k |violation of the unprojected command|^2``.

    What the policy *asks for* -- the previous command plus this step's rate,
    before the projection -- so the penalty still sees a violation the
    projection kept from being executed.  A function of the command alone, so
    it joins the surrogate directly, with no adjoint.
    """
    if bounds is None:
        return torch.zeros((), device=device)
    T = len(u_list)
    lams = torch.arange(T, dtype=u0.dtype, device=device)[:, None] * dlam
    prev = torch.stack([u0] + list(u_list[:-1]))[:, k:]
    raw = prev + dlam * net(lams)
    lo = torch.tensor(bounds[0], dtype=u0.dtype, device=device)
    hi = torch.tensor(bounds[1], dtype=u0.dtype, device=device)
    viol = torch.relu(raw - hi) + torch.relu(lo - raw)
    return weight * (viol ** 2).sum(dim=1).mean()


# --------------------------------------------------------------- warm start

def fit_policy(net, u0, U_target, k, dim=3, iters=3000, lr=1e-2):
    """Warm start: make the policy's integrated command reproduce ``U_target``.

    The rollout integrates ``u_j = u_0 + dlam * sum_{i<=j} net(i/N)`` (see
    :func:`rollout`), so this fits that cumulative sum, not the rates: errors
    in the rates would otherwise pile up along the path.  ``dim`` only splits
    the reported residual into its position and angle halves.
    """
    N = len(U_target)
    dlam = 1.0 / N
    lam = torch.arange(N, dtype=torch.float32).unsqueeze(1) / N
    tgt = torch.tensor(U_target[:, k:], dtype=torch.float32)
    base = u0[k:].detach()
    opt = optim.Adam(net.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 1e-2)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        u = base + dlam * torch.cumsum(net(lam), dim=0)
        loss = ((u - tgt) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        u = base + dlam * torch.cumsum(net(lam), dim=0)
        err = (u - tgt).abs()
    return {"fit_mse": float(loss.detach()),
            "fit_max_pos": float(err[:, :dim].max()),
            "fit_max_ang": float(err[:, dim:].max())}
