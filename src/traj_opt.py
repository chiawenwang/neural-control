# -*- coding: utf-8 -*-
"""Trajectory optimization for terminal objectives: Adam and L-BFGS on full rollouts.

This is the optimizer side of the shape-formation experiment
(scripts/train_shape.py), written against a flat parameter vector ``xi`` so
that the same code trains

    * ``NetRates``    the network of the paper (1-64-64-n_u, tanh), float64,
                      ``xi`` = its weights (NumPy, PyTorch's layout and init), and
    * ``DirectRates`` no network, ``xi`` = the rates themselves (the "direct
                      rates" baseline; numpy only).

One evaluation of ``xi`` is one full rollout of the quasi-static simulator
from the checkpointed initial equilibrium, the terminal loss, and its exact
state-based gradient (paper eq. (terminal): one adjoint solve at the final
step), or the frozen-tangent proxy for the baseline.  Both optimizers follow
the RSS protocol:

``adam``   fixed learning rate, every update taken (no line search).
``lbfgs``  limited-memory BFGS (Liu & Nocedal 1989): two-loop recursion,
           strong-Wolfe line search on full rollouts.

Stopping (whichever comes first): the shape RMS of an admissible rollout
reaches ``stop_rms`` | ``max_iters`` updates | ``max_rollouts`` rollouts
(every rollout counts, failed ones and line-search trials included) |
``max_failures`` failed rollouts | the optimizer cannot make progress.

A *failed* rollout is one the physics or the task rejects: a Newton failure at
the finest subdivision, a step that ends on an unstable equilibrium, an
exception from the model (e.g. a vertical gripper tangent), or -- with
``workspace="reject"`` -- a command outside the workspace box (detected
before simulating, so it costs no rollout).  A snap-through is admitted, this
being a terminal objective.  Adam reverts to the last admissible iterate and
takes its next step from there; L-BFGS treats the trial as ``f = +inf`` (the
line search shortens the step).
"""
from __future__ import annotations

import time

import numpy as np

from implicit_grad_tools import multistep_adjoint_gradient, N_ADJOINT_SOLVES

OPTIMIZERS = ("adam", "lbfgs", "none")
PARAMETERIZATIONS = ("net", "rates", "endpose")


class StopTraining(Exception):
    """Raised by the evaluator when a budget or the threshold ends the run."""


# ===================================================================== parameterizations

class DirectRates:
    """``xi`` = the rates r_k of the controlled gripper, ``(M, k)`` flattened."""

    kind = "rates"

    def __init__(self, M, k):
        self.M, self.k = int(M), int(k)
        self.P = self.M * self.k

    def rates(self, xi):
        return np.asarray(xi, dtype=float).reshape(self.M, self.k)

    def vjp(self, xi, G):
        """``dL/dxi`` from ``G = dL/dr`` (M, k)."""
        return np.asarray(G, dtype=float).reshape(-1).copy()

    def from_rates(self, R):
        """The ``xi`` whose rates are ``R`` -- exact for this parameterization."""
        return np.asarray(R, dtype=float).reshape(-1).copy(), {"fit_rms_pose": 0.0}


class NetRates:
    """The network of the paper, ``lambda -> rate``: 1-hidden-hidden-k, tanh, float64.

    Written in NumPy (forward pass and its vector-Jacobian product by hand),
    so the pipeline needs no PyTorch.  Layout and initialization are those of
    ``torch.nn.Sequential(Linear(1,h), Tanh(), Linear(h,h), Tanh(), Linear(h,k))``:
    ``xi = [W1 (h,1), b1, W2 (h,h), b2, W3 (k,h), b3]`` row-major, as
    ``parameters_to_vector`` orders them, and every weight and bias of a layer
    with ``fan_in`` inputs drawn from U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    (PyTorch's default), here from ``numpy.random.default_rng(seed)``.  The
    input of the rate of step k (k = 0..M-1) is ``lambda = k / M``, as in
    neural_control.rollout.
    """

    kind = "net"

    def __init__(self, M, k, hidden=64, seed=0):
        self.M, self.k, self.h = int(M), int(k), int(hidden)
        h = self.h
        self.shapes = [(h, 1), (h,), (h, h), (h,), (k, h), (k,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.P = int(sum(self.sizes))
        rng = np.random.default_rng(int(seed))
        parts = []
        for (shape, fan_in) in zip(self.shapes, (1, 1, h, h, h, h)):
            b = 1.0 / np.sqrt(fan_in)
            parts.append(rng.uniform(-b, b, size=shape).reshape(-1))
        self._xi = np.concatenate(parts)
        self.lam = (np.arange(self.M, dtype=float) / self.M)[:, None]

    def get(self):
        return self._xi.copy()

    def unpack(self, xi):
        xi = np.asarray(xi, dtype=float)
        out, i = [], 0
        for shape, n in zip(self.shapes, self.sizes):
            out.append(xi[i:i + n].reshape(shape))
            i += n
        return out

    def _forward(self, xi):
        W1, b1, W2, b2, W3, b3 = self.unpack(xi)
        h1 = np.tanh(self.lam @ W1.T + b1)
        h2 = np.tanh(h1 @ W2.T + b2)
        return h1, h2, h2 @ W3.T + b3

    def rates(self, xi):
        return self._forward(xi)[2]

    def vjp(self, xi, G):
        """``dL/dxi`` from ``G = dL/d(rates)`` (M, k): backpropagation through the MLP."""
        W1, b1, W2, b2, W3, b3 = self.unpack(xi)
        h1, h2, _ = self._forward(xi)
        G = np.asarray(G, dtype=float)
        dW3, db3 = G.T @ h2, G.sum(0)
        dz2 = (G @ W3) * (1.0 - h2 * h2)
        dW2, db2 = dz2.T @ h1, dz2.sum(0)
        dz1 = (dz2 @ W2) * (1.0 - h1 * h1)
        dW1, db1 = dz1.T @ self.lam, dz1.sum(0)
        return np.concatenate([a.reshape(-1) for a in (dW1, db1, dW2, db2, dW3, db3)])

    def from_rates(self, R, maxiter=20000):
        """Fit the network so that its *integrated* command reproduces ``R``'s.

        As neural_control.fit_policy: the loss is on the pose offsets
        ``dlam * cumsum(rates)``, not on the rates, so that errors do not pile
        up along the path; scaled per control so that metres and radians
        weigh alike.  L-BFGS-B (scipy) from this network's initial weights.
        Returns ``(xi, info)``; ``fit_rms_pose`` is the RMS pose error.
        """
        from scipy.optimize import minimize
        R = np.asarray(R, dtype=float)
        dlam = 1.0 / self.M
        tgt = dlam * np.cumsum(R, axis=0)
        scale = np.maximum(np.abs(tgt).max(axis=0), 1e-3)
        n = tgt.size

        def fun(xi):
            C = (dlam * np.cumsum(self.rates(xi), axis=0) - tgt) / scale
            dC = 2.0 * C / scale / n
            G = dlam * np.cumsum(dC[::-1], axis=0)[::-1]
            return float(np.sum(C * C) / n), self.vjp(xi, G)

        res = minimize(fun, self._xi, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxfun": 2 * maxiter, "ftol": 1e-20,
                                "gtol": 1e-14, "maxcor": 30})
        xi = res.x
        err = dlam * np.cumsum(self.rates(xi), axis=0) - tgt
        return xi.copy(), {"fit_rms_pose": float(np.sqrt(np.mean(err ** 2))),
                           "fit_max_pos": float(np.abs(err[:, :3]).max()),
                           "fit_max_ang": float(np.abs(err[:, 3:]).max()),
                           "fit_iters": int(res.nit)}


class EndPose:
    """The "end pose only" baseline: a base trajectory with its final pose moved.

    ``xi`` (k,) is an offset of the gripper pose, ramped in linearly over the
    last ``window`` steps of the base commands ``U_base`` (M, n_u)::

        u_k = u_base,k + w_k * xi,   w_k = clip((k + 1 - (M - window)) / window, 0, 1)

    so the first ``M - window`` steps are those of the base trajectory and
    the final pose is ``u_base,M + xi``.  Rates follow by differencing (with
    ``u_{-1} = u0``), so the evaluator and both gradients apply unchanged.
    """

    kind = "endpose"

    def __init__(self, M, k, U_base, u0, window):
        self.M, self.k, self.window = int(M), int(k), int(window)
        if not 1 <= self.window <= self.M:
            raise ValueError(f"window must be in 1..{self.M}, got {window}")
        self.P = self.k
        self.k_end = len(u0) - self.k
        self.U_base = np.asarray(U_base, dtype=float)
        self.u0 = np.asarray(u0, dtype=float)
        idx = np.arange(1, self.M + 1)
        self.w = np.clip((idx - (self.M - self.window)) / self.window, 0.0, 1.0)
        self.dw = np.diff(np.concatenate([[0.0], self.w])) * self.M       # (w_k - w_{k-1}) / dlam
        V = np.vstack([self.u0[None, :], self.U_base])[:, self.k_end:]
        self.R_base = np.diff(V, axis=0) * self.M

    def rates(self, xi):
        return self.R_base + self.dw[:, None] * np.asarray(xi, dtype=float)[None, :]

    def vjp(self, xi, G):
        return (self.dw[:, None] * np.asarray(G, dtype=float)).sum(axis=0)

    def aimed(self, u_end):
        """The offset that puts the final pose at ``u_end`` (gripper part)."""
        return np.asarray(u_end, dtype=float)[self.k_end:] - self.U_base[-1, self.k_end:]


def make_parameterization(kind, M, k, hidden=64, seed=0, U_base=None, u0=None, window=20):
    if kind == "rates":
        return DirectRates(M, k)
    if kind == "net":
        return NetRates(M, k, hidden=hidden, seed=seed)
    if kind == "endpose":
        return EndPose(M, k, U_base, u0, window)
    raise ValueError(f"parameterization must be one of {PARAMETERIZATIONS}, got {kind!r}")


# ===================================================================== rollout

def commands(u0, R, k_end, dlam):
    """Absolute commands ``U`` (M, n_u): ``u_k = u0 + dlam * sum_{i<=k} r_i`` on the gripper."""
    U = np.repeat(np.asarray(u0, float)[None, :], len(R), axis=0)
    U[:, k_end:] += dlam * np.cumsum(R, axis=0)
    return U


def workspace_violation(U, k_end, bounds):
    """Largest amount by which a gripper command leaves the box (0 inside; None: no box)."""
    if bounds is None:
        return 0.0
    lo, hi = bounds
    V = U[:, k_end:]
    return float(np.max(np.maximum(V - hi, 0.0) + np.maximum(lo - V, 0.0)))


def pegs_of(env):
    return [o for o in env.obstacles if hasattr(o, "report")]


def tunneled(peg, q_prev, q_new):
    """Whether the rod passed through ``peg`` between two committed states
    (contact.PegObstacle.crossed); the solvers already prevent it step by
    step with ``newton.max_node_step``, this is the final guard."""
    return peg.crossed(q_prev, q_new)


def self_clearance(X, r_rod, skip=2):
    """Smallest surface-to-surface distance between non-neighbouring edges of the rod.

    ``X`` (nv, 3) node positions; edges i and j with ``|i - j| > skip`` are
    compared (closer ones share a node or bend smoothly into each other).
    Self-contact is not modeled, so a negative value means the rod passes
    through itself and the state is not physical.
    """
    P0, P1 = X[:-1], X[1:]
    n = len(P0)
    I, J = np.triu_indices(n, k=skip + 1)
    a0, a1, b0, b1 = P0[I], P1[I], P0[J], P1[J]
    d1, d2, r = a1 - a0, b1 - b0, a0 - b0
    A = np.sum(d1 * d1, 1); E = np.sum(d2 * d2, 1); F = np.sum(d2 * r, 1)
    C = np.sum(d1 * r, 1); B = np.sum(d1 * d2, 1)
    den = A * E - B * B
    s = np.where(den > 1e-14, np.clip((B * F - C * E) / np.maximum(den, 1e-300), 0, 1), 0.0)
    t = (B * s + F) / E
    t = np.clip(t, 0, 1)
    s = np.clip((B * t - C) / A, 0, 1)
    t = np.clip((B * s + F) / E, 0, 1)
    d = np.linalg.norm(a0 + d1 * s[:, None] - b0 - d2 * t[:, None], axis=1)
    return float(d.min() - 2.0 * r_rod) if d.size else np.inf


def rollout(env, checkpoint, U, keep_J="last", record=False):
    """Execute the commands ``U`` from ``checkpoint``; stop at the first failed step.

    ``keep_J``: ``last`` keeps the equilibrium Hessian of the final step only
    (all a terminal state gradient needs), ``all`` keeps every step's (the
    frozen-tangent proxy needs them).  ``record`` adds the per-step arrays the
    figures and the analysis need (states, contact per peg and edge).
    Returns a dict; ``ok`` is False with ``reason`` on a failed rollout.
    """
    from rodframes import holonomy
    env.set_state(checkpoint)
    M = len(U)
    pegs = pegs_of(env)
    nseg = env.nv - 1
    out = {"ok": True, "reason": "", "steps": 0, "J_list": [None] * M, "snaps": 0, "snap_steps": [],
           "max_substeps": 0, "newton_iterations": 0, "unstable": 0, "max_penetration": 0.0,
           "max_strain": 0.0, "max_abs_holonomy": 0.0, "end_contact": False,
           "min_self_clearance": np.inf,
           "onset": [None] * len(pegs), "q_list": []}
    if record:
        out["rec"] = {"min_clearance": np.full((M, len(pegs)), np.nan),
                      "force": np.full((M, len(pegs)), np.nan),
                      "touching": np.zeros((M, len(pegs), nseg), dtype=bool),
                      "snap": np.zeros(M, dtype=bool), "substeps": np.zeros(M, dtype=int),
                      "holonomy": np.full(M, np.nan), "twist": np.full(M, np.nan)}
    for k in range(M):
        try:
            _, _, info = env.step(U[k])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
            out.update(ok=False, reason=f"error at step {k}: {type(exc).__name__}: {exc}"[:160])
            break
        out["steps"] = k + 1
        out["newton_iterations"] += int(info.get("newton_iterations", 0) or 0)
        out["max_substeps"] = max(out["max_substeps"], int(info.get("substeps", 1) or 1))
        if not info["converged"]:
            out.update(ok=False, reason=f"newton_failed at step {k}")
            break
        if info.get("stable") is False:
            out["unstable"] += 1
            out.update(ok=False, reason=f"unstable at step {k}")
            break
        q = info["q"]
        q_prev = out["q_list"][-1] if out["q_list"] else checkpoint["q"]
        hit = [j for j, peg in enumerate(pegs) if tunneled(peg, q_prev, q)]
        if hit:
            out.update(ok=False, reason=f"tunneled through peg {hit} at step {k}")
            break
        if info.get("snap"):
            out["snaps"] += 1
            out["snap_steps"].append(k)
        out["q_list"].append(q)
        if keep_J == "all" or k == M - 1:
            out["J_list"][k] = info["J"]
        X = env.positions(q)
        strain = np.abs(np.linalg.norm(np.diff(X, axis=0), axis=1) / env.deltaL - 1.0).max()
        out["max_strain"] = max(out["max_strain"], float(strain))
        out["min_self_clearance"] = min(out["min_self_clearance"],
                                        self_clearance(X, getattr(env, "R_outer", 0.0)))
        H = float(env.energy.holonomy(q)) if getattr(env, "twist", False) else 0.0   # lifted
        out["max_abs_holonomy"] = max(out["max_abs_holonomy"], abs(H))
        for j, peg in enumerate(pegs):
            rep = peg.report(q)
            out["max_penetration"] = max(out["max_penetration"], rep["max_penetration"])
            out["end_contact"] = out["end_contact"] or rep["end_contact"]
            if out["onset"][j] is None and rep["touching_edges"].any():
                out["onset"][j] = k
            if record:
                out["rec"]["min_clearance"][k, j] = rep["min_clearance"]
                out["rec"]["force"][k, j] = rep["force"]
                out["rec"]["touching"][k, j] = rep["touching_edges"]
        if record:
            out["rec"]["snap"][k] = bool(info.get("snap"))
            out["rec"]["substeps"][k] = int(info.get("substeps", 1) or 1)
            out["rec"]["holonomy"][k] = H
            if getattr(env, "twist", False):
                out["rec"]["twist"][k] = float(env.energy.total_twist(q))
    if out["ok"] and pegs:
        reps = [p.report(out["q_list"][-1]) for p in pegs]
        out["final_touching"] = [bool(r["touching_edges"].any()) for r in reps]
        out["final_force"] = [r["force"] for r in reps]
    return out


def free_nodes(env):
    """Indices of the nodes whose three position DOFs are all free."""
    free = set(env.free_index.tolist())
    return np.array([i for i in range(env.nv) if all(3 * i + c in free for c in range(3))])


def terminal_loss(env, q, target_shape, nodes):
    """``L = mean_i |x_i - x_i*|^2`` over the free nodes, and ``dL/dq_free``."""
    X = env.positions(q)
    d = X[nodes] - target_shape[nodes]
    L = float(np.mean(np.sum(d * d, axis=1)))
    full = np.zeros(env.n_dof)
    for i, row in zip(nodes, d):
        full[3 * i:3 * i + 3] = 2.0 / len(nodes) * row
    return L, full[env.free_index], X


# ===================================================================== evaluator

class Evaluator:
    """``xi -> (loss, dL/dxi)`` by full rollouts, with every budget and the log.

    Raises :class:`StopTraining` when a budget is exhausted or the threshold
    is reached; ``self.stop`` then says why, and ``self.best`` /
    ``self.threshold_eval`` keep the evaluations needed for the result.
    """

    LOG_KEYS = ("loss", "rms", "gnorm", "ok", "reason", "time", "rollouts", "snaps",
                "max_substeps", "newton_iterations", "max_penetration", "max_strain",
                "max_abs_holonomy", "min_self_clearance", "ws_violation", "end_contact", "onset",
                "final_touching",
                "final_force")

    def __init__(self, env, checkpoint, param, u0, k_end, target_shape, nodes, *,
                 grad_method="state", jac_reg=0.0, ws_bounds=None, workspace="reject",
                 stop_rms=0.0, max_rollouts=500, max_failures=20, verbose=True, tag=""):
        self.env, self.checkpoint, self.param = env, checkpoint, param
        self.u0, self.k_end = np.asarray(u0, float), int(k_end)
        self.M = env.Nsteps
        self.dlam = 1.0 / self.M
        self.target, self.nodes = target_shape, nodes
        self.grad_method, self.jac_reg = grad_method, float(jac_reg)
        self.ws_bounds, self.workspace = ws_bounds, workspace
        self.stop_rms = float(stop_rms)
        self.max_rollouts, self.max_failures = int(max_rollouts), int(max_failures)
        self.verbose, self.tag = verbose, tag
        self.n_rollouts = self.n_failures = self.n_evals = 0
        self.t0 = time.time()
        self.log = {k: [] for k in self.LOG_KEYS}
        self.stop = None
        self.best = None                 # lowest-loss admissible evaluation
        self.threshold_eval = None
        self.grad_time = []

    # -------------------------------------------------------------- core
    def rollout_xi(self, xi, record=False):
        R = self.param.rates(xi)
        U = commands(self.u0, R, self.k_end, self.dlam)
        ws = workspace_violation(U, self.k_end, self.ws_bounds)
        if self.workspace == "reject" and ws > 1e-12:
            return U, {"ok": False, "reason": f"workspace (+{ws:.3g})", "skipped": True,
                       "ws_violation": ws, "steps": 0}
        res = rollout(self.env, self.checkpoint, U,
                      keep_J="all" if self.grad_method == "proxy" else "last", record=record)
        res["ws_violation"] = ws
        return U, res

    def gradient(self, xi, U, res):
        """``dL/dxi`` of an admissible rollout (state or proxy)."""
        t = time.time()
        M = self.M
        L, a_last, X = terminal_loss(self.env, res["q_list"][-1], self.target, self.nodes)
        a = np.zeros((M, len(self.env.free_index)))
        a[-1] = a_last
        g_seq = multistep_adjoint_gradient(
            U, res["J_list"], self.env.free_index, self.env.fixed_index, self.env.deltaL, a,
            jac_reg=self.jac_reg, dim=3, twist=getattr(self.env, "twist", False),
            boundary_rows=getattr(self.env, "boundary_rows", None), grad_method=self.grad_method)
        g_seq = g_seq[:, self.k_end:]
        if self.grad_method == "state":           # dL/dr_j = dlam * sum_{k>=j} p_k
            G = self.dlam * np.cumsum(g_seq[::-1], axis=0)[::-1]
        else:                                     # proxy: dL/dr_j = dlam * q_j
            G = self.dlam * g_seq
        g = self.param.vjp(xi, G)
        self.grad_time.append(time.time() - t)
        return L, g, X

    def __call__(self, xi):
        """One evaluation: ``(ok, loss, grad, info)``; may raise StopTraining."""
        if self.n_rollouts >= self.max_rollouts:
            self._halt(f"rollout budget ({self.max_rollouts}) used")
        xi = np.asarray(xi, dtype=float).copy()
        U, res = self.rollout_xi(xi)
        if not res.get("skipped"):
            self.n_rollouts += 1
        self.n_evals += 1
        ev = {"xi": xi, "U": U, "ok": res["ok"], "reason": res["reason"], "loss": np.inf,
              "g": None, "rms": np.nan, "res": res, "idx": self.n_evals - 1}
        if res["ok"]:
            L, g, X = self.gradient(xi, U, res)
            ev.update(loss=L, g=g, rms=float(np.sqrt(L)), shape=X)
            if self.best is None or L < self.best["loss"]:
                self.best = ev
        else:
            self.n_failures += 1
        self._record(ev)
        if not res["ok"] and self.n_failures >= self.max_failures:
            self._halt(f"{self.n_failures} failed rollouts (last: {res['reason']})")
        if res["ok"] and ev["rms"] <= self.stop_rms:
            self.threshold_eval = ev
            self._halt(f"rms {ev['rms'] * 1e3:.4f} mm <= stop_rms {self.stop_rms * 1e3:g} mm")
        return ev

    def _halt(self, why):
        self.stop = why
        raise StopTraining(why)

    def _record(self, ev):
        res = ev["res"]
        lg = self.log
        lg["loss"].append(ev["loss"])
        lg["rms"].append(ev["rms"])
        lg["gnorm"].append(np.nan if ev["g"] is None else float(np.linalg.norm(ev["g"])))
        lg["ok"].append(bool(ev["ok"]))
        lg["reason"].append(ev["reason"])
        lg["time"].append(time.time() - self.t0)
        lg["rollouts"].append(self.n_rollouts)
        for k in ("snaps", "max_substeps", "newton_iterations", "max_penetration", "max_strain",
                  "max_abs_holonomy", "min_self_clearance", "ws_violation", "end_contact"):
            lg[k].append(res.get(k, np.nan))
        lg["onset"].append([(-1 if o is None else o) for o in res.get("onset", [])])
        lg["final_touching"].append(res.get("final_touching", []))
        lg["final_force"].append(res.get("final_force", []))


# ===================================================================== optimizers

def run_adam(ev_fn, xi0, lr, max_iters, betas=(0.9, 0.999), eps=1e-8, log_every=10, tag=""):
    """Adam at a fixed rate, every admissible update taken.

    A failed rollout is not an iterate: the parameters go back to the last
    admissible ones and the next step is taken from there (with the moments
    updated once more by that iterate's gradient).  Returns the list of
    accepted iterates (dicts with ``eval``) -- iterate 0 is the start.
    """
    b1, b2 = betas
    cur = ev_fn(xi0)
    if not cur["ok"]:
        ev_fn._halt(f"the initial rollout failed ({cur['reason']})")
    m = np.zeros_like(cur["xi"])
    v = np.zeros_like(cur["xi"])
    t = 0
    iters = ev_fn.iters
    iters.append({"eval": cur["idx"], "loss": cur["loss"], "rms": cur["rms"]})
    ev_fn.current = cur
    while len(iters) - 1 < max_iters:
        g = cur["g"]
        t += 1
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        step = -lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        trial = ev_fn(cur["xi"] + step)
        if trial["ok"]:
            cur = trial
            ev_fn.current = cur
            iters.append({"eval": cur["idx"], "loss": cur["loss"], "rms": cur["rms"]})
            n = len(iters) - 1
            if n % log_every == 0:
                _say(ev_fn, tag, n, cur)
        else:
            _say(ev_fn, tag, len(iters) - 1, trial, failed=True)
    ev_fn.stop = ev_fn.stop or f"iteration cap ({max_iters}) reached"
    return iters


def run_lbfgs(ev_fn, xi0, max_iters, history=10, c1=1e-4, c2=0.9, max_ls=20,
              first_step=0.1, log_every=1, tag=""):
    """L-BFGS with a strong-Wolfe line search on full rollouts.

    Direction: two-loop recursion over the last ``history`` pairs, initial
    Hessian ``(s.y / y.y) I``; ``-g`` when the memory is empty or the result
    is not a descent direction.  Step: 1, except on an empty memory, where
    the first trial moves ``xi`` by ``first_step`` (norm).  Line search:
    Nocedal & Wright Alg. 3.5/3.6 with cubic interpolation; a failed rollout
    counts as ``f = +inf`` (an upper bracket).  A pair is stored only if
    ``s.y > 0``.  If the search ends without a Wolfe point, the best Armijo
    point is taken if there is one; otherwise the memory is cleared and the
    search retried along ``-g`` once, and the run stops if that fails too.
    """
    cur = ev_fn(xi0)
    if not cur["ok"]:
        ev_fn._halt(f"the initial rollout failed ({cur['reason']})")
    S, Y = [], []
    iters = ev_fn.iters
    iters.append({"eval": cur["idx"], "loss": cur["loss"], "rms": cur["rms"], "alpha": None,
                  "ls_evals": 0})
    ev_fn.current = cur
    retried = False
    while len(iters) - 1 < max_iters:
        g = cur["g"]
        d = -_two_loop(g, S, Y)
        if not S or g @ d >= 0.0:
            d = -g
            S, Y = [], []
        a0 = 1.0 if S else min(1.0, first_step / max(np.linalg.norm(d), 1e-300))
        n_before = ev_fn.n_evals
        new = _wolfe_search(ev_fn, cur, d, a0, c1, c2, max_ls)
        if new is None:
            if S and not retried:
                S, Y, retried = [], [], True
                continue
            ev_fn.stop = "line search found no admissible decrease"
            break
        retried = False
        s = new["xi"] - cur["xi"]
        y = new["g"] - g
        if s @ y > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
            S.append(s)
            Y.append(y)
            if len(S) > history:
                S.pop(0)
                Y.pop(0)
        cur = new
        ev_fn.current = cur
        iters.append({"eval": cur["idx"],
                      "loss": cur["loss"], "rms": cur["rms"], "alpha": new["alpha"],
                      "ls_evals": ev_fn.n_evals - n_before})
        if (len(iters) - 1) % log_every == 0:
            _say(ev_fn, tag, len(iters) - 1, cur, extra=f"alpha={new['alpha']:.3g} "
                                                          f"ls={ev_fn.n_evals - n_before}")
    if ev_fn.stop is None:
        ev_fn.stop = f"iteration cap ({max_iters}) reached"
    return iters


def _two_loop(g, S, Y):
    q = g.copy()
    alphas = []
    for s, y in zip(reversed(S), reversed(Y)):
        rho = 1.0 / (y @ s)
        a = rho * (s @ q)
        alphas.append((rho, a))
        q -= a * y
    if S:
        q *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
    for (s, y), (rho, a) in zip(zip(S, Y), reversed(alphas)):
        b = rho * (y @ q)
        q += (a - b) * s
    return q


def _wolfe_search(ev_fn, cur, d, a0, c1, c2, max_ls):
    """Strong-Wolfe search along ``d`` from ``cur``; returns the accepted eval (with ``alpha``) or None."""
    f0, dphi0 = cur["loss"], float(cur["g"] @ d)
    if not dphi0 < 0:
        return None
    best_armijo = [None]

    def phi(a):
        e = ev_fn(cur["xi"] + a * d)
        ev_fn.last_eval = e
        e["alpha"] = a
        if e["ok"]:
            e["dphi"] = float(e["g"] @ d)
            if e["loss"] <= f0 + c1 * a * dphi0 and (best_armijo[0] is None
                                                       or e["loss"] < best_armijo[0]["loss"]):
                best_armijo[0] = e
        return e

    def zoom(lo, hi, n):
        while n < max_ls:
            a_lo, a_hi = lo["alpha"], hi["alpha"]
            a = _interp(lo, hi)
            e = phi(a)
            n += 1
            if (not e["ok"]) or e["loss"] > f0 + c1 * a * dphi0 or e["loss"] >= lo["loss"]:
                hi = e
            else:
                if abs(e["dphi"]) <= -c2 * dphi0:
                    return e
                if e["dphi"] * (a_hi - a_lo) >= 0:
                    hi = lo
                lo = e
            if abs(hi["alpha"] - lo["alpha"]) < 1e-12 * max(1.0, lo["alpha"]):
                break
        return best_armijo[0]

    prev = {"alpha": 0.0, "loss": f0, "dphi": dphi0, "ok": True}
    a = a0
    for i in range(max_ls):
        e = phi(a)
        if (not e["ok"]) or e["loss"] > f0 + c1 * a * dphi0 or (i > 0 and e["loss"] >= prev["loss"]):
            return zoom(prev, e, i + 1)
        if abs(e["dphi"]) <= -c2 * dphi0:
            return e
        if e["dphi"] >= 0:
            return zoom(e, prev, i + 1)
        prev, a = e, 2.0 * a
    return best_armijo[0]


def _interp(lo, hi):
    """Trial step in the bracket: cubic interpolation if both ends are finite, else bisection."""
    a, b = lo["alpha"], hi["alpha"]
    mid = 0.5 * (a + b)
    if not (hi.get("ok") and np.isfinite(hi["loss"]) and "dphi" in hi and "dphi" in lo):
        return mid
    fa, fb, da, db = lo["loss"], hi["loss"], lo["dphi"], hi["dphi"]
    d1 = da + db - 3 * (fa - fb) / (a - b)
    disc = d1 * d1 - da * db
    if disc < 0:
        return mid
    d2 = np.sign(b - a) * np.sqrt(disc)
    t = b - (b - a) * (db + d2 - d1) / (db - da + 2 * d2)
    lo_, hi_ = min(a, b), max(a, b)
    width = hi_ - lo_
    if not np.isfinite(t) or t < lo_ + 0.1 * width or t > hi_ - 0.1 * width:
        return mid
    return float(t)


def _say(ev_fn, tag, it, e, failed=False, extra=""):
    if not ev_fn.verbose:
        return
    if failed:
        print(f"[{tag} it {it:03d}] FAILED rollout ({e['reason']}); failures {ev_fn.n_failures}"
              f"/{ev_fn.max_failures}, rollouts {ev_fn.n_rollouts}", flush=True)
        return
    print(f"[{tag} it {it:03d}] rms={e['rms'] * 1e3:8.3f} mm  rollouts={ev_fn.n_rollouts}  "
          f"snaps={e['res']['snaps']}  pen={e['res']['max_penetration'] * 1e3:.3f} mm  "
          f"t={time.time() - ev_fn.t0:7.1f}s  {extra}", flush=True)


def optimize(ev_fn, xi0, optimizer, max_iters, opt_cfg, tag=""):
    """Run ``optimizer`` and return ``(iters, final_eval)``; never raises StopTraining.

    ``final_eval``: the evaluation that reached ``stop_rms`` if any, else the
    optimizer's current (last accepted) iterate.
    """
    ev_fn.current = None
    ev_fn.last_eval = None
    ev_fn.iters = []
    try:
        if optimizer == "adam":
            run_adam(ev_fn, xi0, float(opt_cfg.get("lr", 1e-3)), max_iters,
                             log_every=int(opt_cfg.get("log_every", 10)), tag=tag)
        elif optimizer == "none":                # evaluate the start only (e.g. end pose, aimed)
            cur = ev_fn(xi0)
            ev_fn.current = cur if cur["ok"] else None
            ev_fn.iters.append({"eval": cur["idx"], "loss": cur["loss"], "rms": cur["rms"]})
            ev_fn.stop = "no optimization (evaluation only)" if cur["ok"] else f"failed: {cur['reason']}"
        elif optimizer == "lbfgs":
            run_lbfgs(ev_fn, xi0, max_iters, history=int(opt_cfg.get("history", 10)),
                              c1=float(opt_cfg.get("c1", 1e-4)), c2=float(opt_cfg.get("c2", 0.9)),
                              max_ls=int(opt_cfg.get("max_ls", 20)),
                              first_step=float(opt_cfg.get("first_step", 0.1)), tag=tag)
        else:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {optimizer!r}")
    except StopTraining:
        pass
    final = ev_fn.threshold_eval or ev_fn.current
    return ev_fn.iters, final
