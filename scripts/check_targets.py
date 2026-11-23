"""Target checking: is a path / point target reachable, and does it work with the obstacle?

Run this BEFORE training on a new tracking or point-to-point scene.  The rod's
reachable mid-node positions are not obvious with a bent start pose and an
obstacle (a peg), and a target that sits on or too close to the obstacle cannot
be tracked: the mid node cannot be inside the peg, and if the peg touches the
rod only where the mid node is, tracking and touching exclude each other.

  1. SAMPLE.  Workers explore from the start pose: they perturb the right
     gripper's command a little, solve the quasi-static equilibrium
     (dlo_rrt.stable_step: only stable equilibria are kept), and keep what
     they find -- the mid node's position and the whole rod.  For a path the
     peg is left out here, so the cloud is the free-space reachable set; for
     p2p it is sampled with the peg in place.
  2. TARGET.
       path  a family of smooth half-sine paths x(t) = x0 + dx t,
             y(t) = y0 + dy t, z(t) = z0 + amp sin(2 pi periods t) (the keys
             of task.tracking.sine_xz) is scored against the cloud: every
             point needs a sample within --tol, the curvature must be low, the
             path must travel a worthwhile distance, and x must not increase
             (that stretches the rod).
       p2p   the config's target (or --target) is checked the same way; if it
             is not supported, the nearest supported point is proposed.
  3. PEG (path only).  For each good path the rod shapes the cloud holds along
     it say where a peg of the config's radius would be touched: the peg is
     placed so that the mid node stays --min-clearance from it all along,
     while the rod overlaps it by at most --max-penetration over a middle
     stretch (--contact-range of the path) and clears it elsewhere.  With
     --keep-peg the config's peg is only checked, not moved.
  4. EXECUTE.  The best (path, peg) pairs are followed on the real rod -- with
     the peg and contact physics on -- by a damped least-squares tracker
     (finite-difference mid-node Jacobian, one step per control step), and the
     tracking error and the contact pattern are reported.  The first pair that
     tracks within --exec-tol wins.

Shape mode (--mode shape) finds TWO pegs and shape targets for
scripts/train_shape.py: end shapes that touch both pegs while the start does
not.  The cloud is sampled from the settled u0 (train_shape.py's start, no
ready motion), free of pegs.  Then:

  P. PEGS.  On a grid of peg centers (x, z; center_y, radius, half length
     from the config's peg block) clear of the start rod by --start-clearance,
     each sampled rod shape is matched with the pegs it overlaps by
     --shape-penetration (MIN, MAX) at one interior node, on the cylinder's
     side, ends clear.  One peg takes the clamp-side half of the rod, the
     other the gripper-side half (--mid-gap apart); the pair most sampled
     shapes fit both of wins.
  V. VERIFY.  Diverse shapes of that pair give final commands u_T; the
     straight move u0 -> u_T is rolled out with the full physics by
     train_shape.build_target, once free (the rod must never overlap a peg by
     more than MAX on the way, nor touch it in the first --free-start of the
     path, and must end at least MIN into both) and once with both pegs in
     place (every step converged and stable, no snap, touching both pegs at
     the end and neither in the first --free-start).  The first --n-targets
     verified moves, chosen far apart, are the targets P1, P2, ...

    python scripts/check_targets.py --config configs/task_shape_pegs_3d.yaml --mode shape
    python scripts/check_targets.py --config configs/task_shape_pegs_3d.yaml --mode shape --keep-peg

Outputs (in --out): report.json, cloud.npz, suggested_target.yaml (the
`sine_xz` and `peg` / target block to paste into the config) and
target_check.png; in shape mode also targets/P*.npz|json, the targets as
train_shape.py builds them (train with --out pointing at the same directory).

    python scripts/check_targets.py --config configs/task_tracking_peg_3d.yaml
    python scripts/check_targets.py --config CFG --keep-peg
    python scripts/check_targets.py --config CFG --mode p2p --target 0.1 0.0 0.5
    python scripts/check_targets.py --config CFG --no-execute
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from blas_threads import limit_blas_threads
limit_blas_threads()

import argparse
import copy
import itertools
import json
import multiprocessing as mp
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import load_config
from dlo_rrt import stable_step
from env_factory import (
    build_env, reset_env, settle_initial, env_dim, env_twist, end_control_dim, initial_control,
)
from neural_control import workspace_bounds


# ------------------------------------------------------------------ scene

def find_peg(env):
    """The peg (``env.obstacles`` entry with an axis), or None."""
    return next((o for o in env.obstacles if hasattr(o, "half_length")), None)


def axis_clearance(pts, cx, cy, cz, radius, half_length):
    """Distance of points ``(..., 3)`` from the surface of a y-axis cylinder (negative: inside)."""
    pts = np.asarray(pts, dtype=float)
    yc = np.clip(pts[..., 1], cy - half_length, cy + half_length)
    d = np.stack([pts[..., 0] - cx, pts[..., 1] - yc, pts[..., 2] - cz], axis=-1)
    return np.linalg.norm(d, axis=-1) - radius


def peg_clearance(peg, pts):
    """Clearance of each point from ``peg``'s surface; inf without a peg."""
    pts = np.atleast_2d(pts)
    if peg is None:
        return np.full(len(pts), np.inf)
    return axis_clearance(pts, peg.cx, peg.cy, peg.cz, peg.radius, peg.half_length)


def rod_contact_summary(env, q, peg):
    """``(touching, min clearance)`` of the rod against the peg."""
    if peg is None:
        return False, np.inf
    P = np.asarray(env.positions(q)).reshape(-1, 3)
    clr = peg_clearance(peg, P)
    return bool(np.any(clr < 0.0)), float(clr.min())


def mid_of(env, q):
    return np.asarray(env.positions(q))[env.mid_node].astype(float)


def cap_solver(env, max_substeps=4, max_newton_iter=150):
    """Cheaper, stricter solves for sampling / tracking.

    The start pose needs the full continuation budget (it is a large move from
    the straight rod), but a sampled move that needs many substeps, a snap
    descent and ~1500 Newton iterations can cost 40 s for one sample: it is
    cheaper to reject it, the next perturbation is a small one anyway.
    """
    env.newton_cfg["max_substeps"] = max_substeps
    env.newton_cfg["snap_rule"] = False          # a snap-through is a jump, not a tracked state
    env.max_newton_iter = max_newton_iter


def ready_state(env, cfg, dx, steps=20):
    """Settle the start pose, then move the gripper ``dx`` towards the fixed
    end with stable steps (the same ready pose as train_tracking.py).

    Returns the ready control; ``env`` is left in the ready state.
    """
    u0 = initial_control(cfg)
    k = end_control_dim(env.dim, env_twist(cfg))
    u_r = u0.copy()
    u_r[k] -= dx
    for j in range(1, steps + 1):
        info = stable_step(env, u0 + (u_r - u0) * j / steps)
        if not info["stable"]:
            raise RuntimeError(f"the ready motion fails at step {j}; the start pose has no "
                               "stable equilibrium the continuation can reach")
    return u_r


def start_state(env, cfg, ready_dx):
    """The ready pose (``ready_dx``), or with ``ready_dx`` None the settled u0
    train_shape.py starts from.  Returns that control; ``env`` is left there."""
    if ready_dx is not None:
        return ready_state(env, cfg, ready_dx)
    settle_initial(env, cfg)
    return initial_control(cfg)


def with_peg(cfg, center_xz=None, center_y=None, enabled=True):
    """A copy of ``cfg`` with the peg moved / switched."""
    c = copy.deepcopy(cfg)
    c["peg"]["enabled"] = bool(enabled)
    if center_xz is not None:
        c["peg"]["center_xz"] = [float(center_xz[0]), float(center_xz[1])]
    if center_y is not None:
        c["peg"]["center_y"] = float(center_y)
    return c


# --------------------------------------------------------------- sampling

def explore(job):
    """One worker: a random tree of stable states around the ready pose.

    Returns arrays over the kept samples: right-gripper command, mid position,
    the rod's node positions, and (with a peg) the touching flag.
    """
    cfg, seed, n, ready_dx, step_pos, step_ang, time_limit = job
    t_end = time.time() + time_limit
    rng = np.random.default_rng(seed)
    env = build_env(cfg)
    reset_env(env, cfg)
    u_ready = start_state(env, cfg, ready_dx)
    cap_solver(env)
    k = end_control_dim(env.dim, env_twist(cfg))
    lo, hi = workspace_bounds(cfg, env.dim, k)
    peg = find_peg(env)

    def record(q):
        P = np.asarray(env.positions(q)).reshape(-1, 3)
        return P[env.mid_node].astype(float), P.astype(np.float32), rod_contact_summary(env, q, peg)[0]

    st0 = env.get_state()
    m, P, touch = record(st0["q"])
    nodes = [(u_ready.copy(), st0)]
    out = dict(u=[u_ready[k:]], mid=[m], rod=[P], touch=[touch])
    m0 = m
    fails = 0
    for _ in range(n):
        if time.time() > t_end:
            break
        d = np.linalg.norm(np.array(out["mid"]) - m0, axis=1)
        w = 1.0 + d / 0.05                      # lean towards the frontier
        i = rng.choice(len(nodes), p=w / w.sum())
        u_par, st_par = nodes[i]
        u = u_par.copy()
        u[k:k + 3] += rng.normal(0.0, step_pos, 3)
        u[k + 3:k + 5] += rng.normal(0.0, step_ang, 2)
        u[k:2 * k] = np.clip(u[k:2 * k], lo, hi)
        env.set_state(st_par)
        info = stable_step(env, u)
        if not info["stable"]:
            fails += 1
            continue
        m, P, touch = record(info["q"])
        nodes.append((u, env.get_state()))
        out["u"].append(u[k:])
        out["mid"].append(m)
        out["rod"].append(P)
        out["touch"].append(touch)
    res = {key: np.array(v) for key, v in out.items()}
    res["fails"] = fails
    return res


def sample_cloud(cfg, args):
    ready_dx = None if args.mode == "shape" else args.ready_dx
    jobs = [(cfg, args.seed * 1000 + w, args.samples, ready_dx, args.step_pos,
             args.step_ang, args.time_limit) for w in range(args.workers)]
    t0 = time.time()
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            parts = pool.map(explore, jobs)
    else:
        parts = [explore(jobs[0])]
    cloud = {key: np.concatenate([p[key] for p in parts]) for key in ("u", "mid", "rod", "touch")}
    cloud["fails"] = int(sum(p["fails"] for p in parts))
    print(f"sampled {len(cloud['mid'])} stable states ({cloud['fails']} failed solves) "
          f"in {time.time() - t0:.0f} s", flush=True)
    return cloud


def region_summary(cloud):
    M = cloud["mid"]
    return {"n": int(len(M)), "contact_fraction": float(cloud["touch"].mean()),
            "min": np.round(M.min(0), 4).tolist(), "max": np.round(M.max(0), 4).tolist(),
            "p05": np.round(np.percentile(M, 5, axis=0), 4).tolist(),
            "p95": np.round(np.percentile(M, 95, axis=0), 4).tolist()}


class Support:
    """Nearest-sample queries on the cloud."""

    def __init__(self, cloud):
        from scipy.spatial import cKDTree
        self.cloud = cloud
        self.tree = cKDTree(cloud["mid"])

    def nearest(self, pts):
        return self.tree.query(pts)


# ------------------------------------------------------------------- path

def sine_path(m0, N, amp, dx, dy, periods=0.5):
    """Row k = target of step k (t = (k+1)/N), the sine_xz of make_target_trajectory."""
    t = np.arange(1, N + 1) / N
    return np.column_stack([m0[0] + dx * t, m0[1] + dy * t,
                            m0[2] + amp * np.sin(2.0 * np.pi * periods * t)])


def max_curvature(P):
    """Largest curvature of a polyline (1/m), from the turning angle per arc length."""
    d = np.diff(P, axis=0)
    ds = np.linalg.norm(d, axis=1)
    keep = ds > 1e-9
    d, ds = d[keep], ds[keep]
    if len(d) < 3:
        return 0.0
    t = d / ds[:, None]
    ang = np.arccos(np.clip(np.sum(t[1:] * t[:-1], axis=1), -1.0, 1.0))
    return float(np.max(ang / (0.5 * (ds[1:] + ds[:-1]))))


def score_paths(support, m0, args):
    """Sine candidates supported by the cloud, best first.

    Supported: every checked point has a sample within ``--tol``.  Ranked by
    how much the path moves in z and in total (a worthwhile tracking task)
    among the paths whose curvature is at most ``--max-curvature``.
    """
    amps = [a for a in np.arange(-0.08, 0.0801, 0.01) if abs(a) > 1e-9]
    dxs = np.arange(-0.15, 0.0001, 0.025)
    dys = np.arange(-0.15, 0.1501, 0.03)
    rows = []
    for amp, dx, dy in itertools.product(amps, dxs, dys):
        P = sine_path(m0, args.check_points, amp, dx, dy)
        d, _ = support.nearest(P)
        length = float(np.sum(np.linalg.norm(np.diff(np.vstack([m0, P]), axis=0), axis=1)))
        rows.append({"amp": float(amp), "dx": float(dx), "dy": float(dy), "periods": 0.5,
                     "supported": float(np.mean(d <= args.tol)), "max_gap": float(d.max()),
                     "curvature": max_curvature(np.vstack([m0, P])), "length": length})
    for r in rows:
        r["ok"] = bool(r["supported"] >= args.min_support and r["curvature"] <= args.max_curvature
                       and r["length"] >= args.min_length)
    rows.sort(key=lambda r: (not r["ok"], -(abs(r["amp"]) + 0.3 * r["length"]) + 0.002 * r["curvature"]))
    return rows


# -------------------------------------------------------------------- peg

def place_peg(cloud, support, m0, path_row, peg_cfg, args, centers=None, rods=None):
    """Where a peg touches the rod along the path only in a middle stretch.

    For every path point the nearest sample's rod shape stands for the rod
    there.  A candidate peg center (x, z) -- y fixed at the path's middle --
    gets, per point, the clearance of the mid node and the smallest clearance
    of any rod node (negative: the rod would overlap the peg by that much).
    Accepted: the mid node always at least ``--min-clearance`` away; the rod
    overlapping by at most ``--max-penetration`` over one stretch (one run of
    contiguous points) of ``--contact-range`` of the path that starts and ends
    inside it (free at both ends).  Returns the accepted candidates, best first (centered stretch,
    shallow overlap, large mid clearance).  ``centers`` is a list of
    ``(x, z, y)`` to test instead of the grid.  ``rods`` (n, nv, 3) are the
    rod shapes actually seen along the path (an executed free-space rollout):
    exact, where the nearest samples are only within ``--tol``; the overlap
    must then also be at least ``--min-penetration`` so that the contact is real.
    """
    if rods is None:
        n = args.check_points
        P = sine_path(m0, n, path_row["amp"], path_row["dx"], path_row["dy"])
        _, idx = support.nearest(P)
        rods = cloud["rod"][idx].astype(float)          # (n, nv, 3)
    rods = np.asarray(rods, dtype=float)
    n = len(rods)
    mids = rods[:, rods.shape[1] // 2, :]
    P = mids
    r, hl = float(peg_cfg["radius"]), float(peg_cfg["half_length_y"])
    cy_default = float(P[n // 2, 1])
    if centers is None:
        gx = np.arange(m0[0] - 0.25, m0[0] + 0.3, 0.005)
        gz = np.arange(m0[2] - 0.2, m0[2] + 0.25, 0.005)
        centers = [(x, z, cy_default) for x in gx for z in gz]
    lo_f, hi_f = args.contact_range
    edge = max(3, n // 10)
    out = []
    for cx, cz, cy in centers:
        mclr = axis_clearance(mids, cx, cy, cz, r, hl)
        if mclr.min() < args.min_clearance:
            continue
        rclr = axis_clearance(rods, cx, cy, cz, r, hl).min(axis=1)
        touch = rclr < 0.0
        frac = float(touch.mean())
        if not (lo_f <= frac <= hi_f):
            continue
        where = np.where(touch)[0]
        if where[-1] - where[0] + 1 != len(where):          # one stretch only
            continue
        if where[0] < edge or where[-1] > n - 1 - edge:     # free at both ends: on, then off again
            continue
        pen = float(-rclr.min())
        if pen > args.max_penetration or pen < args.min_penetration:
            continue
        centered = abs(0.5 * (where[0] + where[-1]) / (n - 1) - 0.5)
        score = centered + 5.0 * pen + 0.5 * abs(frac - 0.5 * (lo_f + hi_f)) - 2.0 * float(mclr.min())
        out.append({"center_xz": [float(cx), float(cz)], "center_y": float(cy),
                    "contact_fraction": frac, "penetration": pen,
                    "mid_clearance": float(mclr.min()), "score": float(score)})
    out.sort(key=lambda c: c["score"])
    return out


# --------------------------------------------------------------- execution

def execute(env, u_ready, state0, ref, bounds, k, peg, args):
    """Follow ``ref`` (N rows, row j = target of step j) with damped least squares."""
    dim = env.dim
    lo, hi = bounds
    S = np.ones(k)
    S[dim:] = 1.0 / args.w_rot
    h = np.array([2e-3] * dim + [2e-2] * (k - dim))
    env.set_state(state0)
    u = u_ready.copy()
    mids, touch, clr, errs, rods = [], [], [], [], []
    failed = 0
    G = None
    t_end = time.time() + args.exec_time_limit
    for j, tgt in enumerate(ref):
        if time.time() > t_end:                 # a contact state the solver crawls through
            print(f"    execution stopped at step {j}: over --exec-time-limit", flush=True)
            break
        if j % 25 == 0:
            print(f"    step {j}/{len(ref)}", flush=True)
        st = env.get_state()
        mid = mid_of(env, st["q"])
        e = tgt - mid
        if G is None or j % args.jac_every == 0:
            J = np.zeros((dim, k))
            for c in range(k):
                up = u.copy()
                up[k + c] += h[c]
                inf = stable_step(env, up)
                if inf["stable"]:
                    J[:, c] = (mid_of(env, inf["q"]) - mid) / h[c]
                env.set_state(st)
            G = J * S
            s_max = max(np.linalg.svd(G, compute_uv=False)[0], 1e-9)
        du = S * (G.T @ np.linalg.solve(G @ G.T + (args.damping * s_max) ** 2 * np.eye(dim), e))
        n = np.linalg.norm(du[:dim])
        if n > args.max_step:
            du *= args.max_step / n
        accepted = False
        for frac in (1.0, 0.5, 0.25):
            un = u.copy()
            un[k:2 * k] = np.clip(un[k:2 * k] + frac * du, lo, hi)
            info = stable_step(env, un)
            if info["stable"]:
                accepted, u = True, un
                break
            env.set_state(st)
        if not accepted:                       # hold the previous command
            failed += 1
            info = stable_step(env, u)
            if not info["stable"]:
                env.set_state(st)
                break
        q = info["q"]
        m = mid_of(env, q)
        t, c = rod_contact_summary(env, q, peg)
        mids.append(m)
        rods.append(np.asarray(env.positions(q)).reshape(-1, 3))
        touch.append(t)
        clr.append(c)
        errs.append(np.linalg.norm(m - tgt))
    n_done = len(mids)
    errs = np.array(errs)
    tch = np.array(touch, dtype=bool)
    res = {"steps_done": n_done, "steps": len(ref), "failed_steps": failed,
           "rms": float(np.sqrt(np.mean(errs ** 2))) if n_done else float("inf"),
           "max": float(errs.max()) if n_done else float("inf"),
           "final": float(errs[-1]) if n_done else float("inf"),
           "contact_fraction": float(tch.mean()) if n_done else 0.0,
           "contact_transitions": int(np.sum(tch[1:] != tch[:-1])) if n_done > 1 else 0,
           "max_penetration": float(max(0.0, -min(clr))) if n_done else 0.0}
    res["ok"] = bool(n_done == len(ref) and res["rms"] <= args.exec_tol)
    if peg is not None:                        # a peg scene must actually be touched
        res["ok"] = bool(res["ok"] and res["contact_fraction"] >= args.min_contact)
    return res, np.array(mids), tch, np.array(rods)


def run_execution(cfg, args, ref, peg_center=None, peg_y=None, peg_enabled=None):
    """Build the scene (peg moved to ``peg_center``), ready it, and track ``ref``."""
    cfg_x = with_peg(cfg, peg_center, peg_y,
                     enabled=bool(cfg["peg"]["enabled"]) if peg_enabled is None else peg_enabled)
    env = build_env(cfg_x)
    reset_env(env, cfg_x)
    k = end_control_dim(3, env_twist(cfg_x))
    bounds = workspace_bounds(cfg_x, 3, k)
    u_ready = ready_state(env, cfg_x, args.ready_dx)
    cap_solver(env)
    state0 = env.get_state()
    peg = find_peg(env)
    return execute(env, u_ready, state0, ref, bounds, k, peg, args)


# ------------------------------------------------------------------ shape

def with_pegs(cfg, centers):
    """A copy of ``cfg`` with pegs at ``centers`` ((x, z) each: the first is
    ``peg:``, the others ``peg.extra``, all with the block's center_y, radius
    and half length), or with none when ``centers`` is None."""
    c = copy.deepcopy(cfg)
    c["peg"]["enabled"] = centers is not None
    if centers is not None:
        cy = float(c["peg"]["center_y"])
        c["peg"]["center_xz"] = [float(v) for v in centers[0]]
        c["peg"]["extra"] = [{"center_xz": [float(v) for v in xz], "center_y": cy} for xz in centers[1:]]
    return c


def config_pegs(cfg):
    """The (x, z) centers of the config's pegs, ``peg:`` first, then ``peg.extra``."""
    p = cfg["peg"]
    return [tuple(p["center_xz"])] + [tuple(e["center_xz"]) for e in (p.get("extra") or [])]


def overlap_table(rods, centers, cy, r, hl, y_margin, edge, chunk=48):
    """How deep each rod shape reaches into a peg at each center.

    ``rods`` (S, nv, 3); ``centers`` (C, 2), the axes in x-z of y-axis
    cylinders of radius ``r`` over ``cy +- hl``.  Returns, each (C, S):
    ``pen`` the deepest overlap of any node (negative: the clearance),
    ``node`` that node, ``side`` whether it meets the cylinder's side (its y
    at least ``y_margin`` inside the axis range, not on an end cap), and
    ``ends`` the clearance of the ``edge`` nodes at either end.
    """
    rods = np.asarray(rods, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    S, nv, _ = rods.shape
    ay = np.abs(rods[..., 1] - cy)
    dy2 = np.maximum(ay - hl, 0.0) ** 2
    inside = ay <= hl - y_margin
    C = len(centers)
    pen = np.empty((C, S), np.float32)
    node = np.empty((C, S), np.int16)
    ends = np.empty((C, S), np.float32)
    for a in range(0, C, chunk):
        c = centers[a:a + chunk]
        clr = np.sqrt((rods[None, ..., 0] - c[:, 0, None, None]) ** 2
                      + (rods[None, ..., 2] - c[:, 1, None, None]) ** 2 + dy2[None]) - r
        i = clr.argmin(-1)
        node[a:a + chunk] = i
        pen[a:a + chunk] = -np.take_along_axis(clr, i[..., None], -1)[..., 0]
        ends[a:a + chunk] = np.minimum(clr[..., :edge].min(-1), clr[..., nv - edge:].min(-1))
    side = inside[np.arange(S)[None, :], node]
    return pen, node, side, ends


def place_peg_pair(rods, start_rod, build_rod, peg_cfg, args, centers=None):
    """Step P: two pegs that many of ``rods`` (S, nv, 3) touch, one per half of the rod.

    A shape fits a peg when its deepest node is interior (10 % of the rod
    from either end), on the cylinder's side, overlapping by
    ``--shape-penetration`` (MIN, MAX), with both ends clear.  The first peg
    of a pair is touched on the clamp-side half (at least ``--mid-gap`` of
    the rod below the mid node), the second on the gripper-side half.
    Candidates are the grid points clear of the start rod by
    ``--start-clearance`` (and of ``build_rod``, the rod build_env checks a
    peg against).  Returns the pairs, best first, each ``{"centers": [(x, z),
    (x, z)], "samples": indices of the shapes fitting both, "n": their
    number}``.  ``centers`` (``--keep-peg``) is the one pair to check.
    """
    r, hl, cy = float(peg_cfg["radius"]), float(peg_cfg["half_length_y"]), float(peg_cfg["center_y"])
    nv = rods.shape[1]
    mid = nv // 2
    gap = max(1, int(round(args.mid_gap * (nv - 1))))
    edge = max(2, int(round(0.1 * (nv - 1))))
    lo_p, hi_p = args.shape_penetration
    if centers is None:
        g = np.round(np.arange(-0.2, 1.0 + 1e-9, args.peg_grid), 6)   # the centers the yaml writes
        grid = np.array([(x, z) for x in g for z in g])
    else:
        grid = np.array(centers, dtype=float)
    tab = lambda R, G: overlap_table(R, G, cy, r, hl, 0.02, edge)
    start_clr = -tab(start_rod[None], grid)[0][:, 0]
    build_clr = -tab(build_rod[None], grid)[0][:, 0]
    clear = (start_clr >= args.start_clearance) & (build_clr > 0.0)
    if centers is not None and not clear.all():
        raise SystemExit(f"--keep-peg: the start rod is {np.round(start_clr * 1e3, 1).tolist()} mm from the "
                         f"pegs (needs >= {args.start_clearance * 1e3:.0f} mm), or a peg cuts the rod "
                         "build_env checks it against")
    grid = grid[clear]
    pen, node, side, ends = tab(rods, grid)
    fit = ((pen >= lo_p) & (pen <= hi_p) & side & (ends >= 0.0)
           & (node >= edge) & (node <= nv - 1 - edge))
    low, high = fit & (node <= mid - gap), fit & (node >= mid + gap)
    if centers is not None:
        both = fit[0] & fit[1] & (np.abs(node[0].astype(int) - node[1]) >= 2 * gap)
        return [{"centers": [tuple(c) for c in grid], "samples": np.where(both)[0], "n": int(both.sum())}]
    il, ih = np.where(low.any(1))[0], np.where(high.any(1))[0]
    print(f"peg grid: {clear.sum()} of {len(clear)} centers clear of the start rod; {len(il)} fit a "
          f"clamp-side contact, {len(ih)} a gripper-side one", flush=True)
    if len(il) == 0 or len(ih) == 0:
        return []
    counts = low[il].astype(np.float32) @ high[ih].T.astype(np.float32)
    dist = np.linalg.norm(grid[il][:, None, :] - grid[ih][None, :, :], axis=-1)
    counts[dist < 2 * r + args.peg_gap] = 0.0
    pairs = []
    for flat in np.argsort(counts, axis=None)[::-1]:
        a, b = np.unravel_index(flat, counts.shape)
        if counts[a, b] < 1 or len(pairs) >= args.max_pairs:
            break
        ca, cb = grid[il[a]], grid[ih[b]]
        if any(np.linalg.norm(ca - p["centers"][0]) < args.pair_sep
               and np.linalg.norm(cb - p["centers"][1]) < args.pair_sep for p in pairs):
            continue
        pairs.append({"centers": [tuple(ca), tuple(cb)],
                      "samples": np.where(low[il[a]] & high[ih[b]])[0], "n": int(counts[a, b])})
    return pairs


def farthest(X, n, first):
    """``n`` row indices of ``X`` chosen far apart, starting at ``first``."""
    X = X.reshape(len(X), -1)
    pick = [int(first)]
    d = np.linalg.norm(X - X[first], axis=1)
    while len(pick) < min(n, len(X)):
        j = int(np.argmax(d))
        if d[j] <= 0.0:
            break
        pick.append(j)
        d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
    return pick


def verify_move(job):
    """Step V for one straight move u0 -> u0 + du: rolled out free, then with the pegs."""
    cfg, du, centers = job
    from train_shape import build_target        # the rollout train_shape.py builds its targets with
    out = {"du": du}
    for label, c in (("free", with_pegs(cfg, None)), ("pegs", with_pegs(cfg, centers))):
        env = build_env(c)
        reset_env(env, c)
        settle_initial(env, c)
        try:
            rec, arrs = build_target(env, env.get_state(), initial_control(c), "candidate", "",
                                     lambda l: l * du)
        except RuntimeError as e:
            out[label] = {"error": str(e)}
            break
        out[label] = {"rec": rec, "shapes": arrs["shapes"].astype(np.float32), "U_ref": arrs["U_ref"],
                      "shape": arrs["shape"]}
    return out


def judge_move(res, start_rod, centers, peg_cfg, args):
    """Step V's rules for one verified move: ``(ok, info)``, ``info["reasons"]`` what failed."""
    r, hl, cy = float(peg_cfg["radius"]), float(peg_cfg["half_length_y"]), float(peg_cfg["center_y"])
    lo_p, hi_p = args.shape_penetration
    info = {"du": np.round(res["du"], 4).tolist(), "reasons": []}
    for label in ("free", "pegs"):
        if label not in res or "error" in res[label]:
            info["reasons"].append(f"{label} rollout: {res.get(label, {}).get('error', 'not run')}")
            return False, info
    N = len(res["pegs"]["shapes"])
    first = int(np.ceil(args.free_start * N))
    free = np.concatenate([start_rod[None], res["free"]["shapes"]])
    cont = np.concatenate([start_rod[None], res["pegs"]["shapes"]])
    rec = res["pegs"]["rec"]
    info.update(snap_steps=rec["snap_steps"], max_substeps=rec["max_substeps"],
                newton_iterations=rec["newton_iterations"], rollout_time=rec["rollout_time"],
                change=float(np.sqrt(np.mean(np.sum((cont[-1] - start_rod) ** 2, axis=1)))),
                peg_effect=float(np.sqrt(np.mean(np.sum((cont[-1] - free[-1]) ** 2, axis=1)))),
                pegs=[])
    if rec["snap_steps"] and not args.allow_snap:
        info["reasons"].append(f"snaps at steps {rec['snap_steps']} with the pegs")
    for j, (cx, cz) in enumerate(centers):
        cf = axis_clearance(free, cx, cy, cz, r, hl).min(axis=1)
        cp = axis_clearance(cont, cx, cy, cz, r, hl).min(axis=1)
        on = np.where(cp >= 0.0)[0]
        onset = int(on[-1] + 1) if len(on) else 0           # touching from this step to the end
        clr_end = axis_clearance(cont[-1], cx, cy, cz, r, hl)
        info["pegs"].append({"free_final_overlap": float(-cf[-1]), "free_max_overlap": float(-cf.min()),
                             "final_penetration": float(-cp[-1]), "contact_from_step": onset,
                             "contact_node": int(np.argmin(clr_end)),
                             "contact_force": float(peg_cfg["k_contact"] * np.sum(np.maximum(-clr_end, 0.0)))})
        tag = f"peg {j + 1}"
        if -cf.min() > hi_p:
            info["reasons"].append(f"{tag}: the free rod overlaps it by {-cf.min() * 1e3:.0f} mm on the way")
        if -cf[-1] < lo_p:
            info["reasons"].append(f"{tag}: the free rod ends only {-cf[-1] * 1e3:.1f} mm into it")
        if np.any(cf[:first + 1] <= 0.0) or np.any(cp[:first + 1] <= 0.0):
            info["reasons"].append(f"{tag}: touched in the first {args.free_start * 100:.0f} % of the path")
        if cp[-1] >= 0.0:
            info["reasons"].append(f"{tag}: not touched at the end with the pegs in place")
    return not info["reasons"], info


def run_shape(cfg, args, cloud, start_rod, build_rod, out):
    """Steps P and V; writes targets/, suggested_target.yaml and the figure; returns the report part."""
    peg_cfg = dict(cfg["peg"])
    rods = cloud["rod"].astype(np.float32)
    L = float(cfg["rod"]["rod_length"])
    change = np.sqrt(np.mean(np.sum((rods - start_rod[None]) ** 2, axis=2), axis=1))
    stretch = np.sum(np.linalg.norm(np.diff(rods, axis=1), axis=2), axis=1) / L - 1.0
    elig = np.where((change >= args.min_change) & (stretch <= args.max_stretch))[0]
    rng = np.random.default_rng(args.seed)
    if len(elig) > args.pair_samples:
        elig = np.sort(rng.choice(elig, args.pair_samples, replace=False))
    print(f"{len(elig)} sampled shapes eligible (>= {args.min_change * 1e3:.0f} mm rms from the start, "
          f"stretch <= {args.max_stretch * 100:g} %)", flush=True)
    t0 = time.time()
    pairs = place_peg_pair(rods[elig], start_rod, build_rod, peg_cfg, args,
                           centers=config_pegs(cfg) if args.keep_peg else None)
    print(f"{len(pairs)} peg pair(s) ({time.time() - t0:.0f} s)", flush=True)
    rep = {"eligible_samples": int(len(elig)), "pairs": []}
    k = end_control_dim(3, env_twist(cfg))
    u0 = initial_control(cfg)[k:]
    best = None
    for p in pairs:
        cen = p["centers"]
        print(f"pegs at (x, z) = {np.round(cen[0], 3).tolist()} and {np.round(cen[1], 3).tolist()}: "
              f"{p['n']} sampled shapes touch both", flush=True)
        if p["n"] == 0:
            continue
        idx = elig[p["samples"]]
        cand = idx[farthest(rods[idx], args.candidates, int(np.argmax(change[idx])))]
        # du rounded as suggested_target.yaml writes it, so the config rebuilds these targets exactly
        jobs = [(cfg, np.round(cloud["u"][i] - u0, 4), cen) for i in cand]
        t0 = time.time()
        if args.workers > 1:
            with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
                results = pool.map(verify_move, jobs)
        else:
            results = [verify_move(j) for j in jobs]
        judged = [judge_move(res, start_rod, cen, peg_cfg, args) for res in results]
        good = [i for i, (ok, _) in enumerate(judged) if ok]
        for i, (ok, inf) in enumerate(judged):
            pg = inf.get("pegs") or []
            print(f"    du {inf['du']}: {'OK' if ok else 'rejected'}"
                  + (f", change {inf['change'] * 1e3:.0f} mm, pegs move it {inf['peg_effect'] * 1e3:.0f} mm, "
                     f"contact from steps {[q['contact_from_step'] for q in pg]} at nodes "
                     f"{[q['contact_node'] for q in pg]}" if pg else "")
                  + ("" if ok else f" -- {'; '.join(inf['reasons'])}"), flush=True)
        print(f"    {len(good)} of {len(jobs)} moves verified ({time.time() - t0:.0f} s)", flush=True)
        rep["pairs"].append({"centers": [list(map(float, c)) for c in cen], "n_samples": p["n"],
                             "moves": [inf for _, inf in judged], "verified": len(good)})
        if best is None or len(good) > len(best[1]):
            best = (cen, good, results, judged)
        if len(good) >= args.n_targets:
            break
    if best is None or not best[1]:
        print("no peg pair with a verified target", flush=True)
        return rep
    cen, good, results, judged = best
    finals = np.array([results[i]["pegs"]["shape"] for i in good])
    first = int(np.argmax([judged[i][1]["change"] for i in good]))
    chosen = [good[j] for j in farthest(finals, args.n_targets, first)]
    tdir = out / "targets"
    tdir.mkdir(parents=True, exist_ok=True)
    targets = []
    for n, i in enumerate(chosen, 1):
        name, res, inf = f"P{n}", results[i], judged[i][1]
        du = np.asarray(res["du"], float)
        desc = (f"straight move du = {np.round(du, 3).tolist()} onto both pegs (touching from steps "
                f"{[q['contact_from_step'] for q in inf['pegs']]}, nodes {[q['contact_node'] for q in inf['pegs']]})")
        rec = dict(res["pegs"]["rec"], name=name, description=desc, pegs=inf["pegs"],
                   peg_centers_xz=[list(map(float, c)) for c in cen])
        np.savez_compressed(tdir / f"{name}.npz", shape=res["pegs"]["shape"], U_ref=res["pegs"]["U_ref"],
                            shapes=res["pegs"]["shapes"], free_shape=res["free"]["shape"])
        (tdir / f"{name}.json").write_text(json.dumps(rec, indent=1, default=float))
        targets.append({"name": name, "du": du, "description": desc, "shape": res["pegs"]["shape"],
                        "free_shape": res["free"]["shape"], **inf})
        print(f"[{name}] {desc}; rms change {inf['change'] * 1e3:.0f} mm (the pegs account for "
              f"{inf['peg_effect'] * 1e3:.0f} mm), contact forces "
              f"{[round(q['contact_force'], 2) for q in inf['pegs']]} N", flush=True)
    (out / "suggested_target.yaml").write_text(snippet_shape(cen, peg_cfg, targets))
    shape_figure(out / "target_check.png", start_rod, cen, peg_cfg, targets)
    rep["chosen"] = {"centers": [list(map(float, c)) for c in cen],
                     "targets": [{key: v for key, v in t.items() if key not in ("shape", "free_shape")}
                                 for t in targets]}
    return rep


def snippet_shape(centers, peg_cfg, targets):
    cy = float(peg_cfg["center_y"])
    s = (f"peg:\n  enabled: true\n  center_xz: [{centers[0][0]:.3f}, {centers[0][1]:.3f}]\n"
         f"  center_y: {cy:.3f}\n  extra:\n")
    for c in centers[1:]:
        s += f"    - {{center_xz: [{c[0]:.3f}, {c[1]:.3f}], center_y: {cy:.3f}}}\n"
    s += "task:\n  shape_targets:\n"
    for t in targets:
        s += (f"    {t['name']}:\n      du: [{', '.join(f'{v:.4f}' for v in t['du'])}]\n"
              f"      description: \"{t['description']}\"\n")
    return s


def shape_figure(path, start_rod, centers, peg_cfg, targets):
    """x-z and x-y projections and a 3D view: the start rod, the pegs, each
    target (solid) and the same move without the pegs (dashed)."""
    r, hl, cy = float(peg_cfg["radius"]), float(peg_cfg["half_length_y"]), float(peg_cfg["center_y"])
    fig = plt.figure(figsize=(17, 5.4))
    axs = [fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3, projection="3d")]
    colors = plt.cm.tab10(np.arange(len(targets)))
    for ax, (a, b, xl, yl) in zip(axs[:2], [(0, 2, "x [m]", "z [m]"), (0, 1, "x [m]", "y [m]")]):
        for j, (cx, cz) in enumerate(centers):
            if b == 2:
                ax.add_patch(plt.Circle((cx, cz), r, fc="steelblue", alpha=0.45,
                                        label="pegs" if j == 0 else None))
            else:
                ax.add_patch(plt.Rectangle((cx - r, cy - hl), 2 * r, 2 * hl, fc="steelblue", alpha=0.3))
        ax.plot(start_rod[:, a], start_rod[:, b], color="0.45", lw=2.2, label="start")
        for t, c in zip(targets, colors):
            ax.plot(t["free_shape"][:, a], t["free_shape"][:, b], "--", color=c, lw=1.0)
            ax.plot(t["shape"][:, a], t["shape"][:, b], "-o", color=c, lw=1.8, ms=2.5, label=t["name"])
        ax.plot(0, 0, "ks", ms=6)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8, loc="best")
    ax = axs[2]
    th = np.linspace(0, 2 * np.pi, 32)
    for cx, cz in centers:
        T, Y = np.meshgrid(th, [cy - hl, cy + hl])
        ax.plot_surface(cx + r * np.cos(T), Y, cz + r * np.sin(T), color="steelblue", alpha=0.35, linewidth=0)
    ax.plot(*start_rod.T, color="0.45", lw=2.2)
    for t, c in zip(targets, colors):
        ax.plot(*t["shape"].T, color=c, lw=1.8)
    P = np.vstack([start_rod] + [t["shape"] for t in targets])
    lo, hi = P.min(0) - 0.08, P.max(0) + 0.08
    c, h = 0.5 * (lo + hi), 0.5 * (hi - lo).max()
    ax.set_xlim(c[0] - h, c[0] + h)
    ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_zlim(c[2] - h, c[2] + h)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(18, -60)
    fig.suptitle("shape targets: solid = with both pegs (the target), dashed = the same move without pegs",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ output

def snippet_path(c, peg=None):
    s = ("task:\n  tracking:\n    trajectory_type: sine_xz\n    sine_xz:\n"
         f"      amp: {c['amp']:.3f}\n      periods: {c['periods']}\n"
         f"      dx: {c['dx']:.3f}\n      dy: {c['dy']:.3f}\n      bow: 0.0\n")
    if peg is not None:
        s += (f"peg:\n  center_xz: [{peg['center_xz'][0]:.3f}, {peg['center_xz'][1]:.3f}]\n"
              f"  center_y: {peg['center_y']:.3f}\n")
    return s


def make_figure(path, cloud, peg_geom, m0, ref, mids, touch, title):
    """Three projections: the cloud, the peg, the target path and the executed mid node."""
    M = cloud["mid"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    views = [(0, 2, "x [m]", "z [m]"), (0, 1, "x [m]", "y [m]"), (1, 2, "y [m]", "z [m]")]
    for ax, (a, b, xl, yl) in zip(axs, views):
        ax.scatter(M[:, a], M[:, b], s=2, c="0.75", label="reachable mid node")
        if peg_geom is not None:
            cx, cy, cz, r, hl = peg_geom
            c = np.array([cx, cy, cz])
            if (a, b) == (0, 2):
                ax.add_patch(plt.Circle((c[0], c[2]), r, fc="steelblue", alpha=0.45, label="peg"))
            elif (a, b) == (0, 1):
                ax.add_patch(plt.Rectangle((c[0] - r, c[1] - hl), 2 * r, 2 * hl, fc="steelblue", alpha=0.45))
            else:
                ax.add_patch(plt.Rectangle((c[1] - hl, c[2] - r), 2 * hl, 2 * r, fc="steelblue", alpha=0.45))
        if ref is not None:
            R = np.vstack([m0, ref])
            ax.plot(R[:, a], R[:, b], "b--", lw=2, label="target path")
        if mids is not None and len(mids):
            ax.plot(mids[:, a], mids[:, b], "r-", lw=1.4, label="executed mid node")
            if touch is not None and np.any(touch):
                ax.plot(mids[touch, a], mids[touch, b], "k.", ms=4, label="rod on peg")
        ax.plot(m0[a], m0[b], "go", ms=7, label="start")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="check a path / point target against sampled reachability")
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=("path", "p2p", "shape"), default="path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target", type=float, nargs=3, default=None, help="p2p target (default: the config's)")
    ap.add_argument("--keep-peg", action="store_true",
                    help="path / shape mode: check the config's peg(s) instead of placing them")
    ap.add_argument("--load-cloud", default=None, help="reuse the cloud.npz of an earlier run instead of sampling")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--samples", type=int, default=3000, help="most attempts per worker")
    ap.add_argument("--time-limit", type=float, default=240.0, help="[s] sampling time per worker")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ready-dx", type=float, default=0.01)
    ap.add_argument("--step-pos", type=float, default=0.02, help="[m] sampling perturbation of the gripper position")
    ap.add_argument("--step-ang", type=float, default=0.06, help="[rad] sampling perturbation of theta, phi")
    ap.add_argument("--tol", type=float, default=0.015, help="[m] a target point is supported by a sample within this distance")
    ap.add_argument("--min-support", type=float, default=0.97, help="fraction of path points that must be supported")
    ap.add_argument("--max-curvature", type=float, default=25.0, help="[1/m]")
    ap.add_argument("--min-length", type=float, default=0.10, help="[m] shortest path worth tracking")
    ap.add_argument("--min-clearance", type=float, default=0.015, help="[m] mid node distance from the peg surface")
    ap.add_argument("--min-penetration", type=float, default=0.003, help="[m] with exact rod shapes, the overlap that makes the contact real")
    ap.add_argument("--max-penetration", type=float, default=0.006, help="[m] how far the free rod shape may overlap the peg")
    ap.add_argument("--contact-range", type=float, nargs=2, default=(0.2, 0.6),
                    help="share of the path the rod should be on the peg")
    ap.add_argument("--check-points", type=int, default=60)
    ap.add_argument("--no-execute", action="store_true")
    ap.add_argument("--max-exec", type=int, default=6, help="(path, peg) pairs tried, best first")
    ap.add_argument("--exec-tol", type=float, default=0.005, help="[m] rms error that counts as tracked")
    ap.add_argument("--min-contact", type=float, default=0.10, help="share of the executed steps the rod must be on the peg")
    ap.add_argument("--w-rot", type=float, default=0.2)
    ap.add_argument("--damping", type=float, default=0.03)
    ap.add_argument("--max-step", type=float, default=0.03)
    ap.add_argument("--jac-every", type=int, default=4)
    ap.add_argument("--exec-time-limit", type=float, default=300.0, help="[s] per executed candidate")
    sh = ap.add_argument_group("shape mode")
    sh.add_argument("--n-targets", type=int, default=3)
    sh.add_argument("--candidates", type=int, default=12, help="moves verified per peg pair")
    sh.add_argument("--max-pairs", type=int, default=4, help="peg pairs tried, best first")
    sh.add_argument("--shape-penetration", type=float, nargs=2, default=(0.005, 0.02),
                    help="[m] MIN MAX overlap of the free rod's end shape with each peg")
    sh.add_argument("--start-clearance", type=float, default=0.03, help="[m] start rod to either peg")
    sh.add_argument("--free-start", type=float, default=0.3,
                    help="share of the path, from the start, in which no peg is touched")
    sh.add_argument("--mid-gap", type=float, default=0.1,
                    help="share of the rod either side of the mid node neither contact may use")
    sh.add_argument("--peg-grid", type=float, default=0.01, help="[m] peg center grid in x-z")
    sh.add_argument("--peg-gap", type=float, default=0.03, help="[m] least gap between the two pegs")
    sh.add_argument("--pair-sep", type=float, default=0.05,
                    help="[m] a pair this close to a better one (both pegs) is not tried")
    sh.add_argument("--min-change", type=float, default=0.05, help="[m] rms shape change from the start")
    sh.add_argument("--max-stretch", type=float, default=0.01, help="largest rod stretch of a sampled shape")
    sh.add_argument("--pair-samples", type=int, default=6000, help="sampled shapes used to place the pegs")
    sh.add_argument("--allow-snap", action="store_true", help="admit a target rollout with a snap")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if env_dim(cfg) != 3:
        raise SystemExit("check_targets.py handles the 3D scenes (dim: 3)")
    out = Path(args.out) if args.out else _ROOT / "experiments" / "target_check" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    peg_cfg = dict(cfg["peg"])
    peg_on = bool(peg_cfg.get("enabled", False))
    shape = args.mode == "shape"

    # the scene as configured: start pose, workspace, step count (shape: without its pegs,
    # which are placed or checked later; the start is the settled u0 of train_shape.py)
    env = build_env(with_peg(cfg, enabled=False) if shape else cfg)
    build_rod = np.asarray(env.nodes, dtype=float).reshape(-1, 3).copy()
    reset_env(env, cfg)
    k = end_control_dim(3, env_twist(cfg))
    if workspace_bounds(cfg, 3, k) is None:
        raise SystemExit("the scene has no workspace; enable workspace in the config")
    start_state(env, cfg, None if shape else args.ready_dx)
    cap_solver(env)
    m0 = mid_of(env, env.get_state()["q"])
    start_rod = np.asarray(env.positions(env.get_state()["q"]), dtype=float).reshape(-1, 3)
    N = env.Nsteps
    print(f"start: mid node {np.round(m0, 4)}, N = {N} steps", flush=True)

    free_sampling = args.mode in ("path", "shape")   # free-space cloud, the peg(s) placed afterwards
    cfg_s = with_peg(cfg, enabled=False) if (free_sampling and (peg_on or shape)) else cfg
    if args.load_cloud:
        cloud = dict(np.load(args.load_cloud))
        print(f"loaded {len(cloud['mid'])} samples from {args.load_cloud}", flush=True)
    else:
        cloud = sample_cloud(cfg_s, args)
    support = Support(cloud)
    region = region_summary(cloud)
    print(f"reachable mid-node region (5-95 %): x {region['p05'][0]:.3f}..{region['p95'][0]:.3f}, "
          f"y {region['p05'][1]:.3f}..{region['p95'][1]:.3f}, z {region['p05'][2]:.3f}..{region['p95'][2]:.3f}",
          flush=True)
    np.savez_compressed(out / "cloud.npz", mid=cloud["mid"], u=cloud["u"], touch=cloud["touch"], rod=cloud["rod"])
    report = {"config": str(args.config), "mode": args.mode, "m0": m0.tolist(), "region": region,
              "args": vars(args)}

    ref = chosen = chosen_peg = executed = mids = touch = None
    peg_geom = None
    if shape:
        report["start_rod"] = start_rod.tolist()
        report.update(run_shape(cfg, args, cloud, start_rod, build_rod, out))
    elif args.mode == "path":
        rows = score_paths(support, m0, args)
        good = [r for r in rows if r["ok"]]
        print(f"{len(rows)} candidate paths, {len(good)} supported (>= {args.min_support * 100:.0f} % of points within "
              f"{args.tol * 1e3:.0f} mm of a sample, curvature <= {args.max_curvature:g} 1/m, length >= "
              f"{args.min_length} m)", flush=True)
        report["n_candidates"], report["n_ok"], report["top"] = len(rows), len(good), rows[:10]
        # Candidate by candidate: track the path on the FREE rod first (that shows it is
        # reachable, and records the rod's exact shape along it), place the peg on those
        # shapes, then track the path again with the peg and contact physics on.
        tried = []
        for r in good[:args.max_exec]:
            ref_c = sine_path(m0, N, r["amp"], r["dx"], r["dy"])
            name = f"amp {r['amp']:+.2f} dx {r['dx']:+.3f} dy {r['dy']:+.2f}"
            if args.no_execute:
                if chosen is None:
                    pegs = place_peg(cloud, support, m0, r, peg_cfg, args) if peg_on else []
                    chosen, chosen_peg, ref = r, (pegs[0] if pegs else None), ref_c
                continue
            t0 = time.time()
            free, mids_f, _, rods_f = run_execution(cfg, args, ref_c, peg_enabled=False)
            print(f"{name} (curvature {r['curvature']:.1f} 1/m, length {r['length']:.2f} m), free rod: "
                  f"rms {free['rms'] * 1e3:.1f} mm, max {free['max'] * 1e3:.1f} mm ({time.time() - t0:.0f} s)", flush=True)
            if not free["ok"] and not (peg_on and free["rms"] <= args.exec_tol and free["steps_done"] == free["steps"]):
                tried.append({**r, "free": free, "peg": None})
                continue
            if not peg_on:
                tried.append({**r, "free": free, "peg": None, "execution": free})
                chosen, ref, executed, mids, touch = r, ref_c, free, mids_f, np.zeros(len(mids_f), bool)
                break
            pegs = place_peg(cloud, support, m0, r, peg_cfg, args, rods=rods_f)
            if not pegs:
                print("    no peg position touches the rod in a middle stretch only", flush=True)
                tried.append({**r, "free": free, "peg": None})
                continue
            p = pegs[0]
            print(f"    peg at x {p['center_xz'][0]:.3f}, z {p['center_xz'][1]:.3f}, y {p['center_y']:.3f}: on peg "
                  f"{p['contact_fraction'] * 100:.0f} %, overlap {p['penetration'] * 1e3:.1f} mm, mid clearance "
                  f"{p['mid_clearance'] * 1e3:.0f} mm", flush=True)
            t0 = time.time()
            res, mids_c, touch_c, _ = run_execution(cfg, args, ref_c, p["center_xz"], p["center_y"])
            print(f"    with the peg: rms {res['rms'] * 1e3:.1f} mm, max {res['max'] * 1e3:.1f} mm, on peg "
                  f"{res['contact_fraction'] * 100:.0f} % ({res['contact_transitions']} transitions), failed steps "
                  f"{res['failed_steps']}, {res['steps_done']}/{res['steps']} steps ({time.time() - t0:.0f} s)", flush=True)
            tried.append({**r, "free": free, "peg": p, "execution": res})
            if chosen is None or (res["ok"] and not executed["ok"]) or (
                    res["ok"] == executed["ok"] and res["rms"] < executed["rms"]):
                chosen, chosen_peg, ref, executed, mids, touch = r, p, ref_c, res, mids_c, touch_c
            if res["ok"]:
                break
        report["tried"] = tried
        if peg_on:
            pc = chosen_peg if chosen_peg is not None else {
                "center_xz": peg_cfg["center_xz"], "center_y": peg_cfg["center_y"]}
            peg_geom = (pc["center_xz"][0], pc["center_y"], pc["center_xz"][1],
                        peg_cfg["radius"], peg_cfg["half_length_y"])
        if chosen is None:
            print("nothing to execute", flush=True)
            make_figure(out / "target_check.png", cloud, peg_geom, m0, None, None, None,
                        "no supported path / peg combination")
        else:
            report["chosen"] = {**chosen, "peg": chosen_peg, "execution": executed}
            (out / "suggested_target.yaml").write_text(snippet_path(chosen, chosen_peg))
            title = (f"path amp {chosen['amp']:+.2f} dx {chosen['dx']:+.3f} dy {chosen['dy']:+.2f}"
                     + (f" | executed rms {executed['rms'] * 1e3:.1f} mm, on peg "
                        f"{executed['contact_fraction'] * 100:.0f} %" if executed else " | not executed"))
            make_figure(out / "target_check.png", cloud, peg_geom, m0, ref, mids, touch, title)
    else:
        tgt = np.array(args.target if args.target is not None else cfg.task["target_xyz"], float)
        d, i = support.nearest(tgt[None, :])
        peg_s = find_peg(env)
        clr = float(peg_clearance(peg_s, tgt)[0])
        ok = bool(d[0] <= args.tol and clr >= args.min_clearance)
        print(f"target {np.round(tgt, 4)}: nearest sample {d[0] * 1e3:.1f} mm away, peg clearance "
              f"{clr * 1e3:.1f} mm -> {'reachable' if ok else 'NOT reachable / too close to the peg'}", flush=True)
        report["target"] = {"xyz": tgt.tolist(), "nearest_sample_mm": float(d[0] * 1e3),
                            "peg_clearance_mm": clr * 1e3, "ok": ok}
        goal = tgt
        if not ok:
            far = peg_clearance(peg_s, cloud["mid"]) >= args.min_clearance
            good = np.where(far)[0]
            j = good[np.argmin(np.linalg.norm(cloud["mid"][good] - tgt, axis=1))]
            goal = cloud["mid"][j]
            print(f"nearest reachable point clear of the peg: {np.round(goal, 4)} "
                  f"({np.linalg.norm(goal - tgt) * 1e3:.0f} mm from the target)", flush=True)
            report["suggested_target"] = goal.tolist()
        ref = m0 + (np.arange(1, N + 1) / N)[:, None] * (goal - m0)
        if not args.no_execute:
            t0 = time.time()
            executed, mids, touch, _ = run_execution(cfg, args, ref)
            print(f"executed the straight path to {np.round(goal, 4)}: final error {executed['final'] * 1e3:.1f} mm, "
                  f"rms {executed['rms'] * 1e3:.1f} mm, on peg {executed['contact_fraction'] * 100:.0f} %, "
                  f"{executed['steps_done']}/{executed['steps']} steps ({time.time() - t0:.0f} s)", flush=True)
            report["execution"] = executed
        (out / "suggested_target.yaml").write_text(
            f"task:\n  target_xyz: [{goal[0]:.4f}, {goal[1]:.4f}, {goal[2]:.4f}]\n")
        peg_geom = None if peg_s is None else (peg_s.cx, peg_s.cy, peg_s.cz, peg_s.radius, peg_s.half_length)
        make_figure(out / "target_check.png", cloud, peg_geom, m0, ref, mids, touch,
                    f"p2p target {np.round(goal, 3)}")

    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
