# -*- coding: utf-8 -*-
"""Redundancy audit for the expensive inner-loop computations.

Why this exists
---------------
A training run spends nearly all of its time inside a handful of numerical
kernels: the equilibrium Hessian assembly, the Newton solves that call it, and
the adjoint's linear solves.  When a run feels too slow, the first question is
not "which kernel is slow" (a profiler answers that) but **"how much of that
work was already done before"** -- either literally the same input (a genuine
recomputation that a cache would remove) or an input so close to an earlier one
that the result could plausibly be reused or warm-started.

This module answers exactly that.  It wraps the hot functions, and for every
call it records

  * the *input* it was given (the thing that decides the result),
  * how long the call took, exclusive of instrumented calls nested inside it,
  * where in training it happened (epoch / RHC segment, rollout step, Newton
    iteration).

Then, for each call, it finds the **nearest earlier input of the same kind** and
reports the distribution of that distance.  Exact repeats are detected exactly
(hash of the raw bytes); near repeats are detected with random-projection LSH
plus a Johnson-Lindenstrauss sketch, so the cost per call is O(1) in the number
of calls already seen instead of O(n).

What the report tells you
-------------------------
Per kernel::

    kind                       calls     self_s   %self   exact    <=1e-8   <=1e-4   <=1e-2
    energy.force_and_jacobian  91_400    612.3    58.1%   47.9%     48.0%    52.7%    73.1%

Read that as: 47.9% of the Hessian assemblies were called on an input *byte-for
byte identical* to one already seen -- pure waste, removable with a one-entry
cache -- and 73.1% were within 1e-2 relative distance of an earlier input, which
is the budget available to an approximate/reuse scheme (freezing the Jacobian
across Newton iterations, reusing it across epochs, and so on).

The per-epoch table answers the "is every epoch redoing the previous epoch"
question directly: it pairs each rollout step with the same step of the previous
epoch and reports how far apart their inputs actually are.

Usage
-----
    from compute_audit import ComputeAudit

    audit = ComputeAudit(enabled=True)
    audit.install(env)                       # wraps the kernels in place
    ...
    with audit.context(epoch=ep):
        rollout(...)                         # env.step tags itself with `step`
    ...
    print(audit.text_report())
    audit.save(run_dir)

Everything is off unless ``enabled=True``; with it off, ``install`` does nothing
and there is no wrapper in the hot path at all.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from contextlib import contextmanager
from hashlib import blake2b
from time import perf_counter

import numpy as np

#: Relative-distance ladder the report buckets nearest-neighbour distances into.
DEFAULT_TOLERANCES = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)

#: Inputs longer than this are compared through a fixed random coordinate
#: subset.  The exact-duplicate hash always uses the full input, so this only
#: affects the *near*-duplicate distance, and 1024 coordinates estimate a
#: relative distance to well within the tolerance ladder's resolution.
MAX_KEY_DIM = 1024


class _Kind:
    """Per-kernel accumulator: counters, LSH tables, sketches, per-call rows."""

    def __init__(self, name, audit, detail=True, near=True):
        self.name = name
        self.audit = audit
        self.detail = detail            # keep one row per call
        self.near = near                # run near-duplicate search
        self.calls = 0
        self.total_time = 0.0
        self.self_time = 0.0
        self.exact_dups = 0
        self.exact_dup_time = 0.0
        self.key_dim = None

        self._seen = {}                 # digest -> first call index
        self._tables = None             # list of dict: bucket key -> [ids]
        self._proj = None               # LSH projections (L*k, d)
        self._offsets = None
        self._sketch_proj = None        # JL projection (sketch_dim, d)
        # Sketches live in one doubling buffer rather than a list: the nearest
        # neighbour search indexes it on every call, and restacking a list each
        # time would make the audit itself O(n^2).
        self._sk = np.zeros((1024, audit.sketch_dim), dtype=np.float32)
        self._snorm = np.zeros(1024, dtype=np.float32)
        self._n = 0                     # sketches stored
        self._scale = None              # frozen input scale
        self._calib = []                # norms collected before freezing

        # per-call rows (only when detail)
        self.t_self = []
        self.t_total = []
        self.dist = []                  # nearest earlier input, relative
        self.sk_idx = []                # row of this call's input in the index
        self.ctx_epoch = []
        self.ctx_step = []
        self.ctx_seg = []

    # ---------- key preparation ----------

    def _prepare(self, key):
        v = np.asarray(key, dtype=np.float64).ravel()
        digest = blake2b(v.tobytes(), digest_size=16).digest()
        if self.key_dim is None:
            self.key_dim = int(v.size)
        if v.size > MAX_KEY_DIM:
            if self._sub is None:
                rng = np.random.default_rng(abs(hash(self.name)) % (2 ** 31))
                self._sub = np.sort(rng.choice(v.size, MAX_KEY_DIM, replace=False))
            v = v[self._sub]
        return v, digest

    _sub = None

    def _build_projections(self, d):
        rng = np.random.default_rng(abs(hash(self.name)) % (2 ** 31) + 7)
        L, k = self.audit.lsh_tables, self.audit.lsh_bits
        # Standard E2LSH: unit-variance entries, so a difference of the inputs
        # of size r shows up in each projection with standard deviation r.  Two
        # inputs closer than `lsh_width` therefore almost always share a cell,
        # and distant ones almost never do.
        self._proj = rng.standard_normal((L * k, d))
        self._offsets = rng.uniform(0.0, self.audit.lsh_width, size=L * k)
        self._tables = [defaultdict(list) for _ in range(L)]
        self._sketch_proj = rng.standard_normal((self.audit.sketch_dim, d)) / np.sqrt(
            self.audit.sketch_dim)

    # ---------- the per-call hook ----------

    def add(self, key, t_total, t_self, ctx):
        idx = self.calls
        self.calls += 1
        self.total_time += t_total
        self.self_time += t_self

        v, digest = self._prepare(key)

        # An exact repeat is at distance 0 by definition, and the input it
        # repeats is already in the index -- re-indexing it would only make the
        # buckets heavier, so it reuses the first occurrence's row.
        first = self._seen.get(digest)
        if first is None:
            d_near, sk = self._nearest(v) if self.near else (float("nan"), -1)
            self._seen[digest] = sk
        else:
            self.exact_dups += 1
            self.exact_dup_time += t_self
            d_near, sk = 0.0, first

        if self.detail:
            self.t_self.append(t_self)
            self.t_total.append(t_total)
            self.dist.append(d_near)
            self.sk_idx.append(sk)
            self.ctx_epoch.append(ctx[0])
            self.ctx_step.append(ctx[1])
            self.ctx_seg.append(ctx[2])

    def _nearest(self, v):
        """``(relative distance to the nearest earlier input, this input's row)``.

        The row is -1 once the index is full; the call is still counted and
        timed, it just stops taking part in the distance search.
        """
        if self._proj is None:
            self._build_projections(v.size)

        # Freeze a scale from the first few inputs so "relative" means the same
        # thing for every call; without it an input that drifts in magnitude
        # would silently change what a tolerance refers to.
        n = float(np.linalg.norm(v))
        if self._scale is None:
            self._calib.append(n)
            if len(self._calib) >= self.audit.calibration:
                med = float(np.median(self._calib))
                self._scale = med if med > 0.0 else 1.0
            ref = float(np.median(self._calib)) if self._calib else 1.0
            scale = ref if ref > 0.0 else 1.0
        else:
            scale = self._scale

        vn = v / scale
        sk = (self._sketch_proj @ vn).astype(np.float32)
        snorm = float(np.linalg.norm(vn))

        k = self.audit.lsh_bits
        proj = self._proj @ vn
        buckets = np.floor((proj + self._offsets) / self.audit.lsh_width).astype(np.int64)
        keys = [buckets[t * k:(t + 1) * k].tobytes() for t in range(len(self._tables))]

        best = float("inf")
        if self._n:
            cands = []
            for table, key in zip(self._tables, keys):
                cands.extend(table.get(key, ())[-self.audit.bucket_cap:])
            if cands:
                cands = np.unique(np.asarray(cands))[-self.audit.max_candidates:]
                S = self._sk[cands]
                dd = np.linalg.norm(S - sk, axis=1)
                j = int(np.argmin(dd))
                denom = max(snorm, float(self._snorm[int(cands[j])]), 1e-30)
                best = float(dd[j] / denom)

        row = -1
        if self._n < self.audit.max_indexed:
            row = self._n
            if row == len(self._sk):
                grown = np.zeros((2 * row, self.audit.sketch_dim), dtype=np.float32)
                grown[:row] = self._sk
                self._sk = grown
                gn = np.zeros(2 * row, dtype=np.float32)
                gn[:row] = self._snorm
                self._snorm = gn
            self._sk[row] = sk
            self._snorm[row] = snorm
            self._n += 1
            for table, key in zip(self._tables, keys):
                b = table[key]
                b.append(row)
                if len(b) > 4 * self.audit.bucket_cap:
                    del b[:len(b) - 2 * self.audit.bucket_cap]
        return best, row


class ComputeAudit:
    """Records every instrumented call and reports how much of it was a repeat.

    Parameters
    ----------
    enabled : bool
        When false every entry point is a no-op and ``install`` patches nothing.
    tolerances : sequence of float
        Relative-distance ladder used by the report.
    trace_kernels : bool
        Also instrument the per-stencil kernels (``_bend_stencil``,
        ``psi_grad_hess``, ...).  These are called tens of millions of times in
        a full run, so the instrumentation itself becomes a measurable cost --
        useful to answer "does the Hessian assembly redo the gradient's work",
        but off by default.
    max_indexed : int
        Cap on inputs kept for near-duplicate search, per kind.  Exact
        duplicates and timings keep being counted past the cap.
    """

    def __init__(self, enabled=True, tolerances=DEFAULT_TOLERANCES,
                 trace_kernels=False, max_indexed=200_000, sketch_dim=32,
                 lsh_tables=12, lsh_bits=6, lsh_width=0.1, bucket_cap=32,
                 max_candidates=64, calibration=32, report_every=0):
        self.enabled = bool(enabled)
        self.tolerances = tuple(sorted(float(t) for t in tolerances))
        self.trace_kernels = bool(trace_kernels)
        self.max_indexed = int(max_indexed)
        self.sketch_dim = int(sketch_dim)
        self.lsh_tables = int(lsh_tables)
        self.lsh_bits = int(lsh_bits)
        self.lsh_width = float(lsh_width)
        self.bucket_cap = int(bucket_cap)
        self.max_candidates = int(max_candidates)
        self.calibration = int(calibration)
        self.report_every = int(report_every)

        self.kinds = {}
        #: epoch = gradient step, step = rollout step within it, seg = RHC
        #: segment (-1 when there is none).  Callers tag with context().
        self._ctx = {"epoch": -1, "step": -1, "seg": -1}
        self._stack = []            # (start_time, accumulated child time)
        self._patched = []          # (owner, attr, original)
        self.wall_start = perf_counter()
        self.wall_total = None

    # ---------- context ----------

    @contextmanager
    def context(self, **kw):
        """Tag every call made inside the block with e.g. ``epoch=3``."""
        if not self.enabled:
            yield
            return
        old = {k: self._ctx.get(k, -1) for k in kw}
        self._ctx.update({k: int(v) for k, v in kw.items()})
        try:
            yield
        finally:
            self._ctx.update(old)

    def set_context(self, **kw):
        if self.enabled:
            self._ctx.update({k: int(v) for k, v in kw.items()})

    # ---------- recording ----------

    def kind(self, name, detail=True, near=True):
        k = self.kinds.get(name)
        if k is None:
            k = self.kinds[name] = _Kind(name, self, detail=detail, near=near)
        return k

    @contextmanager
    def timed(self, name, key, detail=True, near=True):
        """Time a block and attribute it to ``name`` with input ``key``."""
        if not self.enabled:
            yield
            return
        self._stack.append([perf_counter(), 0.0])
        try:
            yield
        finally:
            start, child = self._stack.pop()
            total = perf_counter() - start
            if self._stack:
                self._stack[-1][1] += total
            self.kind(name, detail, near).add(
                key, total, total - child,
                (self._ctx["epoch"], self._ctx["step"], self._ctx["seg"]))

    # ---------- patching ----------

    def _wrap(self, owner, attr, name, keyfn, detail=True, near=True):
        orig = getattr(owner, attr)
        kind = self.kind(name, detail, near)
        stack = self._stack
        ctx = self._ctx

        def wrapped(*a, **kw):
            stack.append([perf_counter(), 0.0])
            try:
                out = orig(*a, **kw)
            finally:
                start, child = stack.pop()
                total = perf_counter() - start
                if stack:
                    stack[-1][1] += total
            kind.add(keyfn(a, kw), total, total - child,
                     (ctx["epoch"], ctx["step"], ctx["seg"]))
            return out

        wrapped.__name__ = getattr(orig, "__name__", attr)
        wrapped.__doc__ = getattr(orig, "__doc__", None)
        setattr(owner, attr, wrapped)
        self._patched.append((owner, attr, orig))

    def _wrap_global(self, module, attr, name, keyfn, detail=True, near=True):
        """Wrap a module-level function *everywhere it is already bound*.

        ``from implicit_grad_tools import multistep_adjoint_gradient`` copies the
        function object into the importing module's namespace, so patching only
        the defining module would leave those callers running the original and
        the audit would quietly report zero calls.
        """
        import sys

        orig = getattr(module, attr, None)
        if orig is None:
            return
        owners = [m for m in list(sys.modules.values())
                  if m is not None and getattr(m, attr, None) is orig]
        if module not in owners:
            owners.append(module)
        kind_args = (name, keyfn, detail, near)
        first = True
        for owner in owners:
            if first:
                self._wrap(owner, attr, *kind_args)
                wrapped = getattr(owner, attr)
                first = False
            else:
                self._patched.append((owner, attr, orig))
                setattr(owner, attr, wrapped)

    def install(self, env):
        """Wrap the kernels of a live environment (and the modules it calls).

        Safe to call once per run; ``restore()`` puts everything back.
        """
        if not self.enabled:
            return self

        import implicit_grad_tools as igt
        energy = env.energy

        # -- the simulator's outer step: the input is the boundary control --
        def _step_key(a, kw):
            c = kw.get("control", a[0] if a else None)
            return np.asarray(c, dtype=float).ravel()

        self._wrap(env, "step", "env.step", _step_key)

        # -- Newton solve: the input is the starting configuration --
        import newton
        self._wrap_global(newton, "solve_equilibrium", "newton.solve",
                          lambda a, kw: np.asarray(kw.get("q_start", a[0] if a else None),
                                                   dtype=float).ravel())

        # -- the assembly itself: force+Hessian, and the force-only variant --
        key_q = lambda a, kw: np.asarray(kw.get("q", a[0] if a else None), dtype=float).ravel()
        self._wrap(energy, "force_and_jacobian", "energy.force_and_jacobian", key_q)
        self._wrap(energy, "force", "energy.force", key_q)
        for attr, nm in (("gradient", "energy.gradient"), ("hessian", "energy.hessian")):
            if hasattr(energy, attr):
                self._wrap(energy, attr, nm, key_q)

        # -- adjoint: the per-step implicit solve, and the whole backward pass --
        self._wrap_global(igt, "local_dxf_dxb", "adjoint.local_dxf_dxb",
                          lambda a, kw: np.asarray(kw.get("J", a[0] if a else None),
                                                   dtype=float).ravel())
        self._wrap_global(igt, "multistep_adjoint_gradient", "adjoint.multistep",
                          lambda a, kw: np.asarray(kw.get("u_seq", a[0] if a else None),
                                                   dtype=float).ravel())

        if self.trace_kernels:
            self._install_kernels(energy)
        return self

    def _install_kernels(self, energy):
        """Per-stencil kernels.  High call counts: only enable deliberately."""
        import rodframes

        bend = getattr(energy, "_bend", energy)
        for attr, nm in (("_bend_stencil", "kernel.bend_stencil"),
                         ("_stretch_stencil", "kernel.stretch_stencil")):
            if hasattr(bend, attr):
                self._wrap(bend, attr, nm,
                           lambda a, kw: np.concatenate([np.asarray(x, float).ravel()
                                                         for x in a]),
                           detail=False)
        if hasattr(rodframes, "psi_grad_hess"):
            self._wrap_global(rodframes, "psi_grad_hess", "kernel.psi_grad_hess",
                       lambda a, kw: np.concatenate([np.asarray(x, float).ravel()
                                                     for x in a]),
                       detail=False)
        if hasattr(rodframes, "RodFrames") and hasattr(rodframes.RodFrames, "reference_twist"):
            self._wrap(rodframes.RodFrames, "reference_twist", "kernel.reference_twist",
                       lambda a, kw: np.asarray(a[1], dtype=float).ravel(),
                       detail=False)

    def restore(self):
        for owner, attr, orig in reversed(self._patched):
            setattr(owner, attr, orig)
        self._patched.clear()

    # ---------- reporting ----------

    def _kind_summary(self, k: _Kind):
        dist = np.asarray(k.dist, dtype=float) if k.detail else None
        t_self = np.asarray(k.t_self, dtype=float) if k.detail else None
        rec = {
            "calls": k.calls,
            "key_dim": k.key_dim,
            "total_time_s": k.total_time,
            "self_time_s": k.self_time,
            "unique_exact": len(k._seen),
            "exact_dup_calls": k.exact_dups,
            "exact_dup_frac": k.exact_dups / k.calls if k.calls else 0.0,
            "exact_dup_self_time_s": k.exact_dup_time,
            "exact_dup_time_frac": (k.exact_dup_time / k.self_time
                                    if k.self_time > 0 else 0.0),
            "indexed": int(k._n),
            "near": {},
        }
        if dist is not None and dist.size:
            for tol in self.tolerances:
                m = dist <= tol
                rec["near"][f"{tol:g}"] = {
                    "calls": int(m.sum()),
                    "call_frac": float(m.mean()),
                    "self_time_s": float(t_self[m].sum()),
                    "time_frac": float(t_self[m].sum() / t_self.sum()) if t_self.sum() else 0.0,
                }
            finite = dist[np.isfinite(dist)]
            if finite.size:
                rec["nearest_distance_percentiles"] = {
                    p: float(np.percentile(finite, q))
                    for p, q in (("p1", 1), ("p10", 10), ("p50", 50),
                                 ("p90", 90), ("p99", 99))
                }
        return rec

    def _epoch_repeat(self, k: _Kind):
        """How far apart are the same rollout step of consecutive epochs?

        This is the "is every epoch redoing the last one" question.  It pairs
        calls that share a step index and sit in adjacent epochs, and reports
        the relative distance between their inputs.
        """
        if not k.detail or not k._n:
            return None
        ep = np.asarray(k.ctx_epoch)
        st = np.asarray(k.ctx_step)
        sg = np.asarray(k.ctx_seg)
        ok = (ep >= 0) & (st >= 0)
        if ok.sum() < 4 or len(np.unique(ep[ok])) < 2:
            return None
        S, norms = k._sk, k._snorm
        sk_idx = np.asarray(k.sk_idx)

        by = {}
        for i in range(len(ep)):
            if ok[i] and sk_idx[i] >= 0:
                by.setdefault((int(sg[i]), int(ep[i]), int(st[i])), int(sk_idx[i]))
        ds = []
        for (g, e, s), i in by.items():
            j = by.get((g, e - 1, s))
            if j is None:
                continue
            denom = max(float(norms[i]), float(norms[j]), 1e-30)
            ds.append(float(np.linalg.norm(S[i] - S[j]) / denom))
        if not ds:
            return None
        ds = np.asarray(ds)
        return {
            "pairs": int(ds.size),
            "p10": float(np.percentile(ds, 10)),
            "median": float(np.median(ds)),
            "p90": float(np.percentile(ds, 90)),
            "frac_within": {f"{t:g}": float((ds <= t).mean()) for t in self.tolerances},
        }

    def summary(self):
        self.wall_total = perf_counter() - self.wall_start
        kinds = {name: self._kind_summary(k) for name, k in self.kinds.items()}
        for name, k in self.kinds.items():
            rep = self._epoch_repeat(k)
            if rep is not None:
                kinds[name]["epoch_over_epoch"] = rep
        audited = sum(v["self_time_s"] for v in kinds.values())
        return {
            "wall_time_s": self.wall_total,
            "audited_self_time_s": audited,
            "audited_frac_of_wall": audited / self.wall_total if self.wall_total else 0.0,
            "tolerances": list(self.tolerances),
            "kinds": kinds,
        }

    def text_report(self, summary=None):
        s = summary or self.summary()
        tols = s["tolerances"]
        out = []
        out.append("=" * 100)
        out.append("COMPUTE REDUNDANCY AUDIT")
        out.append("=" * 100)
        out.append(f"wall time              : {s['wall_time_s']:.1f} s")
        out.append(f"inside audited kernels : {s['audited_self_time_s']:.1f} s "
                   f"({s['audited_frac_of_wall'] * 100:.1f}% of wall, exclusive of nesting)")
        if self.trace_kernels:
            out.append("NOTE: per-stencil tracing is on.  Those kernels are called hundreds of "
                       "thousands of")
            out.append("      times, so the wrapper's own cost inflates their share of the time "
                       "columns.  The")
            out.append("      duplicate *fractions* are call counts and are unaffected.")
        out.append("")
        out.append("Share of CALLS whose input repeats an earlier one "
                   "(exact = identical bytes; the rest are cumulative, relative distance):")
        head = f"{'kind':<28}{'calls':>10}{'self_s':>10}{'%self':>8}{'exact':>9}"
        head += "".join(f"{('<=' + f'{t:g}'):>10}" for t in tols)
        out.append(head)
        out.append("-" * len(head))
        order = sorted(s["kinds"].items(), key=lambda kv: -kv[1]["self_time_s"])
        for name, r in order:
            pct = (r["self_time_s"] / s["audited_self_time_s"] * 100
                   if s["audited_self_time_s"] else 0.0)
            line = (f"{name:<28}{r['calls']:>10}{r['self_time_s']:>10.1f}"
                    f"{pct:>7.1f}%{r['exact_dup_frac'] * 100:>8.1f}%")
            for t in tols:
                cell = r["near"].get(f"{t:g}")
                line += f"{cell['call_frac'] * 100:>9.1f}%" if cell else f"{'-':>10}"
            out.append(line)

        out.append("")
        out.append("Share of TIME spent on those repeated calls "
                   "(what a cache / reuse scheme could recover):")
        head2 = f"{'kind':<28}{'self_s':>10}{'exact_s':>10}{'exact%':>9}"
        head2 += "".join(f"{('<=' + f'{t:g}'):>10}" for t in tols)
        out.append(head2)
        out.append("-" * len(head2))
        for name, r in order:
            line = (f"{name:<28}{r['self_time_s']:>10.1f}"
                    f"{r['exact_dup_self_time_s']:>10.1f}"
                    f"{r['exact_dup_time_frac'] * 100:>8.1f}%")
            for t in tols:
                cell = r["near"].get(f"{t:g}")
                line += f"{cell['time_frac'] * 100:>9.1f}%" if cell else f"{'-':>10}"
            out.append(line)

        out.append("")
        out.append("Distance to the nearest earlier input (relative, percentiles):")
        for name, r in order:
            p = r.get("nearest_distance_percentiles")
            if p:
                out.append(f"  {name:<28} p1={p['p1']:.2e}  p10={p['p10']:.2e}  "
                           f"p50={p['p50']:.2e}  p90={p['p90']:.2e}  p99={p['p99']:.2e}")

        reps = [(n, r["epoch_over_epoch"]) for n, r in order if "epoch_over_epoch" in r]
        if reps:
            out.append("")
            out.append("Epoch over epoch: same rollout step, consecutive epochs -- "
                       "how far apart are the inputs?")
            for name, rep in reps:
                within = "  ".join(f"<={t}:{rep['frac_within'][t] * 100:.0f}%"
                                   for t in rep["frac_within"])
                out.append(f"  {name:<28} pairs={rep['pairs']:<8} "
                           f"p10={rep['p10']:.2e} median={rep['median']:.2e} "
                           f"p90={rep['p90']:.2e}")
                out.append(f"  {'':<28} {within}")

        out.append("")
        out.append("How to read this: `exact%` is work a memo-cache removes with no "
                   "change in results.")
        out.append("The `<=tol` columns are the reuse budget -- the share of calls whose "
                   "input was already")
        out.append("seen to within that relative distance, so a frozen/stale Jacobian or a "
                   "warm start would")
        out.append("be that accurate.  Counts past the index cap are still timed but not "
                   "distance-checked.")
        out.append("=" * 100)
        return "\n".join(out)

    def checkpoint_report(self, directory, epoch):
        """Rewrite the text/JSON report mid-run, every ``report_every`` epochs.

        A full training run takes hours; this is what makes the audit readable
        while it is still going, with `watch cat .../compute_audit.txt`.  The
        per-call table is skipped -- compressing it is the slow part, and it is
        only useful once the run is over.
        """
        if not (self.enabled and self.report_every and directory):
            return False
        if epoch < 0 or (epoch + 1) % self.report_every:
            return False
        self.save(directory, save_records=False)
        return True

    def save(self, directory, save_records=True):
        """Write ``compute_audit.json`` / ``.txt`` (and the per-call table)."""
        if not self.enabled:
            return None
        os.makedirs(directory, exist_ok=True)
        s = self.summary()
        with open(os.path.join(directory, "compute_audit.json"), "w") as f:
            json.dump(s, f, indent=2, default=float)
        text = self.text_report(s)
        with open(os.path.join(directory, "compute_audit.txt"), "w") as f:
            f.write(text + "\n")
        if save_records:
            arrays = {}
            for name, k in self.kinds.items():
                if not k.detail or not k.calls:
                    continue
                tag = name.replace(".", "_")
                arrays[f"{tag}__self_time"] = np.asarray(k.t_self, dtype=np.float32)
                arrays[f"{tag}__total_time"] = np.asarray(k.t_total, dtype=np.float32)
                arrays[f"{tag}__nearest"] = np.asarray(k.dist, dtype=np.float32)
                arrays[f"{tag}__epoch"] = np.asarray(k.ctx_epoch, dtype=np.int32)
                arrays[f"{tag}__step"] = np.asarray(k.ctx_step, dtype=np.int32)
                arrays[f"{tag}__seg"] = np.asarray(k.ctx_seg, dtype=np.int32)
            if arrays:
                np.savez_compressed(os.path.join(directory, "compute_audit_records.npz"),
                                    **arrays)
        return text


def from_config(cfg):
    """Build a :class:`ComputeAudit` from the ``audit:`` block of a config."""
    a = (cfg.get("audit") or {}) if hasattr(cfg, "get") else {}
    return ComputeAudit(
        enabled=bool(a.get("enabled", False)),
        tolerances=tuple(a.get("tolerances") or DEFAULT_TOLERANCES),
        trace_kernels=bool(a.get("trace_kernels", False)),
        max_indexed=int(a.get("max_indexed", 200_000)),
        report_every=int(a.get("report_every", 0) or 0),
    )
