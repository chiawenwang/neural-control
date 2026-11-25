"""Sweep the number of helix turns: where does greedy shape control stop working?

For every turn count ``T`` in ``--turns-start .. --turns-stop`` (step
``--turns-step``) this runs ``scripts/greedy_helix.py run`` for each
handedness and seed, in parallel, and then tabulates success, getting stuck
and error against ``T``.  Each run keeps its own gif, figures and data in
``OUT/t<T>/``.

Comparable targets across ``T``:

* pitch ratio ``h = 1 - (1 - h1) / T`` with ``h1`` the config's one-turn
  value (0.7): the Bishop holonomy ``2 pi T (1 - h)`` stays at the one-turn
  value (0.6 pi), clear of the twist model's wrap at pi (src/rodframes.py);
* ``nv = edges_per_turn * T + 1`` nodes, so every turn gets the same
  resolution (radius ``sqrt(1 - h^2) / (2 pi T)`` shrinks with ``T``).

Seeds perturb the right gripper's start pose (1 cm / 0.02 rad sd; seed 0 is
the configured pose), so each ``T`` has ``len(hands) * len(seeds)`` distinct
runs.  A run is

* ``success``  -- lam reached 1 and the final rms error to the helix is
  below ``--success-tol``;
* ``stuck``    -- it stopped before lam = 1 (trust radius collapsed, time
  limit or max steps), i.e. the greedy could not follow the shape family;
* ``inexact``  -- lam reached 1 but the rms error stayed above the tolerance;
* ``crashed``  -- no result (see the run's log).

Usage
-----
    python scripts/greedy_helix_sweep.py run --out experiments/helix_turns
    python scripts/greedy_helix_sweep.py run --out DIR --turns-step 0.25 --seeds 0 1 2 3 --time-limit 1200
    python scripts/greedy_helix_sweep.py report --out DIR

``run`` skips runs whose result already exists (``--force`` reruns them) and
ends with the report.  The report writes ``summary.json``, ``runs.csv`` and
the figure ``sweep`` (.pdf/.svg/.png) into ``OUT``.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

from config import load_config
import tro_style as ts

DEFAULT_CONFIG = _ROOT / "configs" / "task_helix_3d.yaml"
SCRIPT = _ROOT / "scripts" / "greedy_helix.py"
STATUSES = ("success", "inexact", "stuck", "crashed")
COLORS = {"success": "#1baf7a", "inexact": "#e8a33d", "stuck": "#d6452a", "crashed": "#7a7975"}


def turn_values(start, stop, step):
    n = int(round((stop - start) / step))
    return [round(start + i * step, 6) for i in range(n + 1)]


def tdir(out, T):
    return Path(out) / f"t{T:g}"


def job_list(args):
    cfg = load_config(args.config)
    h1 = float(cfg.task.helix.pitch_ratio) if cfg.task.helix.turns == 1.0 else None
    if h1 is None:
        raise SystemExit("the config's task.helix.turns must be 1 (its pitch_ratio is the one-turn value)")
    jobs = []
    for T in turn_values(args.turns_start, args.turns_stop, args.turns_step):
        h = round(1.0 - (1.0 - h1) / T, 6)
        nv = int(round(args.edges_per_turn * T)) + 1
        for hand in args.hands:
            for seed in args.seeds:
                name = f"t{T:g}_{hand}_s{seed}"
                cmd = [sys.executable, str(SCRIPT), "run", "--config", str(args.config),
                       "--out", str(tdir(args.out, T)), "--name", name, "--turns", f"{T:g}",
                       "--pitch", f"{h:g}", "--nv", str(nv), "--hand", hand, "--seed", str(seed),
                       "--time-limit", f"{args.time_limit:g}"]
                jobs.append({"T": T, "h": h, "nv": nv, "hand": hand, "seed": seed, "name": name,
                             "cmd": cmd})
    return jobs


def run_job(job, out, force):
    d = tdir(out, job["T"])
    (d / "logs").mkdir(parents=True, exist_ok=True)
    if (d / f"{job['name']}.json").exists() and not force:
        return job["name"], "skipped", 0.0
    t0 = time.time()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with open(d / "logs" / f"{job['name']}.log", "w") as log:
        rc = subprocess.run(job["cmd"], stdout=log, stderr=subprocess.STDOUT, cwd=_ROOT, env=env).returncode
    return job["name"], ("ok" if rc == 0 else f"exit {rc}"), time.time() - t0


def cmd_run(args):
    jobs = job_list(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({k: v for k, v in vars(args).items() if k != "func"},
              open(out / "sweep_args.json", "w"), indent=1, default=str)
    workers = args.workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 2))
    print(f"{len(jobs)} runs on {workers} workers, time limit {args.time_limit:g} s each -> {out}")
    for T in sorted({j["T"] for j in jobs}):
        j = next(j for j in jobs if j["T"] == T)
        r = np.sqrt(1 - j["h"] ** 2) / (2 * np.pi * T)
        print(f"  turns {T:g}: pitch ratio {j['h']:g}, nv {j['nv']}, radius {1e3 * r:.0f} mm, "
              f"{sum(1 for x in jobs if x['T'] == T)} runs")
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(run_job, j, out, args.force) for j in jobs]
        for f in futures:
            name, status, dt = f.result()
            print(f"  [{time.time() - t0:6.0f}s] {name:22s} {status}" + (f" ({dt:.0f}s)" if dt else ""),
                  flush=True)
    report(out, args.success_tol)


# ---------------------------------------------------------------- report

def classify(r, tol):
    if r is None:
        return "crashed"
    if r["lam_final"] < 1.0:
        return "stuck"
    return "success" if r["rms_final"] < tol else "inexact"


def collect(out):
    """Every run of the sweep: its job description and its result (None if crashed)."""
    rows = []
    for d in sorted(Path(out).glob("t*"), key=lambda p: float(p.name[1:])):
        T = float(d.name[1:])
        names = {p.stem for p in (d / "logs").glob("*.log")} | {p.stem for p in d.glob("t*_s*.json")}
        for name in sorted(names):
            p = d / f"{name}.json"
            r = json.load(open(p)) if p.exists() else None
            hand, seed = name.split("_")[1], int(name.split("_s")[-1])
            rows.append({"T": T, "name": name, "hand": hand, "seed": seed, "dir": d, "res": r})
    return rows


def report(out, tol):
    out = Path(out)
    rows = collect(out)
    if not rows:
        raise SystemExit(f"no runs in {out}")
    for row in rows:
        row["status"] = classify(row["res"], tol)

    print(f"\nPer run (success: lam = 1 and rms < {1e3 * tol:g} mm)")
    head = (f"{'run':18s} {'h':>5s} {'nv':>3s} {'status':>8s} {'lam':>5s} {'rms mm':>7s} {'max mm':>7s} "
            f"{'steps':>5s} {'snaps':>9s} {'yaw':>6s} {'time s':>6s}  stop")
    print(head)
    print("-" * len(head))
    csv_rows = []
    for row in rows:
        r = row["res"]
        if r is None:
            print(f"{row['name']:18s} {'':>5s} {'':>3s} {'crashed':>8s}  (see {row['dir'] / 'logs' / row['name']}.log)")
            csv_rows.append({"turns": row["T"], "run": row["name"], "hand": row["hand"],
                             "seed": row["seed"], "status": "crashed"})
            continue
        print(f"{row['name']:18s} {r['helix']['pitch_ratio']:5.3g} {r['nv']:3d} {row['status']:>8s} "
              f"{r['lam_final']:5.2f} {1e3 * r['rms_final']:7.1f} {1e3 * r['max_err_final']:7.1f} "
              f"{r['steps']:5d} {r['snaps']:4d}/{r.get('snap_trials', 0):<4d} "
              f"{r['yaw_final'] if r['yaw_final'] is not None else float('nan'):+6.2f} "
              f"{r['time']:6.0f}  {r['stop']}")
        csv_rows.append({"turns": row["T"], "run": row["name"], "hand": row["hand"], "seed": row["seed"],
                         "status": row["status"], "pitch_ratio": r["helix"]["pitch_ratio"],
                         "nv": r["nv"], "radius_mm": 1e3 * r["helix"]["radius"],
                         "lam_final": r["lam_final"], "rms_mm": 1e3 * r["rms_final"],
                         "max_err_mm": 1e3 * r["max_err_final"], "steps": r["steps"],
                         "snaps_kept": r["snaps"], "snap_trials": r.get("snap_trials", 0),
                         "yaw_final": r["yaw_final"], "twist_final": r["twist_final"],
                         "time_s": r["time"], "stop": r["stop"],
                         "gif": str(row["dir"] / f"{row['name']}.gif")})

    turns = sorted({row["T"] for row in rows})
    per_T = []
    print("\nPer turn count")
    head = (f"{'turns':>5s} {'runs':>4s} {'success':>8s} {'stuck':>6s} {'inexact':>7s} {'crashed':>7s} "
            f"{'rms mm (median)':>15s} {'lam (median)':>12s} {'snap trials':>11s} {'time s':>6s}")
    print(head)
    print("-" * len(head))
    for T in turns:
        rs = [row for row in rows if row["T"] == T]
        n = len(rs)
        cnt = {s: sum(row["status"] == s for row in rs) for s in STATUSES}
        ok = [row["res"] for row in rs if row["res"] is not None]
        med = lambda key, scale=1.0: float(np.median([scale * r[key] for r in ok])) if ok else float("nan")  # noqa: E731
        rec = {"turns": T, "runs": n, **{f"n_{s}": cnt[s] for s in STATUSES},
               "success_rate": cnt["success"] / n, "stuck_rate": cnt["stuck"] / n,
               "rms_mm_median": med("rms_final", 1e3), "lam_median": med("lam_final"),
               "snap_trials_median": med("snap_trials"), "time_s_median": med("time")}
        per_T.append(rec)
        print(f"{T:5g} {n:4d} {cnt['success']:4d} ({100 * rec['success_rate']:3.0f}%) {cnt['stuck']:6d} "
              f"{cnt['inexact']:7d} {cnt['crashed']:7d} {rec['rms_mm_median']:15.1f} {rec['lam_median']:12.2f} "
              f"{rec['snap_trials_median']:11.0f} {rec['time_s_median']:6.0f}")

    json.dump({"success_tol": tol, "per_turns": per_T,
               "runs": [{k: (str(v) if isinstance(v, Path) else v) for k, v in c.items()} for c in csv_rows]},
              open(out / "summary.json", "w"), indent=1, default=float)
    keys = sorted({k for c in csv_rows for k in c}, key=lambda k: list(csv_rows[0]).index(k)
                  if k in csv_rows[0] else 99)
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(csv_rows)
    plot_sweep(rows, per_T, tol, out / "sweep")
    print(f"\nsummary.json, runs.csv, sweep.pdf/.svg/.png -> {out}; gifs in {out}/t<turns>/")


def plot_sweep(rows, per_T, tol, stem):
    """Left: share of runs per outcome against turns.  Right: each run's final
    rms (filled) and lam reached (label on stuck runs) against turns."""
    turns = [p["turns"] for p in per_T]
    x = np.arange(len(turns))
    with ts.style():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(ts.COL_W * 2, 2.2))
        bottom = np.zeros(len(turns))
        for s in STATUSES:
            frac = np.array([p[f"n_{s}"] / p["runs"] for p in per_T])
            if frac.any():
                a1.bar(x, frac, 0.6, bottom=bottom, color=COLORS[s], label=s, edgecolor="white",
                       linewidth=0.5)
                bottom += frac
        a1.set_xticks(x, [f"{t:g}" for t in turns])
        a1.set_xlabel("Turns")
        a1.set_ylabel("Share of runs")
        a1.set_ylim(0, 1)
        a1.legend(fontsize=ts.FS_TICK, frameon=False, ncol=len(STATUSES), loc="lower center",
                  bbox_to_anchor=(0.5, 1.0))
        ts.style_2d(a1)

        rng = np.random.default_rng(0)
        for row in rows:
            r = row["res"]
            if r is None:
                continue
            i = turns.index(row["T"])
            xi = i + rng.uniform(-0.15, 0.15)
            a2.plot(xi, 1e3 * r["rms_final"], "o", color=COLORS[row["status"]], markersize=4,
                    markeredgecolor="white", markeredgewidth=0.4)
            if row["status"] == "stuck":
                a2.annotate(f"$\\lambda$={r['lam_final']:.2f}", (xi, 1e3 * r["rms_final"]),
                            xytext=(4, 0), textcoords="offset points", fontsize=ts.FS_TICK - 1,
                            va="center")
        a2.axhline(1e3 * tol, color="#7a7975", linewidth=0.8, linestyle=(0, (4, 2)))
        a2.text(len(turns) - 0.55, 1e3 * tol, f"success < {1e3 * tol:g} mm", fontsize=ts.FS_TICK - 1,
                ha="right", va="bottom", color="#52514e")
        a2.set_yscale("log")
        a2.set_xticks(x, [f"{t:g}" for t in turns])
        a2.set_xlim(-0.5, len(turns) - 0.5)
        a2.set_xlabel("Turns")
        a2.set_ylabel("Final RMS error to helix (mm)")
        ts.style_2d(a2)
        fig.tight_layout()
        ts.save(fig, stem)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="greedy helix: success against the number of turns")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the sweep (skips finished runs), then report")
    r.add_argument("--config", default=str(DEFAULT_CONFIG), help="one-turn config (task.helix.turns: 1)")
    r.add_argument("--out", required=True)
    r.add_argument("--turns-start", type=float, default=1.0)
    r.add_argument("--turns-stop", type=float, default=2.0)
    r.add_argument("--turns-step", type=float, default=0.5)
    r.add_argument("--hands", nargs="+", choices=("right", "left"), default=["right", "left"])
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--edges-per-turn", type=int, default=20)
    r.add_argument("--time-limit", type=float, default=900.0,
                   help="[s] per run; a run still short of lam = 1 then counts as stuck")
    r.add_argument("--success-tol", type=float, default=0.015, help="[m] rms to the helix")
    r.add_argument("--workers", type=int, default=None, help="default: cpu count - 2")
    r.add_argument("--force", action="store_true", help="rerun finished runs")
    p = sub.add_parser("report", help="tables and figure of a finished (or partial) sweep")
    p.add_argument("--out", required=True)
    p.add_argument("--success-tol", type=float, default=0.015, help="[m] rms to the helix")
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    else:
        report(args.out, args.success_tol)


if __name__ == "__main__":
    main()
