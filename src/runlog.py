# -*- coding: utf-8 -*-
"""Per-run training log: what happened, as it happens, on disk.

Every run gets a self-contained directory -- including its own TensorBoard
event file, so runs are separate, comparable entries rather than piling into
one directory where a later run buries an earlier one::

    experiments/<experiment_name>/runs/<timestamp>/
        events.out.tfevents.*   TensorBoard scalars, flushed per record
        config.yaml             the fully resolved config, as actually used
        meta.json               git commit, device, rod/env description
        log.jsonl               one JSON record per epoch or segment
        summary.json            final results, and how the run ended
        log.txt                 the same records as readable lines

Watch them live with::

    tensorboard --logdir experiments

:meth:`RunLogger.log` is the single place a result goes: it writes the JSON
record, the readable line, *and* the TensorBoard scalars, so the three can
never drift apart.  Scalars are flushed on every record, which is what makes
the curve update while the run is still going.

``log.jsonl`` is flushed on every record, so a run that crashes or is killed
still leaves everything up to that point -- which is exactly when the log is
worth having.  ``summary.json`` records *how* the run ended, so an incomplete
run is never mistaken for a successful one that merely scored badly: a run that
dies is closed out as ``incomplete`` by an ``atexit`` hook, and using the logger
as a context manager upgrades that to ``interrupted`` or ``error`` with the
traceback.

Typical use::

    log = RunLogger(cfg, env=env, root=_ROOT)
    log.log(epoch=ep, loss=L, midpoint=mid, newton_failures=env.n_newton_failures)
    ...
    log.finish(final_loss=..., final_error=..., target=...)
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import atexit
import sys
import time
import traceback
from datetime import datetime

import numpy as np

#: Scalar names in TensorBoard.  Grouped by prefix so the sidebar collapses
#: into loss / error / physics rather than one flat list, and named for what
#: the number means rather than for the variable that happened to hold it.
_TB_NAME = {
    "loss": "loss/final",
    "total_loss": "loss/total",
    "best_loss": "loss/best",
    "error": "error/distance",
    "mean_track_error": "error/track_mean",
    "max_track_error": "error/track_max",
    "converged": "physics/converged",
    "converged_ratio": "physics/converged_ratio",
    "newton_failures": "physics/newton_failures",
    "steps": "physics/steps_completed",
    "horizon": "physics/horizon",
    "iterations": "train/iterations",
}

#: Fields that are recorded but make no sense as a curve.
_TB_SKIP = {"i", "t", "note", "target", "losses", "u_final", "epoch", "segment"}


def _jsonable(v):
    """Make numpy scalars/arrays and the config namespace JSON-serialisable."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return str(v)


def _git_commit(root):
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        commit = out.stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except Exception:
        return {"commit": None, "dirty": None}


def _fmt(v):
    """Compact human-readable rendering for the .txt mirror."""
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


class RunLogger:
    """Writes one run's directory.  Cheap enough to always leave on."""

    def __init__(self, cfg, env=None, root=None, tag=None, enabled=True):
        self.enabled = bool(enabled)
        self.t0 = time.time()
        self.n_records = 0
        self.dir = None
        self._jsonl = None
        self._txt = None
        self._writer = None
        self._cfg = cfg
        self._env = env

        root = root or os.getcwd()
        exp = (cfg.get("output") or {}).get("experiment_name", "run")
        #: Where a run's own figures belong. With logging off there is no run
        #: directory, so they fall back to the experiment directory.
        self.experiment_dir = os.path.join(str(root), "experiments", str(exp))
        self.artifact_dir = self.experiment_dir
        if not self.enabled:
            os.makedirs(self.experiment_dir, exist_ok=True)
            self._writer = self._make_writer(self.experiment_dir)
            return

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if tag:
            stamp = f"{stamp}-{tag}"
        self.dir = os.path.join(self.experiment_dir, "runs", stamp)
        self.artifact_dir = self.dir
        os.makedirs(self.dir, exist_ok=True)

        self._write_config(cfg)
        self._write_meta(cfg, env, root)
        self._jsonl = open(os.path.join(self.dir, "log.jsonl"), "w")
        self._txt = open(os.path.join(self.dir, "log.txt"), "w")
        self._writer = self._make_writer(self.dir)
        if self._writer is not None:
            # The config as text so it is readable from TensorBoard itself,
            # not just from the run directory.
            try:
                with open(os.path.join(self.dir, "config.yaml")) as f:
                    self._writer.add_text("config", "```yaml\n" + f.read() + "\n```", 0)
            except Exception:
                pass
        # If the process dies before finish(), still close the run out rather
        # than leaving a directory with no summary at all.
        atexit.register(self._atexit)
        print(f"Run log -> {self.dir}")

    @staticmethod
    def _make_writer(log_dir):
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(log_dir=log_dir)
        except Exception:
            return None

    @property
    def writer(self):
        """The run's SummaryWriter, for anything log() cannot express."""
        return self._writer

    def _atexit(self):
        if self._jsonl is not None:
            self.finish(status="incomplete")

    # ---------- setup ----------

    def _write_config(self, cfg):
        """The resolved config, so a run can be reproduced from its own log."""
        try:
            import yaml
            text = yaml.safe_dump(_jsonable(dict(cfg)), sort_keys=False,
                                  default_flow_style=False, allow_unicode=True)
        except Exception:
            text = json.dumps(_jsonable(dict(cfg)), indent=2)
        with open(os.path.join(self.dir, "config.yaml"), "w") as f:
            f.write(text)

    def _write_meta(self, cfg, env, root):
        meta = {
            "started": datetime.now().isoformat(timespec="seconds"),
            "git": _git_commit(root),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "argv": sys.argv,
        }
        if env is not None:
            meta["env"] = {
                "class": type(env).__name__,
                "dim": getattr(env, "dim", None),
                "twist": bool(getattr(env, "twist", False)),
                "nv": getattr(env, "nv", None),
                "n_dof": int(getattr(env, "n_dof", np.size(env.q) if env.q is not None else 0)),
                "n_free": int(len(env.free_index)),
                "control_size": getattr(env, "control_size", None),
                "rod_length": getattr(env, "rod_length", None),
                "deltaL": getattr(env, "deltaL", None),
                "Nsteps": getattr(env, "Nsteps", None),
                "EI": getattr(env, "EI", None),
                "EA": getattr(env, "EA", None),
                "GJ": getattr(getattr(env, "energy", None), "GJ", None),
                "gravity": getattr(env, "gravity", None),
                "newton_tol": getattr(env, "tol", None),
                "energy": type(getattr(env, "energy", None)).__name__,
            }
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(_jsonable(meta), f, indent=2)

    # ---------- during the run ----------

    def log(self, **fields):
        """Record one epoch or segment: JSON, readable line, and TensorBoard.

        Everything is flushed immediately, so the curve updates while the run
        is still going and a killed run keeps what it had.
        """
        rec = {"i": self.n_records, "t": round(time.time() - self.t0, 3)}
        rec.update(_jsonable(fields))

        # The x-axis is the epoch (or RHC segment) when there is one, so skipped
        # epochs leave a visible gap rather than silently shifting the curve.
        step = int(rec.get("epoch", rec.get("segment", self.n_records)))
        self._tb_scalars(rec, step)

        if self.enabled:
            self._jsonl.write(json.dumps(rec) + "\n")
            self._jsonl.flush()
            self._txt.write("  ".join(f"{k}={_fmt(v)}" for k, v in rec.items()) + "\n")
            self._txt.flush()
        self.n_records += 1

    def _tb_scalars(self, rec, step):
        w = self._writer
        if w is None:
            return
        for k, v in rec.items():
            if k in _TB_SKIP:
                continue
            name = _TB_NAME.get(k, f"train/{k}")
            if isinstance(v, bool):
                w.add_scalar(name, float(v), step)
            elif isinstance(v, (int, float)) and np.isfinite(v):
                w.add_scalar(name, float(v), step)
            elif (isinstance(v, (list, tuple)) and 0 < len(v) <= 3
                  and all(isinstance(x, (int, float)) for x in v)):
                for axis, x in zip("xyz", v):
                    w.add_scalar(f"{k}/{axis}", float(x), step)
        w.flush()

    # ---------- end of run ----------

    def finish(self, status="finished", error=None, **fields):
        """Write summary.json and close the files.

        ``status`` records how the run ended, so an interrupted run is not
        mistaken for a successful one that happened to score badly.
        """
        self._tb_hparams(status, fields)
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
        if not self.enabled:
            return
        summary = {
            "status": status,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(time.time() - self.t0, 3),
            "records": self.n_records,
        }
        if error is not None:
            summary["error"] = error
        summary.update(_jsonable(fields))
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        self._txt.write("\n" + "\n".join(f"{k}: {_fmt(v)}" for k, v in summary.items()) + "\n")
        for fh in (self._jsonl, self._txt):
            try:
                fh.close()
            except Exception:
                pass
        self._jsonl = self._txt = None
        try:
            atexit.unregister(self._atexit)
        except Exception:
            pass
        print(f"Run summary -> {os.path.join(self.dir, 'summary.json')}  ({status})")

    def _tb_hparams(self, status, fields):
        """One HPARAMS row per run, so runs can be compared in a table."""
        w = self._writer
        if w is None:
            return
        cfg, env = self._cfg or {}, self._env
        rod = cfg.get("rod") or {}
        train = cfg.get("train") or {}
        hp = {
            "dim": int(getattr(env, "dim", (cfg.get("env") or {}).get("dim", 2))),
            "twist": bool(getattr(env, "twist", rod.get("twist", False))),
            "nv": int(rod.get("nv", getattr(env, "nv", 0) or 0)),
            "material": str(rod.get("material", "custom")),
            "lr": float(train.get("lr", 0.0)),
            "epochs": int(train.get("epochs", 0)),
            "rhc": bool((cfg.get("rhc") or {}).get("enabled", False)),
            "task": str(fields.get("task", (cfg.get("task") or {}).get("type", ""))),
            "status": str(status),
        }
        metrics = {f"hparam/{k}": float(fields[k])
                   for k in ("final_loss", "final_error", "best_loss")
                   if isinstance(fields.get(k), (int, float))}
        try:
            w.add_hparams(hp, metrics or {"hparam/records": float(self.n_records)})
        except Exception:
            pass

    # ---------- context manager: record crashes too ----------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._jsonl is None:           # finish() already called
            return False
        if exc_type is KeyboardInterrupt:
            self.finish(status="interrupted")
        elif exc_type is not None:
            self.finish(status="error",
                        error="".join(traceback.format_exception(exc_type, exc, tb))[-4000:])
        else:
            self.finish()
        return False


def load_run(run_dir):
    """Read a run directory back: ``(meta, config_text, records, summary)``.

    Handy for comparing runs or plotting several together::

        meta, cfg, recs, summary = load_run("experiments/dlo_3d/runs/2026...")
        losses = [r["loss"] for r in recs if "loss" in r]
    """
    def _maybe(path, loader):
        p = os.path.join(run_dir, path)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return loader(f)

    meta = _maybe("meta.json", json.load)
    config = _maybe("config.yaml", lambda f: f.read())
    summary = _maybe("summary.json", json.load)
    records = _maybe("log.jsonl",
                     lambda f: [json.loads(line) for line in f if line.strip()]) or []
    return meta, config, records, summary
