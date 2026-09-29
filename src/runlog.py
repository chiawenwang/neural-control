# -*- coding: utf-8 -*-
"""Per-run training log: what happened, as it happens, on disk.

TensorBoard event files are good for eyeballing a curve but poor for going
back and answering "what exactly did that run do?" -- they hold a handful of
scalars, nothing about the configuration, and every run of the same experiment
piles into the same directory.  This module writes, per run, a self-contained
directory that answers that question::

    experiments/<experiment_name>/runs/<timestamp>/
        config.yaml    the fully resolved config, as actually used
        meta.json      git commit, device, rod/env description, start time
        log.jsonl      one JSON record per epoch or segment, flushed as it goes
        summary.json   final results, written when the run ends
        log.txt        the same records as readable lines

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
        if not self.enabled:
            return

        root = root or os.getcwd()
        exp = (cfg.get("output") or {}).get("experiment_name", "run")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if tag:
            stamp = f"{stamp}-{tag}"
        self.dir = os.path.join(str(root), "experiments", str(exp), "runs", stamp)
        os.makedirs(self.dir, exist_ok=True)

        self._write_config(cfg)
        self._write_meta(cfg, env, root)
        self._jsonl = open(os.path.join(self.dir, "log.jsonl"), "w")
        self._txt = open(os.path.join(self.dir, "log.txt"), "w")
        # If the process dies before finish(), still close the run out rather
        # than leaving a directory with no summary at all.
        atexit.register(self._atexit)
        print(f"Run log -> {self.dir}")

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
        """Append one record. Flushed immediately so a killed run keeps it."""
        if not self.enabled:
            return
        rec = {"i": self.n_records, "t": round(time.time() - self.t0, 3)}
        rec.update(_jsonable(fields))
        self._jsonl.write(json.dumps(rec) + "\n")
        self._jsonl.flush()
        self._txt.write("  ".join(f"{k}={_fmt(v)}" for k, v in rec.items()) + "\n")
        self._txt.flush()
        self.n_records += 1

    # ---------- end of run ----------

    def finish(self, status="finished", error=None, **fields):
        """Write summary.json and close the files.

        ``status`` records how the run ended, so an interrupted run is not
        mistaken for a successful one that happened to score badly.
        """
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
