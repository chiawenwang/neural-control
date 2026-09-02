"""Global YAML-based configuration loading for the training scripts.

Usage
-----
    from config import load_config
    cfg = load_config(args.config)   # args.config may be None
    cfg.rod.nv, cfg.task.target_xy, cfg.rhc.enabled, ...

A custom config file only needs to specify the fields it wants to override;
it is deep-merged on top of configs/default.yaml.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional, Union

import yaml

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _ROOT / "configs" / "default.yaml"


class ConfigNamespace(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return ConfigNamespace(value) if isinstance(value, dict) else value

    def __setattr__(self, name, value):
        self[name] = value


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Optional[Union[str, Path]] = None) -> ConfigNamespace:
    """Load configs/default.yaml, optionally deep-merged with a user config."""
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if path is not None:
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)

    return ConfigNamespace(cfg)
