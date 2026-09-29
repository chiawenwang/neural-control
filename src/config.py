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


#: Subtrees whose keys are open-ended, so an unknown name there is legitimate:
#: user-defined materials, and the kwargs forwarded to a custom energy class.
_OPEN_SUBTREES = {("materials",), ("energy", "params")}


def _unknown_keys(user, default, prefix=()):
    """Keys in the user config that configs/default.yaml does not define."""
    if prefix in _OPEN_SUBTREES:
        return []
    bad = []
    for key, value in (user or {}).items():
        path = prefix + (key,)
        if key not in (default or {}):
            bad.append(".".join(path))
        elif isinstance(value, dict) and isinstance(default[key], dict):
            bad.extend(_unknown_keys(value, default[key], path))
    return bad


def _suggest(key, default):
    """Nearest defined key, so a typo says what was probably meant."""
    import difflib
    flat = []

    def walk(d, pre=()):
        for k, v in (d or {}).items():
            if pre in _OPEN_SUBTREES:
                continue
            flat.append(".".join(pre + (k,)))
            if isinstance(v, dict):
                walk(v, pre + (k,))
    walk(default)
    near = difflib.get_close_matches(key, flat, n=1, cutoff=0.6)
    return f" (did you mean {near[0]!r}?)" if near else ""


def load_config(path: Optional[Union[str, Path]] = None) -> ConfigNamespace:
    """Load configs/default.yaml, optionally deep-merged with a user config.

    A user config may only set keys the default file defines.  Anything else is
    a typo or a leftover from a renamed option, and silently ignoring it is how
    a run ends up not doing what its config appears to say -- so it raises.
    """
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if path is not None:
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        unknown = _unknown_keys(user_cfg, cfg)
        if unknown:
            lines = "\n".join(f"  - {k}{_suggest(k, cfg)}" for k in unknown)
            raise ValueError(
                f"{path} sets keys that configs/default.yaml does not define:\n"
                f"{lines}\nRemove them, or add them to the default config first."
            )
        cfg = _deep_merge(cfg, user_cfg)

    return ConfigNamespace(cfg)
