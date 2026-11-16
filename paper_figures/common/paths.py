"""Repository paths, ``src/`` on ``sys.path``, and the package's output folders.

Nothing here imports NumPy: :func:`use_src` has to run before it is loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent      # paper_figures/
REPO_ROOT = PKG_ROOT.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFIGS_DIR = REPO_ROOT / "configs"
OUT_ROOT = PKG_ROOT / "out"
CACHE_DIR = PKG_ROOT / "cache"
ROLLOUTS_DIR = PKG_ROOT / "data" / "rollouts"


def use_src() -> None:
    """Put ``src/`` on ``sys.path`` as the scripts do, and pin BLAS to one thread."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from blas_threads import limit_blas_threads
    limit_blas_threads()


def fig_dirs(fig: str) -> dict:
    """``out/<fig>/{components,preview,data}``, created on demand."""
    base = OUT_ROOT / fig
    dirs = {"base": base, "components": base / "components",
            "preview": base / "preview", "data": base / "data"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def rel(path) -> str:
    """``path`` relative to the repository root, for manifests and logs."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()
