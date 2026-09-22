# -*- coding: utf-8 -*-
"""Pin the BLAS thread count before NumPy is imported.

Why this exists
---------------
Every Newton iteration and every adjoint step solves a small dense system:
``(n_free x n_free)``, with ``n_free = dim*nv - 4*dim`` — 94 for the 2D
default (nv=51) and 141 for the 3D one.

OpenBLAS switches from its serial kernel to a threaded one at about n = 100,
and for matrices this small the thread spawn/synchronise overhead is far
larger than the factorization itself.  Measured on a 16-core machine:

    size        default threads     OPENBLAS_NUM_THREADS=1
    94 x 94          0.057 ms             0.057 ms
    100 x 100       96.4   ms             0.063 ms      <- threshold
    141 x 141      260.8   ms             0.127 ms
    256 x 256      604.8   ms             0.615 ms

So the 2D default (94) lands just under the threshold and is fast, while the
3D default (141) lands just over it and pays a ~2000x penalty — which is why
3D looked dramatically slower than the 1.5x DOF increase can explain.  The
matrices are identical in structure; only their size changed.

This module must be imported *before* NumPy, because the BLAS backend reads
these variables when its shared library is loaded.  ``setdefault`` is used so
a value already exported in the shell always wins::

    OPENBLAS_NUM_THREADS=4 python scripts/train_adjoint.py
"""
from __future__ import annotations

import os

_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def limit_blas_threads(n: int = 1) -> None:
    """Set the BLAS thread-count variables to ``n`` unless already set.

    Call this at the top of an entry point, before ``import numpy``.
    """
    for var in _VARS:
        os.environ.setdefault(var, str(n))


def blas_thread_settings() -> dict:
    """The thread-count variables currently in effect (for logging)."""
    return {var: os.environ.get(var) for var in _VARS}
