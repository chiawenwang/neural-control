#!/usr/bin/env python3
"""
Stateful Tangent Operator (STO) — adapted for neural-control-main
==================================================================
Uses the same 6D control parameterisation and boundary_node_jacobian (8x6)
as main.py / implicit_grad_tools.py.

Interface:
    sto = StatefulTangentOperator(...)
    sto.query(J_full, free_index, fixed_index, a_free, control_6d, deltaL)
        -> (g_u_6d, used_cache, kappa, reason)

The class caches J_ff^{-T} at a converged equilibrium and reuses it for
subsequent gradient queries until the condition number indicates the
cached factorisation is stale (branch switch / bifurcation).

Invalidation criteria (any triggers re-init):
    1. kappa(J_ff_new) > kappa_threshold
    2. kappa(J_ff_new) / kappa_cached > kappa_ratio
    3. reuse count since last init > max_reuse
"""

import numpy as np
from implicit_grad_tools import boundary_node_jacobian, implicit_final_control_grad


class StatefulTangentOperator:
    """
    Cached implicit adjoint operator for 6D boundary control.

    Initialisation cost:  O(n_f^3)   (one matrix inverse)
    Per-query cost:       O(n_f^2)   (matvecs only)
    Exact adjoint cost:   O(n_f^3)   per query (full solve each time)
    """

    def __init__(self, kappa_abs=1e6, kappa_ratio=3.0, max_reuse=1000): # Determine the thresholds based on...?
        self.kappa_abs   = kappa_abs
        self.kappa_ratio = kappa_ratio
        self.max_reuse   = max_reuse

        # cached state
        self._valid      = False
        self._J_ff_T_inv = None        # (n_f, n_f)
        self._J_fb       = None        # (n_f, 8)
        self._J_map      = None        # (8, 6)
        self._kappa      = None        # condition number at init
        self._free_idx   = None
        self._fixed_idx  = None
        self._reuse_cnt  = 0

        # statistics
        self.n_init      = 0           # number of (re-)initialisations
        self.n_cached    = 0           # queries answered from cache
        self.n_fallback  = 0           # queries that needed exact solve
        self.n_total     = 0           # total queries

    # ------------------------------------------------------------------
    # INITIALISE — factorise J_ff^T and cache coupling matrices
    # ------------------------------------------------------------------
    def initialise(self, J_full, free_index, fixed_index, control_6d, deltaL):
        """
        Extract an implicit adjoint operator at a converged equilibrium.
        Args:
            J_full:      full Jacobian (ndof x ndof) from Newton solver
            free_index:  array of free DOF indices
            fixed_index: array of fixed DOF indices (8 DOFs, both ends)
            control_6d:  current 6D control [x_l, y_l, th_l, x_r, y_r, th_r]
            deltaL:      segment length
        Returns:
            True if initialisation succeeded, False otherwise
        """
        self._free_idx  = free_index.copy()
        self._fixed_idx = fixed_index.copy()

        #Extract Jacobian sub-blocks
        J_ff = J_full[np.ix_(free_index, free_index)] # (n_f × n_f)

        # Check for NaN/Inf before any linear algebra
        if not np.all(np.isfinite(J_ff)):
            self._valid = False
            return False

        self._J_fb = J_full[np.ix_(free_index, fixed_index)].copy() # (n_f × 8)

        # boundary_node_jacobian from implicit_grad_tools: (8, 6)
        self._J_map = boundary_node_jacobian(control_6d, deltaL) # Go to 6D control space

        # invert J_ff^T
        J_ff_T = J_ff.T.copy()
        try:
            self._J_ff_T_inv = np.linalg.inv(J_ff_T)
        except np.linalg.LinAlgError:
            self._valid = False
            return False

        # condition number via SVD
        try:
            s = np.linalg.svd(J_ff, compute_uv=False)
        except np.linalg.LinAlgError:
            self._valid = False
            return False
        self._kappa = s[0] / max(s[-1], 1e-30)

        self._valid = True
        self.n_init += 1
        self._reuse_cnt = 0
        return True

    # ------------------------------------------------------------------
    # APPLY — cached tangent operator: a_free -> g_u (6D)
    # ------------------------------------------------------------------
    def _apply_cached(self, a_free):

        v      = self._J_ff_T_inv @ a_free       # (n_f,), O(n_f^2)
        g_qb   = -(self._J_fb.T @ v)             # (8,), O(n_f * 8)
        g_u    = self._J_map.T @ g_qb            # (6,), O(8 * 6)

        """
        In implicit_grad_tools:
        v    = np.linalg.solve(J_ff.T, a_free)   # ← O(n_f^3) linear solve
        g_qb = -(J_fb.T @ v)                     #
        g_u  = J_map.T @ g_qb                    # 
        """

        self.n_cached   += 1
        self._reuse_cnt += 1
        return g_u

    # ------------------------------------------------------------------
    # VALIDATE — check condition number at new equilibrium
    # ------------------------------------------------------------------
    def validate(self, J_full_new):
        """
        Check whether the cached STO is still valid at a new equilibrium.

        Returns:
            (is_valid, kappa_new, reason_string)
        """
        if not self._valid:
            return False, np.inf, "already_invalid"

        # Extract J_ff and compute condition number (with NaN/SVD protection)
        J_ff_new = J_full_new[np.ix_(self._free_idx, self._free_idx)]
        if not np.all(np.isfinite(J_ff_new)):
            return False, np.inf, "J_ff contains NaN/Inf"
        try:
            s = np.linalg.svd(J_ff_new, compute_uv=False)
        except np.linalg.LinAlgError:
            return False, np.inf, "SVD did not converge"
        kappa_new = s[0] / max(s[-1], 1e-30)

        # max reuse exceeded
        if self._reuse_cnt >= self.max_reuse:
            return False, kappa_new, f"max_reuse={self.max_reuse} reached"

        # absolute threshold
        if kappa_new > self.kappa_abs:
            return False, kappa_new, f"kappa={kappa_new:.1e} > abs_thresh={self.kappa_abs:.1e}"

        # ratio threshold
        ratio = kappa_new / max(self._kappa, 1e-30)
        if ratio > self.kappa_ratio:
            return False, kappa_new, f"kappa_ratio={ratio:.1f} > {self.kappa_ratio:.1f}"

        return True, kappa_new, "valid"

    # ------------------------------------------------------------------
    # INVALIDATE
    # ------------------------------------------------------------------
    def invalidate(self):
        self._valid = False
        self._J_ff_T_inv = None

    # ------------------------------------------------------------------
    # QUERY — main entry point (cache hit or fallback + re-init)
    # ------------------------------------------------------------------
    def query(self, J_full, free_index, fixed_index, a_free, control_6d, deltaL):
        """
        Compute dL/du (6D) using cache when valid, else exact solve + re-init.

        Args:
            J_full:      full Jacobian from Newton solver
            free_index:  free DOF indices
            fixed_index: fixed DOF indices (8 DOFs)
            a_free:      dL/dq_free
            control_6d:  current 6D control vector
            deltaL:      segment length

        Returns:
            (g_u, used_cache, kappa, reason)
            g_u:         gradient dL/du (6D)
            used_cache:  bool, True if cache was used
            kappa:       condition number at this equilibrium
            reason:      string describing what happened
        """
        self.n_total += 1

        if self._valid:
            ok, kappa_new, reason = self.validate(J_full)
            if ok:
                # ---- CACHE HIT ----
                g_u = self._apply_cached(a_free)
                return g_u, True, kappa_new, "cache_hit"
            else:
                # ---- INVALIDATED -> fallback to exact + re-init ----
                self.invalidate()
                g_u = implicit_final_control_grad(
                    J=J_full,
                    free_index=free_index,
                    fixed_index=fixed_index,
                    a_free=a_free,
                    control_6d=control_6d,
                    deltaL=deltaL,
                )
                self.n_fallback += 1
                self.initialise(J_full, free_index, fixed_index, control_6d, deltaL)
                return g_u, False, kappa_new, f"invalidated({reason})->re-init"
        else:
            # ---- COLD START -> exact + init ----
            g_u = implicit_final_control_grad(
                J=J_full,
                free_index=free_index,
                fixed_index=fixed_index,
                a_free=a_free,
                control_6d=control_6d,
                deltaL=deltaL,
            )
            self.n_fallback += 1
            self.initialise(J_full, free_index, fixed_index, control_6d, deltaL)
            return g_u, False, self._kappa, "cold_start->init"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @property
    def is_valid(self):
        return self._valid

    @property
    def kappa(self):
        return self._kappa if self._kappa else np.inf

    def stats_str(self):
        hit_rate = self.n_cached / max(self.n_total, 1) * 100
        return (f"STO stats — init: {self.n_init}  cached: {self.n_cached}  "
                f"fallback: {self.n_fallback}  total: {self.n_total}  "
                f"hit_rate: {hit_rate:.1f}%")
