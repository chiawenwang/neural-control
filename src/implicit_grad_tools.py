import numpy as np

def boundary_node_jacobian_3d(control_10d, deltaL):
    """
    Analytically computes J_map = d(q_b)/d(u) for the 3D environment, shape 12x10.
    - u   = [x_l, y_l, z_l, th_l, ph_l, x_r, y_r, z_r, th_r, ph_r]
    - q_b = [node_0, node_1, node_{n-2}, node_{n-1}]  (3 coords each, flattened),
      matching SimulatorEnv_3D._control_node_from_pose:

        node_0     = p_l
        node_1     = p_l + dL * d(th_l, ph_l)
        node_{n-2} = p_r - dL * d(th_r, ph_r)
        node_{n-1} = p_r

      with d(th, ph) = (cos(ph)cos(th), cos(ph)sin(th), sin(ph)).

    Returns:
      J_map: shape (12, 10)
    """
    x_l, y_l, z_l, th_l, ph_l, x_r, y_r, z_r, th_r, ph_r = control_10d

    J = np.zeros((12, 10), dtype=float)
    I3 = np.eye(3)

    def _dd(th, ph):
        """d(tangent)/d(th), d(tangent)/d(ph)."""
        ct, st = np.cos(th), np.sin(th)
        cp, sp = np.cos(ph), np.sin(ph)
        d_dth = np.array([-cp * st, cp * ct, 0.0])
        d_dph = np.array([-sp * ct, -sp * st, cp])
        return d_dth, d_dph

    # ----- left end: node_0 (rows 0:3), node_1 (rows 3:6); pose cols 0:5 -----
    J[0:3, 0:3] = I3                                  # node_0 = p_l
    J[3:6, 0:3] = I3                                  # node_1 = p_l + dL*d_l
    d_dth, d_dph = _dd(th_l, ph_l)
    J[3:6, 3] = deltaL * d_dth
    J[3:6, 4] = deltaL * d_dph

    # ----- right end: node_{n-2} (rows 6:9), node_{n-1} (rows 9:12); cols 5:10 -----
    J[6:9, 5:8] = I3                                  # node_{n-2} = p_r - dL*d_r
    J[9:12, 5:8] = I3                                 # node_{n-1} = p_r
    d_dth, d_dph = _dd(th_r, ph_r)
    J[6:9, 8] = -deltaL * d_dth
    J[6:9, 9] = -deltaL * d_dph

    return J


def boundary_node_jacobian(control_6d, deltaL):
    """
    Analytically computes J_map = d(q_b)/d(u), shape 8x6.
    - u = [x_l, y_l, th_l, x_r, y_r, th_r]
    - q_b = [X0,Y0,X1,Y1,Xn_2,Yn_2,Xn_1,Yn_1]  (matches env._control_node_from_xytheta)

    Returns:
      J_map: shape (8,6)
    """
    x_l, y_l, th_l, x_r, y_r, th_r = control_6d

    J = np.zeros((8, 6), dtype=float)

    # -----------------------------
    # Left end:
    # node0 = (x_l, y_l)
    # node1 = (x_l + cos(th_l)*dL, y_l + sin(th_l)*dL)
    # -----------------------------
    # X0, Y0
    J[0, 0] = 1.0  # dX0/dx_l
    J[1, 1] = 1.0  # dY0/dy_l

    # X1, Y1 w.r.t x_l, y_l
    J[2, 0] = 1.0
    J[3, 1] = 1.0

    # X1, Y1 w.r.t th_l
    J[2, 2] = -np.sin(th_l) * deltaL
    J[3, 2] =  np.cos(th_l) * deltaL

    # -----------------------------
    # Right end:
    # node_{n-2} = (x_r - cos(th_r)*dL, y_r - sin(th_r)*dL)
    # node_{n-1} = (x_r, y_r)
    # -----------------------------
    # Xn_2, Yn_2 w.r.t x_r, y_r
    J[4, 3] = 1.0
    J[5, 4] = 1.0

    # Xn_2, Yn_2 w.r.t th_r
    # Xn_2 = x_r - cos(th_r)*dL  -> d/dth = +sin(th_r)*dL
    # Yn_2 = y_r - sin(th_r)*dL  -> d/dth = -cos(th_r)*dL
    J[4, 5] =  np.sin(th_r) * deltaL
    J[5, 5] = -np.cos(th_r) * deltaL

    # Xn_1, Yn_1
    J[6, 3] = 1.0
    J[7, 4] = 1.0

    return J


def boundary_node_jacobian_3d_twist(control_12d, deltaL):
    """``d(q_b)/d(u)`` for the twisting 3D rod, shape 14x12.

    - u   = [x_l, y_l, z_l, th_l, ph_l, yaw_l,  x_r, y_r, z_r, th_r, ph_r, yaw_r]
    - q_b = [node_0, node_1, node_{n-2}, node_{n-1}, yaw_l, yaw_r]

    The 12 position rows are exactly the twist-free map; the two extra rows are
    the clamped yaws, which the control sets directly -- so each is a single 1.
    That directness is the point of measuring yaw against a frame built from
    the gripper's own tangent: no geometry enters here, and the bend/twist
    coupling is carried entirely by the holonomy inside the energy.
    """
    control_12d = np.asarray(control_12d, dtype=float)
    J = np.zeros((14, 12), dtype=float)
    I3 = np.eye(3)

    def _dd(th, ph):
        ct, st = np.cos(th), np.sin(th)
        cp, sp = np.cos(ph), np.sin(ph)
        return (np.array([-cp * st, cp * ct, 0.0]),
                np.array([-sp * ct, -sp * st, cp]))

    th_l, ph_l = control_12d[3], control_12d[4]
    th_r, ph_r = control_12d[9], control_12d[10]

    # left end: node_0 (rows 0:3), node_1 (rows 3:6); pose cols 0:6
    J[0:3, 0:3] = I3
    J[3:6, 0:3] = I3
    d_dth, d_dph = _dd(th_l, ph_l)
    J[3:6, 3] = deltaL * d_dth
    J[3:6, 4] = deltaL * d_dph

    # right end: node_{n-2} (rows 6:9), node_{n-1} (rows 9:12); pose cols 6:12
    J[6:9, 6:9] = I3
    J[9:12, 6:9] = I3
    d_dth, d_dph = _dd(th_r, ph_r)
    J[6:9, 9] = -deltaL * d_dth
    J[6:9, 10] = -deltaL * d_dph

    # the two clamped twist angles are the yaw inputs, verbatim
    J[12, 5] = 1.0
    J[13, 11] = 1.0
    return J


def _boundary_jacobian(control, deltaL, dim=2, twist=False, rows=None):
    """Dimension dispatch for the d(q_b)/d(u) map.

    ``rows`` keeps only the boundary DOFs that are actually clamped (the
    environment's ``boundary_rows``), e.g. just the right end when the left
    end is a free tip.  None -> all of them, the both-ends-clamped case.
    """
    if dim == 2:
        if twist:
            raise ValueError("twist requires dim=3")
        B = boundary_node_jacobian(control, deltaL)
    elif dim == 3:
        B = (boundary_node_jacobian_3d_twist(control, deltaL) if twist
             else boundary_node_jacobian_3d(control, deltaL))
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    return B if rows is None else B[rows]


def implicit_final_control_grad(J, free_index, fixed_index, a_free, control_6d, deltaL,
                                 dim=2, twist=False, boundary_rows=None):
    """
    Implicit gradient via the adjoint method (single/final step only).
    Given:
      - the Jacobian sub-blocks of the equilibrium residual r_f(q_f, q_b) = 0, J = df/dq
      - a_free = dL/dq_free  (nonzero only on the free DOFs)
    Returns:
      g_u : dL/du — 6D [x_l, y_l, th_l, x_r, y_r, th_r] when dim=2,
            10D [x_l, y_l, z_l, th_l, ph_l, x_r, y_r, z_r, th_r, ph_r] when dim=3

    Formula:
      J_ff^T v = a_free
      g_qb = - J_fb^T v
      g_u  = (d q_b / d u)^T g_qb
    """
    # Extract sub-blocks
    J_ff = J[np.ix_(free_index, free_index)]         # shape (nf,nf)
    J_fb = J[np.ix_(free_index, fixed_index)]        # shape (nf, 4*dim)

    # Adjoint solve
    v = np.linalg.solve(J_ff.T, a_free)              # shape (nf,)

    # dL/dq_b (4*dim,)
    g_qb = - (J_fb.T @ v)

    # map boundary DOFs -> end-effector pose DOFs
    J_map = _boundary_jacobian(control_6d, deltaL, dim, twist, boundary_rows)   # (n_qb, n_u)
    g_u = J_map.T @ g_qb                                  # (n_u,)

    return g_u


def local_dxf_dxb(J, free_index, fixed_index, jac_reg=0.0):
    """
    Single-step implicit sensitivity dq_free/dq_b (determined purely by that
    step's equilibrium Jacobian, near the current equilibrium point).

    From the implicit function theorem applied to the equilibrium residual
    r_f(q_f, q_b) = 0:
        J_ff @ dq_f + J_fb @ dq_b = 0
        => dq_f/dq_b = -J_ff^{-1} @ J_fb

    Returns:
      dxf_dxb : shape (n_free, 4*dim)
    """
    J_ff = J[np.ix_(free_index, free_index)]
    J_fb = J[np.ix_(free_index, fixed_index)]
    if jac_reg:
        J_ff = J_ff + jac_reg * np.eye(J_ff.shape[0], dtype=J_ff.dtype)
    try:
        dxf_dxb = -np.linalg.solve(J_ff, J_fb)
    except np.linalg.LinAlgError:
        dxf_dxb = -np.linalg.lstsq(J_ff, J_fb, rcond=None)[0]
    return dxf_dxb


#: The multi-step adjoint gradients ``train.grad_method`` selects between (see
#: :func:`multistep_adjoint_gradient`); the first is the default.
GRAD_METHODS = ("state", "proxy")


def multistep_adjoint_gradient(u_seq, J_list, free_index, fixed_index, deltaL,
                                a_free_per_step, jac_reg=0.0, dim=2, twist=False,
                                boundary_rows=None, *, grad_method):
    """
    Per-step adjoint vectors of a T-step quasi-static rollout, for either
    gradient method (``train.grad_method``).

    Notation, for the steps k = 0..T-1:
      - r_k   the rate the policy outputs; the executed command is
              u_k = u_{k-1} + dlam * r_k (integrated in torch's graph)
      - B_k   d(q_b)/d(u) at u_k (:func:`_boundary_jacobian`)
      - S_k   dq_free/dq_b = -J_ff^{-1} J_fb at step k's equilibrium
              (:func:`local_dxf_dxb`, with its regularization and lstsq
              fallback)
      - a_k   dL/dq_free_k; zero on a step that carries no loss term

    Both methods take the same forward pass (the states the warm-started
    Newton solves actually reached), solve with each step's own Jacobian, and
    give a lower-triangular dL/dr_j summed from the back.  They differ only in
    which step's S projects a later step's loss, and so in what the result is
    attached to:

    ``state`` -- exact; the default
        On one equilibrium branch the state depends only on the current
        boundary, q_k = X(q_b,k), so each a_k is projected by its own S_k::

            p_k = B_k^T S_k^T a_k                     (row k of the result)

        p_k is dL/du_k.  Attached to the executed command u_k, autograd's pass
        back through the integration sums it onto every earlier rate,
        dL/dr_j = dlam * sum_{k>=j} p_k.  No third derivatives appear, and on
        the branch the rollout followed this is the exact gradient.  A step
        with a_k = 0 needs no solve, so a terminal loss costs one.

    ``proxy`` -- frozen tangent
        Differentiate q_free = q_free(0) + int S B r dlam and drop the term in
        dS/dtheta (it needs the energy's third derivative).  The future
        losses are summed in state space first and projected by the S of the
        step where the control was applied::

            A_j = a_j + A_{j+1},   A_T = 0
            q_j = B_j^T S_j^T A_j                     (row j of the result)

        dL/dr_j = dlam * q_j, so q_j belongs on the *rate* r_j: attached to
        u_j, autograd would sum the future losses a second time.  It treats
        the state as remembering a perturbation perfectly
        (dq_{k+1}/dq_k = I), which is exact only while S stays constant along
        the path; finer steps do not make the error go away.  It is the
        backward recursion of the original submission code
        (adjoint_RHC_any_node.py), which put it on the network's rate output.

    Parameters
    ----------
    u_seq : np.ndarray, shape (T, n_u)
        The actual accumulated (absolute) control applied at each step:
        [x_l,y_l,th_l,x_r,y_r,th_r] in 2D (n_u=6), or
        [x_l,y_l,z_l,th_l,ph_l,x_r,y_r,z_r,th_r,ph_r] in 3D (n_u=10).
    J_list : list[np.ndarray], length T
        The full equilibrium Jacobian returned by objfun at each step.
    free_index, fixed_index : np.ndarray
        Free / boundary DOF indices (consistent with the environment).
    deltaL : float
        Discretization length of the rod.
    a_free_per_step : np.ndarray, shape (T, n_free)
        Gradient of step k's loss w.r.t. that step's free DOFs, dL_k/dq_free_k;
        rows are zero for steps with no attached loss term (e.g. the
        point-to-point task only has a loss at the final step).
    jac_reg : float
        Regularization coefficient for the equilibrium Jacobian (numerical
        stability).
    dim : int
        Spatial dimension of the environment, 2 or 3. Selects the
        d(q_b)/d(u) map.
    twist : bool
        Whether the rod carries torsion, i.e. whether the control has a yaw
        per end and the boundary has a clamped twist angle per end.
    boundary_rows : np.ndarray or None
        Rows of d(q_b)/d(u) that correspond to ``fixed_index`` (the
        environment's ``boundary_rows``). None -> both ends clamped.
    grad_method : str
        ``state`` or ``proxy`` (:data:`GRAD_METHODS`).  Required, so a caller
        cannot fall back on a default without saying so.

    Returns
    -------
    g_seq : np.ndarray, shape (T, n_u)
        ``state``: p_k = dL/du_k, for the executed commands.
        ``proxy``: q_j, for the rates (dL/dr_j = dlam * q_j).
    """
    if grad_method not in GRAD_METHODS:
        raise ValueError(f"grad_method must be one of {GRAD_METHODS}, got {grad_method!r}")
    accumulate = grad_method == "proxy"
    u_seq = np.asarray(u_seq, dtype=float)
    a_free_per_step = np.asarray(a_free_per_step, dtype=float)
    T, nu = u_seq.shape

    g_seq = np.zeros((T, nu), dtype=float)
    A = np.zeros(len(free_index), dtype=float)

    for i in range(T - 1, -1, -1):
        A = (A if accumulate else 0.0) + a_free_per_step[i]     # proxy: A_i; state: a_i
        if not A.any():                     # S_i^T 0 = 0: nothing to solve for
            continue

        # S_i^T A = -J_fb^T eta with J_ff eta = A: one adjoint solve with the
        # (symmetric) equilibrium Hessian instead of the n_qb solves that
        # forming S_i would take (paper eq. adjoint).
        g_qb_i = adjoint_pull(J_list[i], free_index, fixed_index, A, jac_reg)
        B_i = _boundary_jacobian(u_seq[i], deltaL, dim, twist, boundary_rows)   # (n_qb, n_u)
        g_seq[i] = B_i.T @ g_qb_i                          # (n_u,)

    return g_seq


#: How many adjoint solves :func:`adjoint_pull` has made (H1 cost bookkeeping).
N_ADJOINT_SOLVES = [0]


def adjoint_pull(J, free_index, fixed_index, a_free, jac_reg=0.0):
    """``dL/dq_b = S^T a = -J_fb^T eta`` with ``J_ff eta = a``.

    ``J`` is the Hessian of the total potential at the realized equilibrium
    (``newton.solve_equilibrium``'s Jacobian: symmetric, and positive definite
    at a stable equilibrium), so the solve uses its Cholesky factor -- the
    factorization the stability check makes.  If the factorization fails (an
    unstable state, which the rollout should have rejected) it falls back to
    an LU solve, as the old code did.  ``jac_reg`` adds ``jac_reg * I`` to
    ``J_ff``; keep it 0 for the exact gradient.
    """
    from scipy.linalg import cho_factor, cho_solve, LinAlgError
    J_ff = J[np.ix_(free_index, free_index)]
    J_fb = J[np.ix_(free_index, fixed_index)]
    if jac_reg:
        J_ff = J_ff + jac_reg * np.eye(J_ff.shape[0], dtype=J_ff.dtype)
    N_ADJOINT_SOLVES[0] += 1
    try:
        eta = cho_solve(cho_factor(0.5 * (J_ff + J_ff.T), lower=True, check_finite=False),
                        a_free, check_finite=False)
    except LinAlgError:
        try:
            eta = np.linalg.solve(J_ff.T, a_free)
        except np.linalg.LinAlgError:
            eta = np.linalg.lstsq(J_ff.T, a_free, rcond=None)[0]
    return -(J_fb.T @ eta)
