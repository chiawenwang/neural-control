import numpy as np

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


def implicit_final_control_grad(J, free_index, fixed_index, a_free, control_6d, deltaL):
    """
    Implicit gradient via the adjoint method (single/final step only).
    Given:
      - the Jacobian sub-blocks of the equilibrium residual r_f(q_f, q_b) = 0, J = df/dq
      - a_free = dL/dq_free  (nonzero only on the free DOFs)
    Returns:
      g_u : dL/du (6D), corresponding to [x_l, y_l, th_l, x_r, y_r, th_r]

    Formula:
      J_ff^T v = a_free
      g_qb = - J_fb^T v
      g_u  = (d q_b / d u)^T g_qb
    """
    # Extract sub-blocks
    J_ff = J[np.ix_(free_index, free_index)]         # shape (nf,nf)
    J_fb = J[np.ix_(free_index, fixed_index)]        # shape (nf,8)

    # Adjoint solve
    v = np.linalg.solve(J_ff.T, a_free)              # shape (nf,)

    # dL/dq_b (8D)
    g_qb = - (J_fb.T @ v)                            # shape (8,)

    # map 8D -> 6D
    J_map = boundary_node_jacobian(control_6d, deltaL)  # (8,6)
    g_u = J_map.T @ g_qb                               # (6,)

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
      dxf_dxb : shape (n_free, 8)
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


def multistep_adjoint_gradient(u_seq, J_list, free_index, fixed_index, deltaL,
                                a_free_per_step, jac_reg=0.0):
    """
    Exact multi-step (receding-horizon) adjoint gradient: computes dL/du_k for
    every step k in the horizon, instead of computing it once at the final
    step and reusing that same gradient for every step.

    This is a direct port of the backward recursion in the original
    submission code (adjoint_RHC_any_node.py / adjoint_RHC_middle_tracking.py),
    adapted to this repository's control parameterization:
      - In the original code, the boundary state xb_k is accumulated linearly
        over time from a "control rate"
        (xb_k = xb_{k-1} + dlam * B @ u_rate_k), so the backward recursion
        needs an explicit state-transition matrix A (always zero here).
      - In this repository the policy network outputs a rate u_dot, but the
        integration from rate to accumulated control u_k
        (u_k = u_0 + dlam * cumsum(u_dot)) already happens inside torch's
        computational graph. So the numpy side here only needs to supply
        dL/du_k (the gradient w.r.t. the accumulated control), which is then
        dotted with u_k to build a surrogate loss that autograd backprops
        through — there's no need to reimplement the A/B state recursion.

    Physical assumption (identical to the original code, not a new
    approximation): the equilibrium is treated as evolving "along a
    continuously tracked branch", i.e. a small boundary perturbation δxb_i at
    step i is assumed to propagate approximately unchanged, through the
    subsequent warm-started Newton solves, to every later step's free DOFs
    (δxf_k ≈ δxf_i, k>=i). Under this assumption, any step's loss term
    a_free_k = dL_k/dxf_k should be accumulated backward onto every earlier
    step i<=k, using step i's own local Jacobian dxf_dxb_i to compute its
    contribution — rather than, as in the "approximate" version, using only
    the final step's Jacobian to stand in for every step.

    Parameters
    ----------
    u_seq : np.ndarray, shape (T, 6)
        The actual accumulated (absolute) control applied at each step,
        [x_l,y_l,th_l,x_r,y_r,th_r].
    J_list : list[np.ndarray], length T
        The full equilibrium Jacobian returned by objfun at each step.
    free_index, fixed_index : np.ndarray
        Free / boundary DOF indices (consistent with SimulatorEnv_2D).
    deltaL : float
        Discretization length of the rod.
    a_free_per_step : np.ndarray, shape (T, n_free)
        Gradient of step k's loss w.r.t. that step's free DOFs, dL_k/dq_free_k;
        rows are zero for steps with no attached loss term (e.g. the
        point-to-point task only has a loss at the final step).
    jac_reg : float
        Regularization coefficient for the equilibrium Jacobian (numerical
        stability).

    Returns
    -------
    g_u_seq : np.ndarray, shape (T, 6)
        dL_total/du_k, for every step's accumulated control.
    """
    u_seq = np.asarray(u_seq, dtype=float)
    a_free_per_step = np.asarray(a_free_per_step, dtype=float)
    T, nu = u_seq.shape

    g_u_seq = np.zeros((T, nu), dtype=float)
    lam_f_cum = np.zeros(len(free_index), dtype=float)

    for i in range(T - 1, -1, -1):
        lam_f_cum = lam_f_cum + a_free_per_step[i]

        dxf_dxb_i = local_dxf_dxb(J_list[i], free_index, fixed_index, jac_reg)
        B_i = boundary_node_jacobian(u_seq[i], deltaL)   # (8,6)

        g_qb_i = dxf_dxb_i.T @ lam_f_cum                  # (8,)
        g_u_seq[i] = B_i.T @ g_qb_i                        # (6,)

    return g_u_seq
