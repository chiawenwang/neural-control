# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from render_neurips import render_dlo_neurips

############################

def crossMat(a):
    A = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])
    return A

def gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """
    Returns the derivative of bending energy E_k^b with respect to
    x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.

    Parameters:
    xkm1, ykm1 : float
        Coordinates of the previous node (x_{k-1}, y_{k-1}).
    xk, yk : float
        Coordinates of the current node (x_k, y_k).
    xkp1, ykp1 : float
        Coordinates of the next node (x_{k+1}, y_{k+1}).
    curvature0 : float
        Discrete natural curvature at node (xk, yk).
    l_k : float
        Voronoi length of node (xk, yk).
    EI : float
        Bending stiffness.

    Returns:
    dF : np.ndarray
        Derivative of bending energy.
    """

    # Nodes in 3D
    node0 = np.array([xkm1, ykm1, 0.0])
    node1 = np.array([xk, yk, 0])
    node2 = np.array([xkp1, ykp1, 0])

    # Unit vectors along z-axis
    m2e = np.array([0, 0, 1])
    m2f = np.array([0, 0, 1])

    kappaBar = curvature0

    # Initialize gradient of curvature
    gradKappa = np.zeros(6)

    # Edge vectors
    ee = node1 - node0
    ef = node2 - node1

    # Norms of edge vectors
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)

    # Unit tangents
    te = ee / norm_e
    tf = ef / norm_f

    # Curvature binormal
    kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))

    chi = 1.0 + np.dot(te, tf)
    if chi < 1e-8:
        chi = 1e-8  # Prevent division by zero
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    # Curvature
    kappa1 = kb[2]

    # Gradient of kappa1 with respect to edge vectors
    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))

    # Populate the gradient of kappa
    gradKappa[0:2] = -Dkappa1De[0:2]
    gradKappa[2:4] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6] = Dkappa1Df[0:2]

    # Gradient of bending energy
    dkappa = kappa1 - kappaBar
    dF = gradKappa * EI * dkappa / l_k

    return dF

def hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """
    Returns the Hessian (second derivative) of bending energy E_k^b
    with respect to x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.

    Parameters:
    xkm1, ykm1 : float
        Coordinates of the previous node (x_{k-1}, y_{k-1}).
    xk, yk : float
        Coordinates of the current node (x_k, y_k).
    xkp1, ykp1 : float
        Coordinates of the next node (x_{k+1}, y_{k+1}).
    curvature0 : float
        Discrete natural curvature at node (xk, yk).
    l_k : float
        Voronoi length of node (xk, yk).
    EI : float
        Bending stiffness.

    Returns:
    dJ : np.ndarray
        Hessian of bending energy.
    """

    # Nodes in 3D
    node0 = np.array([xkm1, ykm1, 0])
    node1 = np.array([xk, yk, 0])
    node2 = np.array([xkp1, ykp1, 0])

    # Unit vectors along z-axis
    m2e = np.array([0, 0, 1])
    m2f = np.array([0, 0, 1])

    kappaBar = curvature0

    # Initialize gradient of curvature
    gradKappa = np.zeros(6)

    # Edge vectors
    ee = node1 - node0
    ef = node2 - node1

    # Norms of edge vectors
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)
    if norm_e < 1e-8 or norm_f < 1e-8:
        return np.zeros(6)  # Prevent division by zero

    # Unit tangents
    te = ee / norm_e
    tf = ef / norm_f

    # Curvature binormal
    kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))

    chi = 1.0 + np.dot(te, tf)
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    # Curvature
    kappa1 = kb[2]

    # Gradient of kappa1 with respect to edge vectors
    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))

    # Populate the gradient of kappa
    gradKappa[0:2] = -Dkappa1De[0:2]
    gradKappa[2:4] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6] = Dkappa1Df[0:2]

    # Compute the Hessian (second derivative of kappa)
    DDkappa1 = np.zeros((6, 6))

    norm2_e = norm_e**2
    norm2_f = norm_f**2

    Id3 = np.eye(3)

    # Helper matrices for second derivatives
    tt_o_tt = np.outer(tilde_t, tilde_t)
    tmp = np.cross(tf, tilde_d2)
    tf_c_d2t_o_tt = np.outer(tmp, tilde_t)
    kb_o_d2e = np.outer(kb, m2e)

    D2kappa1De2 = (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tf_c_d2t_o_tt.T) / norm2_e - \
                  kappa1 / (chi * norm2_e) * (Id3 - np.outer(te, te)) + \
                  (kb_o_d2e + kb_o_d2e.T) / (4 * norm2_e)

    tmp = np.cross(te, tilde_d2)
    te_c_d2t_o_tt = np.outer(tmp, tilde_t)
    tt_o_te_c_d2t = te_c_d2t_o_tt.T
    kb_o_d2f = np.outer(kb, m2f)

    D2kappa1Df2 = (2 * kappa1 * tt_o_tt + te_c_d2t_o_tt + te_c_d2t_o_tt.T) / norm2_f - \
                  kappa1 / (chi * norm2_f) * (Id3 - np.outer(tf, tf)) + \
                  (kb_o_d2f + kb_o_d2f.T) / (4 * norm2_f)
    D2kappa1DeDf = -kappa1 / (chi * norm_e * norm_f) * (Id3 + np.outer(te, tf)) \
                  + 1.0 / (norm_e * norm_f) * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt + \
                  tt_o_te_c_d2t - crossMat(tilde_d2))
    D2kappa1DfDe = D2kappa1DeDf.T

    # Populate the Hessian of kappa
    DDkappa1[0:2, 0:2] = D2kappa1De2[0:2, 0:2]
    DDkappa1[0:2, 2:4] = -D2kappa1De2[0:2, 0:2] + D2kappa1DeDf[0:2, 0:2]
    DDkappa1[0:2, 4:6] = -D2kappa1DeDf[0:2, 0:2]
    DDkappa1[2:4, 0:2] = -D2kappa1De2[0:2, 0:2] + D2kappa1DfDe[0:2, 0:2]
    DDkappa1[2:4, 2:4] = D2kappa1De2[0:2, 0:2] - D2kappa1DeDf[0:2, 0:2] - \
                         D2kappa1DfDe[0:2, 0:2] + D2kappa1Df2[0:2, 0:2]
    DDkappa1[2:4, 4:6] = D2kappa1DeDf[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]
    DDkappa1[4:6, 0:2] = -D2kappa1DfDe[0:2, 0:2]
    DDkappa1[4:6, 2:4] = D2kappa1DfDe[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]
    DDkappa1[4:6, 4:6] = D2kappa1Df2[0:2, 0:2]

    # Hessian of bending energy
    dkappa = kappa1 - kappaBar
    dJ = 1.0 / l_k * EI * np.outer(gradKappa, gradKappa)
    dJ += 1.0 / l_k * dkappa * EI * DDkappa1

    return dJ

def gradEs(xk, yk, xkp1, ykp1, l_k, EA):
    """
    Calculate the gradient of the stretching energy with respect to the coordinates.

    Args:
    - xk (float): x coordinate of the current point
    - yk (float): y coordinate of the current point
    - xkp1 (float): x coordinate of the next point
    - ykp1 (float): y coordinate of the next point
    - l_k (float): reference length
    - EA (float): elastic modulus

    Returns:
    - F (np.array): Gradient array
    """
    F = np.zeros(4)
    F[0] = -(1.0 - np.sqrt((xkp1 - xk)**2.0 + (ykp1 - yk)**2.0) / l_k) * ((xkp1 - xk)**2.0 + (ykp1 - yk)**2.0)**(-0.5) / l_k * (-2.0 * xkp1 + 2.0 * xk)
    F[1] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1) / l_k * (-0.2e1 * ykp1 + 0.2e1 * yk)
    F[2] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1) / l_k * (0.2e1 * xkp1 - 0.2e1 * xk)
    F[3] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1) / l_k * (0.2e1 * ykp1 - 0.2e1 * yk)

    F = 0.5 * EA * l_k * F  # Scale by EA and l_k

    return F

def hessEs(xk, yk, xkp1, ykp1, l_k, EA):
    """
    This function returns the 4x4 Hessian of the stretching energy E_k^s with
    respect to x_k, y_k, x_{k+1}, and y_{k+1}.
    """
    J = np.zeros((4, 4))  # Initialize the Hessian matrix
    J11 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * xkp1 + 2 * xk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((-2 * xkp1 + 2 * xk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J12 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * ykp1 + 2 * yk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (-2 * ykp1 + 2 * yk) / 0.2e1
    J13 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (2 * xkp1 - 2 * xk) / 0.2e1 + 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J14 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (2 * ykp1 - 2 * yk) / 0.2e1
    J22 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * ykp1 + 2 * yk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((-2 * ykp1 + 2 * yk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J23 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) * (-2 * ykp1 + 2 * yk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * ykp1 + 2 * yk) * (2 * xkp1 - 2 * xk) / 0.2e1
    J24 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (-2 * ykp1 + 2 * yk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * ykp1 + 2 * yk) * (2 * ykp1 - 2 * yk) / 0.2e1 + 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J33 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((2 * xkp1 - 2 * xk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J34 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (2 * xkp1 - 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (2 * xkp1 - 2 * xk) * (2 * ykp1 - 2 * yk) / 0.2e1
    J44 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((2 * ykp1 - 2 * yk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k

    J = np.array([[J11, J12, J13, J14],
                   [J12, J22, J23, J24],
                   [J13, J23, J33, J34],
                   [J14, J24, J34, J44]])

    J *= 0.5 * EA * l_k

    return J

"""# Functions to create elastic force vector and its Hessian"""

def getFs(q, EA, deltaL):
  # q - DOF vector of size N
  # EA - stretching stiffness
  # deltaL - undeformed reference length (assume to be a scalar for this simple example)
  # Output:
  # Fs - a vector (negative gradient of elastic stretching force)
  # Js - a matrix (negative hessian of elastic stretching force)

  ndof = q.size # Number of DOFs
  N = ndof // 2 # Number of nodes

  Fs = np.zeros(ndof) # stretching force
  Js = np.zeros((ndof, ndof))

  for k in range(0, N-1):
      # May need to modify if network of beams
      # k-th stretching spring (USE A LOOP for the general case
      xkm1 = q[2*k] # x coordinate of the first node
      ykm1 = q[2*k+1] # y coordinate of the first node
      xk = q[2*k+2] # x coordinate of the second node
      yk = q[2*k+3] # y coordinate of the second node
      ind = np.arange(2*k, 2*k+4) # 0, 1, 2, 3 for k = 0
      gradEnergy = gradEs(xkm1, ykm1, xk, yk, deltaL, EA)
      hessEnergy = hessEs(xkm1, ykm1, xk, yk, deltaL, EA)

      Fs[ind] -= gradEnergy # force = - gradient of energy. Fs is the stretching force
      Js[np.ix_(ind, ind)] -= hessEnergy # index vector: 0:4

  return Fs, Js

def getFb(q, EI, deltaL):
  # q - DOF vector of size N
  # EI - bending stiffness
  # deltaL - undeformed Voronoi length (assume to be a scalar for this simple example)
  # Output:
  # Fb - a vector (negative gradient of elastic stretching force)
  # Jb - a matrix (negative hessian of elastic stretching force)

  ndof = q.size # Number of DOFs
  N = ndof // 2 # Number of nodes

  Fb = np.zeros(ndof) # bending force
  Jb = np.zeros((ndof, ndof))

  # First bending spring (USE A LOOP for the general case)
  for k in range(1, N-1):
    xkm1 = q[2*k-2] # x coordinate of the first node
    ykm1 = q[2*k-1] # y coordinate of the first node
    xk = q[2*k] # x coordinate of the second node
    yk = q[2*k+1] # y coordinate of the second node
    xkp1 = q[2*k+2] # x coordinate of the third node
    ykp1 = q[2*k+3] # y coordinate of the third node
    ind = np.arange(2*k-2, 2*k+4)
    gradEnergy = gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI)
    hessEnergy = hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI)

    Fb[ind] -= gradEnergy # force = - gradient of energy. Fb is the stretching force
    Jb[np.ix_(ind, ind)] -= hessEnergy # index vector: 0:6

  return Fb, Jb


def objfun(q_old, u_old, dt, tol, maximum_iter,
           m, mMat, EI, EA, W, deltaL, free_index,
           control_node=None,
           use_inertia=False):
    """
    单步隐式时间积分（Newton-Raphson）
    """
    if control_node is None:
        raise ValueError("Control node position must be provided.")

    q_new = q_old.copy()
    q_new[:4] = control_node[:4]  # left end nodes
    q_new[-4:] = control_node[4:]  # right end nodes

    iter_count = 0
    error = tol * 10
    flag = 1

    while error > tol:
        # inertia term should be 0 for quasi-static assumption
        if use_inertia:
            F_inertia = m / dt * ((q_new - q_old) / dt - u_old)
            J_inertia = mMat / (dt ** 2)
        else:
            F_inertia = np.zeros_like(q_new)
            J_inertia = np.zeros((q_new.size, q_new.size))

        Fs, Js = getFs(q_new, EA, deltaL)
        Fb, Jb = getFb(q_new, EI, deltaL)
        F_elastic = Fs + Fb
        J_elastic = Js + Jb

        f = F_inertia - F_elastic - W
        J = J_inertia - J_elastic

        f_free = f[free_index]
        J_free = J[np.ix_(free_index, free_index)]

        dq_free = np.linalg.solve(J_free, f_free)
        q_new[free_index] = q_new[free_index] - dq_free

        error = np.linalg.norm(f_free)
        iter_count += 1
        if iter_count > maximum_iter:
            flag = -1
            print("Maximum number of iterations reached.")
            break

    return q_new, flag, J


############################
# SimulatorEnv_2D
############################

class SimulatorEnv_2D:
    """
    一个规范化的弹性杆仿真环境：
      - 控制量: (x, y, theta) 由外部给定，对应右端末端的位置与方向
      - 状态: 内部杆节点位置 q, 速度 u, 当前时间 ctime 等
      - 接口: reset(), step(control), render()
    """

    def __init__(self,
                 nv=51,
                 dt=0.01,
                #  dt=1.0,
                 rod_length=1.0,
                 total_time=10.0,
                 R_outer=0.013,
                 r_inner=0.011,
                 E_al=70e9,
                 rho_al=2700,
                 max_newton_iter=1000,
                 save_history=False):
        self.nv = nv
        self.dt = dt
        self.rod_length = rod_length
        self.total_time = total_time
        self.R_outer = R_outer
        self.r_inner = r_inner
        self.E_al = E_al
        self.rho_al = rho_al
        self.max_newton_iter = max_newton_iter
        self.save_history = save_history

        # 截面与材料参数
        self.I_sec = np.pi * (R_outer ** 4 - r_inner ** 4) / 4
        self.A_sec = np.pi * (R_outer ** 2 - r_inner ** 2)
        self.EI = E_al * self.I_sec
        self.EA = E_al * self.A_sec

        # 空间离散
        self.deltaL = rod_length / (nv - 1)

        # 节点坐标（初始直杆）
        self.nodes = np.zeros((nv, 2))
        for c in range(nv):
            self.nodes[c, 0] = c * self.deltaL
            self.nodes[c, 1] = 0.0

        # 质量
        m_node = np.pi * (R_outer ** 2 - r_inner ** 2) * rod_length * rho_al / (nv - 1)
        self.m = np.zeros(2 * nv)
        for k in range(nv):
            self.m[2 * k] = m_node
            self.m[2 * k + 1] = m_node
        self.mMat = np.diag(self.m)

        # 重力
        self.W = np.zeros(2 * nv)
        self.W[1::2] = -9.81 * m_node

        # 牛顿迭代容差
        self.tol = self.EI / rod_length ** 2 * 1e-3

        # 约束 DOF
        all_DOFs = np.arange(2 * nv)
        self.fixed_index = np.array([0, 1, 2, 3, 2 * nv - 4, 2 * nv - 3, 2 * nv - 2, 2 * nv - 1])
        self.free_index = np.setdiff1d(all_DOFs, self.fixed_index)

        # 中间节点索引
        # self.mid_node = nv // 2 + 1  # 与原脚本保持一致
        self.mid_node = nv // 2  # 与原脚本保持一致

        # 总步数
        self.Nsteps = int(round(total_time / dt))

        self.dlamda = 1 / self.Nsteps

        # 状态变量（在 reset 中初始化）
        self.q = None
        self.u = None
        self.ctime = None
        self.current_step = None
        self.lamda = None

        # 历史记录（可选）
        self.history = {
            "error": [],
            "control": [],
            "midx": [],
            "midy": [],
            "q1x": [],
            "q1y": [],
            "q3x": [],
            "q3y": [],
        }

    # ---------- inner tools ----------

    def _control_node_from_xytheta(self, control_vec):

        """
          node_0 = (x_l, y_l)
          node_1 = (x_l + np.cos(th_l)*self.deltaL, y_l + np.sin(th_l)*self.deltaL)
          ...
          node_{N-1} = (x_r, y_r)
          node_{N-2} = (x_r - np.cos(th_r)*self.deltaL, y_r - np.sin(th_r)*self.deltaL)
        order: [node_0_x, node_0_y, node_1_x, node_1_y, ..., node_{N-2}_x, node_{N-2}_y, node_{N-1}_x, node_{N-1}_y]
        """
        x_l, y_l, th_l, x_r, y_r, th_r = control_vec
        
        X_0 = x_l
        Y_0 = y_l
        X_1 = x_l + np.cos(th_l) * self.deltaL
        Y_1 = y_l + np.sin(th_l) * self.deltaL
        Xn_2 = x_r - np.cos(th_r) * self.deltaL
        Yn_2 = y_r - np.sin(th_r) * self.deltaL
        Xn_1 = x_r
        Yn_1 = y_r
        return np.array([X_0, Y_0, X_1, Y_1, Xn_2, Yn_2, Xn_1, Yn_1], dtype=float)

    def _get_midpoint(self):
        """返回当前中点的坐标 (midx, midy)。"""
        x_arr = self.q[::2]
        y_arr = self.q[1::2]
        midx = x_arr[self.mid_node]
        midy = y_arr[self.mid_node]
        return midx, midy

    def _get_observation(self):
        """观测量仅包含中点坐标。"""
        midx, midy = self._get_midpoint()
        return {
            "time": self.ctime,
            "midx": midx,
            "midy": midy
        }

    # ---------- API ----------

    def reset(self,
              init_x=None,
              init_y=None,
              init_theta=None):
        """
        重置环境，返回初始观测。
        控制量初值只影响你第一步时施加的控制，这里默认末端在 (rod_length, 0, 0)。
        """
        if init_x is None:
            init_x = self.rod_length
        if init_y is None:
            init_y = 0.0
        if init_theta is None:
            init_theta = 0.0

        self.q = np.zeros(2 * self.nv)
        for c in range(self.nv):
            self.q[2 * c] = self.nodes[c, 0]
            self.q[2 * c + 1] = self.nodes[c, 1]

        self.u = np.zeros_like(self.q)
        self.ctime = 0.0
        self.current_step = 0
        self.lamda = 0.0

        # 清空历史
        if self.save_history:
            for k in self.history:
                self.history[k] = []

        # 不在 reset 里做一步积分，只返回当前观测
        return self._get_observation()

    def step(self, control, use_inertia=False):
        """
        单步仿真：
          control: (x, y, theta)，numpy 数组或可迭代
        返回:
          obs, reward, done, info
        """
        control = np.asarray(control, dtype=float).flatten()
        if control.shape[0] != 6:
            raise ValueError("control must be a 6D vector: [x_l, y_l, th_l, x_r, y_r, th_r].")

        # x, y, theta = control
        control_node = self._control_node_from_xytheta(control)

        # 调用隐式积分器
        q_new, flag, J = objfun(
            q_old=self.q,
            u_old=self.u,
            dt=self.dt,
            tol=self.tol,
            maximum_iter=self.max_newton_iter,
            m=self.m,
            mMat=self.mMat,
            EI=self.EI,
            EA=self.EA,
            W=self.W,
            deltaL=self.deltaL,
            free_index=self.free_index,
            control_node=control_node,
            use_inertia=use_inertia
        )

        # 更新速度、时间
        u_new = (q_new - self.q) / self.dt
        self.q = q_new
        self.u = u_new
        self.ctime += self.dt
        self.current_step += 1
        self.lamda = self.current_step * self.dlamda

        # 观测与误差
        obs = self._get_observation()
        done = (self.current_step >= self.Nsteps) or (flag < 0)

        info = {
            "converged": (flag > 0),
            "control": control,
            "q": self.q.copy(),
            "u": self.u.copy(),
            "J": J,
            "free_index": self.free_index.copy(),
            "fixed_index": self.fixed_index.copy(),
            "control_node": control_node.copy(),  # 8D 边界DOF
        }

        # 保存历史
        if self.save_history:
            self.history["control"].append(control.copy())
            self.history["midx"].append(obs["midx"])
            self.history["midy"].append(obs["midy"])
            q1 = self.nv // 4
            q3 = 3 * self.nv // 4
            self.history["q1x"].append(self.q[2 * q1])
            self.history["q1y"].append(self.q[2 * q1 + 1])
            self.history["q3x"].append(self.q[2 * q3])
            self.history["q3y"].append(self.q[2 * q3 + 1])

        return obs, done, info

    def render(self, saveimg = False, show_target=True, show=True, ax=None):
        """
        可视化当前杆的形状。
        """
        x_arr = self.q[::2]
        y_arr = self.q[1::2]
        midx, midy = self._get_midpoint()

        # 第一次调用时初始化 figure 和 axis
        if not hasattr(self, "_render_initialized") or not self._render_initialized:
            self.fig, self.ax = plt.subplots()
            self._render_initialized = True
            plt.ion()  # 开启交互模式

        ax = self.ax
        ax.cla()
        ax.plot(x_arr, y_arr, 'ko-', label="rod")
        ax.plot(midx, midy, 'ro', markersize=8, label='midpoint')
        # if show_target:
        #     ax.plot(midx_target, midy_target, 'bx', markersize=10, label='target')

        ax.set_title(f"t = {self.ctime:.3f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis('equal')
        ax.set_xlim(-0.2, self.rod_length + 0.3)
        ax.set_ylim(-1.0, 1.0)
        ax.legend()

        if saveimg:
            plt.savefig(f'imgs/render_{self.ctime:.3f}.png')
            print(f"Saved image: imgs/render_{self.ctime:.3f}.png")

        if show:
            plt.pause(0.01)

if __name__ == "__main__":
    # =========================================================
    # 用户配置区
    # =========================================================
    L = 1.0          # 杆长（m），与 rod_length 保持一致
    T = 10.0         # 总仿真时长（s）

    # --- 初始位形 [x_l, y_l, th_l, x_r, y_r, th_r] ---
    start_pos = [0.0, 0.0, np.pi/2,   # 左端：位置(0,0)，角度0
                 0.7*L, 0.6*L, 0.0]   # 右端：位置(L,0)，角度0

    # --- 控制轨迹：关于时间 t 的函数，返回 6D 向量 ---
    def control_traj(t):
        """
        在整个仿真时间 [0, T] 内定义控制输入。
        返回 [x_l, y_l, th_l, x_r, y_r, th_r]。
        左端固定，右端沿正弦轨迹运动。
        """
        x_l  = start_pos[0]
        y_l  = start_pos[1]
        th_l = start_pos[2]

        x_r  = start_pos[3] - 1.2 * L * (t / T)  # 从0.7L线性移动到-0.9L
        y_r  = start_pos[4]
        th_r = 0.0

        return np.array([x_l, y_l, th_l, x_r, y_r, th_r])

    # =========================================================
    # 仿真主循环（一般不需要修改以下部分）
    # =========================================================
    env = SimulatorEnv_2D(nv=31, dt=0.01, rod_length=L,
                          R_outer=0.003, r_inner=0.0,
                          E_al=2e9, rho_al=1200,
                          total_time=T, save_history=True)

    obs = env.reset(init_x=start_pos[3],
                    init_y=start_pos[4],
                    init_theta=start_pos[5])

    done = False
    while not done:
        t = obs["time"]
        control = control_traj(t)

        obs, done, info = env.step(control)
        render_dlo_neurips(env, control=control, show=True,
                           xlim=(-0.9 * L, 0.9 * L),
                           ylim=(-0.15 * L,  0.75 * L))

    print("Simulation finished.")

