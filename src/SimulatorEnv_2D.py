# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

try:
    from render_neurips import render_dlo_neurips
except ImportError:
    render_dlo_neurips = None

from contact import get_contact_forces
from continuation import quasistatic_step
from energy import build_energy
from newton import solve_equilibrium
from matplotlib.patches import Circle

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
           m, mMat, energy, W, deltaL, free_index,
           control_node=None,
           use_inertia=False,
           obstacles=None,
           newton_cfg=None,
           fixed_index=None,
           boundary_rows=None):
    """Single-step implicit time integration.

    Thin wrapper around :func:`newton.solve_equilibrium`: it only applies the
    clamped boundary DOFs and hands the rest to the shared globalized Newton
    solver (line search + adaptive regularization, see src/newton.py).
    ``newton_cfg`` is the ``newton:`` block of the YAML config; with
    ``line_search: false`` this reproduces the original undamped iteration.

    ``fixed_index`` / ``boundary_rows`` say which DOFs are clamped and which
    entries of the 8D ``control_node`` they take; left out, both ends are
    clamped (the original behaviour).
    """
    if control_node is None:
        raise ValueError("Control node position must be provided.")

    q_start = q_old.copy()
    if fixed_index is None:
        q_start[:4] = control_node[:4]  # left end nodes
        q_start[-4:] = control_node[4:]  # right end nodes
    else:
        q_start[fixed_index] = control_node[boundary_rows]

    cfg = dict(newton_cfg or {})
    q_new, flag, J, info = solve_equilibrium(
        q_start,
        energy=energy,
        W=W,
        free_index=free_index,
        dim=2,
        tol=tol,
        obstacles=obstacles or [],
        max_iter=maximum_iter,
        q_old=q_old,
        u_old=u_old,
        dt=dt,
        m=m,
        mMat=mMat,
        use_inertia=use_inertia,
        line_search=cfg.get("line_search", True),
        max_backtrack=cfg.get("max_backtrack", 12),
        nonmonotone_memory=cfg.get("nonmonotone_memory", 8),
        reg=cfg.get("reg", 0.0),
    )
    return q_new, flag, J, info


############################
# SimulatorEnv_2D
############################

class SimulatorEnv_2D:
    """
    A standardized elastic-rod simulation environment:
      - Control input: (x, y, theta) given externally, corresponding to the
        position and orientation of the right end.
      - State: internal rod node positions q, velocity u, current time ctime, etc.
      - Interface: reset(), step(control), render()

    See SimulatorEnv_3D for the 3D counterpart; both expose the same API and
    a ``dim`` attribute so generic code can branch on it.
    """

    #: Spatial dimension — lets generic code branch without isinstance checks.
    dim = 2

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
                 save_history=False,
                 energy_cfg=None,
                 energy=None,
                 newton_cfg=None,
                 warn_on_newton_failure=False,
                 gravity=(0.0, -9.81),
                 grasp_mode="both"):
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
        # Newton solver options (the `newton:` block of the YAML config) and
        # a running count of steps whose equilibrium solve did not converge.
        self.newton_cfg = dict(newton_cfg or {})
        self.warn_on_newton_failure = warn_on_newton_failure
        self.n_newton_failures = 0
        # Steps that ended on an unstable equilibrium, and steps in which the
        # rod snapped through a fold (see src/continuation.py).
        self.n_unstable = 0
        self.n_snaps = 0

        # Cross-section and material parameters
        self.I_sec = np.pi * (R_outer ** 4 - r_inner ** 4) / 4
        self.A_sec = np.pi * (R_outer ** 2 - r_inner ** 2)
        self.EI = E_al * self.I_sec
        self.EA = E_al * self.A_sec

        # Spatial discretization
        self.deltaL = rod_length / (nv - 1)

        # Elastic energy model. Defaults to the built-in DER; `energy_cfg` is
        # the `energy:` block of the YAML config, and `energy` lets a caller
        # inject a RodEnergy instance directly (overrides energy_cfg).
        self.energy = energy if energy is not None else build_energy(
            energy_cfg, EI=self.EI, EA=self.EA, deltaL=self.deltaL, nv=nv)

        # Node coordinates (initially a straight rod)
        self.nodes = np.zeros((nv, 2))
        for c in range(nv):
            self.nodes[c, 0] = c * self.deltaL
            self.nodes[c, 1] = 0.0

        # Mass
        m_node = np.pi * (R_outer ** 2 - r_inner ** 2) * rod_length * rho_al / (nv - 1)
        self.m = np.zeros(2 * nv)
        for k in range(nv):
            self.m[2 * k] = m_node
            self.m[2 * k + 1] = m_node
        self.mMat = np.diag(self.m)

        # Gravity (in-plane). The default is the original -y; a rod lying on a
        # table, with gravity normal to the plane, uses (0, 0).
        self.gravity = np.asarray(gravity, dtype=float).reshape(2)
        self.W = np.zeros(2 * nv)
        self.W[0::2] = self.gravity[0] * m_node
        self.W[1::2] = self.gravity[1] * m_node

        # Newton iteration tolerance
        self.tol = self.energy.stiffness_scale / rod_length ** 2 * 1e-3

        # Constrained DOFs.
        #   both  -> both ends clamped (the original setup)
        #   right -> only the right end is gripped; the left end is free, so
        #            node 0 is a tip that moves only through the rod's response.
        # boundary_rows picks the clamped entries of the 8D control_node, i.e.
        # the rows of d(q_b)/d(u) that the adjoint must use.
        all_DOFs = np.arange(2 * nv)
        self.grasp_mode = grasp_mode
        if grasp_mode == "both":
            self.fixed_index = np.array([0, 1, 2, 3, 2 * nv - 4, 2 * nv - 3, 2 * nv - 2, 2 * nv - 1])
            self.boundary_rows = np.arange(8)
        elif grasp_mode == "right":
            self.fixed_index = np.array([2 * nv - 4, 2 * nv - 3, 2 * nv - 2, 2 * nv - 1])
            self.boundary_rows = np.arange(4, 8)
        else:
            raise ValueError(f"grasp_mode must be 'both' or 'right', got {grasp_mode!r}")
        self.free_index = np.setdiff1d(all_DOFs, self.fixed_index)
        #: Every DOF is a node position (no twist angles in 2D).
        self.n_pos = 2 * nv

        # Mid-node index
        # self.mid_node = nv // 2 + 1  # kept consistent with the original script
        self.mid_node = nv // 2  # kept consistent with the original script

        # Total number of steps
        self.Nsteps = int(round(total_time / dt))

        self.dlamda = 1 / self.Nsteps

        self.obstacles = []
        # Rigid environment geometry (e.g. contact.LPipe2D), kept for
        # diagnostics and drawing even when its contact force is switched off;
        # only what is in self.obstacles acts on the rod.
        self.pipe = None

        # State variables (initialized in reset())
        self.q = None
        self.u = None
        self.ctime = None
        self.current_step = None
        self.lamda = None
        #: The pose the current state was solved under: the start of the pose
        #: path the next step follows (src/continuation.py).
        self.u_prev = None

        # History log (optional)
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

    # ---------- obstacle management ----------

    def add_obstacle(self, obstacle):
        """Add a rigid circular obstacle to the simulation."""
        self.obstacles.append(obstacle)

    def remove_obstacle(self, obstacle):
        """Remove a specific obstacle from the simulation."""
        self.obstacles.remove(obstacle)

    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles.clear()

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

    # The dimension-neutral name the continuation driver calls (Gamma of the paper).
    _control_node_from_pose = _control_node_from_xytheta

    def _rest_control(self):
        """The pose that clamps the undeformed straight rod where reset() puts it."""
        n0, n1 = self.nodes[0], self.nodes[-1]
        return np.array([n0[0], n0[1], 0.0, n1[0], n1[1], 0.0], dtype=float)

    def _apply_boundary(self, q, control_node):
        """A copy of ``q`` with the clamped DOFs set from ``control_node``."""
        q_new = np.asarray(q, dtype=float).copy()
        q_new[self.fixed_index] = control_node[self.boundary_rows]
        return q_new

    def _solve(self, q_old, control_node, use_inertia=False):
        """One equilibrium solve at ``control_node``, warm-started from ``q_old``."""
        return objfun(
            q_old=q_old,
            u_old=self.u,
            dt=self.dt,
            tol=self.tol,
            maximum_iter=self.max_newton_iter,
            m=self.m,
            mMat=self.mMat,
            energy=self.energy,
            W=self.W,
            deltaL=self.deltaL,
            free_index=self.free_index,
            control_node=control_node,
            use_inertia=use_inertia,
            obstacles=self.obstacles,
            newton_cfg=self.newton_cfg,
            fixed_index=self.fixed_index,
            boundary_rows=self.boundary_rows,
        )

    def _get_midpoint(self):
        """Returns the current midpoint coordinates (midx, midy)."""
        x_arr = self.q[::2]
        y_arr = self.q[1::2]
        midx = x_arr[self.mid_node]
        midy = y_arr[self.mid_node]
        return midx, midy

    def _get_observation(self):
        """The observation contains only the midpoint coordinates."""
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
        Resets the environment and returns the initial observation.
        The initial control value only affects the control applied on the
        first step; by default the end is at (rod_length, 0, 0).
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
        # The straight rod is what the pose below clamps: the first step's
        # continuation starts from it.
        self.u_prev = self._rest_control()

        # Clear history
        if self.save_history:
            for k in self.history:
                self.history[k] = []

        # No integration step is taken in reset(); just return the current observation.
        return self._get_observation()

    @staticmethod
    def _check_control(control):
        control = np.asarray(control, dtype=float).flatten()
        if control.shape[0] != 6:
            raise ValueError("control must be a 6D vector: [x_l, y_l, th_l, x_r, y_r, th_r].")
        return control

    def _commit(self, control, res):
        """Bookkeeping shared by step() and settle(): counters, warnings, info."""
        flag, ninfo = res["flag"], res["newton"]
        if flag < 0:
            self.n_newton_failures += 1
            if self.warn_on_newton_failure:
                print(f"[newton] step {self.current_step}: {res['reason'] or ninfo.get('reason', '')} "
                      f"(residual {ninfo['residual']:.3e} vs tol {self.tol:.3e}, "
                      f"{res['newton_iterations']} iters)")
        if res["stable"] is False:
            self.n_unstable += 1
        if res["snap"]:
            self.n_snaps += 1
        self.u_prev = control.copy()
        return {
            "converged": (flag > 0),
            "control": control,
            "J": res["J"],
            "free_index": self.free_index.copy(),
            "fixed_index": self.fixed_index.copy(),
            "control_node": self._control_node_from_xytheta(control),  # 8D boundary DOFs
            "newton": ninfo,
            # continuation / stability / snap diagnostics (src/continuation.py)
            "substeps": res["substeps"],
            "newton_iterations": res["newton_iterations"],
            "stable": res["stable"],
            "snap": res["snap"],
            "snap_by": res.get("snap_by"),
            "snaps": res["snaps"],
            "max_displacement": res.get("max_displacement", 0.0),
            "boundary_displacement": res.get("boundary_displacement", 0.0),
            "reason": res["reason"],
        }

    def settle(self, control):
        """Bring the rod to equilibrium at ``control`` without advancing the horizon.

        Used before a rollout starts, so that its first step leaves from an
        equilibrium under the initial command rather than from the undeformed
        rod (``newton.initial_equilibrium``).  The pose path from the state's
        previous pose to ``control`` is followed by continuation, exactly as a
        step would; the step counter, time and lambda stay where they are and
        the velocity is reset.  Returns the same ``info`` as :meth:`step`.
        """
        control = self._check_control(control)
        res = quasistatic_step(self, control, use_inertia=False)
        info = self._commit(control, res)
        self.q = res["q"]
        self.u = np.zeros_like(self.q)
        info["q"] = self.q.copy()
        info["u"] = self.u.copy()
        return info

    def step(self, control, use_inertia=False):
        """
        Single simulation step.
          control: (x, y, theta), a numpy array or iterable
        Returns:
          obs, reward, done, info
        """
        control = self._check_control(control)

        # The equilibrium at the new boundary, reached along the pose path
        # from the previous one (continuation, stability check, snap rule --
        # src/continuation.py; a single Newton solve with all three off).
        res = quasistatic_step(self, control, use_inertia=use_inertia)
        q_new, flag = res["q"], res["flag"]
        info = self._commit(control, res)

        # Update velocity and time
        u_new = (q_new - self.q) / self.dt
        self.q = q_new
        self.u = u_new
        self.ctime += self.dt
        self.current_step += 1
        self.lamda = self.current_step * self.dlamda

        # Observation and status
        obs = self._get_observation()
        done = (self.current_step >= self.Nsteps) or (flag < 0)

        info["q"] = self.q.copy()
        info["u"] = self.u.copy()

        # Save history
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

    # ---------- RHC state checkpointing ----------

    def get_state(self):
        """Snapshot the current simulator state (used to warm-start RHC segments)."""
        return {
            "q": self.q.copy(),
            "u": self.u.copy(),
            "ctime": self.ctime,
            "current_step": self.current_step,
            "lamda": self.lamda,
            "u_prev": None if self.u_prev is None else self.u_prev.copy(),
        }

    def set_state(self, state):
        """Restore a state previously returned by get_state()."""
        self.q = state["q"].copy()
        self.u = state["u"].copy()
        self.ctime = state["ctime"]
        self.current_step = state["current_step"]
        self.lamda = state["lamda"]
        u_prev = state.get("u_prev")
        self.u_prev = None if u_prev is None else np.asarray(u_prev, dtype=float).copy()

    def render(self, saveimg=False, show_target=True, show=True, ax=None,
               xlim=None, ylim=None):
        """
        Visualizes the current rod shape.
        """
        x_arr = self.q[::2]
        y_arr = self.q[1::2]
        midx, midy = self._get_midpoint()

        # Initialize the figure and axis on the first call
        if not hasattr(self, "_render_initialized") or not self._render_initialized:
            self.fig, self.ax = plt.subplots()
            self._render_initialized = True
            plt.ion()  # enable interactive mode

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
        ax.set_xlim(*(xlim if xlim else (-0.2, self.rod_length + 0.3)))
        ax.set_ylim(*(ylim if ylim else (-1.0, 1.0)))
        ax.legend()

        for obs in self.obstacles:
            if hasattr(obs, "draw"):
                continue
            patch = Circle((obs.cx, obs.cy), obs.radius,
                           color='steelblue', alpha=0.55, zorder=5)
            ax.add_patch(patch)
        if self.pipe is not None:
            self.pipe.draw(ax)

        if saveimg:
            plt.savefig(f'imgs/render_{self.ctime:.3f}.png')
            print(f"Saved image: imgs/render_{self.ctime:.3f}.png")

        if show:
            plt.pause(0.01)

if __name__ == "__main__":
    from contact import CircleObstacle

    # =========================================================
    # Contact demo: rigid ball moves downward and deforms a
    # horizontally pinned elastic rod.
    # =========================================================
    L = 1.0   # rod length (m)
    T = 6.0   # total simulation time (s)

    r_ball    = 0.07 * L   # ball radius
    k_contact = 5e3        # penalty stiffness – tune relative to rod EI/L^2

    env = SimulatorEnv_2D(nv=31, dt=0.01, rod_length=L,
                          R_outer=0.003, r_inner=0.0,
                          E_al=2e9, rho_al=1200,
                          total_time=T, save_history=True)

    # Ball starts above the rod center and descends until it contacts and
    # deforms the rod downward.
    ball = CircleObstacle(cx=L / 2, cy=0.25, radius=r_ball,
                          k_contact=k_contact)
    env.add_obstacle(ball)

    obs = env.reset()

    x_r_start  = L          # natural (undeformed) right-end position
    x_r_target = 0.9 * L   # compressed target position
    t_ramp     = 0.3 * T   # ramp duration: reach target in first 30% of T

    save_video = True          # set False to skip saving
    video_path = "contact_simulation.gif"   # change to .mp4 after: pip install imageio[ffmpeg]
    fps        = min(25, int(1.0 / env.dt))  # match simulation timestep
    frames     = []

    done = False
    while not done:
        t = obs["time"]

        # Gradually move right end from L → 0.9L during the ramp phase,
        # then hold at target.  This avoids a cold-start discontinuity.
        alpha = min(t / t_ramp, 1.0)
        x_r = x_r_start + alpha * (x_r_target - x_r_start)

        control = np.array([0.0, 0.0, 0.0,   # left:  pos (0, 0), angle 0
                             x_r, 0.0, 0.0])  # right: ramped to 0.9L

        # Ball rises from y = -0.25 to y = 0.20 over the simulation
        cy_new = -0.25 + 0.25 * (t / T)
        ball.move_to(L / 2, cy_new)

        obs, done, info = env.step(control)
        env.render(show=True,
                   xlim=(-0.1 * L, 1.1 * L),
                   ylim=(-0.35, 0.35))

        if save_video:
            env.fig.canvas.draw()
            frame = np.asarray(env.fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(frame)

    print("Contact simulation finished.")

    if save_video and frames:
        print(f"Saving animation ({len(frames)} frames @ {fps} fps)...")
        try:
            import imageio
            imageio.mimsave(video_path, frames, fps=fps)
            print(f"Saved MP4: {video_path}")
        except ImportError:
            # Fallback: save as GIF using Pillow
            from PIL import Image
            gif_path = video_path.replace(".mp4", ".gif")
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                         loop=0, duration=int(1000 / fps))
            print(f"Saved GIF: {gif_path}  (install imageio for MP4)")
        plt.close("all")

