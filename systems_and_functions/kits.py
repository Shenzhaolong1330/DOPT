##############################
## Author ## Shen Zhaolong ###
##  Date  ##    2024-03    ###
##############################

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd.functional import jacobian
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Tuple, Optional, List
import scipy.linalg

"""
#### Continuous Time LQR
1. select Q and R
2. solve CARE to get P: $A^TP+PA-PBR^{-1}B^TP+Q = 0$
3. $K = R^{-1}B^TP$
4. $u = -Kx$

#### Discrete Time LQR
1. set iteration range N
2. set inital iteration value $P_N=Q_f$, in which $Q_f = Q$
3. iteration loop: from $N$ to 1, $P_{t-1} = Q+A^TP_tA-A^TP_tB(R+B^TP_tB)^{-1}B^TP_tA$
4. from 0 to $N-1$, calculate $K_t = (R+B^TP_{t+1}B)^{-1}B^TP_{t+1}A$
5. u_t = -K_tX_t
"""

def continuous_lqr(
    A: np.array,
    B: np.array,
    Q: np.array,
    R: np.array,
    return_eigs: bool = False,
):
    """Solve the continuous time lqr controller.

    Based on Bertsekas, p.151

    Yields the control law u = -K x
    """

    # first, try to solve the ricatti equation
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)

    # compute the LQR gain
    # K = scipy.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)
    K = scipy.linalg.inv(R)@B.T@P
    if not return_eigs:
        return K
    else:
        eigVals, _ = scipy.linalg.eig(A - B * K)
        return K, eigVals
    
def discrete_lqr(
    A: np.array,
    B: np.array,
    Q: np.array,
    R: np.array,
    return_eigs: bool = False,
):
    """Solve the discrete time lqr controller.

    x_{t+1} = A x_t + B u_t

    cost = sum x.T*Q*x + u.T*R*u

    Code adapted from Mark Wilfred Mueller's continuous LQR code at
    http://www.mwm.im/lqr-controllers-with-python/

    Based on Bertsekas, p.151

    Yields the control law u = -K x
    """

    # first, try to solve the ricatti equation
    X = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = scipy.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    if not return_eigs:
        return K
    else:
        eigVals, _ = scipy.linalg.eig(A - B * K)
        return K, eigVals