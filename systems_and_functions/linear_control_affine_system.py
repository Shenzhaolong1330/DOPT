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
from systems_and_functions.kits import continuous_lqr
from systems_and_functions.control_affine_system import ControlAffineSystem
from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
)

class LinearControlAffineSystem(ControlAffineSystem):
    N_DIMS = 2
    N_CONTROLS = 2
    # A = torch.zeros([N_DIMS, N_DIMS])
    # B = torch.zeros([N_DIMS, N_CONTROLS])
    # # F = torch.zeros([N_DIMS, 1])
    # K = torch.zeros([N_CONTROLS, N_DIMS])
    A = np.zeros([N_DIMS, N_DIMS])
    B = np.zeros([N_DIMS, N_CONTROLS])
    # K = np.zeros([N_CONTROLS, N_DIMS])
    K = torch.tensor(np.zeros([N_CONTROLS, N_DIMS]), dtype=torch.float, requires_grad = True)

    def __init__(
            self,
            system_params: dict,
            controller_params: Optional[dict] = None,
            dt: float = 0.01,
            controller_period: float = 0.01
    ):
        super().__init__(system_params, controller_params, dt, controller_period)
        # system_params: tuple(system matrix, input matrix) or tuple(system matrix, input matrix, noise matrix)
        self.A = system_params['A']
        self.B = system_params['B']
        self.N_DIMS = self.A.shape[0]
        self.N_CONTROLS = self.B.shape[1]
        if controller_params is None:
            print('No controller is involved.')
            self.K = torch.tensor(np.zeros([self.N_CONTROLS, self.N_DIMS]),dtype=torch.float)
        else:
            print('Controller is involved.')
            self.K = torch.tensor(controller_params['K'], dtype=torch.float, requires_grad = True)

        if self.K.t().detach().numpy().shape != self.B.shape:
            raise ValueError("Error: Dimension mismatch! ")
        else:
            print('Dimension matched.')


    def _f(self, x: torch.Tensor):
        A = torch.tensor(self.A).type_as(x)
        return A @ x


    def _g(self, x: torch.Tensor):
        B = torch.tensor(self.B).type_as(x)
        return B


    def controller(
            self,
            x: torch.tensor
    ) -> torch.tensor:
        # K = torch.tensor(self.K).type_as(x)
        K = self.K
        return K.float() @ x.float()
    

    def controller_batch(
            self,
            x: torch.Tensor = torch.zeros(100, N_DIMS, 1)
    ) -> torch.tensor:
        """
        K [N_CONTROLS, N_DIMS]
        x [sample num, n dims, 1]
        return [sample num, n controls, 1]
        """
        K = self.K
        return (x.permute(0, 2, 1).float()@K.t().float()).permute(0, 2, 1).float()


    def compute_LQR_controller(
            self,
            A: np.array = None,
            B: np.array = None,
            Q: np.array = None,
            R: np.array = None
    ) -> np.array:
        if A is None:
            A = self.A
        if B is None:
            B = self.B
        if Q is None:
            q = [10, 1]
            Q = np.diag(q)
        if R is None:
            R = np.eye(B.shape[1])
        K_np = continuous_lqr(A, B, Q, R)
        return -K_np

    def use_LQR_controller(
            self,
            A: np.array = None,
            B: np.array = None,
            Q: np.array = None,
            R: np.array = None
    ):
        self.K = torch.tensor(self.compute_LQR_controller(A, B, Q, R), requires_grad = True)

    def system_eigVals(
            self
    ):
        eigVals, _ = scipy.linalg.eig(self.A + self.B * self.K.detach().numpy())
        return eigVals

    # property method: save memory
    @property
    def state_dims(self) -> int:
        return LinearControlAffineSystem.N_DIMS

    @property
    def control_dims(self) -> int:
        return LinearControlAffineSystem.N_CONTROLS
