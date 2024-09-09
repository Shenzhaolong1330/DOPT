##############################
## Author ## Shen Zhaolong ###
##  Date  ##    2024-03    ###
##############################

## Modification # 2024-05-14 ##
## one_step_euler: controller passes in as a Formal parameter (default to system initial controller), 
##                 which is convenient to choose other external controllers.
## simulate_euler: ~
## simulate_rk4: ~

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
from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
)

class ControlAffineSystem(ABC):
    """
    Represents a generic control affine system.
    in the form of x_dot = f(x) + g(x) u
    """

    def __init__(
            self,
            system_params: dict,
            controller_params: Optional[dict] = None,
            dt: float = 0.01,
            controller_period: float = 0.01,
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ):
        self.device = device
        self.dt = dt
        self.system_params = system_params
        self.controller_params = controller_params
        self.controller_period = controller_period

    @abstractmethod
    def _f(self, x: torch.Tensor):
        pass

    @abstractmethod
    def _g(self, x: torch.Tensor):
        pass

    @abstractmethod
    def state_dims(self) -> int:
        pass

    @abstractmethod
    def control_dims(self) -> int:
        pass

    @abstractmethod
    def controller(
            self,
            x: torch.tensor
    ) -> torch.tensor:
        pass

    def x_dot(
            self,
            x: torch.Tensor,
            u: torch.Tensor
    ):
        f = self._f(x).to(self.device).float()
        g = self._g(x).to(self.device).float()
        x_dot = f + g @ u
        return x_dot

    def linearized_ct_system(
            self,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        goal_point = goal_point.to(self.device)
        u_eq = u_eq.to(self.device)
        dynamics = lambda x: self.x_dot(x, u_eq).squeeze()
        A = jacobian(dynamics, goal_point)[0].squeeze().cpu().detach().numpy()
        A = np.reshape(A, (self.N_DIMS, self.N_DIMS))

        B = self._g(goal_point).squeeze().cpu().numpy()
        B = np.reshape(B, (self.N_DIMS, self.N_CONTROLS))
        print("linearized_ct_system:", A, B)
        return A, B


    def linearized_dt_system(
            self,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        Act, Bct = self.linearized_ct_system(goal_point, u_eq)

        A = np.eye(self.N_DIMS) + self.controller_period * Act
        B = self.controller_period * Bct
        print("linearized_dt_system:", A, B)
        return A, B

    def compute_LQR_controller(
            self,
            A: np.array = None,
            B: np.array = None,
            Q: np.array = None,
            R: np.array = None
    ) -> np.array:
        if Q == None:
            Q = np.eye(A.shape[0])
        if R == None:
            R = np.eye(B.shape[1])
        K_np = continuous_lqr(A, B, Q, R)
        return K_np


    def one_step_euler(
            self,
            x_0: torch.Tensor,
            use_controller: bool,
            the_controller = None
    ):
        if the_controller is None:
            the_controller = self.controller
        data_sim = torch.zeros(3, 2, x_0.shape[0], x_0.shape[1]).to(self.device)
        data_sim[0, 0, :, :] = x_0.to(self.device)
        controller_update_freq = int(self.controller_period / self.dt)
        for i in range(1, 3):
            x_current = data_sim[i - 1, 0, :, :].to(self.device)
            if use_controller == False:
                u = torch.zeros(self.control_dims(), 1).to(self.device)
            elif i == 1 or i % controller_update_freq == 0:
                u = the_controller(x_current.to(self.device)).to(self.device)

            x_dot = self.x_dot(x_current, u).to(self.device)
            data_sim[i - 1, 1, :, :] = x_dot
            data_sim[i, 0, :, :] = x_current + self.dt * x_dot
        return data_sim[0:2]

    def simulate_euler(
            self,
            x_initial: torch.Tensor,
            step_number: int,
            use_controller: bool,
            the_controller = None
    ) -> torch.Tensor:
        if the_controller is None:
            the_controller = self.controller
        else:
            the_controller = the_controller.Controller()
        """
        x_initial: initial state
        step_number: simulate steps
        controller:
            a method mapping from state to action, it accepts a torch Tensor type parameter and return a torch Tensor type object
        controller_period:
            the period determining how often the controller is run (in seconds). If none, defaults to self.dt
        """
        data_sim = torch.zeros(step_number, 2, x_initial.shape[0], x_initial.shape[1])
        data_sim[0, 0, :, :] = x_initial
        controller_update_freq = int(self.controller_period / self.dt)
        u = torch.zeros(self.N_CONTROLS, 1)
        for i in range(1, step_number):
            # x_current = x_sim[i-1,:,:]
            x_current = data_sim[i - 1, 0, :, :]
            if not use_controller:
                u = torch.zeros(self.N_CONTROLS, 1)
            elif i == 1 or i % controller_update_freq == 0:
                # u = self.controller(x_current)
                u = the_controller(x_current)
            # print('u',u)
            x_dot = self.x_dot(x_current, u).to(self.device)
            # x_dot_sim[i-1, :, :] = x_dot
            # x_sim[i, :, :] = x_current + self.dt * x_dot
            data_sim[i - 1, 1, :, :] = x_dot
            data_sim[i, 0, :, :] = x_current + self.dt * x_dot
        # data_sim = torch.cat((x_sim.unsqueeze(0), x_dot_sim.unsqueeze(0)), dim=0)
        # print('simulation using forward euler.')
        return data_sim
        # 1000, 2, 2, 1

    def simulate_rk4(
            self,
            x_initial: torch.Tensor,
            step_number: int,
            use_controller: bool,
            the_controller = None # 指定控制器，默认为类自带控制器
    ) -> torch.Tensor:
        """
        Data Structure
        data_sim:[
            step_idx,
            0 for x, 1 for x_dot,
            state.shape[0],
            state.shape[1]
            ]
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # 确保初始状态在GPU上
        x_initial = x_initial.to(device)

        if the_controller is None:
            the_controller = self.controller
        data_sim = torch.zeros(step_number, 2, x_initial.shape[0], x_initial.shape[1], dtype=torch.float32).to(device)
        data_sim[0, 0, :, :] = x_initial
        controller_update_freq = int(self.controller_period / self.dt)
        u = torch.zeros(self.N_CONTROLS, 1).to(device)
        for i in range(1, step_number):
            x_current = data_sim[i - 1, 0, :, :]
            if not use_controller:
                u = torch.zeros(self.N_CONTROLS, 1).device('cpu')
            elif i == 1 or i % controller_update_freq == 0:
                u = the_controller(x_current).to(device)
            k1 = self.x_dot(x_current, u)
            k2 = self.x_dot(x_current + self.dt / 2 * k1, u)
            k3 = self.x_dot(x_current + self.dt / 2 * k2, u)
            k4 = self.x_dot(x_current + self.dt * k3, u)
            x_next = x_current + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            data_sim[i - 1, 1, :, :] = k1  # Store k1 (optional, for analysis)
            data_sim[i, 0, :, :] = x_next
        # print('simulation using rk4.')

        return data_sim


    def convergence_judgment(
            self,
            data_sim: torch.Tensor,
            set_point: torch.Tensor = None,
            epsilon: float = 0.2  #0.2
    ):
        """
        judge if the system converge to set point
        """
        step2converge = 0
        step2unitball = 0
        step2norm2ball = 0
        if set_point == None:
            set_point = torch.zeros(data_sim[0].shape, dtype=torch.float32)
        data_sim = data_sim.cpu().detach().numpy()
        set_point = set_point.detach().numpy()
        for i in range(data_sim.shape[0]):
            if np.linalg.norm(data_sim[i]-set_point) <= 2:
                step2norm2ball = i
                break
            else:
                step2norm2ball = float("inf")
        for i in range(data_sim.shape[0]):
            if np.linalg.norm(data_sim[i]-set_point) <= 1:
                step2unitball = i
                break
            else:
                step2unitball = float("inf")
        for i in range(data_sim.shape[0]):
            if np.linalg.norm(data_sim[i]-set_point) <= epsilon:
                step2converge = i
                break
            else:
                step2converge = float("inf")
        print('-----------------Convergence Speed and Judgment-----------------')
        print('--------------It takes {} steps to norm 2 ball;--------------\n---------------It takes {} steps to unit ball;---------------\n----------------It takes {} steps to converge.--------------'.format(step2norm2ball, step2unitball, step2converge))
        return step2norm2ball,step2unitball,step2converge


    def plot_phase_portrait(
            self,
            data_sim: torch.Tensor,
            arrow_on: bool = False,
            title = 'System Phase Portrait',
            save_fig = False,
            save_path = 'saved_files/figs/'
    ):
        
        # data_sim = torch.cat((x_sim.unsqueeze(0), x_dot_sim.unsqueeze(0)), dim=0)
        # x_sim = data_sim[0]
        # x_dot_sim = data_sim[1]

        # step2converge = self.convergence_judgment(data_sim)
        # print('-----------------It takes {} steps to converge.------------------'.format(step2converge))
        data_sim = data_sim.detach().numpy()
        x_sim = data_sim[:, 0, :, :]
        x_dot_sim = data_sim[:, 1, :, :]
        plt.figure()
        # data is in the form of torch.Tensor
        num = x_sim.shape[0]
        dim = x_sim.shape[1]
        # state tensors are column vectors as default
        x_value = x_sim[:, 0, 0]
        y_value = x_sim[:, 1, 0]
        plt.plot(x_value, y_value, label='State trajectory')
        if arrow_on is True:
            interval = int(num/50)  # 每隔 interval 个点进行绘制
            for i in range(0, num, interval):
                x_dot_x = x_dot_sim[i, 0, 0]  # x 维度上的导数
                x_dot_y = x_dot_sim[i, 1, 0]  # y 维度上的导数
                dx = 2 * x_dot_x / ((x_dot_x ** 2 + x_dot_y ** 2) ** 0.5)
                dy = 2 * x_dot_y / ((x_dot_x ** 2 + x_dot_y ** 2) ** 0.5)
                plt.arrow(x_sim[i, 0, 0], x_sim[i, 1, 0], dx/3, dy/3, head_width=0.3, head_length=0.4, fc='r', ec='r')
        ax = plt.gca()
        ax.set_aspect(1)
        plt.xlabel(r"$\mathregular{x_{1}}$")
        plt.ylabel(r"$\mathregular{x_{2}}$")
        plt.title(title)
        plt.grid(True)
        if save_fig:
            plt.savefig(save_path+title+'.png', dpi=600)
        plt.show()


    def plot_state(
            self,
            data_sim: torch.Tensor,
            state_idx: int,
            title = 'System State'
    ):
        data_sim = data_sim.detach().numpy()
        x_sim = data_sim[:, 0, :, :]
        data = x_sim[:, state_idx, :]
        num_samples = data.shape[0]
        time_labels = np.arange(0, num_samples * self.dt, self.dt)
        plt.plot(time_labels, data)
        plt.xlabel('Time (s)')
        plt.ylabel('State')
        plt.title(title)
        plt.grid(True)
        plt.show()