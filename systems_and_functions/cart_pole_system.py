##############################
## Author ## Shen Zhaolong ###
##  Date  ##   2024-05-13   ##
##############################


import cvxpy as cp
import numpy as np
import math
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

from systems_and_functions.control_affine_system import ControlAffineSystem
from systems_and_functions.linear_control_affine_system import LinearControlAffineSystem


class CartPole(ControlAffineSystem, nn.Module):
    N_DIMS = 4
    N_CONTROLS = 1
    __g = 9.8
    
    def __init__(
        self,
        system_params: dict = {'M':2.0,'m': 1.0,'L': 1.0},
        controller_params: Optional[dict] = None,
        dt: float = 0.01,
        controller_period: float = 0.01,
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ):
        ControlAffineSystem.__init__(self, system_params, controller_params, dt, controller_period)
        # system_params{'m': mass,'L': length, 'b': friction}
        self.device = device
        self.M =  self.system_params["M"]
        self.m =  self.system_params["m"]
        self.L =  self.system_params["L"]
        if controller_params is None:
            print('No controller is involved.')
            self.K = torch.zeros([self.N_CONTROLS, self.N_DIMS])
        else:
            print('Controller is involved.')
            self.K = torch.Tensor(controller_params['K'])

    
    def _f(self,x: torch.Tensor):
        p = x[0]
        p_dot = x[1]
        theta = x[2]
        theta_dot = x[3]
        M, m, L = self.M, self.m, self.L
        g = self.__g
        f = torch.zeros(self.N_DIMS, 1)

        f[0, 0] = p_dot
        f[1, 0] = (m*(g*torch.cos(theta)-theta_dot**2*L)*torch.sin(theta))/(M+m-m*torch.cos(theta)**2)
        f[2, 0] = theta_dot
        f[3, 0] = (g*(M+m)*torch.sin(theta)-theta_dot**2*L*m*torch.sin(theta)*torch.cos(theta))/(L*(M+m-m*torch.cos(theta)**2))
        return f
        
    def _g(self,x: torch.Tensor):
        p = x[0]
        p_dot = x[1]
        theta = x[2]
        theta_dot = x[3]
        M, m, L = self.M, self.m, self.L
        # g = self.__g

        g = torch.zeros(self.N_DIMS, self.N_CONTROLS)
        g = g.type_as(x)

        g[1, 0] = 1 / (M+m-m*torch.cos(theta)**2)
        g[3, 0] = torch.cos(theta) / (L*(M+m-m*torch.cos(theta)**2))
        return g

    
    def x_dot(
            self,
            x: torch.Tensor,
            u: torch.Tensor
    ):
        f = self._f(x).to(self.device).float()
        g = self._g(x).to(self.device).float()
        x_dot = f + (g @ u).reshape(-1, 1)
        return x_dot 


    def linearized_ct_system(
            self,
            goal_point = torch.zeros(N_DIMS,1),
            u_eq = torch.zeros(N_CONTROLS)
    ) -> Tuple[np.ndarray, np.ndarray]:
        goal_point = goal_point.to(self.device)
        u_eq = u_eq.to(self.device)
        dynamics = lambda x: self.x_dot(x, u_eq).squeeze()
        A = jacobian(dynamics, goal_point).squeeze().cpu().detach().numpy()
        A = np.reshape(A, (self.N_DIMS, self.N_DIMS))

        B = self._g(goal_point).squeeze().cpu().detach().numpy()
        B = np.reshape(B, (self.N_DIMS, self.N_CONTROLS))
        print("linearized_ct_system: \n A{}, \n B{}".format(A, B))
        return A, B

    
    def state_dims(self)->int:
        return CartPole.N_DIMS
        
 
    def control_dims(self)->int:
        return CartPole.N_CONTROLS
        
    
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
            q = [0, 100, 100, 0]
            Q = np.diag(q)
        if R is None:
            R = np.eye(B.shape[1])
        K_np = continuous_lqr(A, B, Q, R)
        return K_np


    def linearize_and_compute_LQR(self):
        goal_point = torch.zeros((4,1), requires_grad=True)
        u_eq = torch.zeros(1, requires_grad=True)
        Act, Bct = self.linearized_ct_system(goal_point, u_eq)
        # Adt, Bdt = self.linearized_dt_system(goal_point, u_eq)
        K_np = self.compute_LQR_controller(Act, Bct)
        self.K = torch.tensor(K_np, dtype=torch.float)
        return self.K


    def controller(
        self,
        x: torch.tensor
    )->torch.tensor:
        K = (self.K).type_as(x)
        u = -K@x
        return u

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
        return -(x.permute(0, 2, 1).float()@K.t().float()).permute(0, 2, 1).float()


    def construct_nn_controller(
        self
    ):
        """
        initialize a nn controller by defining the structure and activation function
        the parameters can be accessed through CartPole.parameters()
        """
        nn.Module.__init__(self)
        self.nn_controller_fc1 = nn.Linear(self.N_DIMS, 128)
        self.nn_controller_fc2 = nn.Linear(128, 64)
        self.nn_controller_fc3 = nn.Linear(64, self.N_CONTROLS)
        self.nn_controller_parameters = self.parameters()


    def nn_controller(
        self,
        x: torch.tensor
    )->torch.tensor:
        x1 = nn.functional.relu(self.nn_controller_fc1(x))
        x2 = nn.functional.relu(self.nn_controller_fc2(x1))
        x3 = self.nn_controller_fc3(x2)
        return x3


    def nn_controller_imitate_existed_controller(
        self,
        learning_rate = 0.01,
        epoch_num = 1000
    ):
        """
        force nn controller to imitate the existed linear controller 
        which can stablize the system.
        
        2024.5.20
        not sure if this is effective !!
        """
        # DONE: not sure if this is effective !!
        training_data = 10 * torch.rand(1000, 4, 1) - 5
        reshaped_training_data = torch.squeeze(training_data)
        y_train = torch.squeeze(self.controller_batch(training_data))
        y_pred = torch.squeeze(self.nn_controller(reshaped_training_data))
        print('y_train',y_train,'y_pred',y_pred)
        criterion = nn.MSELoss()
        # loss = criterion(y_pred,y_train)
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        for epoch in range(epoch_num):
            y_pred = self.nn_controller(reshaped_training_data)
            loss = criterion(y_pred,y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {}'.format(epoch+1, epoch_num, loss.item()))
        print('Final Loss of nn_controller_imitate_existed_controller: {}'.format(loss))

 
    def plot_phase_portrait(
        self, 
        data_sim: torch.Tensor,
        arrow_on: bool = False,
        title = 'Cart Pole'
    ):
        x_sim = data_sim[:, 0, :, :]
        x_dot_sim = data_sim[:, 1, :, :]
        plt.figure()
        num = x_sim.shape[0]
        dim = x_sim.shape[1]
        x_value = x_sim[:, 0, 0]
        y_value = x_sim[:, 1, 0]
        plt.plot(x_value, y_value, label='State trajectory')
        if arrow_on is True:
            interval = 200  # 每隔 interval 个点进行绘制
            for i in range(0, num, interval):
                x_dot_x = x_dot_sim[i, 0, 0]  # x 维度上的导数
                x_dot_y = x_dot_sim[i, 1, 0]  # y 维度上的导数
                dx = 2*x_dot_x/((x_dot_x**2+x_dot_y**2)**0.5)
                dy = 2*x_dot_y/((x_dot_x**2+x_dot_y**2)**0.5)
                plt.arrow(x_sim[i, 0, 0], x_sim[i, 1, 0], dx, dy, head_width=3, head_length=4, fc='r', ec='r')
        ax = plt.gca()
        ax.set_aspect(1)
        plt.xlabel(r"$\mathregular{\theta}$")
        plt.ylabel(r"$\mathregular{\dot{\theta}}$")
        plt.title(title)
        plt.show()

    
    def plot_state(
            self,
            data_sim: torch.Tensor,
            state_idx: int
    ):
        data_sim = data_sim.detach().numpy()
        x_sim = data_sim[:, 0, :, :]
        data = x_sim[:, state_idx, :]
        num_samples = data.shape[0]
        time_labels = np.arange(0, num_samples * self.dt, self.dt)
        match state_idx:
            case 0:
                title = 'System State: Position (p)'
                ylabel = 'p'
            case 1:
                title = 'System State: Velocity (p_dot)'
                ylabel = 'p_dot'
            case 2:
                title = 'System State: Angle (theta)'
                ylabel = 'theta'
            case 3:
                title = 'System State: Angular Velocity (theta_dot)'
                ylabel = 'theta_dot'
        plt.plot(time_labels, data)
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()
    

    def flatten(self, a):
        """
        将多维数组降为一维
        """
        return np.array(a).flatten()

    def plot_cart(self, xt, theta):
        """
        画图
        """
        cart_w = 1.0
        cart_h = 0.5
        radius = 0.1
        l_bar = 2.0 
        cx = np.array([-cart_w / 2.0, cart_w / 2.0, cart_w /
                        2.0, -cart_w / 2.0, -cart_w / 2.0])
        cy = np.array([0.0, 0.0, cart_h, cart_h, 0.0])
        cy += radius * 2.0

        cx = cx + xt

        bx = np.array([0.0, l_bar * math.sin(-theta)])
        bx += xt
        by = np.array([cart_h, l_bar * math.cos(-theta) + cart_h])
        by += radius * 2.0

        angles = np.arange(0.0, math.pi * 2.0, math.radians(3.0))
        ox = np.array([radius * math.cos(a) for a in angles])
        oy = np.array([radius * math.sin(a) for a in angles])

        rwx = np.copy(ox) + cart_w / 4.0 + xt
        rwy = np.copy(oy) + radius
        lwx = np.copy(ox) - cart_w / 4.0 + xt
        lwy = np.copy(oy) + radius

        wx = np.copy(ox) + bx[-1]
        wy = np.copy(oy) + by[-1]

        plt.plot(self.flatten(cx), self.flatten(cy), "-b")
        plt.plot(self.flatten(bx), self.flatten(by), "-k")
        plt.plot(self.flatten(rwx), self.flatten(rwy), "-k")
        plt.plot(self.flatten(lwx), self.flatten(lwy), "-k")
        plt.plot(self.flatten(wx), self.flatten(wy), "-k")
        plt.title(f"Position: {xt:.2f} , Angle: {math.degrees(theta):.2f}")

        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        plt.axis("equal")


    def show_animation(
            self,
            data_sim: torch.Tensor
    ):
        data_sim = data_sim.detach().numpy()
        x_sim = data_sim[:, 0, :, :]
        length = x_sim.shape[0]
        for i in range(length):
            plt.clf()  # Clear the current figure
            px = float(x_sim[i,0,0])  # 将一个字符串或数字转换为浮点数。输出位置
            theta = float(x_sim[i,2,0])  # 输出角度
            self.plot_cart(px, theta)  # 调用函数
            plt.xlim([px - 5.0, px + 5.0])
            plt.pause(0.001)  # 暂停间隔
        plt.show()