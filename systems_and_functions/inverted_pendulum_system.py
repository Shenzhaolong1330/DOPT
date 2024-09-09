##############################
## Author ## Shen Zhaolong ###
##  Date  ##   2024-06-20   ##
##############################


import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd.functional import jacobian
from typing import Callable, Tuple, Optional, List
from systems_and_functions.control_affine_system import ControlAffineSystem


class InvertedPendulum(ControlAffineSystem):
    N_DIMS = 2
    N_CONTROLS = 1
    __g = 9.8
    
    def __init__(
        self,
        system_params: dict = {'m': 2.0,'L': 1.0, 'b': 0.01},
        controller_params: Optional[dict] = None,
        dt: float = 0.01,
        controller_period: float = 0.01,
        controller_bound: float = 1000.0,
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ):
        super().__init__(system_params, controller_params, dt, controller_period)
        # system_params{'m': mass,'L': length, 'b': friction}
        self.device = device
        self.controller_bound = controller_bound
        self.m =  self.system_params["m"]
        self.L =  self.system_params["L"]
        self.b =  self.system_params["b"]
        if controller_params is None:
            print('No controller is involved.')
            self.K = np.zeros([self.N_CONTROLS, self.N_DIMS])
        else:
            print('Controller is involved.')
            self.K = controller_params['K']

    # DONE: 建模，正负号
    def _f(self,x: torch.Tensor):
        theta = x[0]
        theta_dot = x[1]
        m, L, b = self.m, self.L, self.b
        g = self.__g
        f = torch.zeros(self.N_DIMS, 1)
        f[0, 0] = theta_dot
        f[1, 0] = (g / L)*torch.sin(theta) - b * theta_dot / (m * L**2) 
        return f
        
    def _g(self,x: torch.Tensor):
        g = torch.zeros(self.N_DIMS, self.N_CONTROLS)
        g = g.type_as(x)
        m, L = self.m, self.L
        g[1, 0] = 1 / (m * L ** 2)
        return g

    
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
        print(dynamics(goal_point))
        A = jacobian(dynamics, goal_point).squeeze().cpu().detach().numpy()
        A = np.reshape(A, (self.N_DIMS, self.N_DIMS))
        B = self._g(goal_point).squeeze().cpu().numpy()
        B = np.reshape(B, (self.N_DIMS, self.N_CONTROLS))
        print("linearized_ct_system:\n A{},\n B{}".format(A, B))
        return A, B

    
    def state_dims(self)->int:
        return InvertedPendulum.N_DIMS
        
 
    def control_dims(self)->int:
        return InvertedPendulum.N_CONTROLS
        
    
    def linearize_and_compute_LQR(self):
        goal_point = torch.Tensor([[0.0],[0.0]]).to(self.device)
        u_eq = torch.Tensor([[0.0]]).to(self.device)
        """
        u_eq should be 
        torch.Tensor([[0.0]])
        instead of 
        torch.Tensor([0.0])
        """
        Act, Bct = self.linearized_ct_system(goal_point, u_eq)
        # Adt, Bdt = self.linearized_dt_system(goal_point, u_eq)
        K_np = self.compute_LQR_controller(Act, Bct)
        self.K = torch.tensor(K_np, dtype=torch.float)
        print('computed LQR controller is {}'.format(K_np))
        # return self.K


    def controller(
        self,
        x: torch.tensor
    )->torch.tensor:
        K = torch.tensor(self.K).type_as(x)
        u = -K@x
        u = torch.clamp(u, -self.controller_bound, self.controller_bound)
        return u

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
        
        if x_initial[0] > torch.pi:
            x_initial[0] = x_initial[0]-2*torch.pi
        elif x_initial[0] < -torch.pi:
            x_initial[0] = x_initial[0]+2*torch.pi

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
            
            if x_next[0] > torch.pi:
                x_next[0] = x_next[0]-2*torch.pi
            elif x_next[0] < -torch.pi:
                x_next[0] = x_next[0]+2*torch.pi
                
            data_sim[i - 1, 1, :, :] = k1  # Store k1 (optional, for analysis)
            data_sim[i, 0, :, :] = x_next
        # print('simulation using rk4.')

        return data_sim

    def plot_phase_portrait(
        self, 
        data_sim: torch.Tensor,
        arrow_on: bool = False,
        title = 'inverted pendulum phase portrait',
        save_fig = False,
        save_path = 'saved_files/figs/'
    ):
        # data_sim = torch.cat((x_sim.unsqueeze(0), x_dot_sim.unsqueeze(0)), dim=0)
        # x_sim = data_sim[0]
        # x_dot_sim = data_sim[1]
        data_sim = data_sim.cpu().detach().numpy()
        x_sim = data_sim[:, 0, :, :]
        x_dot_sim = data_sim[:, 1, :, :]
        # plt.figure(figsize=(4, 8))
        fig, ax = plt.subplots(figsize=(6, 6))
        # data is in the form of torch.Tensor
        num = x_sim.shape[0]
        dim = x_sim.shape[1]
        # state tensors are column vectors as default
        x_value = x_sim[:, 0, 0]
        y_value = x_sim[:, 1, 0]
        circle_2 = plt.Circle((0, 0), 2, color='goldenrod', fill=True, alpha=0.5, label='||x||<2')
        circle_1 = plt.Circle((0, 0), 1, color='limegreen', fill=True, alpha=0.5, label='||x||<1')
        dot_0 = plt.scatter(0, 0, s = 60, color='green')
        ax.add_artist(circle_2)
        ax.add_artist(circle_1)
        ax.add_artist(dot_0)
        ax.scatter(x_value, y_value, s = 10, color='purple', label='trajectory')
        if arrow_on is True:
            interval = 200  # 每隔 interval 个点进行绘制
            for i in range(0, num, interval):
                x_dot_x = x_dot_sim[i, 0, 0]  # x 维度上的导数
                x_dot_y = x_dot_sim[i, 1, 0]  # y 维度上的导数
                dx = 2*x_dot_x/((x_dot_x**2+x_dot_y**2)**0.5)
                dy = 2*x_dot_y/((x_dot_x**2+x_dot_y**2)**0.5)
                ax.arrow(x_sim[i, 0, 0], x_sim[i, 1, 0], dx, dy, head_width=3, head_length=4, fc='r', ec='r', label='Gradient')
        # ax.xlim( (-np.pi, np.pi))
        # ax.ylim( (-np.pi, np.pi))
        ax = plt.gca()
        ax.set_aspect(1)
        ax.set_xlabel(r"$\mathregular{\theta}$")
        ax.set_ylabel(r"$\mathregular{\dot{\theta}}$")
        ax.set_title(title)
        ax.grid(True)
        # ax.legend(loc='upper right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.3)
        if save_fig:
            fig.savefig(save_path+title+'.png', dpi=600)
        fig.show()