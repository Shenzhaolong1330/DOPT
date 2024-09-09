##############################
## Author ## Shen Zhaolong ###
##  Date  ##   2024-07-10   ##
##############################


import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import torch
from torch.autograd.functional import jacobian
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Tuple, Optional, List
import scipy.linalg
from systems_and_functions.kits import continuous_lqr, discrete_lqr
from systems_and_functions.car_parameters import VehicleParameters
plt.rcParams.update({'font.size': 20})  # 设置全局字体大小为16

from systems_and_functions.control_affine_system import ControlAffineSystem

class SingleTrackCar(ControlAffineSystem):
    N_DIMS = 7
    N_CONTROLS = 2
    __g = 9.81
    
    # State indices
    SXE = 0 # x方向位置误差，偏离期望路径距离
    SYE = 1 # y方向位置误差
    DELTA = 2 # 转向角，前轮转向角度
    VE = 3 # 纵向速度误差，前进方向速度
    PSI_E = 4 # 方向误差角
    PSI_E_DOT = 5 # 方向误差角的导数
    BETA = 6 # 侧向偏转角，当前方向与车辆中心线之间的夹角

    # Control indices
    VDELTA = 0 # 转向角速度
    ALONG = 1 # 纵向加速度


    def __init__(
        self,
        system_params: dict,  
        # = {'psi_ref': 0, 'v_ref': 0, 'a_ref': 0, 'omega_ref': 0, 'mu_scale': 0}
        controller_params: Optional[dict] = None,
        dt: float = 0.01,
        controller_period: float = 0.01,
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ):
        ControlAffineSystem.__init__(self, system_params, controller_params, dt, controller_period)
        # system_params{'m': mass,'L': length, 'b': friction}
        self.device = device
        self.car_params = VehicleParameters()
        self.goal_point = torch.zeros((1, SingleTrackCar.N_DIMS))

        # # DONE: 处理 nominal_params: a dictionary giving the parameter values for the system.
        #                     Requires keys ["psi_ref", "v_ref", "a_ref",
        #                     "omega_ref", "mu_scale"] (_c and _s denote cosine and sine)
        #                     "mu_scale" is optional

        if controller_params is None:
            print('No controller is involved.')
            self.K = torch.zeros([self.N_CONTROLS, self.N_DIMS])
        else:
            print('Controller is involved.')
            self.K = torch.Tensor(controller_params['K'])
    
    def modifiy_param(self, param):
        self.system_params = param

    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[SingleTrackCar.SXE] = 1.0
        upper_limit[SingleTrackCar.SYE] = 1.0
        upper_limit[SingleTrackCar.DELTA] = self.car_params.steering_max
        upper_limit[SingleTrackCar.VE] = 1.0
        upper_limit[SingleTrackCar.PSI_E] = np.pi / 2
        upper_limit[SingleTrackCar.PSI_E_DOT] = np.pi / 2
        upper_limit[SingleTrackCar.BETA] = np.pi / 3

        lower_limit = -1.0 * upper_limit
        lower_limit[SingleTrackCar.DELTA] = self.car_params.steering_min

        return (upper_limit, lower_limit)

    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = 10 * torch.tensor(
            [
                5.0,  # self.car_params.steering_v_max,
                self.car_params.longitudinal_a_max,
            ]
        )
        lower_limit = 10 * torch.tensor(
            [
                -5.0,  # self.car_params.steering_v_min,
                -self.car_params.longitudinal_a_max,
            ]
        )

        return (upper_limit, lower_limit)

    def _f(self,x: torch.Tensor):

        batch_size = x.shape[0]
        f = torch.zeros((batch_size, SingleTrackCar.N_DIMS, 1))
        f = f.type_as(x)
        
        params = self.system_params
        v_ref = torch.tensor(params["v_ref"])
        a_ref = torch.tensor(params["a_ref"])
        omega_ref = torch.tensor(params["omega_ref"])
        if "mu_scale" in params:
            mu_scale = torch.tensor(params["mu_scale"])
        else:
            mu_scale = torch.tensor(1.0)

        # Extract the state variables and adjust for the reference
        v = x[:, SingleTrackCar.VE] + v_ref
        psi_e = x[:, SingleTrackCar.PSI_E]
        psi_e_dot = x[:, SingleTrackCar.PSI_E_DOT]
        psi_dot = psi_e_dot + omega_ref
        beta = x[:, SingleTrackCar.BETA]
        delta = x[:, SingleTrackCar.DELTA]
        sxe = x[:, SingleTrackCar.SXE]
        sye = x[:, SingleTrackCar.SYE]

        g = SingleTrackCar.__g  # [m/s^2]

        # create equivalent bicycle parameters
        mu = mu_scale * self.car_params.tire_p_dy1
        C_Sf = -self.car_params.tire_p_ky1 / self.car_params.tire_p_dy1
        C_Sr = -self.car_params.tire_p_ky1 / self.car_params.tire_p_dy1
        lf = self.car_params.a
        lr = self.car_params.b
        m = self.car_params.m
        Iz = self.car_params.I_z

        # We want to express the error in x and y in the reference path frame, so
        # we need to get the dynamics of the rotated global frame error
        dsxe_r = v * torch.cos(psi_e + beta) - v_ref + omega_ref * sye
        dsye_r = v * torch.sin(psi_e + beta) - omega_ref * sxe

        f[:, SingleTrackCar.SXE, 0] = dsxe_r
        f[:, SingleTrackCar.SYE, 0] = dsye_r
        f[:, SingleTrackCar.VE, 0] = -a_ref
        f[:, SingleTrackCar.DELTA, 0] = 0.0

        # Use the single-track dynamics if the speed is high enough, otherwise fall back
        # to the kinematic model (since single-track becomes singular for small v)
        use_kinematic_model = v.abs() < 0.1

        # Single-track dynamics
        f[:, SingleTrackCar.PSI_E, 0] = psi_e_dot
        # Sorry this is a mess (it's ported from the commonroad models)
        f[:, SingleTrackCar.PSI_E_DOT, 0] = (
        # f[:, SingleTrackCar.PSI_E_DOT] = (
            -(mu * m / (v * Iz * (lr + lf)))
            * (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf)
            * psi_dot
            + (mu * m / (Iz * (lr + lf)))
            * (lr * C_Sr * g * lf - lf * C_Sf * g * lr)
            * beta
            + (mu * m / (Iz * (lr + lf))) * (lf * C_Sf * g * lr) * delta
        )
        f[:, SingleTrackCar.BETA, 0] = (
        # f[:, SingleTrackCar.BETA] = (
            (
                (mu / (v ** 2 * (lr + lf))) * (C_Sr * g * lf * lr - C_Sf * g * lr * lf)
                - 1
            )
            * psi_dot
            - (mu / (v * (lr + lf))) * (C_Sr * g * lf + C_Sf * g * lr) * beta
            + mu / (v * (lr + lf)) * (C_Sf * g * lr) * delta
        )
        # Kinematic dynamics
        lwb = lf + lr
        km = use_kinematic_model
        f[km, SingleTrackCar.PSI_E, 0] = (
        # f[km, SingleTrackCar.PSI_E] = (
            v[km] * torch.cos(beta[km]) / lwb * torch.tan(delta[km]) - omega_ref
        ) 
        f[km, SingleTrackCar.PSI_E_DOT, 0] = 0.0
        f[km, SingleTrackCar.BETA, 0] = 0.0
        # f[km, SingleTrackCar.PSI_E_DOT] = 0.0
        # f[km, SingleTrackCar.BETA] = 0.0
        return f
    
    # def _f_batch(self,x: torch.Tensor):

    #     batch_size = x.shape[0]
    #     f = torch.zeros((batch_size, SingleTrackCar.N_DIMS, 1))
    #     f = f.type_as(x)
        
    #     params = self.system_params
    #     v_ref = torch.tensor(params["v_ref"])
    #     a_ref = torch.tensor(params["a_ref"])
    #     omega_ref = torch.tensor(params["omega_ref"])
    #     if "mu_scale" in params:
    #         mu_scale = torch.tensor(params["mu_scale"])
    #     else:
    #         mu_scale = torch.tensor(1.0)

    #     # Extract the state variables and adjust for the reference
    #     v = x[:, SingleTrackCar.VE] + v_ref
    #     psi_e = x[:, SingleTrackCar.PSI_E]
    #     psi_e_dot = x[:, SingleTrackCar.PSI_E_DOT]
    #     psi_dot = psi_e_dot + omega_ref
    #     beta = x[:, SingleTrackCar.BETA]
    #     delta = x[:, SingleTrackCar.DELTA]
    #     sxe = x[:, SingleTrackCar.SXE]
    #     sye = x[:, SingleTrackCar.SYE]

    #     g = SingleTrackCar.__g  # [m/s^2]

    #     # create equivalent bicycle parameters
    #     mu = mu_scale * self.car_params.tire_p_dy1
    #     C_Sf = -self.car_params.tire_p_ky1 / self.car_params.tire_p_dy1
    #     C_Sr = -self.car_params.tire_p_ky1 / self.car_params.tire_p_dy1
    #     lf = self.car_params.a
    #     lr = self.car_params.b
    #     m = self.car_params.m
    #     Iz = self.car_params.I_z

    #     # We want to express the error in x and y in the reference path frame, so
    #     # we need to get the dynamics of the rotated global frame error
    #     dsxe_r = v * torch.cos(psi_e + beta) - v_ref + omega_ref * sye
    #     dsye_r = v * torch.sin(psi_e + beta) - omega_ref * sxe

    #     # f[:, SingleTrackCar.SXE, 0] = dsxe_r
    #     # f[:, SingleTrackCar.SYE, 0] = dsye_r
    #     # f[:, SingleTrackCar.VE, 0] = -a_ref
    #     # f[:, SingleTrackCar.DELTA, 0] = 0.0

    #     f[:, SingleTrackCar.SXE] = dsxe_r
    #     f[:, SingleTrackCar.SYE] = dsye_r
    #     f[:, SingleTrackCar.VE] = -a_ref
    #     f[:, SingleTrackCar.DELTA] = 0.0

    #     # Use the single-track dynamics if the speed is high enough, otherwise fall back
    #     # to the kinematic model (since single-track becomes singular for small v)
    #     use_kinematic_model = v.abs() < 0.1

    #     # Single-track dynamics
    #     # f[:, SingleTrackCar.PSI_E, 0] = psi_e_dot
    #     f[:, SingleTrackCar.PSI_E] = psi_e_dot
    #     # Sorry this is a mess (it's ported from the commonroad models)
    #     # f[:, SingleTrackCar.PSI_E_DOT, 0] = (
    #     f[:, SingleTrackCar.PSI_E_DOT] = (
    #         -(mu * m / (v * Iz * (lr + lf)))
    #         * (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf)
    #         * psi_dot
    #         + (mu * m / (Iz * (lr + lf)))
    #         * (lr * C_Sr * g * lf - lf * C_Sf * g * lr)
    #         * beta
    #         + (mu * m / (Iz * (lr + lf))) * (lf * C_Sf * g * lr) * delta
    #     )
    #     # f[:, SingleTrackCar.BETA, 0] = (
    #     f[:, SingleTrackCar.BETA] = (
    #         (
    #             (mu / (v ** 2 * (lr + lf))) * (C_Sr * g * lf * lr - C_Sf * g * lr * lf)
    #             - 1
    #         )
    #         * psi_dot
    #         - (mu / (v * (lr + lf))) * (C_Sr * g * lf + C_Sf * g * lr) * beta
    #         + mu / (v * (lr + lf)) * (C_Sf * g * lr) * delta
    #     )
    #     # Kinematic dynamics
    #     lwb = lf + lr
    #     km = use_kinematic_model
    #     # f[km, SingleTrackCar.PSI_E, 0] = (
    #     f[km, SingleTrackCar.PSI_E] = (
    #         v[km] * torch.cos(beta[km]) / lwb * torch.tan(delta[km]) - omega_ref
    #     ) 
    #     # f[km, SingleTrackCar.PSI_E_DOT, 0] = 0.0
    #     # f[km, SingleTrackCar.BETA, 0] = 0.0
    #     f[km, SingleTrackCar.PSI_E_DOT] = 0.0
    #     f[km, SingleTrackCar.BETA] = 0.0
    #     return f
     
    def _g(self,x: torch.Tensor):
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, SingleTrackCar.N_DIMS, SingleTrackCar.N_CONTROLS))
        g = g.type_as(x)

        # Extract the parameters
        params = self.system_params
        v_ref = torch.tensor(params["v_ref"])
        omega_ref = torch.tensor(params["omega_ref"])
        if "mu_scale" in params:
            mu_scale = torch.tensor(params["mu_scale"])
        else:
            mu_scale = torch.tensor(1.0)

        # Extract the state variables and adjust for the reference
        v = x[:, SingleTrackCar.VE] + v_ref
        psi_e_dot = x[:, SingleTrackCar.PSI_E_DOT]
        psi_dot = psi_e_dot + omega_ref
        beta = x[:, SingleTrackCar.BETA]
        delta = x[:, SingleTrackCar.DELTA]

        # create equivalent bicycle parameters
        mu = mu_scale * self.car_params.tire_p_dy1
        C_Sf = -self.car_params.tire_p_ky1 / self.car_params.tire_p_dy1
        C_Sr = -self.car_params.tire_p_ky1 / self.car_params.tire_p_dy1
        lf = self.car_params.a
        lr = self.car_params.b
        h = self.car_params.h_s
        m = self.car_params.m
        Iz = self.car_params.I_z

        # Use the single-track dynamics if the speed is high enough, otherwise fall back
        # to the kinematic model (since single-track becomes singular for small v)
        use_kinematic_model = v.abs() < 0.1

        # Single-track dynamics
        g[:, SingleTrackCar.DELTA, SingleTrackCar.VDELTA] = 1.0
        g[:, SingleTrackCar.VE, SingleTrackCar.ALONG] = 1.0

        g[:, SingleTrackCar.PSI_E_DOT, SingleTrackCar.ALONG] = (
            -(mu * m / (v * Iz * (lr + lf)))
            * (-(lf ** 2) * C_Sf * h + lr ** 2 * C_Sr * h)
            * psi_dot
            + (mu * m / (Iz * (lr + lf))) * (lr * C_Sr * h + lf * C_Sf * h) * beta
            - (mu * m / (Iz * (lr + lf))) * (lf * C_Sf * h) * delta
        )
        g[:, SingleTrackCar.BETA, SingleTrackCar.ALONG] = (
            (mu / (v ** 2 * (lr + lf))) * (C_Sr * h * lr + C_Sf * h * lf) * psi_dot
            - (mu / (v * (lr + lf))) * (C_Sr * h - C_Sf * h) * beta
            - mu / (v * (lr + lf)) * C_Sf * h * delta
        )

        # Kinematic dynamics
        lwb = lf + lr
        km = use_kinematic_model
        beta_dot = (
            1
            / (1 + (torch.tan(delta) * lr / lwb) ** 2)
            * lr
            / (lwb * torch.cos(delta) ** 2)
        )
        g[km, SingleTrackCar.PSI_E_DOT, SingleTrackCar.ALONG] = (
            1 / lwb * (torch.cos(beta[km]) * torch.tan(delta[km]))
        )
        g[km, SingleTrackCar.PSI_E_DOT, SingleTrackCar.VDELTA] = (
            1
            / lwb
            * (
                -v[km] * torch.sin(beta[km]) * torch.tan(delta[km]) * beta_dot[km]
                + v[km] * torch.cos(beta[km]) / torch.cos(delta[km]) ** 2
            )
        )
        g[km, SingleTrackCar.BETA, 0] = beta_dot[km]
        
        return g

    def controller(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        # x = x.t()
        params = self.system_params
        # print('params unsed in controller',params)

        # Compute the LQR gain matrix for the nominal parameters
        # create equivalent bicycle parameters
        g = SingleTrackCar.__g  # [m/s^2]
        mu = self.car_params.tire_p_dy1
        C_Sf = -self.car_params.tire_p_ky1 / self.car_params.tire_p_dy1
        C_Sr = -self.car_params.tire_p_ky1 / self.car_params.tire_p_dy1
        lf = self.car_params.a
        lr = self.car_params.b
        m = self.car_params.m
        Iz = self.car_params.I_z

        # Linearize the system about the path
        # self.goal_point = torch.zeros((1, SingleTrackCar.N_DIMS))
        x0 = self.goal_point
        x0[0, SingleTrackCar.PSI_E_DOT] = params["omega_ref"]
        x0[0, SingleTrackCar.DELTA] = (
            (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf)
            / (lf * C_Sf * g * lr)
            * params["omega_ref"]
            / params["v_ref"]
        )
        x0[0, SingleTrackCar.DELTA] /= lf * C_Sf * g * lr
        A = np.zeros((SingleTrackCar.N_DIMS, SingleTrackCar.N_DIMS))
        A[SingleTrackCar.SXE, SingleTrackCar.VE] = 1.0
        A[SingleTrackCar.SXE, SingleTrackCar.SYE] = params["omega_ref"]

        A[SingleTrackCar.SYE, SingleTrackCar.SXE] = -params["omega_ref"]
        A[SingleTrackCar.SYE, SingleTrackCar.PSI_E] = params["v_ref"]
        A[SingleTrackCar.SYE, SingleTrackCar.BETA] = params["v_ref"]

        A[SingleTrackCar.PSI_E, SingleTrackCar.PSI_E_DOT] = 1.0

        A[SingleTrackCar.PSI_E_DOT, SingleTrackCar.VE] = (
            (mu * m / (params["v_ref"] ** 2 * Iz * (lr + lf)))
            * (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf)
            * params["omega_ref"]
        )
        A[SingleTrackCar.PSI_E_DOT, SingleTrackCar.PSI_E_DOT] = -(
            mu * m / (params["v_ref"] * Iz * (lr + lf))
        ) * (lf ** 2 * C_Sf * g * lr + lr ** 2 * C_Sr * g * lf)
        A[SingleTrackCar.PSI_E_DOT, SingleTrackCar.BETA] = +(mu * m / (Iz * (lr + lf))) * (
            lr * C_Sr * g * lf - lf * C_Sf * g * lr
        )
        A[SingleTrackCar.PSI_E_DOT, SingleTrackCar.DELTA] = (mu * m / (Iz * (lr + lf))) * (
            lf * C_Sf * g * lr
        )

        A[SingleTrackCar.BETA, SingleTrackCar.VE] = (
            -2
            * (mu / (params["v_ref"] ** 3 * (lr + lf)))
            * (C_Sr * g * lf * lr - C_Sf * g * lr * lf)
            * params["omega_ref"]
            - mu
            / (params["v_ref"] ** 2 * (lr + lf))
            * (C_Sf * g * lr)
            * x0[0, SingleTrackCar.DELTA]
        )
        A[SingleTrackCar.BETA, SingleTrackCar.PSI_E_DOT] = (mu / (params["v_ref"] ** 2 * (lr + lf))) * (
            C_Sr * g * lf * lr - C_Sf * g * lr * lf
        ) - 1
        A[SingleTrackCar.BETA, SingleTrackCar.BETA] = -(mu / (params["v_ref"] * (lr + lf))) * (
            C_Sr * g * lf + C_Sf * g * lr
        )
        A[SingleTrackCar.BETA, SingleTrackCar.DELTA] = (
            mu / (params["v_ref"] * (lr + lf)) * (C_Sf * g * lr)
        )
        # A = np.eye(SingleTrackCar.N_DIMS) + self.controller_period * A
        B = self._g(self.goal_point).squeeze().cpu().numpy()
        # B = self.controller_period * B
        # Define cost matrices as identity
        # Q = np.eye(SingleTrackCar.N_DIMS)
        a = [10, 10, 1, 1, 10, 10, 1]
        Q = np.diag(a)
        R = np.eye(SingleTrackCar.N_CONTROLS)
        # Get feedback matrix (only recompute if we need to)
        K = torch.tensor(continuous_lqr(A, B, Q, R))
        self.K = K
        # 在跟踪轨迹加速度不变的情况下，控制器是时不变的
        # Compute nominal control from feedback + equilibrium control
        x0 = x0.type_as(x)
        u_nominal = -(K.type_as(x) @ (x - x0).T).T
        u_eq = torch.zeros_like(u_nominal)
        u = u_nominal + u_eq
        # Clamp given the control limits
        upper_u_lim, lower_u_lim = self.control_limits()
        for dim_idx in range(SingleTrackCar.N_CONTROLS):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )
        return u

    def x_dot(
            self,
            x: torch.Tensor,
            u: torch.Tensor
    ):
        u = u.view(2, 1)
        f = self._f(x).to(self.device).float()
        g = self._g(x).to(self.device).float()
        x_dot = f + (g @ u).reshape(-1, 1)
        # g = torch.zeros((batch_size, SingleTrackCar.N_DIMS, SingleTrackCar.N_CONTROLS))
        # u = 
        # f = torch.zeros((batch_size, SingleTrackCar.N_DIMS, 1))
        return x_dot.view(x.shape)
    
    def state_dims(self)->int:
        return SingleTrackCar.N_DIMS
        
    def control_dims(self)->int:
        return SingleTrackCar.N_CONTROLS

    def run_rk4(self,
                t_sim: int = 10,
                the_controller = None,
                delta_x: torch.tensor = torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
                sample_mode: bool = False
                ):
        if the_controller is None:
            the_controller = self.controller
            # print('using internal controller')
        else:
            the_controller = the_controller
            # print('using external controller')

        device = self.device
        delta_t = self.dt
        T = int(t_sim // delta_t)
        n_dims = self.state_dims()
        n_controls = self.control_dims()
        x_current = torch.zeros(1, n_dims, device=device)
        delta_x = delta_x.to(device)
        
        # delta_x = torch.tensor([0, 0, -1, 0, 0, 0, 0], dtype=torch.float32).to(device)
        # 采样范围：
        # SXE = 0 # x方向位置误差，偏离期望路径距离
        # SYE = 1 # y方向位置误差
        # DELTA = 2 # 转向角，前轮转向角度 -1
        # VE = 3 # 纵向速度误差，前进方向速度 -0.5 - 5
        # PSI_E = 4 # 方向误差角 -1 - 1
        # PSI_E_DOT = 5 # 方向误差角的导数 -5 - 5
        # BETA = 6 # 侧向偏转角，当前方向与车辆中心线之间的夹角 -3 - 3
        x_current = x_current + (self.goal_point.type_as(x_current) + delta_x)
        u_current = torch.zeros(1, n_controls).type_as(x_current)

        # And create a place to store the reference path
        x_ref = 0.0
        y_ref = 0.0
        psi_ref = self.system_params["psi_ref"]
        omega_ref = self.system_params["omega_ref"]
        a_ref = self.system_params["a_ref"]
        v_ref = self.system_params["v_ref"]

        state_sim = torch.zeros(T, 2, x_current.shape[0], x_current.shape[1], dtype=torch.float32).to(device)
        control_output_sim = torch.zeros(T, u_current.shape[0], u_current.shape[1], dtype=torch.float32).to(device)
        traj_sim = np.zeros((T,5))
        controller_update_freq = int(self.controller_period / delta_t)
        if sample_mode:
            prog_bar_range = range(0, T)
        else:
            prog_bar_range = tqdm.trange(0, T, desc="S-Curve", leave=True)
        for tstep in prog_bar_range:
            # Update the reference path to trace an S curve
            # omega_ref = 0.4 * np.sin(tstep * delta_t)
            psi_ref += delta_t * omega_ref
            v_ref += delta_t * a_ref
            self.system_params["omega_ref"] = omega_ref
            self.system_params["psi_ref"] = psi_ref
            self.system_params["v_ref"] = v_ref
            x_ref += delta_t * self.system_params["v_ref"] * np.cos(psi_ref)
            y_ref += delta_t * self.system_params["v_ref"] * np.sin(psi_ref)

            state_sim[tstep,0] = x_current
            # Update the controller if it's time
            if tstep % controller_update_freq == 0:
                u_current = the_controller(x_current).view(2, 1)
            
            k1 = self.x_dot(x_current, u_current)
            k2 = self.x_dot(x_current + self.dt / 2 * k1, u_current)
            k3 = self.x_dot(x_current + self.dt / 2 * k2, u_current)
            k4 = self.x_dot(x_current + self.dt * k3, u_current) 
            x_current = x_current + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            
            # one step euler
            # x_dot = self.x_dot(x_current, u_current.t())
            # x_current = x_current + delta_t * x_dot
            
            state_sim[tstep,1] = x_current
            control_output_sim[tstep] = u_current.t()

            x_err = x_current[0, SingleTrackCar.SXE].cpu().item()
            y_err = x_current[0, SingleTrackCar.SYE].cpu().item()
            psi_err = x_current[0, SingleTrackCar.PSI_E].cpu().item()
            x = x_ref + x_err * np.cos(psi_ref) - y_err * np.sin(psi_ref)
            y = y_ref + x_err * np.sin(psi_ref) + y_err * np.cos(psi_ref)
            # err2 = x_err ** 2 + y_err ** 2 + psi_err ** 2
            err2 = x_err ** 2 + y_err ** 2
            err = err2 ** 0.5
            traj_sim[tstep] = np.array([x, y, x_ref, y_ref, err])
            # respectively store1 x, y, x_ref, y_ref, err
            # store2 state
            # state ([100, 2, 7, 1])
            # dim1: sample index
            # dim2: x  x_next,nixiang
            # dim3: SXE SYE DELTA VE PSI_E PSI_E_DOT
            # dim4: value

        return state_sim, control_output_sim, traj_sim

    def plot_traj(
            self,
            traj_sim: np.array,
            title = 'Tracking',
            save_fig = False,
            save_path = 'saved_files/figs/'
    ):
        sns.set_theme(context="talk", style="white")

        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(16, 8)

        ax = axs[0]
        tracking_trajectory_color = sns.color_palette("pastel")[4]
        start_point_color = sns.color_palette("pastel")[3]
        x_ref = traj_sim[:,2]
        y_ref = traj_sim[:,3]
        x = traj_sim[:,0]
        y = traj_sim[:,1]
        x0 = x[0]
        y0 = y[0]
        err = traj_sim[:,4]
        ax.plot(
            x_ref,
            y_ref,
            linestyle="dotted",
            label="Reference",
            color="black",
            linewidth = 4
        )
        ax.plot(
            x,
            y,
            linestyle="solid",
            label="Controller",
            color=tracking_trajectory_color,
            alpha = 0.8,
            linewidth = 4
        )
        ax.scatter(
            x0,
            y0,
            marker = 'o',
            label="Start",
            color=start_point_color
        )
        ax.set_xlabel("$x (m)$")
        ax.set_ylabel("$y (m)$")
        ax.set_title('Tracking Trajectory')
        ax.grid(True)
        ax.legend()
        ax.set_aspect("equal")  

        ax = axs[1]
        ax.plot(np.arange(err.shape[0])*self.dt, err, color = sns.color_palette("pastel")[0],linewidth = 4)
        ax.set_xlabel("$t (s)$")
        ax.set_ylabel("error")
        ax.grid(True)
        ax.set_title('Tracking Error')
        ax.set_xlim(0, (err.shape[0]+50)*self.dt)
        ax.set_ylim(0.0, np.max(err)*1.05)
        fig.suptitle(title)
        if save_fig:
            fig.savefig(save_path + title + '.png', dpi=600)
        fig.show()


    # def plot_multi_traj(
    #         self,
    #         multi_traj_sim: np.array,
    #         title = 'Tracking',
    #         save_fig = False,
    #         save_path = 'saved_files/figs/'
    # ):
    #     num = multi_traj_sim.shape[0]
    #     sns.set_theme(context="talk", style="white")

    #     fig, axs = plt.subplots(1, 2)
    #     fig.set_size_inches(16, 8)

    #     ax = axs[0]
    #     tracking_trajectory_color = sns.color_palette("pastel")[4]
    #     start_point_color = sns.color_palette("pastel")[3]
    #     x_ref = traj_sim[:,2]
    #     y_ref = traj_sim[:,3]
    #     x = traj_sim[:,0]
    #     y = traj_sim[:,1]
    #     x0 = x[0]
    #     y0 = y[0]
    #     err = traj_sim[:,4]
    #     ax.plot(
    #         x_ref,
    #         y_ref,
    #         linestyle="dotted",
    #         label="Reference",
    #         color="black",
    #         linewidth = 4
    #     )
    #     ax.plot(
    #         x,
    #         y,
    #         linestyle="solid",
    #         label="Controller",
    #         color=tracking_trajectory_color,
    #         alpha = 0.8,
    #         linewidth = 4
    #     )
    #     ax.scatter(
    #         x0,
    #         y0,
    #         marker = 'o',
    #         label="Start",
    #         color=start_point_color
    #     )
    #     ax.set_xlabel("$x$")
    #     ax.set_ylabel("$y$")
    #     ax.set_title('Tracking Trajectory')
    #     ax.grid(True)
    #     ax.legend()
    #     ax.set_aspect("equal")  

    #     ax = axs[1]
    #     ax.plot(np.arange(err.shape[0])*self.dt, err, color = sns.color_palette("pastel")[0],linewidth = 4)
    #     ax.set_xlabel("$t / s$")
    #     ax.set_ylabel("error")
    #     ax.grid(True)
    #     ax.set_title('Tracking Error')
    #     ax.set_xlim(0, (err.shape[0]+50)*self.dt)
    #     ax.set_ylim(0.0, np.max(err)*1.05)
    #     fig.suptitle(title)
    #     if save_fig:
    #         fig.savefig(save_path + title + '.png', dpi=600)
    #     fig.show()


    def sample_training_data_stcar(
        self,
        sample_state: dict = {'sxe_range':(-2, 2), 'sye_range': (-2, 2), 'delta_range': (-1, 1), 've_range': (-0.5, 1), 'psi_e_range': (-1, 1), 'psi_e_dot_range': (-3, 3), 'beta_range': (-2, 2)},
        sample_num: int = 1000,
        invariant_sample: bool = True,
        sample_plot: bool = True,
        the_controller = None, # 指定控制器，默认为ControlAffineSystem自带控制器
        title = "Samples"
    )->torch.Tensor:
        print('------------------Sampling Training Data STCar------------------')
        # initialize system: use LQR as 
        # self.system.compute_LQR_controller()
        # self.system.use_LQR_controller()
        # sample initialize ([sample_num, 2, state_dim, 1])
        # DONE: sample function
        if invariant_sample:
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
        # 定义状态空间的稳定采样范围
        # sample_state: dict = {'sxe_range':(-2, 2), 'sye_range': (-2, 2), 'delta_range': (-1, 1), 've_range': (-0.5, 1), 'psi_e_range': (-1, 1), 'psi_e_dot_range': (-3, 3), 'beta_range': (-2, 2)},
        sxe_range = sample_state['sxe_range']  # SXE x方向位置误差
        sye_range = sample_state['sye_range']  # SYE y方向位置误差
        delta_range = sample_state['delta_range']  # DELTA 转向角
        ve_range = sample_state['ve_range']  # VE 纵向速度误差
        psi_e_range = sample_state['psi_e_range']  # PSI_E 方向误差角
        psi_e_dot_range = sample_state['psi_e_dot_range']  # PSI_E_DOT 方向误差角的导数
        beta_range = sample_state['beta_range'] # BETA 侧向偏转角
        # 随机采样
        sxe_samples = torch.rand(sample_num, 1) * (sxe_range[1] - sxe_range[0]) + sxe_range[0]
        sye_samples = torch.rand(sample_num, 1) * (sye_range[1] - sye_range[0]) + sye_range[0]
        delta_samples = torch.rand(sample_num, 1) * (delta_range[1] - delta_range[0]) + delta_range[0]
        ve_samples = torch.rand(sample_num, 1) * (ve_range[1] - ve_range[0]) + ve_range[0]
        psi_e_samples = torch.rand(sample_num, 1) * (psi_e_range[1] - psi_e_range[0]) + psi_e_range[0]
        psi_e_dot_samples = torch.rand(sample_num, 1) * (psi_e_dot_range[1] - psi_e_dot_range[0]) + psi_e_dot_range[0]
        beta_samples = torch.rand(sample_num, 1) * (beta_range[1] - beta_range[0]) + beta_range[0]
        state_samples = torch.cat((sxe_samples, sye_samples, delta_samples, ve_samples, psi_e_samples, psi_e_dot_samples, beta_samples), dim=1).unsqueeze(2)
        state_samples = state_samples.permute(0,2,1)
        # 模型不支持批处理
        sample_state_space = torch.zeros(state_samples.shape[0],2,SingleTrackCar.N_DIMS,1).to(self.device)
        sample_control_output = torch.zeros(state_samples.shape[0],SingleTrackCar.N_CONTROLS,1).to(self.device)
        prog_bar_range = tqdm.trange(0, state_samples.shape[0], desc="sample state space", leave=True)
        for i in prog_bar_range:
            state_sim, control_output_sim, _ = self.run_rk4(t_sim = self.dt,
                                                            the_controller = the_controller,
                                                            delta_x = state_samples[i],
                                                            sample_mode = True)
            sample_state_space[i] = state_sim[0].permute(0,2,1)
            sample_control_output[i] = control_output_sim[0].permute(1,0)


        if sample_plot:
            fig, ax = plt.subplots()
            ax.scatter(state_samples[:,0,0], state_samples[:,0,1], s=10, alpha=0.5, c='c', label='dim 0,1')
            ax = plt.gca()
            ax.set_aspect(1)
            ax.set_xlabel(r'$x_{err}$')#r"$\theta$"
            ax.set_ylabel(r'$y_{err}$')
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
            plt.show()
            plt.clf()
            plt.close()
        # torch.Size([1000, 2, 7, 1])
        # torch.Size([1000, 2, 1])
        return sample_state_space, sample_control_output
    
    def convergence_judgment(
            self,
            traj_sim: np.array,
            epsilon: float = 0.2  #0.2
    ):
        """
        judge if the system converge to set point
        """
        step2converge = 0
        step2unitball = 0
        step2norm2ball = 0
        for i in range(traj_sim.shape[0]):
            if np.linalg.norm(traj_sim[i,4]) <= 2:
                step2norm2ball = i
                break
            else:
                step2norm2ball = float("inf")
        for i in range(traj_sim.shape[0]):
            if np.linalg.norm(traj_sim[i,4]) <= 0.5:
                step2unitball = i
                break
            else:
                step2unitball = float("inf")
        for i in range(traj_sim.shape[0]):
            if np.linalg.norm(traj_sim[i,4]) <= epsilon:
                step2converge = i
                break
            else:
                step2converge = float("inf")
        print('-----------------Convergence Speed and Judgment-----------------')
        print('--------------It takes {} steps to norm 2 ball;--------------\n---------------It takes {} steps to unit ball;---------------\n----------------It takes {} steps to converge.--------------'.format(step2norm2ball, step2unitball, step2converge))
        return step2norm2ball,step2unitball,step2converge


    def plot_phase_portrait(
            self,
            traj_sim: np.array,
            title = 'System Phase Portrait',
            save_fig = False,
            save_path = 'saved_files/figs/'
    ):
        x_ref = traj_sim[:,2]
        y_ref = traj_sim[:,3]
        x = traj_sim[:,0]
        y = traj_sim[:,1]
        x_err = x - x_ref
        y_err = y - y_ref
        fig, ax = plt.subplots(figsize=(8, 5))
        # circle_2 = plt.Circle((0, 0), 2, color='goldenrod', fill=True, alpha=0.5, label='||x||<2')
        # circle_1 = plt.Circle((0, 0), 1, color='limegreen', fill=True, alpha=0.5, label='||x||<1')
        circle_1 = plt.Circle((0, 0), 0.5, color='limegreen', fill=True, alpha=0.5, label='||x||<0.5')
        dot_0 = plt.scatter(0, 0, s = 60, color='green')
        # ax.add_artist(circle_2)
        ax.add_artist(circle_1)
        ax.add_artist(dot_0)
        ax.scatter(x_err, y_err, s = 10, color='purple', label='Position tracking error')
        ax = plt.gca()
        ax.set_aspect(1)
        plt.xlabel(r"$\mathregular{x_{err}}$")
        plt.ylabel(r"$\mathregular{y_{err}}$")
        ax.set_title(title)
        ax.grid(True)
        # ax.legend(loc='upper right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.3)
        if save_fig:
            fig.savefig(save_path+title+'.png', dpi=600)
        fig.show()