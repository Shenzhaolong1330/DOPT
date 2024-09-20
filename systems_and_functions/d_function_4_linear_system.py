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
from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
)

from systems_and_functions.linear_control_affine_system import LinearControlAffineSystem



class DFunction4LinearSystem():
    """
    Represents a D_Function formed with a quadratic Lyapunov function
    """
    N_DIMS = 2
    N_CONTROLS = 1
    P = torch.tensor(np.eye(N_DIMS), dtype=torch.float, requires_grad=True)
    Learned_P = torch.tensor(np.eye(N_DIMS), dtype=torch.float, requires_grad=True)
    Learned_A = torch.tensor(np.eye(N_DIMS), dtype=torch.float, requires_grad=True)
    Learned_B = torch.tensor(np.zeros((N_DIMS, N_CONTROLS)), dtype=torch.float, requires_grad = True)
    Learned_K = torch.tensor(np.zeros((N_CONTROLS, N_DIMS)), dtype=torch.float, requires_grad = True)
    # A = torch.tensor(np.array([[0, -1], [2, 0]]),dtype=torch.float)
    # B = torch.tensor(np.array([[0], [1]]),dtype=torch.float)

    def __init__(
        self, 
        P_ = None,
        dynamic_system = LinearControlAffineSystem
    ):
        # P is a matrix represented by tensor
        if P_ is None:
            P = np.eye(self.N_DIMS)
            P = torch.tensor(P, requires_grad=True)
        else:
            # P = torch.tensor(P, requires_grad=True)
            # P = P.clone().detach().requires_grad_(True)
            P = torch.tensor(P_, requires_grad=True)

        if not P.equal(P.t()):
            raise ValueError("Error: P is not Symmetrical!")
        else:
            self.N_DIMS = P.shape[0]
            self.P = P

        self.system = dynamic_system
        self.Learned_K = torch.tensor(dynamic_system.K.detach().numpy(), dtype=torch.float, requires_grad = True)


    def V_value(self, x: torch.Tensor = torch.zeros(N_DIMS, 1)):
        # return x.t()@self.P@x
        return x.t() @ self.P.to(x.dtype) @ x
    

    def V_value_batch(self, x: torch.Tensor = torch.zeros(100, N_DIMS, 1)):
        """
        x:[sample num, n dims, value]
        return [sumple num, value]
        """
        # P = self.P.to(x.dtype)
        P = self.P
        xT = x.permute(0, 2, 1)
        return torch.matmul(torch.matmul(xT, P), x)
    
    
    def V_jacobian(self, x: torch.Tensor = torch.zeros(N_DIMS, 1)):
        # return 2*self.P@x
        return 2 * self.P.to(x.dtype) @ x
    

    def V_jacobian_batch(self, x: torch.Tensor = torch.zeros(100, N_DIMS, 1)):
        """
        x:[sample num, n dims, 1]
        return [sumple num, 1, n dims]
        """
        P = self.P.to(x.dtype)
        xT = x.permute(0, 2, 1)
        # x:[sample num, 1, n dims]
        # return torch.matmul(xT, P).permute(0, 2, 1)
        return torch.matmul(xT, P)


    def V_dot_analytical(
        self, 
        x: torch.Tensor = torch.zeros(N_DIMS, 1), 
        x_dot: torch.Tensor = torch.zeros(N_DIMS, 1)
    ):
        return x_dot.t()@self.V_jacobian(x)


    def V_dot_analytical_batch(
        self, 
        x: torch.Tensor = torch.zeros(100, N_DIMS, 1), 
        x_dot: torch.Tensor = torch.zeros(100, N_DIMS, 1)
    ):
        """
        x:[sample num, n dims, 1]
        return [sumple num, 1, 1]
        """
        # print('self.V_jacobian(x).shape:', self.V_jacobian_batch(x).shape)
        # print('x_dot.shape:',x_dot.shape)
        return torch.matmul(self.V_jacobian_batch(x), x_dot)


    def V_dot_numerical(
        self,
        x_k: torch.Tensor = torch.zeros(N_DIMS, 1),
        x_k1: torch.Tensor = torch.zeros(N_DIMS, 1),
        step: int = 1
    ):
        return (x_k1 - x_k)/(step*self.system.dt)


    def learned_system(
        self,
        x: torch.Tensor = torch.zeros(N_DIMS, 1)
        # u: torch.Tensor = torch.zeros(N_CONTROLS, 1)
    ):
        # u = self.system.controller(x)
        u = self.Learned_K.type_as(x)@x
        return self.Learned_A@x+self.Learned_B@u


    def learned_system_batch(
        self,
        x: torch.Tensor = torch.zeros(100, N_DIMS, 1)
        # u: torch.Tensor = torch.zeros(100, N_CONTROLS, 1)
    ):
        """
        return [sumple num, n dims, 1]
        """
        # u = self.system.controller_batch(x)
        K = self.Learned_K
        u = (x.permute(0, 2, 1).float()@K.t().float()).permute(0, 2, 1).float()
        Ax = (x.permute(0, 2, 1)@self.Learned_A.t()).permute(0, 2, 1)
        Bu = (u.permute(0, 2, 1)@self.Learned_B.t()).permute(0, 2, 1)
        # return self.Learned_A@x+self.Learned_B@u
        return Ax+Bu


    def sample_training_data(
        self,
        sample_trajectory_number: int = 10,
        sample_number_per_trajectory: int = 100,
        sample_radius: int = 15,
        sample_number_in_radius: int = 200,
        invariant_sample: bool = True,
        sample_draw: bool = True
    )->torch.Tensor:
        print('---------------------Sampling Training Data---------------------')
        """
        sample training data
        data content: x_i, x_dot_i, controller_params
        sample strategy: data set is composed of two parts: 
            samples from trajectory;
            samples from random state;
            (if invariant_sample is true, random state point is Unchanging) 
        """
        # initialize system: use LQR as 
        # self.system.compute_LQR_controller()
        # self.system.use_LQR_controller()
        # sample initialize ([100, 2, 2, 1])
        # dim1: sample index
        # dim2: x x_dot
        # dim3: x1 x2
        # dim4: value
        sample_from_trajectory = torch.zeros(sample_trajectory_number*sample_number_per_trajectory,2,2,1)
        sample_from_radius = torch.zeros(sample_number_in_radius,2,2,1)
        theta = np.linspace(0, 2*np.pi, sample_trajectory_number+1)
        
        # sample in trajectory
        # x_0_traj = torch.zeros(sample_trajectory_number,2,1)
        for i in range(0,sample_trajectory_number):
            # print(i)
            x_0_traj = torch.tensor([[sample_radius*np.cos(theta[i])],[sample_radius*np.sin(theta[i])]],dtype=torch.float)
            # print(x_0_traj)
            sample_from_trajectory[i*sample_number_per_trajectory:(i+1)*sample_number_per_trajectory] = self.system.simulate_rk4(x_0_traj,sample_number_per_trajectory,1)
            # print(sample_from_trajectory[i])
        # sample_from_trajectory = sample_from_trajectory.view(sample_trajectory_number*sample_number_per_trajectory,2,2,1)
        # print('sample_from_trajectory:', sample_from_trajectory)

        # sample randomly in radius
        if invariant_sample == True:
            np.random.seed(42)
        theta_ = np.random.uniform(0, 2*np.pi, sample_number_in_radius)
        r_ = np.sqrt(np.random.uniform(0, sample_radius**2, sample_number_in_radius))
        combined_data = zip(theta_, r_)
        i = 0
        for data in combined_data:
            theta__, r__ = data
            x_0_radius = torch.tensor([[r__ * np.cos(theta__)],[r__ * np.sin(theta__)]])
            # print('sample:',x_0_radius)
            sample_from_radius[i] = self.system.one_step_euler(x_0_radius,1)[1]
            i = i + 1
            # print('sample:',sample_from_radius[i])
        
        if sample_draw == True:
            x1_trajectory = sample_from_trajectory[:,0,0,:].detach().numpy()
            x2_trajectory = sample_from_trajectory[:,0,1,:].detach().numpy()
            plt.scatter(x1_trajectory, x2_trajectory, label='sample_from_trajectory')

            x1_radius = sample_from_radius[:,0,0,:].detach().numpy()
            x2_radius = sample_from_radius[:,0,1,:].detach().numpy()
            plt.scatter(x1_radius, x2_radius, label='sample_from_radius')
            ax = plt.gca()
            ax.set_aspect(1)
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.title('samples')
            plt.legend()
            plt.grid(True)
            plt.show()

        # print('sample_from_radius:', sample_from_radius)
        sample_data = torch.cat((sample_from_trajectory,sample_from_radius),dim=0)
        return sample_data


    def learn_V(
        self,
        sample_data
    ):
        """
        learn a V(x) by constraining V(x)>=0, V_dot(x)<=0
        """
        print('--------------------------Learning V--------------------------')
        sample_data = sample_data.detach().numpy()
        P = cp.Variable((self.system.N_DIMS, self.system.N_DIMS))
        eta = cp.Variable()
        
        constraints = [P.T == P, eta>=0]
        for i in range(sample_data.shape[0]):
            xi = sample_data[i][0]
            xi_dot = sample_data[i][1]
            constraints.append(xi.T@P@xi>=0)
            constraints.append(xi.T@P@xi_dot+xi_dot.T@P@xi<=-eta*np.linalg.norm(xi)**2)

        objective = cp.Minimize(-eta)
        problem = cp.Problem(objective, constraints)
        # 求解优化问题
        problem.solve()
        print("status:",problem.status)
        # print("optimal value",problem.value)
        # 输出解
        print("var eta (eta should be positive):", eta.value)
        print("var P:", P.value)
        self.P = torch.tensor(P.value)


    def verify_lyapunov_P(
            self,
            sample_data
            ):
        """
        verifing lyapunov function xTPx在数据集上是否正定，导数是否负定
        """
        print('---------------------Verifing Lyapunov P----------------------')
        # sample_data = sample_data.numpy()
        A = self.system.A
        B = self.system.B
        K = self.system.K.detach().numpy()
        Acl = A+B@K
        P = self.P.detach().numpy()
        W = Acl.T@P+P@Acl
        eigVals, _ = scipy.linalg.eig(W)
        if all(eig < 0 for eig in eigVals):
            print('P is effective')
            print('eigs of Acl.T@P+P@Acl',eigVals)
        else:
            print('P is INeffective')
            print('eigs of Acl.T@P+P@Acl',eigVals)
        
        for i in range(sample_data.shape[0]):
            xi = sample_data[i][0]
            # print(xi)
            xi_dot = sample_data[i][1]
            # V = xi.T@P@xi
            V = self.V_value(xi)
            # V_dot = xi.T@P@xi_dot+xi_dot.T@P@xi
            V_dot = self.V_dot_analytical(xi,xi_dot)
            # print(xi_dot)
            if V.item() <= 0:
                print('exists negative samples V(x):{}, at x:{}'.format(V.item(), xi.detach().numpy()))   
            if V_dot.item() >= 0:
                print('exists positive samples V_dot(x):{}, at x:{}'.format(V_dot.item(), xi.detach().numpy()))


    def plot_lyapunov(
            self,
            xlim = 10,
            ylim = 10
    ):
        print('-----------------------Plotting Lyapunov-----------------------')
        x = np.linspace(-xlim, xlim, 100)
        y = np.linspace(-ylim, ylim, 100)
        X, Y = np.meshgrid(x, y)
        grid = np.stack([X, Y], axis=-1)
        P = self.P
        Z = np.einsum('...i,ij,...j->...', grid, P, grid)

        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=100)
        plt.colorbar(label='Lyapunov Function')
        plt.xlabel(r"$\mathregular{x_{1}}$")
        plt.ylabel(r"$\mathregular{x_{2}}$")
        plt.title('Lyapunov Function Contour Plot')
        plt.show()


    def D_value(
        self,
        x: torch.Tensor = torch.zeros(N_DIMS, 1)
        # u: torch.Tensor = torch.zeros(N_CONTROLS, 1)
    ):
        
        return self.V_jacobian(x).t()@self.learned_system(x)
    

    def D_value_batch(
        self,
        x: torch.Tensor = torch.zeros(100, N_DIMS, 1)
        # u: torch.Tensor = torch.zeros(100, N_CONTROLS, 1)
    ):
        """
        V_jacobian_batch [sumple num, 1, n dims]
        learned_system_batch [sumple num, n dims, 1]
        return []
        """
        return torch.matmul(self.V_jacobian_batch(x), self.learned_system_batch(x))


    def verify_D_function(
            self,
            sample_data
    ):
        """
        verifing D function xTPx在数据集上是否非正定
        """
        print('----------------------Verifing D Function----------------------')
        flag = 0
        for i in range(sample_data.shape[0]):
            xi = sample_data[i,0,:,:]
            # xi = sample_data[i][0]
            D = self.D_value(xi)
            if D.item() > 0:
                print('exists positive samples D(x):',D)
                flag = 1
        if flag:
            print('D function is flawed')
        else:
            print('D function is flawless')


    def learn_D(
            self,
            sample_data,
            learning_rate = 0.0001,
            epoch_num = 1500
    ):
        x_train = sample_data[:,0,:,:]
        x_dot_train = sample_data[:,1,:,:]
        y_train = self.V_dot_analytical_batch(x_train,x_dot_train)
        y_pred = self.D_value_batch(x_train)
        criterion = nn.MSELoss()
        # print('y_pred',y_pred)
        # print('y_train',y_train)
        loss = criterion(y_pred,y_train)
        optimizer = torch.optim.SGD([self.Learned_A,self.Learned_B], lr=learning_rate)
        print('Inital Loss of D Function: {}'.format(loss))
        print('---------------------Training D Function-----------------------')
        for epoch in range(epoch_num):
            y_pred = self.D_value_batch(x_train)
            loss = criterion(y_pred,y_train)
            loss.backward(retain_graph=True)
            # with torch.no_grad():  
            #     self.Learned_A -= learning_rate * self.Learned_A.grad.clone()
            #     self.Learned_B -= learning_rate * self.Learned_B.grad.clone()
            #     self.Learned_A.grad.zero_()
            #     self.Learned_B.grad.zero_()
            optimizer.step()
            optimizer.zero_grad()
            if (epoch+1) % 100 == 0:
                print('Epoch [{}/{}], Loss: {}'.format(epoch+1, epoch_num, loss.item()))
        print('Final Loss of D Function: {}'.format(loss))
        self.verify_D_function(sample_data)

                
    def test_D(
            self,
            sample_trajectory_number = 10,
            sample_number_per_trajectory = 200,
            sample_radius = 10,
            sample_number_in_radius = 200
    ):
        print('----------------------Testing D Function-----------------------')
        sample_data_test = self.sample_training_data(sample_trajectory_number,sample_number_per_trajectory,sample_radius,sample_number_in_radius)
        x_test = sample_data_test[:,0,:,:]
        x_dot_test = sample_data_test[:,1,:,:]
        y_test = self.V_dot_analytical_batch(x_test,x_dot_test)
        y_pred = self.D_value_batch(x_test)
        # print('y_pred:',y_pred[799])
        # print('y_train:',y_test[799])
        criterion = nn.MSELoss()
        loss_test = criterion(y_pred,y_test)
        print('loss_test:',loss_test)

    def plot_D(
            self,
            xlim = 1,
            ylim = 1
    ):
        print('----------------------ploting D Function-----------------------')
        x_range = np.linspace(-xlim, xlim, 100)
        y_range = np.linspace(-ylim, ylim, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        for i in range(len(x_range)):
            for j in range(len(y_range)):
                x = torch.tensor([[X[i, j]], [Y[i, j]]], dtype=torch.float32)
                Z[i, j] = self.D_value(x).item()

        plt.figure(figsize=(8, 6))
        # contour = plt.contour(X, Y, Z, levels=20)
        plt.contourf(X, Y, Z, levels=100)
        plt.xlabel(r"$\mathregular{x_{1}}$")
        plt.ylabel(r"$\mathregular{x_{2}}$")
        plt.colorbar(label='D Function Value')

    def plot_lyapunov_dot(
            self,
            xlim = 1,
            ylim = 1
    ):
        print('---------------------Plotting Lyapunov dot---------------------')
        x = np.linspace(-xlim, xlim, 100)
        y = np.linspace(-ylim, ylim, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                xi = torch.tensor([[X[i, j]], [Y[i, j]]], dtype=torch.float32)
                ui = self.system.controller(xi)
                x_doti = self.system.x_dot(xi,ui)
                Z[i, j] = self.V_dot_analytical(xi,x_doti).item()

        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=100)
        plt.colorbar(label='Decreasing Rate of Lyapunov Function')
        plt.xlabel(r"$\mathregular{x_{1}}$")
        plt.ylabel(r"$\mathregular{x_{2}}$")
        # plt.title('Lyapunov Function Contour Plot')
        plt.show()

    def upper_bound_loss(self, output, K0):
        positive_penalty = torch.sum(torch.relu(output + 0.1))
        control_effort_penalty = torch.sum(self.Learned_K**2)
        upper_bound = torch.max(output, dim=0).values
        control_deviation_penalty = torch.sum((self.Learned_K - K0)**2)
        # print('upper_bound:{}, positive_penalty:{}'.format(upper_bound, positive_penalty))
        return upper_bound + positive_penalty + control_effort_penalty*0 + control_deviation_penalty*0
    
    def mean_variance_loss(self,output):
        positive_penalty = torch.sum(torch.relu(output))
        mean = torch.mean(output)
        variance = torch.var(output)
        # print('Mean:{}, Variance:{}'.format(mean, variance))
        return mean + variance*0.1 + positive_penalty*1000

    def upper_bound_mean_variance_loss(self,output):
        positive_penalty = torch.sum(torch.relu(output))
        upper_bound = torch.max(output, dim=0).values
        mean = torch.mean(output)
        variance = torch.var(output)
        return upper_bound*10 + mean + variance + positive_penalty*1000
    

    def improve_controller(
            self,
            sample_data,
            learning_rate = 0.01,
            epoch_num = 1000
    ):
        print('--------------------Improveing Controller----------------------')
        x_train = sample_data[:,0,:,:]
        y_pred = self.D_value_batch(x_train)
        # optimizer = torch.optim.SGD([self.Learned_K], lr=learning_rate)
        optimizer = torch.optim.Adam([self.Learned_K], lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        print('Current D1.Learned_K:', self.Learned_K)
        K0 = self.Learned_K.detach()
        # loss = self.upper_bound_loss(y_pred, K0)
        loss = self.upper_bound_mean_variance_loss(y_pred)
        print('Loss before improving:', loss)
        print('Sum of y_pred before improving:', torch.sum(y_pred))
        for epoch in range(epoch_num):
            y_pred = self.D_value_batch(x_train)
            K0 = self.Learned_K.detach()
            # loss = self.upper_bound_loss(y_pred, K0)
            # loss = self.mean_variance_loss(y_pred)
            loss = self.upper_bound_mean_variance_loss(y_pred)
            loss.backward(retain_graph=True)
            with torch.no_grad(): 
                # self.Learned_K -= learning_rate * self.Learned_K.grad
                # self.Learned_K.grad.zero_()
                optimizer.step()
                optimizer.zero_grad()
            if (epoch+1) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.10f}'.format(epoch+1, epoch_num, loss.item()))
                # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epoch_num, loss.item()))
                # print('D1.Learned_K.grad:',self.Learned_K.grad)
                print('D1.Learned_K:', self.Learned_K)
                print('Sum of y_pred:', torch.sum(y_pred))

    def update_controller(
            self,
    ):
        print('----------------------Update Controller------------------------')
        # self.system.K = torch.tensor(self.Learned_K.detach().numpy(),dtype=torch.float,requires_grad = True)
        self.system.K = torch.tensor(self.Learned_K.detach().numpy(), dtype=torch.float, requires_grad = True)
        
    def clear_grad(
            self
    ):
        # self.Learned_A.zero_grad()
        # self.Learned_B.zero_grad()
        # self.Learned_K.zero_grad()
        # Learned_A = self.Learned_A.detach().numpy()
        # Learned_B = self.Learned_B.detach().numpy()
        # Learned_K = self.Learned_K.detach().numpy()
        # self.Learned_A = torch.tensor(Learned_A, dtype=torch.float, requires_grad=True)
        # self.Learned_B = torch.tensor(Learned_B, dtype=torch.float, requires_grad=True)
        # self.Learned_K = torch.tensor(Learned_K, dtype=torch.float, requires_grad=True)
        # self.system.K = torch.tensor(Learned_K, dtype=torch.float, requires_grad=True)

        self.Learned_A = torch.tensor(np.eye(self.N_DIMS), dtype=torch.float, requires_grad=True)
        self.Learned_B = torch.tensor(np.zeros((self.N_DIMS, self.N_CONTROLS)), dtype=torch.float, requires_grad = True)
        # self.Learned_K = torch.tensor(np.zeros((self.N_CONTROLS, self.N_DIMS)), dtype=torch.float, requires_grad = True)
        self.Learned_K = torch.tensor(self.system.K.detach().numpy(), dtype=torch.float, requires_grad = True)
        # self.system.K = torch.tensor(np.zeros((self.N_CONTROLS, self.N_DIMS)), dtype=torch.float, requires_grad=True)