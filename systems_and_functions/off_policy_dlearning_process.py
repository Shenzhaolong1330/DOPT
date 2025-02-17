import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import collections
import random
import cvxpy as cp
# DONE 导入动力学系统，写一个倒立摆系统
from systems_and_functions.control_affine_system import ControlAffineSystem
from systems_and_functions.networks import PolicyNet, LyapunovNet, DFunctionNet
from systems_and_functions.ddpg_process import DDPGProcess

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
random_seed = 3

class OffPolicyDlearningProcess():
    def __init__(
        self, 
        system: ControlAffineSystem,
        actor_bound: float = 50.0,
        n_hiddens_policy: int = 16,
        n_hiddens_lyapunov: int = 128,
        n_hiddens_dfunction: int = 128,
        gamma: float = 0.5,
        tau: float = 0.1,
        replay_buffer_capacity: int = 4000,
        min_training_batch: int = 2000,
        sample_num: int = 2000,
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        save_path = 'saved_files/Algorithm2_Dlearning_based_DDPG/'
    ):
        self.system = system
        self.gamma = gamma
        self.tau = tau
        self.n_states = self.system.state_dims()
        self.n_actions = self.system.control_dims()
        self.device = device
        self.save_path = save_path
        self.min_training_batch = min_training_batch
        self.buffer = collections.deque(maxlen=replay_buffer_capacity)
        self.sample_num = sample_num

        self.actor = PolicyNet(self.system.state_dims(), n_hiddens_policy, self.system.control_dims(), actor_bound).to(device)
        self.lyapunov = LyapunovNet(self.system.state_dims(), n_hiddens_lyapunov).to(device)
        self.dfunction = DFunctionNet(self.system.state_dims(), n_hiddens_dfunction, self.system.control_dims()).to(device)

        self.target_actor = PolicyNet(self.system.state_dims(), n_hiddens_policy, self.system.control_dims(), actor_bound).to(device)
        self.target_lyapunov = LyapunovNet(self.system.state_dims(), n_hiddens_lyapunov).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_lyapunov.load_state_dict(self.lyapunov.state_dict())

        self.step2norm2ball_record = []
        self.step2unitball_record = []
        self.step2converge_record = []

    def model_save(
            self,
            path = None,
            name = ''
    ):
        if path is None:
            file_path = self.save_path + 'model/'
        else:
            file_path = path + 'model/'
        torch.save(self.actor, file_path+'actor'+name+'_model.pth')
        torch.save(self.lyapunov, file_path+'lyapunov'+name+'_model.pth')
        torch.save(self.dfunction, file_path+'dfunction'+name+'_model.pth')
        torch.save(self.target_actor, file_path+'target_actor'+name+'_model.pth')
        torch.save(self.target_lyapunov, file_path+'target_lyapunov'+name+'_model.pth')

    def buffer_add(
            self, 
            state, 
            action, 
            next_state, 
            weight):

        self.buffer.append((state, action, next_state, weight))


    def buffer_size(self):
        return len(self.buffer)
    

    def buffer_sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, next_state, weight = zip(*transitions)
        return state, action, next_state, weight


    def buffer_add_sampledata(
      self,
      sample_data: torch.Tensor,
      index: int = 0
    ):
        sample_data = sample_data.detach().clone()
        N = sample_data.shape[0]
        state = sample_data[:,0].permute(0, 2, 1)
        next_state = sample_data[:,1].permute(0, 2, 1)
        action = self.target_actor.Controller(state.squeeze(1).t()).t()
        weight = self.gamma**index * torch.ones(N)
        for i in range(N):
            self.buffer_add(state[i], action[i], next_state[i], weight[i])


    def direct_update(
            self,
            net: nn.Module,
            target_net: nn.Module,
        ):
            print('-------------------------Direct Update--------------------------')
            
            for param_target, param in zip(target_net.parameters(), net.parameters()):
                param_target.data.copy_(param.data)


    def soft_update(
            self,
            net: nn.Module,
            target_net: nn.Module,
        ):
            print('--------------------------Soft Update---------------------------')
            
            for param_target, param in zip(target_net.parameters(), net.parameters()):
                param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)


    def initialize_policy_net(
            self,
            x_train_lim: int = 10,
            x_test_lim: int = 13,
            sample_num: int = 1000,
            iteration: int = 2*10**4,
            lr: float = 1e-4,
    ):
        print('---------------------Initializing Policy------------------------')

        random_data = [-x_train_lim + torch.rand(sample_num)*2*x_train_lim for _ in range(self.system.state_dims())]
        train_data = torch.stack(random_data, dim=1).to(self.device)

        random_data1 = [-1 + torch.rand(sample_num)*2*1 for _ in range(self.system.state_dims())]
        train_data1 = torch.stack(random_data1, dim=1).to(self.device)

        K = torch.tensor(-self.system.K)
        zero_column = torch.zeros(1, self.n_states)
        K_extended = torch.vstack((K, zero_column)).t().to(self.device)
        K_extended = K_extended.float()
        labels = train_data @ K_extended
        labels1 = train_data1 @ K_extended
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.actor.parameters(), lr = lr)
        for i in range(iteration):
            optimizer.zero_grad()
            # y_train = self.actor(train_data.t())
            # y_train1 = self.actor(train_data1.t())
            y_train = self.actor(train_data)
            y_train1 = self.actor(train_data1)
            loss = loss_fn(y_train, labels) + loss_fn(y_train1, labels1)*10
            # if i % 10000 == 0:
            #     print(f'times {i} - lr {lr} -  loss: {loss.item()}')
            if (i + 1) % 1000 == 0:
                print ('Epoch [{}/{}], Loss: {:.10f}'.format(i + 1, iteration, loss.item()))
            loss.backward()
            optimizer.step()
        self.direct_update(self.actor, self.target_actor)
        # x_ = torch.linspace(-x_test_lim, x_test_lim, sample_num, dtype=torch.float)
        # test_data = torch.stack([x_] * self.system.state_dims(), dim=1).to(self.device)
        # labels = test_data @ K.t().to(self.device)
        # y_test = self.actor.Controller(test_data.t())
        # plt.plot(x.cpu().detach().numpy(), labels.cpu().detach().numpy(), c='red', label='True')
        # plt.plot(x.cpu().detach().numpy(), y_test.t().cpu().detach().numpy(), c='blue', label='Pred')
        # plt.legend(loc='best')
        # plt.show()


    def sample_training_data(
        self,
        sample_radius_trajectory: int = 7,
        sample_trajectory_number: int = 10,
        sample_number_per_trajectory: int = 500,
        sample_radius_random: int = 15,
        sample_number_in_radius: int = 0,
        invariant_sample: bool = True,
        sample_plot: bool = True,
        the_controller = None, 
        title = "Samples"
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
        # dim2: x  x'
        # dim3: x1 x2
        # dim4: value
        sample_from_trajectory = torch.zeros(sample_trajectory_number*sample_number_per_trajectory,2,self.n_states,1).to(self.device)
        sample_from_radius = torch.zeros(sample_number_in_radius,2,self.n_states,1).to(self.device)
        theta = np.linspace(0, 2*np.pi, sample_trajectory_number + 1)
        
        # sample in trajectory
        for i in range(0, sample_trajectory_number):
            x_0_traj = torch.tensor([[sample_radius_trajectory*np.cos(theta[i])],[sample_radius_trajectory*np.sin(theta[i])]],dtype=torch.float).to(self.device)
            simulate_rk4 = self.system.simulate_rk4(x_0_traj,sample_number_per_trajectory,1,the_controller)
            x = simulate_rk4[:,0].unsqueeze(1)
            x_dot = simulate_rk4[:,1].unsqueeze(1)
            sample_from_trajectory[i*sample_number_per_trajectory:(i+1)*sample_number_per_trajectory] = torch.cat((x, x + x_dot*self.system.dt), dim=1)

        # sample randomly in radius
        if invariant_sample == True:
            np.random.seed(42)
        theta_ = np.random.uniform(0, 2*np.pi, sample_number_in_radius)
        r_ = np.sqrt(np.random.uniform(0, sample_radius_random**2, sample_number_in_radius))
        combined_data = zip(theta_, r_)
        i = 0
        for data in combined_data:
            theta__, r__ = data
            x_0_radius = torch.tensor([[r__ * np.cos(theta__)],[r__ * np.sin(theta__)]]).to(self.device)
            one_step_euler = self.system.one_step_euler(x_0_radius,1,the_controller)[1].to(self.device)
            x = one_step_euler[0].unsqueeze(0)
            x_dot = one_step_euler[1].unsqueeze(0)
            sample_from_radius[i] = torch.cat((x, x + x_dot*self.system.dt), dim=0)
            i = i + 1

        if sample_plot == True:
            x1_trajectory = sample_from_trajectory[:,0,0,:].cpu().detach().numpy()
            x2_trajectory = sample_from_trajectory[:,0,1,:].cpu().detach().numpy()
            fig, ax = plt.subplots()
            ax.scatter(x1_trajectory, x2_trajectory, s=10, alpha=0.5, c='c', label='sample_from_trajectory')

            x1_radius = sample_from_radius[:,0,0,:].cpu().detach().numpy()
            x2_radius = sample_from_radius[:,0,1,:].cpu().detach().numpy()
            ax.scatter(x1_radius, x2_radius, s=10, alpha=0.5, c='g', label='sample_from_radius')
            ax = plt.gca()
            ax.set_aspect(1)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
            # fig.show()
            plt.show()
            plt.clf()
            plt.close()

        sample_data = torch.cat((sample_from_trajectory,sample_from_radius),dim=0).to(self.device)
        return sample_data


    def learn_V_LQF(
        self,
        sample_data,
        plot_lyapuonv: bool = True
    ):
        """
        learn a V(x) by constraining V(x)>=0, V_dot(x)<=0
        """
        print('--------------------------Learning V--------------------------')
        sample_data = sample_data.cpu().detach().numpy()
        P = cp.Variable((self.system.state_dims(), self.system.state_dims()))
        eta = cp.Variable()
        
        constraints = [P.T == P, eta>=0]
        for i in range(sample_data.shape[0]):
            xi = sample_data[i][0]
            xi_dot = (sample_data[i][1]-sample_data[i][0])/self.system.dt
            constraints.append(xi.T@P@xi>=0)
            constraints.append(xi.T@P@xi_dot+xi_dot.T@P@xi<=-eta*np.linalg.norm(xi)**2)

        objective = cp.Minimize(-eta)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        print("status:",problem.status)
        print("var eta (eta should be positive):", eta.value)
        print("var P:", P.value)
        self.P = torch.tensor(P.value).float()

        if plot_lyapuonv:
            xlim = 5
            ylim = 5
            print('-----------------------Plotting Lyapunov-----------------------')
            x = np.linspace(-xlim, xlim, 100)
            y = np.linspace(-ylim, ylim, 100)
            X, Y = np.meshgrid(x, y)
            grid = np.stack([X, Y], axis=-1)
            P = self.P
            Z = np.einsum('...i,ij,...j->...', grid, P, grid)

            fig = plt.figure()
            ax = fig.add_subplot(figsize=(8, 6))
            contour = ax.contourf(X, Y, Z, levels=np.arange(0,5.0,0.1), cmap='RdBu', alpha = 1)
            cbar = plt.colorbar(contour)
            cbar.set_label('Lyapunov Function Value')

            ax.set_xlabel(r"$\mathregular{x_{1}}$")
            ax.set_xlabel(r"$\mathregular{x_{2}}$")
            ax.set_title('Lyapunov Function Contour Plot')
            ax.grid(True)
            fig.show()


    def V_value_P_batch(self, x: torch.Tensor):
        """
        x:[sample num, value, n dims]
        return [sumple num, value]
        """
        P = self.P.to(self.device)
        xP = torch.matmul(x, P)
        quadratic_form = torch.sum(xP * x, dim=1, keepdim=True)

        return quadratic_form


    def initialize_lyapunov_net(
            self,
            x_train_lim: int = 15,
            x_test_lim: int = 13,
            sample_num: int = 1000,
            iteration: int = 5*10**4,
            lr: float = 1e-3,
            plot_loss: bool = True,
            plot_lyapuonv: bool = True
    ):
        print('--------------------Initializing Lyapunov---------------------')
        random_data = [-x_train_lim + torch.rand(sample_num) * 2 * x_train_lim for _ in range(self.system.state_dims())]
        training_data = torch.stack(random_data, dim=1).to(self.device)
        V_value_P = self.V_value_P_batch(training_data)
        labels = torch.cat([V_value_P, torch.zeros(V_value_P.shape[0],1).to(self.device)], dim=1).to(self.device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.lyapunov.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        loss_values = []
        for i in range(iteration):
            optimizer.zero_grad()
            y_train = self.lyapunov(training_data)
            param_squares = [p ** 2 for p in self.lyapunov.parameters()]
            param_sum_square = sum(torch.sum(p) for p in param_squares)
            loss = loss_fn(y_train, labels) + 0.01 * param_sum_square
            loss_values.append(loss.item())
            if (i + 1) % 1000 == 0:
                print ('Epoch [{}/{}], Loss: {:.10f}'.format(i + 1, iteration, loss.item()))
            loss.backward()
            optimizer.step()
        
        if plot_loss:
            plt.figure(figsize=(10, 5))
            plt.plot(loss_values, label='Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Initializing V Loss')
            plt.legend() 
            # plt.show()
            plt.show()
            plt.clf()
            plt.close()

        if plot_lyapuonv:
            self.plot_contour(5, 5, 'Lyapunov')


    def learn_V_LNN(
        self,
        sample_data,
        iteration: int = 3*10**3,
        plot_loss: bool = True,
        plot_lyapuonv: bool = True,
        plot_rollout: bool = True,
        lr: float = 1e-4,
        save_fig = False,
        index = 0
    ):
        dt = self.system.dt
        print('--------------------------Learning V----------------------------')
        # sample_data = sample_data.detach().clone()
        # DONE: using sample_data = sample_data.detach().clone()
        # RuntimeError: Trying to backward through the graph a second time 
        # (or directly access saved tensors after they have already been freed).
        # Saved intermediate values of the graph are freed when you call 
        # .backward() or autograd.grad(). Specify retain_graph=True 
        # if you need to backward through the graph a second time or 
        # if you need to access saved tensors after calling backward
        # sample initialize ([100, 2, 2, 1])
        # dim1: sample index
        # dim2: x  x'
        # dim3: x1 x2
        # dim4: value
        state, _, next_state, weight = self.buffer_sample(self.buffer_size())
        states = torch.stack(state).to(self.device)
        next_states = torch.stack(next_state).to(self.device)
        weights = torch.stack(weight).to(self.device)
        N = states.shape[0]
        s = states # torch.Size([2000, 1, 2])
        s_ = next_states # torch.Size([2000, 1, 2])
        s0 = torch.zeros_like(s[0]).to(device) # torch.Size([1, 2])
        # print(s.shape)
        # print(s_.shape)
        # print(s0,s0.shape)
        # s0 = torch.tensor([[0.],[0.]]).t().to(device)
        optimizer = torch.optim.Adam(self.lyapunov.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        # self.lyapunov_optimizer
        loss_values = []

        for i in range(iteration):
            # loss = self.lyapunov(s0)**2 + (sum(F.relu(-self.lyapunov(s.permute(0,2,1)))) + sum(F.relu((self.lyapunov(s_.permute(0,2,1))-self.lyapunov(s.permute(0,2,1)))/dt)))/N
            L0 = self.lyapunov.V(s0)
            Ls = self.lyapunov.V(s.squeeze(1))
            Ls_ = self.lyapunov.V(s_.squeeze(1))
            dL = (Ls_ - Ls) / dt
            # print('self.lyapunov.V(s)',Ls.shape)
            # print('F.relu(dL+0.5)',F.relu(dL+0.5).shape)
            # print(weights.shape)
            SND = torch.sum(F.relu(dL+0.5) * weights)
            PD = torch.sum(F.relu(-Ls) * weights)

            param_squares = [p ** 2 for p in self.lyapunov.parameters()]
            param_sum_square = sum(torch.sum(p) for p in param_squares)
            loss = L0**2 + (PD*10 + SND) + 0.01 * param_sum_square

            # loss = L0**2 + (PD + SND) + 0.01 * torch.sum(p ** 2 for p in self.lyapunov.parameters())
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                optimizer.step()
                optimizer.zero_grad()
            loss_values.append(loss.item())
            if (i + 1) % 1000 == 0:
                print ('Epoch [{}/{}], Loss: {:.10f}'.format(i + 1, iteration, loss.item()))

        if plot_loss:
            fig, ax =plt.subplots(figsize=(10, 5))
            ax.plot(loss_values, label='Loss')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.set_title('Training V Loss')
            ax.legend() 
            plt.show()
            plt.clf()
            plt.close()
        
        if plot_lyapuonv:
            self.plot_contour(xlim = 5,
                              ylim = 10,
                              select = 'Lyapunov',
                              plot_rollout = plot_rollout,
                              save_fig = save_fig,
                              index = index)
        print('L0:{}, \nPD:{}, \ntorch.sum(F.relu(dL)):{}'.format(L0, PD, torch.sum(F.relu(dL)) ))

        if index == 1:
            self.direct_update(net = self.lyapunov, 
                               target_net = self.target_lyapunov)
        else:
            self.soft_update(net = self.lyapunov, 
                             target_net = self.target_lyapunov)


    def plot_contour(
        self,
        xlim = 12,
        ylim = 15,
        select = 'Lyapunov',
        plot_rollout = True,
        save_fig = False,
        index = 0
    ):
        print('-----------------------Plotting {}------------------------'.format(select))
        x = np.linspace(-xlim, xlim, 100)
        y = np.linspace(-ylim, ylim, 120)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        others = torch.rand(1,self.n_states-2).to(device)*0.5
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                xy = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32).to(device)
                if self.n_states > 2:
                    # others = torch.rand(1,self.n_states-2).to(device)*0.1
                    xy = torch.cat((xy, others), dim=1).to(device)
                if select == 'Dfunction':
                    Z[i, j] = self.dfunction.D(xy,self.actor.Controller(xy).t()).item()
                else:
                    Z[i, j] = self.lyapunov.V(xy).item()
        plt.figure(figsize=(8, 8))
        fig = plt.figure(clear=True)
        ax = fig.add_subplot()
        if select == 'Dfunction':
            cmap_forward = cm.get_cmap('Blues') 
            cmap_reverse = cmap_forward.reversed()
            contour = ax.contourf(X, Y, Z, cmap=cmap_reverse, alpha = 1)
            # contour = ax.contourf(X, Y, Z, levels=np.arange(-50.8,0.8,0.8), cmap=cmap_reverse, alpha = 1)
            # contour = ax.contourf(X, Y, Z, cmap='RdBu', alpha = 1)
        else:
            # contour = ax.contourf(X, Y, Z, levels=np.arange(-0.2,9.0,0.1), cmpa='RdBu', alpha = 1)
            contour = ax.contourf(X, Y, Z, levels=25, cmpa='RdBu', alpha = 1)
            # contour = ax.contourf(X, Y, Z, levels=np.arange(-0.4,9.0,0.1), cmap='jet', alpha = 1)
        cbar = plt.colorbar(contour)
        cbar.set_label('Function Value')
        if plot_rollout:
            data = self.sample_training_data(sample_radius_trajectory = 3,
                        sample_trajectory_number = 12,
                        sample_number_per_trajectory = 400,
                        sample_radius_random = 0,
                        sample_number_in_radius = 0,
                        invariant_sample = True,
                        sample_plot = False,
                        the_controller = self.actor.Controller,
                        title = None).detach().clone().cpu()
            x1 = data[:,0,0]
            x2 = data[:,0,1]
            plt.scatter(x1, x2, s = 2, c = "purple")
        ax.set_xlim([-xlim, xlim])
        ax.set_ylim([-ylim, ylim])
        ax.set_aspect(0.5)
        ax.set_xlabel(r"$\mathregular{\theta}(rad)$")
        ax.set_ylabel(r"$\mathregular{\dot{\theta}(rad/s)}$")

        # ax.set_aspect(1)
        # ax.set_xlabel(r"$x_{err}(m)$", fontsize=15)
        # ax.set_ylabel(r"$y_{err}(m)$", fontsize=15)
        # ax.set_title(select +' Function Contour Plot' + ' in Iteration {}'.format(index))
        # ax.set_title(select + ' in Iteration {}'.format(index))
        ax.grid(True)
        if save_fig:
            fig.savefig(self.save_path +'figs/'+ select +'Function Contour' + ' in Iteration {}'.format(index)+'.png', dpi=1200, bbox_inches='tight', pad_inches=0.1)
        # fig.show()
        plt.show()
        plt.clf()
        plt.close()


    def learn_D_DNN(
        self,
        sample_data,
        iteration: int = 10**4,
        plot_loss: bool = True,
        plot_rollout: bool = True,
        plot_dfunction: bool = True,
        lr: float = 1e-4,
        save_fig: bool = False,
        index = 0
    ):
        dt = self.system.dt
        print('--------------------------Learning D----------------------------')

        state, _, next_state, weight = self.buffer_sample(self.buffer_size())
        states = torch.stack(state).to(self.device)
        next_states = torch.stack(next_state).to(self.device)
        weights = torch.stack(weight).to(self.device)
        N = states.shape[0]
        s = states # torch.Size([2000, 1, 2])
        s_ = next_states # torch.Size([2000, 1, 2])
        s0 = torch.zeros_like(s[0]).to(device) # torch.Size([1, 2])
        # sample_data = sample_data.detach().clone()
        # N = sample_data.shape[0]
        # s = sample_data[:,0].permute(0, 2, 1)
        # s_ = sample_data[:,1].permute(0, 2, 1)
        # s0 = torch.tensor([[0.],[0.]]).to(device)
        # s0 = torch.zeros_like(s[0]).to(device)
        a0 = self.actor.Controller(s0)
        Ls = self.lyapunov(s)
        Ls_ = self.lyapunov(s_)
        dL = ((Ls_ - Ls) / dt).squeeze(1)
        # print(dL.shape)
        a = self.actor.Controller(s.squeeze(1)).t()
        # optimizer = self.dfunction_optimizer
        optimizer = torch.optim.Adam(self.dfunction.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        loss_values = []
        loss_fn = nn.MSELoss()
        for i in range(iteration):
            D0 = self.dfunction.D(s0, a0.t())
            # D0 = self.dfunction.D(s0.t(), a0)
            DV_ext = self.dfunction(s.squeeze(1),a)
            DV = self.dfunction.D(s.squeeze(1),a)
            param_squares = [p ** 2 for p in self.dfunction.parameters()]
            param_sum_square = sum(torch.sum(p) for p in param_squares)
            loss = torch.sum(loss_fn(dL, DV_ext)) + torch.sum(F.relu(DV)) + D0**2 + 0.01 * param_sum_square
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                optimizer.step()
                optimizer.zero_grad()
            loss_values.append(loss.item())
            if (i + 1) % 1000 == 0:
                print ('Epoch [{}/{}], Loss: {:.10f}'.format(i + 1, iteration, loss.item()))
        
        if plot_loss:
            plt.figure(figsize=(10, 5))
            plt.plot(loss_values, label='Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training D Loss')
            plt.legend()
            plt.show()
            plt.clf()
            plt.close()
        
        if plot_dfunction:
            # self.plot_contour(select='Dfunction')
            self.plot_contour(xlim = 3,
                              ylim = 3,
                              select = 'Dfunction',
                              plot_rollout = plot_rollout,
                              save_fig = save_fig,
                              index = index)
        print('torch.sum(loss_fn(dL, DV_ext)):{}, \ntorch.sum(F.relu(DV)):{}, \nD0:{}'.format(torch.sum(loss_fn(dL, DV_ext)),torch.sum(F.relu(DV)),D0))


    def mean_variance_loss(self,output):
        positive_penalty = torch.sum(torch.relu(output))
        mean = torch.mean(output)
        variance = torch.var(output)
        # print('Mean:{}, Variance:{}'.format(mean, variance))
        return mean + variance*0.1 + positive_penalty*1000


    def upper_bound_mean_variance_loss(self,output):
        """
        """
        positive_penalty = torch.sum(torch.relu(output))
        upper_bound = torch.max(output, dim=0).values
        lower_bound = torch.min(output, dim=0).values
        mean = torch.mean(output)
        variance = torch.var(output)
        # return upper_bound*100 + lower_bound*0 + mean*30 + variance*0 + positive_penalty*10
        return upper_bound*70 + lower_bound*0 + mean*100 + variance*0 + positive_penalty*1000

    def policy_improvement(
        self,
        sample_data,
        iteration: int = 10**3,
        plot_loss: bool = True,
        lr: float = 1e-4,
        index: int = 1
    ):
        print('------------------------Improveing Policy-----------------------')
        sample_data = sample_data.detach().clone()
        # state, _, _, _ = self.buffer_sample(self.buffer_size())
        # states = torch.stack(state).to(self.device)
        # next_states = torch.stack(next_state).to(self.device)
        # weights = torch.stack(weight).to(self.device)
        # N = states.shape[0]
        # s = states.squeeze(1) # torch.Size([2000, 1, 2])
        N = sample_data.shape[0]
        s = sample_data[:,0].permute(0, 2, 1).squeeze(1)
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        
        loss_values = []
        for i in range(iteration):
            a = self.actor.Controller(s).t()
            DV = self.dfunction.D(s.squeeze(1),a)
            param_squares = [p ** 2 for p in self.actor.parameters()]
            param_sum_square = sum(torch.sum(p) for p in param_squares)
            loss = self.upper_bound_mean_variance_loss(DV) + 0 * param_sum_square
            loss.backward(retain_graph = True)
            with torch.no_grad(): 
                optimizer.step()
                optimizer.zero_grad()
            loss_values.append(loss.item())
            if (i + 1) % 1000 == 0:
                print ('Epoch [{}/{}], Loss: {:.10f}'.format(i + 1, iteration, loss.item()))

        if plot_loss:
            plt.figure(figsize=(10, 5))
            plt.plot(loss_values, label='Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Policy Improvement Loss')
            plt.legend()
            plt.show()
            plt.clf()
            plt.close()
        
        if index == 0:
            self.direct_update(net = self.actor, 
                               target_net = self.target_actor)
        else:
            self.soft_update(net = self.actor, 
            # self.direct_update(net = self.actor, 
                             target_net = self.target_actor)


    def plot_save_converge_steps(
        self,
        path = 'saved_files/Algorithm2_Dlearning_based_DDPG/'
        ):  
            print('----------------------------------Save Data--------------------------------')
            index = np.arange(0,len(self.step2converge_record))

            fig = plt.figure(figsize=(12.5, 6))
            ax = fig.add_subplot()

            ax.plot(index, self.step2converge_record, label = 'steps to convergence', color='c', marker = 'o')
            ax.plot(index, self.step2unitball_record, label = 'steps to unit ball', color='cornflowerblue', marker = 'o')
            ax.plot(index, self.step2norm2ball_record, label = 'steps to norm 2 ball', color='plum', marker = 'o')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.set_xlim([0, len(self.step2converge_record)-1])
            ax.legend()

            ax.set_title("Convergence Speed and Judgment")
            ax.set_xlabel("PI iteration")
            ax.set_ylabel("Steps / 0.01s")
            fig.savefig(self.save_path + 'figs/Convergence Speed and Judgment.png', dpi=600)
            # fig.show()
            plt.show()
            plt.clf()
            plt.close()

            self.model_save()

            print(self.step2converge_record)
            print(self.step2unitball_record)
            print(self.step2norm2ball_record)

            filename = self.save_path + 'data/converge_record_data.txt'

            with open(filename, 'w') as file:
                file.write(f"Step to Converge Record: {self.step2converge_record}\n")
                file.write(f"Step to Unit Ball Record: {self.step2unitball_record}\n")
                file.write(f"Step to Norm 2 Ball Record: {self.step2norm2ball_record}\n")


    def off_policy_dlearning_main_iteration(
        self,
        iteration: int = 20,
        plot_x_initial: torch.tensor = torch.tensor([[2],[-2]]),
        plot_step_num: int = 500,
        ):
            print('-------------------------------Main Iteration------------------------------')
            x_initial = plot_x_initial.to(self.device)
            self.initialize_policy_net(x_train_lim = 10,
                                       x_test_lim = 12,
                                       sample_num = 2000,
                                       iteration = 1*10**3,
                                       lr = 1e-3)
            
            sim_data_ = self.system.simulate_rk4(x_initial = x_initial, 
                                    step_number = plot_step_num,
                                    use_controller = 1,
                                    the_controller = self.target_actor.Controller)

            step2norm2ball,step2unitball,step2converge = self.system.convergence_judgment(sim_data_)

            self.system.plot_phase_portrait(data_sim = sim_data_,
                                     arrow_on = 0,
                                    #  title = 'Phase Portrait with initialized NN controller',
                                     title = 'before_PI',
                                     save_fig = 1,
                                     save_path = self.save_path + 'figs/')
        
            self.step2norm2ball_record.append(step2norm2ball)
            self.step2unitball_record.append(step2unitball)
            self.step2converge_record.append(step2converge)

            for i in range(1, iteration+1):
                print('---------------------------------Iteration {}-------------------------------'.format(i))
                sim_data = self.sample_training_data(sample_radius_trajectory = 6,
                                                sample_trajectory_number = 0,
                                                sample_number_per_trajectory = 0,
                                                sample_radius_random = 6,
                                                sample_number_in_radius = self.sample_num,
                                                invariant_sample = 1,
                                                sample_plot = 0,
                                                the_controller = self.target_actor.Controller,
                                                title = 'After policy initialization but before LQR')
                
                self.buffer_add_sampledata(sim_data, i)
                
                self.learn_V_LNN(sample_data = None,
                            iteration = 1*10**3,
                            plot_loss = 0,
                            plot_lyapuonv = 0,
                            lr = 2 * 1e-3,
                            save_fig = 0,
                            index = i)

                self.learn_D_DNN(sample_data = None,
                            iteration = 1*10**3,
                            plot_loss = 0,
                            plot_dfunction = 0,
                            lr = 2*1e-4,
                            save_fig = 0,
                            index = i)
                
                if i == 1:
                    self.model_save(name = '_initial')

                self.policy_improvement(sample_data = sim_data,
                                        iteration = 450,
                                        plot_loss = 0,
                                        lr = 2*1e-4,
                                        index = i)
                
                sim_data_policy_improvement = self.sample_training_data(sample_radius_trajectory = 3,
                                                sample_trajectory_number = 0,
                                                sample_number_per_trajectory = 0,
                                                sample_radius_random = 0.4,
                                                sample_number_in_radius = 500,
                                                invariant_sample = 1,
                                                sample_plot = 0,
                                                the_controller = self.target_actor.Controller,
                                                title = 'PI training data')

                self.policy_improvement(sample_data = sim_data_policy_improvement,
                                        iteration = 50,
                                        plot_loss = 0,
                                        lr = 2*1e-4,
                                        index = i)
                
                sim_data_ = self.system.simulate_rk4(x_initial = x_initial, 
                                            step_number = plot_step_num,
                                            use_controller = 1,
                                            the_controller = self.target_actor.Controller)
                
                step2norm2ball,step2unitball,step2converge = self.system.convergence_judgment(sim_data_)

                self.step2norm2ball_record.append(step2norm2ball)
                self.step2unitball_record.append(step2unitball)
                self.step2converge_record.append(step2converge)

                self.system.plot_phase_portrait(data_sim = sim_data_,
                                        arrow_on = 0,
                                        title = 'after PI in iteration {}'.format(i),
                                        save_fig = 1,
                                        save_path = self.save_path + 'figs/')
                # filename = 'saved_files/Algorithm2_Dlearning_based_DDPG/data/converge_record_data.txt'
                filename = self.save_path + 'data/converge_record_data.txt'
                with open(filename, 'w') as file:
                    file.write(f"Step to Converge Record: {self.step2converge_record}\n")
                    file.write(f"Step to Unit Ball Record: {self.step2unitball_record}\n")
                    file.write(f"Step to Norm 2 Ball Record: {self.step2norm2ball_record}\n")
                    
                self.model_save()
                self.plot_save_converge_steps()
            plt.clf()
            plt.close()
            
    def off_policy_dlearning_main_iteration_4_stcar(
        self,
        iteration: int = 20,
        delta_x: torch.tensor = torch.tensor([[-1, -1]], dtype=torch.float32),
        plot_time_span: float = 5.0,
        ):
            print('-------------------------------Main Iteration------------------------------')
            torch.manual_seed(random_seed)
            others = torch.rand(1,self.n_states-2).to(self.device)*0.5
            delta_x = delta_x.to(self.device)
            delta_x = torch.cat((delta_x, others), dim=1).to(self.device)
            
            _, _, traj_sim = self.system.run_rk4(t_sim = plot_time_span,
                                                the_controller = None,
                                                delta_x = delta_x,
                                                sample_mode = False)

            step2norm2ball,step2unitball,step2converge = self.system.convergence_judgment(traj_sim)

            self.initialize_policy_net(x_train_lim = 4,
                                       x_test_lim = 7,
                                       sample_num = 2000,
                                       iteration = 10*10**3,
                                       lr = 1e-3)
            
            _, _, traj_sim = self.system.run_rk4(t_sim = plot_time_span,
                                                the_controller = self.target_actor.Controller,
                                                delta_x = delta_x,
                                                sample_mode = False)

            step2norm2ball,step2unitball,step2converge = self.system.convergence_judgment(traj_sim)

            self.system.plot_phase_portrait(traj_sim = traj_sim,
                                            title = 'Phase Portrait with initialized NN controller',
                                            save_fig = 1,
                                            save_path = self.save_path+'figs/')
            
            self.system.plot_traj(traj_sim = traj_sim,
                                  title = 'Tracking',
                                  save_fig = 1,
                                  save_path = self.save_path+'figs/')
        
            self.step2norm2ball_record.append(step2norm2ball)
            self.step2unitball_record.append(step2unitball)
            self.step2converge_record.append(step2converge)

            for i in range(1, iteration+1):
                print('---------------------------------Iteration {}-------------------------------'.format(i))
                sample_state_space, _ = self.system.sample_training_data_stcar(sample_state = {'sxe_range':(-4, 4), 'sye_range': (-4, 4), 'delta_range': (-1, 1), 've_range': (-0.5, 0.5), 'psi_e_range': (-0.5, 0.5), 'psi_e_dot_range': (-0.5, 0.5), 'beta_range': ((-0.5, 0.5))},
                                                                            #    sample_state = {'sxe_range':(-3, 3), 'sye_range': (-3, 3), 'delta_range': (-1.5, 1.5), 've_range': (-0.5, 0.5), 'psi_e_range': (-1, 1), 'psi_e_dot_range': (-1, 1), 'beta_range': ((-1.5, 1.5))},
                                                                                sample_num = self.sample_num,
                                                                                invariant_sample = False,
                                                                                sample_plot = False,
                                                                                the_controller = self.target_actor.Controller,
                                                                                title = "Samples")
                self.buffer_add_sampledata(sample_state_space, i)

                self.learn_V_LNN(sample_data = sample_state_space,
                                iteration = 4*10**3,
                                plot_loss = 0,
                                plot_lyapuonv = 0,
                                plot_rollout = 0,
                                lr = 2*1e-4,
                                save_fig = 0,
                                index = 0)
                
                self.plot_contour(xlim = 5,
                                ylim = 5,
                                select = 'Lyapunov',
                                plot_rollout = False,
                                save_fig = 1,
                                index = i)

                self.learn_D_DNN(sample_data = sample_state_space,
                                iteration = 5*10**3,
                                plot_loss = 0,
                                plot_rollout = 0,
                                plot_dfunction = 1,
                                lr = 2*1e-4,
                                save_fig = 1,
                                index = 0)
                
                if i == 1:
                    self.model_save(name = '_initial')

                sample_state_space_mini, _ = self.system.sample_training_data_stcar(sample_state = {'sxe_range':(-1, 1), 'sye_range': (-1, 1), 'delta_range': (-0.3, 0.3), 've_range': (-0.3, 0.3), 'psi_e_range': (-0.3, 0.3), 'psi_e_dot_range': (-0.3, 0.3), 'beta_range': (-0.3, 0.3)},
                                                                            #    sample_state = {'sxe_range':(-3, 3), 'sye_range': (-3, 3), 'delta_range': (-1.5, 1.5), 've_range': (-0.5, 0.5), 'psi_e_range': (-1, 1), 'psi_e_dot_range': (-1, 1), 'beta_range': (-1.5, 1.5)},
                                                                                sample_num = 500,
                                                                                invariant_sample = False,
                                                                                sample_plot = False,
                                                                                the_controller = self.target_actor.Controller,
                                                                                title = "Samples")

                self.policy_improvement(sample_data = sample_state_space_mini,
                                        iteration = 4 * 10**2,
                                        plot_loss = 0,
                                        lr = 2 * 1e-4)

                _, _, traj_sim = self.system.run_rk4(t_sim = plot_time_span,
                                                    the_controller = self.target_actor.Controller,
                                                    delta_x = delta_x,
                                                    sample_mode = False)
                
                step2norm2ball,step2unitball,step2converge = self.system.convergence_judgment(traj_sim)

                self.step2norm2ball_record.append(step2norm2ball)
                self.step2unitball_record.append(step2unitball)
                self.step2converge_record.append(step2converge)

                self.system.plot_phase_portrait(traj_sim = traj_sim,
                                                title = 'after PI in iteration {}'.format(i),
                                                save_fig = 1,
                                                save_path = self.save_path+'figs/')
                
                self.system.plot_traj(traj_sim = traj_sim,
                                    title = 'Tracking in iteration {}'.format(i),
                                    save_fig = 1,
                                    save_path = self.save_path+'figs/')
                
                filename = self.save_path + 'data/converge_record_data.txt'
                with open(filename, 'w') as file:
                    file.write(f"Step to Converge Record: {self.step2converge_record}\n")
                    file.write(f"Step to Unit Ball Record: {self.step2unitball_record}\n")
                    file.write(f"Step to Norm 2 Ball Record: {self.step2norm2ball_record}\n")
                    
                self.model_save()
                self.plot_save_converge_steps()
            plt.clf()
            plt.close()
            