import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import collections
import random

from systems_and_functions.control_affine_system import ControlAffineSystem
from systems_and_functions.networks import PolicyNet, QValueNet
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class DDPGProcess:
    def __init__(
        self, 
        system: ControlAffineSystem,
        actor_bound: float = 100.0,
        n_hiddens_policy: int = 16,
        n_hiddens_critic: int = 128,
        sigma: float = 0.5,
        tau: float = 0.2,
        gamma: float = 0.9,
        replay_buffer_capacity: int = 4000,
        min_training_batch: int = 2000,
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        save_path = 'saved_files/Algorithm3_DDPG/'
    ):
        # 属性分配
        self.system = system # 动力学系统
        self.sigma = sigma # 噪声标准差
        self.tau = tau # 软更新权重
        self.gamma = gamma
        self.n_states = self.system.state_dims()
        self.n_actions = self.system.control_dims()
        self.device = device
        self.save_path = save_path
        self.min_training_batch = min_training_batch
        self.buffer = collections.deque(maxlen=replay_buffer_capacity)
        # 训练网络
        self.actor = PolicyNet(self.system.state_dims(), n_hiddens_policy, self.system.control_dims(), actor_bound).to(device)
        self.critic = QValueNet(self.system.state_dims(), n_hiddens_critic, self.system.control_dims()).to(device)

        # 目标网络
        self.target_actor = PolicyNet(self.system.state_dims(), n_hiddens_policy, self.system.control_dims(), actor_bound).to(device)
        self.target_critic = QValueNet(self.system.state_dims(), n_hiddens_critic, self.system.control_dims()).to(device)

        # 使训练网络与网络的初始参数相同
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.step2norm2ball_record = []
        self.step2unitball_record = []
        self.step2converge_record = []
        self.test_accumulated_rewards = []

    def model_save(
            self,
            path = None
    ):
        if path is None:
            file_path = self.save_path + 'model/'
        else:
            file_path = path
        torch.save(self.actor, file_path+'actor_model.pth')
        torch.save(self.critic, file_path+'critic_model.pth')
        torch.save(self.target_actor, file_path+'target_actor_model.pth')
        torch.save(self.target_critic, file_path+'target_critic_model.pth')

    
    def plot_save_converge_steps(
        self,
        path = 'saved_files/Algorithm3_DDPG/'
        ):  
            print('----------------------------------Save Data--------------------------------')
            index = np.arange(0,len(self.step2converge_record))

            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot()

            ax1.plot(index, self.step2converge_record, label = 'steps to convergence', color='c', marker = 'o')
            ax1.plot(index, self.step2unitball_record, label = 'steps to unit ball', color='cornflowerblue', marker = 'o')
            ax1.plot(index, self.step2norm2ball_record, label = 'steps to norm 2 ball', color='plum', marker = 'o')

            ax2 = ax1.twinx()
            ax2.plot(index, self.test_accumulated_rewards, label='rewards', color='orange')

            ax1.legend()
            ax2.legend()

            plt.title("Convergence Speed and Reward")
            ax1.set_xlabel("PI iteration")
            ax1.set_ylabel("Steps / 0.01s")
            ax2.set_ylabel("rewards")
            fig.savefig(self.save_path + 'figs/Convergence Speed and Judgment.png', dpi=600)
            fig.show()

            self.model_save()
            filename = self.save_path + 'data/converge_record_data.txt'

            # 打开文件用于写入
            with open(filename, 'w') as file:
                # 写入列表名和数据
                file.write(f"Step to Converge Record: {self.step2converge_record}\n")
                file.write(f"Step to Unit Ball Record: {self.step2unitball_record}\n")
                file.write(f"Step to Norm 2 Ball Record: {self.step2norm2ball_record}\n")
                file.write(f"Accumulated Rewards in Test Record: {self.test_accumulated_rewards}\n")


    def initialize_policy_net(
            self,
            x_train_lim: int = 10,
            x_test_lim: int = 13,
            sample_num: int = 1000,
            iteration: int = 2*10**4,
            lr: float = 1e-4
    ):
        """
        用神经网络拟合专家策略（线性静态状态反馈）
        在接近平衡点处更多采样，使拟合更精准
        """
        print('---------------------Initializing Policy------------------------')

        random_data = [-x_train_lim + torch.rand(400)*2*x_train_lim for _ in range(self.system.state_dims())]
        train_data = torch.stack(random_data, dim=1).to(self.device)

        random_data1 = [-1 + torch.rand(400)*2*1 for _ in range(self.system.state_dims())]
        train_data1 = torch.stack(random_data1, dim=1).to(self.device)

        K = torch.tensor(-self.system.K)
        zero_column = torch.zeros(1, 2)
        K_extended = torch.vstack((K, zero_column)).t().to(self.device)
        labels = train_data @ K_extended
        labels1 = train_data1 @ K_extended
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.actor.parameters(), lr = lr)
        for i in range(iteration):
            optimizer.zero_grad()
            y_train = self.actor(train_data.t())
            y_train1 = self.actor(train_data1.t())
            loss = loss_fn(y_train, labels) + loss_fn(y_train1, labels1)*10
            # if i % 10000 == 0:
            #     print(f'times {i} - lr {lr} -  loss: {loss.item()}')
            if (i + 1) % 1000 == 0:
                print ('Epoch [{}/{}], Loss: {:.10f}'.format(i + 1, iteration, loss.item()))
            loss.backward()
            optimizer.step()
        # test部分
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
        the_controller = None, # 指定控制器，默认为ControlAffineSystem自带控制器
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


    def reward_pendulum(
            self,
            theta,
            theta_dot,
            action
    ):
        """
        theta = 0表示倒立
        reward: 越接近零, 越大越好
        """
        return -theta**2 - 0.1*theta_dot - 0.001*action**2
        
    
    def buffer_add(
            self, 
            state, 
            action,
            reward, 
            next_state, 
            done):
        # 以list类型保存
        self.buffer.append((state, action, reward, next_state, done))


    def buffer_size(self):
        return len(self.buffer)
    

    def buffer_sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 将数据集拆分开来
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done


    def buffer_add_sampledata(
      self,
      sample_data: torch.Tensor
    ):
        sample_data = sample_data.detach().clone()
        N = sample_data.shape[0]
        state = sample_data[:,0].permute(0, 2, 1)
        next_state = sample_data[:,1].permute(0, 2, 1)
        theta = state[:,:,0]
        theta_dot = state[:,:,1]
        action = self.target_actor.Controller(state.squeeze(1).t()).t()
        reward = self.reward_pendulum(theta,theta_dot,action)
        for i in range(N):
            if torch.norm(state[i], p=2)<=0.1:
                done = 1
            else:
                done = 0
            self.buffer_add(state[i], action[i], reward[i], next_state[i], done)


    def record_rollout_reward(self,
                              rollout_data):
        sample_data = rollout_data.detach().clone()
        N = sample_data.shape[0]
        state = sample_data[:,0].permute(0, 2, 1)
        # next_state = sample_data[:,1].permute(0, 2, 1)
        theta = state[:,:,0]
        theta_dot = state[:,:,1]
        action = self.target_actor.Controller(state.squeeze(1).t()).t()
        reward = self.reward_pendulum(theta,theta_dot,action)
        rewards = 0
        for i in range(N):
            if torch.norm(state[i], p=2)>0.1:
                rewards = rewards + reward[i]
        return rewards.item()
           
            
    def soft_update(self, net, target_net):
        # 获取训练网络和目标网络需要更新的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 训练网络的参数更新要综合考虑目标网络和训练网络
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)
        
        
    def update(self,
               batch_size: int = 1500,
               actor_lr: float = 0.01, 
               critic_lr: float = 0.01,
               actor_descent_step: int = 100,
               critic_descent_step: int = 100
               ):
        print('---------------------Updating Actor & Critic--------------------')
        
        state, action, reward, next_state, done = self.buffer_sample(batch_size)
        states = torch.stack(state).squeeze(1).to(self.device)
        actions = torch.stack(action).to(self.device)
        rewards = torch.stack(reward).to(self.device)
        next_states = torch.stack(next_state).squeeze(1).to(self.device)
        dones = torch.tensor(done, dtype=torch.float).view(-1, 1).to(self.device)

        # 价值目标网络获取下一时刻的每个动作价值[b,n_states]-->[b,n_actors]
        next_a_values = self.target_actor.Controller(next_states.t()).view(-1, 1)
        # 策略目标网络获取下一时刻状态选出的动作价值 [b,n_states+n_actions]-->[b,1]
        next_q_values = self.target_critic(next_states, next_a_values)
        # 当前时刻的动作价值的目标值 [b,1]
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        print('-----------------------Updating Critic----------------------')
        for _ in range(critic_descent_step):    
            # 当前时刻动作价值的预测值 [b,n_states+n_actions]-->[b,1]
            q_values = self.critic(states, actions)
            # 预测值和目标值之间的均方差损失
            critic_loss = torch.mean(F.mse_loss(q_values, q_targets))
            # 价值网络梯度
            critic_loss.backward(retain_graph = True)
            with torch.no_grad(): 
                critic_optimizer.step()
                critic_optimizer.zero_grad()
        
        print('------------------------Updating Actor----------------------')
        for _ in range(actor_descent_step):
            # 当前状态的每个动作的价值 [b, n_actions]
            actor_a_values = self.actor.Controller(states.t()).view(-1, 1)
            # 当前状态选出的动作价值 [b,1]
            score = self.critic(states, actor_a_values)
            # 计算损失
            actor_loss = torch.mean(score)
            actor_loss.backward(retain_graph = True)
            # 策略网络梯度
            with torch.no_grad(): 
                actor_optimizer.step()
                actor_optimizer.zero_grad()

        self.soft_update(self.actor, self.target_actor)
        # 软更新价值网络的参数
        self.soft_update(self.critic, self.target_critic)


    def DDPG_main_iteration(
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
                                    iteration = 1*10**2,
                                    lr = 1e-3)
        
        sim_data_ = self.system.simulate_rk4(x_initial = x_initial, 
                                    step_number = plot_step_num,
                                    use_controller = 1,
                                    the_controller = self.actor.Controller)
        
        step2norm2ball,step2unitball,step2converge = self.system.convergence_judgment(sim_data_)
        rewards = self.record_rollout_reward(sim_data_)
        self.step2norm2ball_record.append(step2norm2ball)
        self.step2unitball_record.append(step2unitball)
        self.step2converge_record.append(step2converge)
        self.test_accumulated_rewards.append(rewards)

        self.system.plot_phase_portrait(data_sim = sim_data_,
                                     arrow_on = 0,
                                     title = 'Phase Portrait with initialized NN controller',
                                     save_fig = 1,
                                     save_path = self.save_path + 'figs/')
        
        for i in range(1, iteration+1):
            print('---------------------------------Iteration {}-------------------------------'.format(i))
            
            sim_data = self.sample_training_data(sample_radius_trajectory = 5,
                                                sample_trajectory_number = 0,
                                                sample_number_per_trajectory = 0,
                                                sample_radius_random = 8,
                                                sample_number_in_radius = 1000,
                                                # sample_number_in_radius = 0,
                                                invariant_sample = 0,
                                                sample_plot = 0,
                                                the_controller = self.actor.Controller,
                                                title = 'sample training data')
            self.buffer_add_sampledata(sim_data)

            if self.buffer_size() >= self.min_training_batch:
                # sim_data = self.sample_training_data(sample_radius_trajectory = 0,
                #                                 sample_trajectory_number = 0,
                #                                 sample_number_per_trajectory = 0,
                #                                 sample_radius_random = 8,
                #                                 sample_number_in_radius = 1000,
                #                                 invariant_sample = 0,
                #                                 sample_plot = 0,
                #                                 the_controller = self.actor.Controller,
                #                                 title = 'sample training data')
                self.buffer_add_sampledata(sim_data)
                self.update(batch_size = self.min_training_batch,
                            actor_lr = 0.01, 
                            critic_lr = 0.01,
                            actor_descent_step = 1,
                            critic_descent_step = 1)
                
                sim_data_ = self.system.simulate_rk4(x_initial = x_initial, 
                                            step_number = plot_step_num,
                                            use_controller = 1,
                                            the_controller = self.actor.Controller)
                step2norm2ball,step2unitball,step2converge = self.system.convergence_judgment(sim_data_)
                rewards = self.record_rollout_reward(sim_data_)

                self.step2norm2ball_record.append(step2norm2ball)
                self.step2unitball_record.append(step2unitball)
                self.step2converge_record.append(step2converge)
                self.test_accumulated_rewards.append(rewards)

                self.system.plot_phase_portrait(data_sim = sim_data_,
                                        arrow_on = 0,
                                        title = 'after PI in iteration {}'.format(i),
                                        save_fig = 1,
                                        save_path =  self.save_path + 'figs/')
                
            self.plot_save_converge_steps()
            plt.close()
