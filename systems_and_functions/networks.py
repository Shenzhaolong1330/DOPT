##############################
## Author ## Shen Zhaolong ###
##  Date  ##    2024-07    ###
##############################

import torch
from torch import nn
from torch.nn import functional as F


class PolicyNet(nn.Module):
    def __init__(
        self, 
        n_states: int = 4, 
        n_hiddens: int = 16,
        n_actions: int = 2, 
        action_bound: float = 5.0
    ):
        super(PolicyNet, self).__init__()
        # 环境可以接受的动作最大值
        self.action_bound = action_bound
        self.n_actions = n_actions
        # 只包含一个隐含层
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions + 1)

    # 前向传播
    def forward(self, x):
        # x = x.t()
        if x.shape[1] != 2:
            if x.shape[1] == 7:
                x = x
            else:
                x = x.t()
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
        # x = F.relu(x)
        # x = torch.tanh(x)  # 将数值调整到 [-1,1]
        # x = x * self.action_bound  # 缩放到 [-action_bound, action_bound]
        return x
    
    def Controller(self, x):
        # x = x.t()
        u = self.forward(x).t()[:self.n_actions]
        # u = torch.tanh(u)  # 将数值调整到 [-1,1]
        # u = u * self.action_bound  # 缩放到 [-action_bound, action_bound]
        u = torch.clamp(u, max = self.action_bound, min = -self.action_bound)
        # u = -torch.clamp(-u, max=self.action_bound)
        return u

class LyapunovNet(nn.Module):
    def __init__(
        self,
        n_states:int = 4, 
        n_hiddens:int = 128
    ):
        super(LyapunovNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 2)

    def forward(self, x):
        # x = x.clone().detach().requires_grad_(True)
        s = self.fc1(x)  # -->[b, n_hiddens]
        # s = F.relu(s)
        s = torch.tanh(s)
        s = self.fc2(s)  # -->[b, n_hiddens]
        # s = F.relu(s)
        s = torch.tanh(s)
        V = self.fc3(s)  # -->[b, 2]
        # V = 0.5 * torch.pow(V,2) # semi-positive definite
        return V
    
    def V(self, x):
        # print('self.forward(x)', self.forward(x).shape)
        V = self.forward(x)[:,0]
        # print('self.V(x)', V.shape)
        return V

    def V_with_JV(self, x):
        x = x.clone().detach().requires_grad_(True)
        V = self.V(x)
        JV = torch.autograd.grad(V, x)
        return V, JV

class DFunctionNet(nn.Module):
    def __init__(
        self,
        n_states:int = 4, 
        n_hiddens: int = 128, 
        n_actions: int = 2
    ):
        super(DFunctionNet, self).__init__()
        self.fc1 = nn.Linear(n_states + n_actions, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 2)

    # 前向传播
    def forward(self, x, a):
        # 拼接状态和动作
        cat = torch.cat([x, a], dim=1)  # [b, n_states + n_actions]
        x = self.fc1(cat)  # -->[b, n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # -->[b, n_hiddens]
        x = F.relu(x)
        x = self.fc3(x)  # -->[b, 1]
        return x

    def D(self, x, a):
        return self.forward(x, a)[:,0]

class QValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(QValueNet, self).__init__()
        #
        self.fc1 = nn.Linear(n_states + n_actions, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 1)

    # 前向传播
    def forward(self, x, a):
        # 拼接状态和动作
        cat = torch.cat([x, a], dim=1)  # [b, n_states + n_actions]
        x = self.fc1(cat)  # -->[b, n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # -->[b, n_hiddens]
        x = F.relu(x)
        x = self.fc3(x)  # -->[b, 1]
        return x