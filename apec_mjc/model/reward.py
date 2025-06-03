import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_clamp(
    x : torch.Tensor,
    _min=None,
    _max=None,
) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLPReward(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_sizes=(256,256),
        hid_act='tanh',
        use_bn=False,
        residual=False,
        clamp_magnitude=10.0,
        device=torch.device('cpu'),
        **kwargs
    ):
        super().__init__()

        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()

        self.clamp_magnitude = clamp_magnitude
        self.input_dim = input_dim
        self.device = device
        self.residual = residual

        self.first_fc = nn.Linear(input_dim, hidden_sizes[0])
        self.blocks_list = nn.ModuleList()

        for i in range(len(hidden_sizes) - 1):
            block = nn.ModuleList()
            block.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if use_bn: block.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            block.append(hid_act_class())
            self.blocks_list.append(nn.Sequential(*block))
        
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, batch):
        x = self.first_fc(batch)
        for block in self.blocks_list:
            if self.residual:
                x = x + block(x)
            else:
                x = block(x)
        output = self.last_fc(x)
        output = torch.clamp(output, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return output  

    def r(self, batch):
        r = self.forward(batch)
        # return torch.log(torch.sigmoid(r)) - torch.log(1-torch.sigmoid(r))
        return r

    def get_scalar_reward(self, obs):
        self.eval()
        with torch.no_grad():
            if not torch.is_tensor(obs):
                obs = torch.FloatTensor(obs.reshape(-1,self.input_dim))
            obs = obs.to(self.device)
            reward = self.forward(obs).cpu().detach().numpy().flatten()
        self.train()
        return reward
    
class PrefRewardModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_sizes=(256,256),
        hid_act='tanh',
        use_bn=False,
        residual=False,
        clamp_magnitude=10.0,
        device=torch.device('cpu'),
        **kwargs
    ):
        super().__init__()

        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()
        
        self.norm = nn.LayerNorm(input_dim)

        self.clamp_magnitude = clamp_magnitude
        self.input_dim = input_dim
        self.device = device
        self.residual = residual
        
        self._min_r = nn.Parameter(torch.zeros(1))
        self._max_r = nn.Parameter(torch.ones(1))

        self.first_fc = nn.Linear(input_dim, hidden_sizes[0])
        self.blocks_list = nn.ModuleList()

        for i in range(len(hidden_sizes) - 1):
            block = nn.ModuleList()
            block.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if use_bn: block.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            block.append(hid_act_class())
            self.blocks_list.append(nn.Sequential(*block))
        
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)
        # self.apply(weight_init)

    def forward(self, batch):
        # x = self.norm(batch)
        x = self.first_fc(batch)

        for block in self.blocks_list:
            if self.residual:
                x = x + block(x)
            else:
                x = block(x)
        # output = torch.sigmoid(self.last_fc(x))
        output = self.last_fc(x)
        # output = soft_clamp(output, -10, 10)
        return output  

    def r(self, batch):
        reward = self.forward(batch)
        return reward
    
    def get_scalar_reward(self, obs):
        self.eval()
        with torch.no_grad():
            if not torch.is_tensor(obs):
                obs = torch.FloatTensor(obs.reshape(-1,self.input_dim))
            obs = obs.to(self.device)
            reward = self.r(obs).cpu().detach().numpy().flatten()
        self.train()
        return reward


class GAILMLPReward(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_sizes=(256,256),
        hid_act='tanh',
        use_bn=False,
        residual=False,
        clamp_magnitude=10.0,
        device=torch.device('cpu'),
        **kwargs
    ):
        super().__init__()

        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()

        self.clamp_magnitude = clamp_magnitude
        self.input_dim = input_dim
        self.device = device
        self.residual = residual

        self.first_fc = nn.Linear(input_dim, hidden_sizes[0])
        self.blocks_list = nn.ModuleList()

        for i in range(len(hidden_sizes) - 1):
            block = nn.ModuleList()
            block.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if use_bn: block.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            block.append(hid_act_class())
            self.blocks_list.append(nn.Sequential(*block))
        
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, batch):
        x = self.first_fc(batch)
        for block in self.blocks_list:
            if self.residual:
                x = x + block(x)
            else:
                x = block(x)
        output = self.last_fc(x)
        output = torch.clamp(output, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        prob = torch.sigmoid(output)
        return output, prob

    def get_scalar_reward(self, obs):
        self.eval()
        with torch.no_grad():
            if not torch.is_tensor(obs):
                obs = torch.FloatTensor(obs.reshape(-1,self.input_dim))
            obs = obs.to(self.device)
            logits, prob = self.forward(obs)
            reward = logits + nn.functional.softplus(-logits)
            reward = reward.cpu().detach().numpy().flatten()
        self.train()
        return reward

class DrexRewardModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_sizes=(256,256),
        hid_act='tanh',
        use_bn=False,
        residual=False,
        clamp_magnitude=10.0,
        device=torch.device('cpu'),
        **kwargs
    ):
        super().__init__()

        hid_act_class = nn.ReLU

        self.input_dim = input_dim
        self.device = device
        self.first_fc = nn.Linear(input_dim, hidden_sizes[0])
        self.blocks_list = nn.ModuleList()

        for i in range(len(hidden_sizes) - 1):
            block = nn.ModuleList()
            block.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            block.append(hid_act_class())
            self.blocks_list.append(nn.Sequential(*block))
        
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, batch):
        # x = self.norm(batch)
        x = self.first_fc(batch)
        for block in self.blocks_list:
            x = block(x)
        output = self.last_fc(x)
        return output  

    def r(self, batch):
        return self.forward(batch)

    def get_scalar_reward(self, obs):
        self.eval()
        with torch.no_grad():
            if not torch.is_tensor(obs):
                obs = torch.FloatTensor(obs.reshape(-1,self.input_dim))
            obs = obs.to(self.device)
            reward = self.forward(obs).cpu().detach().numpy().flatten()
        self.train()
        return reward

class SSRRSigmoid(nn.Module):
    def __init__(
        self,
        device=torch.device('cpu'),
    ):
        super().__init__()
        self.sigmoid_params = nn.ParameterList([
            nn.Parameter(torch.empty(1, device=device)),  # c
            nn.Parameter(torch.empty(1, device=device)),  # x0
            nn.Parameter(torch.empty(1, device=device)),  # k
            nn.Parameter(torch.empty(1, device=device))   # y0
        ])

    #     self._init_params()

    # def _init_params(self):
    #     nn.init.uniform_(self.sigmoid_params[0], a=0.5, b=1.5)  
    #     nn.init.uniform_(self.sigmoid_params[1], a=-1.0, b=1.0)  
    #     nn.init.uniform_(self.sigmoid_params[2], a=0.1, b=2.0)  
    #     nn.init.uniform_(self.sigmoid_params[3], a=-0.5, b=0.5)


    def forward(self, x): # from SSRR paper
        c, x0, k, y0 = self.sigmoid_params
        return c / (1 + torch.exp(-k * (x - x0))) + y0

class SupervisedRewardModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_sizes=(256,256),
        hid_act='tanh',
        use_bn=False,
        residual=False,
        clamp_magnitude=10.0,
        device=torch.device('cpu'),
        **kwargs
    ):
        super().__init__()

        hid_act_class = nn.ReLU

        self.input_dim = input_dim
        self.device = device
        self.first_fc = nn.Linear(input_dim, hidden_sizes[0])
        self.blocks_list = nn.ModuleList()

        for i in range(len(hidden_sizes) - 1):
            block = nn.ModuleList()
            block.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            block.append(hid_act_class())
            self.blocks_list.append(nn.Sequential(*block))
        
        self.last_fc = nn.Linear(hidden_sizes[-1], 2)

    def forward(self, batch):
        # x = self.norm(batch)
        x = self.first_fc(batch)
        for block in self.blocks_list:
            x = block(x)
        output = self.last_fc(x)
        return output  

    def r(self, batch):
        output = self.forward(batch)
        r1, r2 =  output[..., 0], output[..., 1]
        w1 = 1
        w2 = 1
        return torch.exp((w1 * r1 + w2 * (1 - r2)).clip(max=2.5))


    def get_scalar_reward(self, obs):
        self.eval()
        with torch.no_grad():
            if not torch.is_tensor(obs):
                obs = torch.FloatTensor(obs.reshape(-1,self.input_dim))
            obs = obs.to(self.device)
            reward = self.r(obs).cpu().detach().numpy().flatten()
        self.train()
        return reward
