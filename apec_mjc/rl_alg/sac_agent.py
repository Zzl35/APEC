'''
Code from spinningup repo.
Refer[Original Code]: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def sigmoid_map(x):
    return 2 * torch.sigmoid(x) - 1

def sigmoid_inv(y):
    return torch.log((1 + y) / (1 - y))

def rational_sigmoid_map(x):
    return x / torch.sqrt(1 + x ** 2)

def rational_sigmoid_inv(y):
    return y / torch.sqrt(1 - y ** 2 + 1e-7)

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None
        
        pi_action_mapped = torch.tanh(pi_action)
        # pi_action_mapped = sigmoid_map(pi_action)
        # pi_action_mapped = rational_sigmoid_map(pi_action)

        pi_action_mapped = self.act_limit * pi_action_mapped

        return pi_action_mapped, logp_pi


    def log_prob(self, obs, act):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std) 
        # 20210306: fix this bug for reversal operation on action. 
        # this may improve AIRL results, leaving for future work.
        act = act / self.act_limit

        # mapping to project [-1,1] to real
        act = torch.atanh(act)
        # act = sigmoid_inv(act)
        # act = rational_sigmoid_inv(act)

        logp_pi = pi_distribution.log_prob(act).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - act - F.softplus(-2*act))).sum(axis=1)

        return logp_pi

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        # print(obs, act)
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, k, hidden_sizes=(256,256), add_time=False,
                 activation=nn.ReLU, device=torch.device("cpu")):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.device = device

        if k == 1:
            self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(self.device)
        # else: # we did not use GMM policy, this is experimental.
        #     self.pi = SquashedGmmMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, k).to(self.device)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(self.device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(self.device)

    def act(self, obs, deterministic=False, get_logprob = False):
        with torch.no_grad():
            a, logpi = self.pi(obs, deterministic, True)
            if get_logprob:
                return a.cpu().data.numpy().flatten(), logpi.cpu().data.numpy()
            else:
                return a.cpu().data.numpy().flatten()

    def act_batch(self, obs, deterministic=False):
        with torch.no_grad():
            a, logpi = self.pi(obs, deterministic, True)
            return a.cpu().data.numpy(), logpi.cpu().data.numpy()

    def log_prob(self, obs, act):
        return self.pi.log_prob(obs, act)

    def act_batch_with_grad(self, obs, deterministic=False):
        a, _ = self.pi(obs, deterministic, True)
        return a
    