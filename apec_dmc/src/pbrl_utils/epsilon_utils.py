import torch
import numpy as np
from torch import nn
import os

class MLPBC(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=256,
        device=torch.device('cpu'),
        act_dim=0,
        env=None,
    ):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
        self.device = device
        self.env = env
        
        self.to(device)

    def forward(self, batch):
        return self.model(batch)
    
    def act(self, batch):
        if isinstance(batch, np.ndarray):
            input = torch.from_numpy(batch).to(self.device).float()
        else:
            input = batch
        output= self.forward(input)
        return np.clip(output.detach().cpu().numpy(), self.env.action_space.low, self.env.action_space.high)

class ActionNoise(object):
    def reset(self):
        pass

class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=0.033, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class NoiseInjectedPolicy(object):
    def __init__(self, env, policy, action_noise_type, noise_level, scale=None):
        self.action_space = env.action_space
        self.policy = policy
        self.action_noise_type = action_noise_type
        self.device = self.policy.device

        # if action_noise_type == 'normal':
        #     mu, std = np.zeros(self.action_space.shape), noise_level*np.ones(self.action_space.shape)
        #     self.action_noise = NormalActionNoise(mu=mu,sigma=std)
        self.epsilon = noise_level
        if scale is not None:
            self.scale = scale
        else:       
            self.scale = self.action_space.high[0]
        # assert np.all(self.scale == self.action_space.high) and \
        #     np.all(self.scale == -1.*self.action_space.low)
        # else:
        #     assert False, "no such action noise type: %s"%(action_noise_type)

    def act(self, obs):
        if np.random.random() < self.epsilon:
            act = np.random.uniform(-self.scale,self.scale,self.action_space.shape)
        else:
            act = self.policy.act(obs)

        return np.clip(act,self.action_space.low,self.action_space.high)

    def reset(self):
        self.action_noise.reset()


class SSRRNoiseInjectedPolicy(object):
    def __init__(self, env, policy, action_noise_type, noise_level, scale=None):
        self.action_space = env.action_space
        self.policy = policy
        self.action_noise_type = action_noise_type
        self.device = self.policy.device

        # if action_noise_type == 'normal':
        #     mu, std = np.zeros(self.action_space.shape), noise_level*np.ones(self.action_space.shape)
        #     self.action_noise = NormalActionNoise(mu=mu,sigma=std)
        self.epsilon = noise_level
        if scale is not None:
            self.scale = scale
        else:       
            self.scale = self.action_space.high[0]
        # assert np.all(self.scale == self.action_space.high) and \
        #     np.all(self.scale == -1.*self.action_space.low)
        # else:
        #     assert False, "no such action noise type: %s"%(action_noise_type)

    def act(self, obs):
        act = self.policy.act(obs)
        if np.random.random() < self.epsilon:
            return np.random.uniform(-self.scale,self.scale,self.action_space.shape)
        return np.clip(act,self.action_space.low,self.action_space.high)
    
    def gen_traj_padding(self, env, segment_len=1000):
        obs, actions, rewards, dones = [env.reset()], [], [], []
        t = 0
        while t < segment_len:
            action_old = self.policy.act(torch.from_numpy(obs[-1]).to(self.device).float().reshape(1, -1))
            
            if np.random.random() < self.epsilon:
                action = np.random.uniform(-self.scale,self.scale,self.action_space.shape)
            else:
                action = action_old
            
            ob, reward, done, _ = env.step(action)

            obs.append(ob)
            actions.append(action_old.reshape(-1)) # ATTENTION ! ssrr stores old action
            rewards.append(reward)
            dones.append(done)

            t += 1
            if done:
                break
        if done:
            obs.pop()
        
        states, action, rewards, dones = np.stack(obs,axis=0), np.array(actions), np.array(rewards), np.array(dones)

        seq_len_1 = len(states)
        seq_len_1 = action.shape[0]
        if seq_len_1 < segment_len:
            start_idx_1 = 0
            # padding with last value
            states = np.concatenate([states, states[-1] * np.ones((segment_len-seq_len_1, states[0].shape[-1]))])
            action = np.concatenate([action, actions[-1] * np.ones((segment_len-seq_len_1, action[0].shape[-1]))])
            rewards = np.concatenate([rewards, np.zeros((segment_len-seq_len_1, ))])
        else:
            start_idx_1 = np.random.randint(seq_len_1 - segment_len + 1)
            states = states[start_idx_1: start_idx_1 + segment_len]
            action = action[start_idx_1: start_idx_1 + segment_len]
            rewards = rewards[start_idx_1: start_idx_1 + segment_len]
        masks = np.ones(segment_len)
        masks[seq_len_1:] *= 0
            
        return states, action, rewards, dones, masks

def gen_traj(env, agent, min_length):
    obs, actions, rewards, dones = [env.reset()], [], [], []
    t = 0
    while t < 1000:
        action = agent.act(torch.from_numpy(obs[-1]).to(agent.device).float().reshape(1, -1)).squeeze()
        # print('action:', action)
        ob, reward, done, _ = env.step(action)

        obs.append(ob)
        actions.append(action.reshape(-1))
        rewards.append(reward)
        dones.append(done)

        t += 1
        if done:
            break

    if done:
        obs.pop()
    
    return np.stack(obs,axis=0), np.array(actions), np.array(rewards), np.array(dones)



def longest_decreasing_subsequence(arr, threshold):
    n = len(arr)
    if n == 0:
        return 0, []
    
    dp = [1] * n
    parent = [-1] * n
    max_len = 1
    max_idx = 0

    for i in range(1, n):
        for j in range(i):
            if arr[j] - arr[i] >= threshold and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j  

        if dp[i] > max_len:
            max_len = dp[i]
            max_idx = i

    sequence_indices = []
    while max_idx != -1:
        sequence_indices.append(max_idx)
        max_idx = parent[max_idx]

    sequence_indices.reverse() 

    return max_len, sequence_indices


def filter_paths(file_dir, n_model=10, filter_times=1, max_epoch=1000):
    '''
        return: loadable paths for models sorted by distance
    '''

    epochs = []
    distances = []
    filenames = []
    for file_name in os.listdir(file_dir):
        epoch = eval(file_name.split('_')[2].replace('epoch', ''))
        if epoch > max_epoch:
            continue
        distance = eval(file_name.split('_')[-1].replace('distance', '').replace('.pt', ''))
        epochs.append(epoch)
        distances.append(distance)
        filenames.append(os.path.join(file_dir, file_name))
    
    epochs = np.array(epochs)
    distances = np.array(distances)[epochs.argsort()]
    filenames = np.array(filenames)[epochs.argsort()]
    
    # for path in filenames:
    #     print(path)
    return filenames

if __name__ == '__main__':
    filter_paths('logfile/train_irl/irl/Ant-v2/0_0_0_1/AntFH-v0_maxentirl_sa_use_pref_False_seed_1_sac_epochs_5_sac_alpha_0.1_0_0_0_1_last_n_samples-DEBUG-NEW/best_model/actors', filter_times=2)