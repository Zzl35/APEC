from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import sys
import rl_alg.sac_agent as core
from collections import deque
import os
import ot
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from utils.epsilon_utils import NoiseInjectedPolicy, gen_traj, MLPBC

EPS = 1e-4


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class ExpertReplayBuffer:
    ### oracle rewards only for test
    def __init__(self, states=None, actions=None, rewards=None, masks=None, init_num_trajs=None, device=None):
        if states is None:
            self.states = self.actions = self.rewards = self.masks = None
        else:
            self.states = states
            self.actions = actions
            self.rewards = rewards
            self.masks = masks
        self.trajs_weight = None
        self.init_num_trajs = init_num_trajs
        self.device = device
    
    def store_batch(self, states, actions, rewards, masks):
        if self.states is None:
            self.states = states
            self.actions = actions
            self.rewards = rewards
            self.masks = masks
        else:
            self.states = np.concatenate([self.states, states], axis=0)
            self.actions = np.concatenate([self.actions, actions], axis=0)
            self.rewards = np.concatenate([self.rewards, rewards], axis=0)
            self.masks = np.concatenate([self.masks, masks], axis=0)

    def update_trajs_weight(self, pref_reward, beta, state_only=False, clip_max=None):
        self.beta = beta
        if not state_only:
            expert_trajs = np.concatenate([self.states, self.actions], axis=-1)
        else:
            expert_trajs = self.states
        with torch.no_grad():
            D_logits = pref_reward.predict_r_hat_torch(torch.FloatTensor(expert_trajs).to(self.device)).squeeze(-1)
        D_logits *= torch.FloatTensor(self.masks).to(self.device)
        self.fake_rewards = D_logits
        self.fake_returns = torch.sum(D_logits, dim=1)
        if clip_max is not None:
            self.fake_returns = torch.clamp(self.fake_returns, -float("inf"), clip_max)
        if beta > 0:
            self.trajs_weight = torch.exp(torch.clamp(self.fake_returns / beta, -float("inf"), 85))
        else:
            self.trajs_weight = torch.exp(self.fake_returns * 0)
        self.trajs_weight /= torch.sum(self.trajs_weight)

    def cail_update_trajs_weight(self, reward, beta, state_only=False, clip_max=None):
        self.beta = beta
        if not state_only:
            expert_trajs = np.concatenate([self.states, self.actions], axis=-1)
        else:
            expert_trajs = self.states
        with torch.no_grad():
            D_logits = reward.predict_r_hat_torch(torch.FloatTensor(expert_trajs).to(self.device)).squeeze(-1)
        D_logits *= torch.FloatTensor(self.masks).to(self.device)
        self.fake_rewards = D_logits
        self.fake_returns = torch.sum(D_logits, dim=1)
        if clip_max is not None:
            self.fake_returns = torch.clamp(self.fake_returns, -float("inf"), clip_max)
        if beta > 0:
            self.trajs_weight = torch.exp(torch.clamp(self.fake_returns / beta, -float("inf"), 85))
        else:
            self.trajs_weight = torch.exp(self.fake_returns * 0)
        self.trajs_weight /= torch.sum(self.trajs_weight)
    
    def sample_all_without_pref(self):
        return np.concatenate([self.states, self.actions], axis=-1), self.masks, None

    def sample_with_pref(self, num_trajs=None):
        if num_trajs is None:
            num_trajs = self.init_num_trajs
        if self.states.shape[0]<=num_trajs:
            return np.concatenate([self.states, self.actions], axis=-1), self.masks, self.trajs_weight
        else:
            index = sorted(range(self.trajs_weight.shape[0]), key=lambda k:self.trajs_weight[k])[-num_trajs:]
            return np.concatenate([self.states, self.actions], axis=-1)[index], self.masks[index], self.trajs_weight[index]

    def draw(self, dir, name):
        if not os.path.exists(dir):
            os.mkdir(dir)
        oracle_returns_draw = np.sum(self.rewards * self.masks, axis=-1)
        fake_returns_draw = self.fake_returns.cpu().numpy()
        plt.scatter(oracle_returns_draw, fake_returns_draw)
        plt.xlabel("oracle returns")
        plt.ylabel("fake returns")
        plt.title("returns2fake returns")
        plt.savefig(f"{dir}/oracle_returns2fake_returns_{name}.png")
        plt.close()
        if self.beta>0:
            oracle_weights_draw = np.exp(np.clip(oracle_returns_draw / self.beta, -float("inf"), 85))
        else:
            oracle_weights_draw = np.exp(oracle_returns_draw * 0)
        oracle_weights_draw /= np.sum(oracle_weights_draw)
        fake_weights_draw = self.trajs_weight.cpu().numpy()
        plt.scatter(oracle_weights_draw, fake_weights_draw)
        plt.xlabel("oracle_weights")
        plt.ylabel("fake_weights")
        plt.title("oracle weights2fake weights")
        plt.savefig(f"{dir}/oracle_weights2fake_weights_{name}.png")
        plt.close()

    def draw_grid(self, dir, name):
        if not os.path.exists(dir):
            os.mkdir(dir)
        oracle_returns_draw = np.sum(self.rewards * self.masks, axis=-1)
        fake_returns_draw = self.fake_returns.cpu().numpy()
        tar_norm = np.linalg.norm(self.states[:, 0, 2:], axis=-1)
        in_index = np.where(tar_norm<=0.75)[0]
        out_index = np.where(tar_norm>0.75)[0]
        plt.scatter(oracle_returns_draw[in_index], fake_returns_draw[in_index], label="In Circle")
        plt.scatter(oracle_returns_draw[out_index], fake_returns_draw[out_index], label="Out of Circle")
        plt.legend()
        plt.xlabel("oracle returns")
        plt.ylabel("fake returns")
        plt.title("returns2fake returns")
        plt.savefig(f"{dir}/oracle_returns2fake_returns_{name}.png")
        plt.close()
        if self.beta>0:
            oracle_weights_draw = np.exp(oracle_returns_draw / self.beta)
        else:
            oracle_weights_draw = np.exp(oracle_returns_draw * 0)
        oracle_weights_draw /= np.sum(oracle_weights_draw)
        fake_weights_draw = self.trajs_weight.cpu().numpy()
        plt.scatter(oracle_weights_draw, fake_weights_draw)
        plt.xlabel("oracle_weights")
        plt.ylabel("fake_weights")
        plt.title("oracle weights2fake weights")
        plt.savefig(f"{dir}/oracle_weights2fake_weights_{name}.png")
        plt.close()
        M_index = list(range(self.rewards.shape[0]//2))
        H_index = list(range(self.rewards.shape[0]//2, self.rewards.shape[0]))
        plt.scatter(oracle_returns_draw[M_index], fake_returns_draw[M_index], label="M")
        plt.scatter(oracle_returns_draw[H_index], fake_returns_draw[H_index], label="H")
        plt.legend()
        plt.xlabel("oracle returns")
        plt.ylabel("fake returns")
        plt.title("returns2fake returns")
        plt.savefig(f"{dir}/oracle_returns2fake_returns_level_{name}.png")
        plt.close()

    def draw_reward(self, dir, name):
        if not os.path.exists(dir):
            os.mkdir(dir)
        valid_index = np.where(self.masks.reshape((-1))==1)[0]
        oracle_rewards_draw = self.rewards.reshape((-1))[valid_index]
        fake_rewards_draw = self.fake_rewards.cpu().numpy().reshape((-1))[valid_index]
        plt.scatter(oracle_rewards_draw, fake_rewards_draw)
        plt.xlabel("oracle rewards")
        plt.ylabel("fake rewards")
        plt.title("rewards2fake rewards")
        plt.savefig(f"{dir}/oracle_rewards2fake_rewards_{name}.png")
        plt.close()


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, device=torch.device('cpu'), size=int(1e6), env_T=1000):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.next_state = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.action = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.reward = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device
        self.traj_begin_point = deque(maxlen=1000)
        self.traj_begin_point.append(0)
        self.env_T = env_T

    def add_traj_begin_point(self):
        self.traj_begin_point.append(self.ptr)

    def store(self, obs, act, rew, next_obs, done):
        self.state[self.ptr] = obs
        self.next_state[self.ptr] = next_obs
        self.action[self.ptr] = act
        self.reward[self.ptr] = rew
        self.done[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.state[idxs],
                     obs2=self.next_state[idxs],
                     act=self.action[idxs],
                     rew=self.reward[idxs],
                     done=self.done[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in batch.items()}

    def sample_batch_sa(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.state[idxs], self.action[idxs]

    def last_n_samples_sa(self, n=5000):
        # assert self.size >= n
        if self.size < n:
            print(f"Use total {self.size} samples in buffer, not enough {n}!")
            return self.state[:self.ptr], self.action[:self.ptr]
        else:
            if self.ptr >= n:
                return self.state[self.ptr-n : self.ptr], self.action[self.ptr-n : self.ptr]
            else:
                ret_state = np.concatenate((self.state[-(n-self.ptr) :], self.state[: self.ptr]), axis=0)
                ret_action = np.concatenate((self.action[-(n-self.ptr) :], self.action[: self.ptr]), axis=0)
                return ret_state, ret_action

    def last_n_trajs(self, n=5):
        assert len(self.traj_begin_point) - 1 >= n
        ret_states = np.zeros((n, self.env_T, self.obs_dim))
        ret_actions = np.zeros((n, self.env_T, self.act_dim))
        ret_rewards = np.zeros((n, self.env_T))
        ret_masks = np.zeros((n, self.env_T))
        for i in range(1, n+1):
            begin, end = self.traj_begin_point[-(i+1)], self.traj_begin_point[-i]
            if begin < end:
                ret_states[i-1, :end-begin, :] = self.state[begin:end, :]
                ret_actions[i-1, :end-begin, :] = self.action[begin:end, :]
                ret_rewards[i-1, :end-begin] = self.reward[begin:end]
                ret_masks[i-1, :end-begin] = 1
            else:
                pre_len = self.max_size - begin
                total_len = pre_len + end
                ret_states[i-1, :pre_len, :] = self.state[begin:, :]
                ret_states[i-1, pre_len:total_len, :] = self.state[:end, :]
                ret_actions[i-1, :pre_len, :] = self.action[begin:, :]
                ret_actions[i-1, pre_len:total_len, :] = self.action[:end, :]
                ret_rewards[i-1, :pre_len] = self.reward[begin:]
                ret_rewards[i-1, pre_len:total_len] = self.reward[:end]
                ret_masks[i-1, :total_len] = 1
        return ret_states, ret_actions, ret_rewards, ret_masks

    def save(self, path):
        np.savez(path, 
                 state=self.state[:self.size], 
                 next_state=self.next_state[:self.size], 
                 action=self.action[:self.size], 
                 reward=self.reward[:self.size], 
                 done=self.done[:self.size])
    
    def load(self, path):
        data = np.load(path)
        state, next_state, action, reward, done = data['state'], data['next_state'], data['action'], data['reward'], data['done']
        sample_num = state.shape[0]
        for i in range(sample_num):
            self.store(state[i], action[i], reward[i], next_state[i], done[i])

    def load_epsilon_buffer(self, path, n=10):
        data = np.load(path, allow_pickle=True)
        available_indexs = np.where(np.array([x[2] for x in data])==0)[0]
        indexs = np.random.choice(available_indexs, size=n)
        for idx in indexs:
            traj = data[idx][3][0]
            states = traj[0]
            actions = traj[1]
            rewards = traj[2]
            dones = traj[3]
            ep_len = states.shape[0]
            for i in range(ep_len-1):
                self.store(states[i], actions[i], rewards[i], states[i+1], dones[i])
            
    def load_expert_buffer(self, expert_buffer):
        states = expert_buffer.states
        actions = expert_buffer.actions
        rewards = expert_buffer.rewards
        masks = expert_buffer.masks
        for i in range(states.shape[0]):
            for j in range(states[i].shape[1] - 1):
                if masks[i, j]:
                    self.store(states[i, j], actions[i, j], rewards[i, j], states[i, j+1], 1-masks[i,j])


class PreferenceBuffer(object):
    def __init__(self, memory_size: int, seed: int = 0) -> None:
        np.random.seed(seed)
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)
        
    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)
    
    def save(self, path):
        buffer_size = len(self.buffer)
        states_1 = np.asarray([self.buffer[i][0] for i in range(buffer_size)])
        states_2 = np.asarray([self.buffer[i][1] for i in range(buffer_size)])
        actions_1 = np.asarray([self.buffer[i][2] for i in range(buffer_size)])
        actions_2 = np.asarray([self.buffer[i][3] for i in range(buffer_size)])
        rewards_1 = np.asarray([self.buffer[i][4] for i in range(buffer_size)])
        rewards_2 = np.asarray([self.buffer[i][5] for i in range(buffer_size)])
        masks_1 = np.asarray([self.buffer[i][6] for i in range(buffer_size)])
        masks_2 = np.asarray([self.buffer[i][7] for i in range(buffer_size)])
        pref_labels = np.asarray([self.buffer[i][8] for i in range(buffer_size)])

        np.savez(path, states_1=states_1, states_2=states_2, actions_1=actions_1, actions_2=actions_2, rewards_1=rewards_1, rewards_2=rewards_2, masks_1=masks_1, masks_2=masks_2, pref_labels=pref_labels)

    def load(self, path):
        data = np.load(path)
        states_1 = data['states_1']
        states_2 = data['states_2']
        actions_1 = data['actions_1']
        actions_2 = data['actions_2']
        rewards_1 = data['rewards_1']
        rewards_2 = data['rewards_2']
        masks_1 = data['masks_1']
        masks_2 = data['masks_2']
        pref_labels = data['pref_labels']
        for i in range(states_1.shape[0]):
            self.add((states_1[i], states_2[i], actions_1[i], actions_2[i], rewards_1[i], rewards_2[i], masks_1[i], masks_2[i], pref_labels[i]))
        
    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
            print(f"Sample num reset to buffer size: {batch_size}")
        if continuous:
            rand = np.random.randint(0, max(1, len(self.buffer) - batch_size)) # dxh
            # rand = np.random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

    def get_samples(self, batch_size, device):
        batch = self.sample(batch_size, False)

        batch_state_1, batch_state_2, batch_action_1, batch_action_2, batch_reward_1, batch_reward_2, batch_mask_1, batch_mask_2, batch_pref_label = zip(*batch)

        batch_state_1 = np.array(batch_state_1)
        batch_state_2 = np.array(batch_state_2)
        batch_action_1 = np.array(batch_action_1)
        batch_action_2 = np.array(batch_action_2)
        batch_reward_1 = np.array(batch_reward_1)
        batch_reward_2 = np.array(batch_reward_2)
        batch_mask_1 = np.array(batch_mask_1)
        batch_mask_2 = np.array(batch_mask_2)

        batch_state_1 = torch.as_tensor(batch_state_1, dtype=torch.float, device=device)
        batch_state_2 = torch.as_tensor(batch_state_2, dtype=torch.float, device=device)
        batch_action_1 = torch.as_tensor(batch_action_1, dtype=torch.float, device=device)
        batch_action_2 = torch.as_tensor(batch_action_2, dtype=torch.float, device=device)
        if batch_action_1.ndim == 1:
            batch_action_1 = batch_action_1.unsqueeze(1)
        if batch_action_2.ndim == 1:
            batch_action_2 = batch_action_2.unsqueeze(1)
        batch_reward_1 = torch.as_tensor(batch_reward_1, dtype=torch.float, device=device)
        batch_reward_2 = torch.as_tensor(batch_reward_2, dtype=torch.float, device=device)
        batch_mask_1 = torch.as_tensor(batch_mask_1, dtype=torch.float, device=device)
        batch_mask_2 = torch.as_tensor(batch_mask_2, dtype=torch.float, device=device)
        batch_pref_label = torch.as_tensor(np.array(batch_pref_label), dtype=torch.float, device=device).unsqueeze(1)

        return batch_state_1, batch_state_2, batch_action_1, batch_action_2, batch_reward_1, batch_reward_2, batch_mask_1, batch_mask_2, batch_pref_label
    
    def get_all(self, device):
        buffer_size = self.size()
        states_1 = np.asarray([self.buffer[i][0] for i in range(buffer_size)])
        states_2 = np.asarray([self.buffer[i][1] for i in range(buffer_size)])
        actions_1 = np.asarray([self.buffer[i][2] for i in range(buffer_size)])
        actions_2 = np.asarray([self.buffer[i][3] for i in range(buffer_size)])
        rewards_1 = np.asarray([self.buffer[i][4] for i in range(buffer_size)])
        rewards_2 = np.asarray([self.buffer[i][5] for i in range(buffer_size)])
        masks_1 = np.asarray([self.buffer[i][6] for i in range(buffer_size)])
        masks_2 = np.asarray([self.buffer[i][7] for i in range(buffer_size)])
        pref_labels = np.asarray([self.buffer[i][8] for i in range(buffer_size)])     

        states_1 = torch.as_tensor(states_1, dtype=torch.float, device=device)
        states_2 = torch.as_tensor(states_2, dtype=torch.float, device=device)
        actions_1 = torch.as_tensor(actions_1, dtype=torch.float, device=device)
        actions_2 = torch.as_tensor(actions_2, dtype=torch.float, device=device)
        if actions_1.ndim == 1:
            actions_1 = actions_1.unsqueeze(1)
        if actions_2.ndim == 1:
            actions_2 = actions_2.unsqueeze(1)
        rewards_1 = torch.as_tensor(rewards_1, dtype=torch.float, device=device)
        rewards_2 = torch.as_tensor(rewards_2, dtype=torch.float, device=device)
        masks_1 = torch.as_tensor(masks_1, dtype=torch.float, device=device)
        masks_2 = torch.as_tensor(masks_2, dtype=torch.float, device=device)
        pref_labels = torch.as_tensor(pref_labels, dtype=torch.float, device=device).unsqueeze(1)

        return states_1, states_2, actions_1, actions_2, rewards_1, rewards_2, masks_1, masks_2, pref_labels

    def get_all_np(self):
        buffer_size = self.size()
        states_1 = np.asarray([self.buffer[i][0] for i in range(buffer_size)])
        states_2 = np.asarray([self.buffer[i][1] for i in range(buffer_size)])
        actions_1 = np.asarray([self.buffer[i][2] for i in range(buffer_size)])
        actions_2 = np.asarray([self.buffer[i][3] for i in range(buffer_size)])
        rewards_1 = np.asarray([self.buffer[i][4] for i in range(buffer_size)])
        rewards_2 = np.asarray([self.buffer[i][5] for i in range(buffer_size)])
        masks_1 = np.asarray([self.buffer[i][6] for i in range(buffer_size)])
        masks_2 = np.asarray([self.buffer[i][7] for i in range(buffer_size)])
        pref_labels = np.asarray([self.buffer[i][8] for i in range(buffer_size)])     

        return states_1, states_2, actions_1, actions_2, rewards_1, rewards_2, masks_1, masks_2, pref_labels  

def optimal_transport_plan(X,
                           Y,
                           cost_matrix,
                           method='sinkhorn',
                           niter=500,
                           epsilon=0.01):
    X_pot = np.ones(X.shape[0]) * (1 / X.shape[0])
    Y_pot = np.ones(Y.shape[0]) * (1 / Y.shape[0])
    transport_plan = ot.sinkhorn(X_pot, Y_pot, cost_matrix, epsilon, numItermax=niter)
    return transport_plan


def cosine_distance(x, y):
    C = np.dot(x, y.T)
    x_norm = np.linalg.norm(x, ord=2, axis=1)
    y_norm = np.linalg.norm(y, ord=2, axis=1)
    x_n = np.expand_dims(x_norm, axis=1)
    y_n = np.expand_dims(y_norm, axis=1)
    norms = np.dot(x_n, y_n.T)
    C = (1 - C / norms)
    return C

def euclidean_distance(x, y):
    x_col = np.expand_dims(x, axis=1)
    y_lin = np.expand_dims(y, axis=0)
    c = np.sqrt(np.sum((np.abs(x_col - y_lin)) ** 2, 2))
    return c

def ot_distance(traj_1, traj_2):
    cost_matrix = cosine_distance(traj_1+EPS, traj_2+EPS)  # Get cost matrix for samples using critic network.
    transport_plan = optimal_transport_plan(traj_1, traj_2, cost_matrix, method='sinkhorn', niter=100).astype(np.float32)  # Getting optimal coupling
    distance = np.mean(np.diag(np.dot(transport_plan, cost_matrix.T)))
    return distance

def expert_distance(traj, expert_trajs, max_ep_len=1000):
    traj_len = len(traj)
    expert_trajs_expand = expert_trajs.reshape(-1, expert_trajs.shape[-2], expert_trajs.shape[-1])
    distance = np.array([ot_distance(traj, expert_trajs_expand[i]) * 1000 for i in range(expert_trajs_expand.shape[0])]).mean()
    return distance

def mse_distance(x, y):
    """
    Compute the MSE-based cost matrix between two sets of vectors.
    """
    x_col = np.expand_dims(x, axis=1)  # Shape: (N, 1, D)
    y_lin = np.expand_dims(y, axis=0)  # Shape: (1, M, D)
    c = np.mean((x_col - y_lin) ** 2, axis=2)  # Compute MSE along feature dimension
    return c

def ot_distance_mse(traj_1, traj_2):
    cost_matrix = mse_distance(traj_1 + EPS, traj_2 + EPS)  # Compute MSE cost matrix
    transport_plan = optimal_transport_plan(traj_1, traj_2, cost_matrix, method='sinkhorn', niter=100).astype(np.float32)
    distance = np.mean(np.diag(np.dot(transport_plan, cost_matrix.T)))
    return distance

def expert_distance_mse(traj, expert_trajs, max_ep_len=1000):
    traj_len = len(traj)
    expert_trajs_expand = expert_trajs.reshape(-1, expert_trajs.shape[-2], expert_trajs.shape[-1])
    distance = np.array([ot_distance_mse(traj, expert_trajs_expand[i]) * 1000 for i in range(expert_trajs_expand.shape[0])]).mean()
    return distance

def make_test_preference_dataset(file_path, max_ep_len, sample_num, segment_len):
    data = np.load(file_path)
    states_, actions_, rewards_, dones_ = data['state'], data['action'], data['reward'], data['done']

    states, actions, rewards = [], [], []
    start_idx, end_idx = 0, 0
    for i in range(int(1e6)): # 4.5e5 for walker
        if dones_[i] or (i - start_idx) == max_ep_len - 1:
            end_idx = i + 1
            states.append(states_[start_idx: end_idx])
            actions.append(actions_[start_idx: end_idx])
            rewards.append(rewards_[start_idx: end_idx])
            start_idx = end_idx
    
    traj_idxs = np.arange(len(states))
    acc = 0
    preference_dataset = PreferenceBuffer(memory_size=sample_num)
    while preference_dataset.size() < sample_num:
        traj_idx_1 = np.random.choice(traj_idxs)
        traj_idx_2 = np.random.choice(traj_idxs)
        
        states_1, actions_1, rewards_1 = states[traj_idx_1], actions[traj_idx_1], rewards[traj_idx_1]
        states_2, actions_2, rewards_2 = states[traj_idx_2], actions[traj_idx_2], rewards[traj_idx_2]

        seq_len_1 = states_1.shape[0]
        if seq_len_1 < segment_len:
            start_idx_1 = 0
            states_1 = np.concatenate([states_1, np.zeros((segment_len-seq_len_1, states[0].shape[-1]))])
            states_1[seq_len_1:] = states_1[-1]
            actions_1 = np.concatenate([actions_1, np.zeros((segment_len-seq_len_1, actions[0].shape[-1]))])
            actions_1[seq_len_1:] = actions_1[-1]
            rewards_1 = np.concatenate([rewards_1, np.zeros((segment_len-seq_len_1, ))])
        else:
            start_idx_1 = np.random.randint(seq_len_1 - segment_len + 1)
            states_1 = states_1[start_idx_1: start_idx_1 + segment_len]
            actions_1 = actions_1[start_idx_1: start_idx_1 + segment_len]
            rewards_1 = rewards_1[start_idx_1: start_idx_1 + segment_len]
        masks_1 = np.ones(segment_len)
        masks_1[seq_len_1:] *= 0

        seq_len_2 = states_2.shape[0]
        if seq_len_2 < segment_len:
            start_idx_2 = 0
            states_2 = np.concatenate([states_2, np.zeros((segment_len-seq_len_2, states[0].shape[-1]))])
            states_2[seq_len_2:] = states_2[-1]
            actions_2 = np.concatenate([actions_2, np.zeros((segment_len-seq_len_2, actions[0].shape[-1]))])
            actions_2[seq_len_2:] = actions_2[-1]
            rewards_2 = np.concatenate([rewards_2, np.zeros((segment_len-seq_len_2, ))])
        else:
            if start_idx_1 < seq_len_2 - segment_len:
                start_idx_2 = start_idx_1
            else:
                start_idx_2 = np.random.randint(seq_len_2 - segment_len + 1)
            states_2 = states_2[start_idx_2: start_idx_2 + segment_len]
            actions_2 = actions_2[start_idx_2: start_idx_2 + segment_len]
            rewards_2 = rewards_2[start_idx_2: start_idx_2 + segment_len]
        masks_2 = np.ones(segment_len)
        masks_2[seq_len_2:] *= 0

        return_1 = rewards_1.sum()
        return_2 = rewards_2.sum()
        if return_1 > return_2:
            pref_label = [1, 0]
        elif return_1 < return_2:
            pref_label = [0, 1]
        else:
            pref_label = [0.5, 0.5]
        preference_dataset.add((states_1, states_2, actions_1, actions_2, rewards_1, rewards_2, masks_1, masks_2, pref_label))
        acc += 1

    accuracy = acc / sample_num
    print('Accuracy: %.4f'%(accuracy))

    return preference_dataset, accuracy 


def epsilon_sample(file_path, expert_states, expert_actions, max_ep_len, sample_num, segment_len, sub_level=8, mode="train", task_type='baseline_drex'):
    preference_dataset = PreferenceBuffer(memory_size=sample_num)

    with open(file_path, 'rb') as f:
        trajs = pickle.load(f)
    D = []
    min_noise_margin = 0.5 # 0.3 from drex paper

    with tqdm(total=sample_num) as pbar:
        pbar.set_description('make preference dataset')

        acc = 0
        for _ in range(sample_num):
            # Pick Two Noise Level Set
            level1 = level2 = 0
            while abs(level1 - level2) < min_noise_margin:
                x_idx, y_idx = np.random.choice(len(trajs),2, replace=False)
                level1 = trajs[x_idx][0]
                level2 = trajs[y_idx][0]
            # x_idx = np.random.choice(len(trajs))
            # y_idx = np.random.choice([i for i in range(len(trajs)) if i - x_idx > sub_level or x_idx - i > sub_level], 1, replace=False)[0]

            # Pick trajectory from each set
            x_traj = trajs[x_idx][1][np.random.choice(len(trajs[x_idx][1]))]
            y_traj = trajs[y_idx][1][np.random.choice(len(trajs[y_idx][1]))]

            # Subsampling from a trajectory
            if len(x_traj[0]) > segment_len:
                ptr = np.random.randint(len(x_traj[0])-segment_len)
                x_slice = slice(ptr, ptr+segment_len)
            else:
                x_slice = slice(len(x_traj[1]))

            if len(y_traj[0]) > segment_len:
                ptr = np.random.randint(len(y_traj[0])-segment_len)
                y_slice = slice(ptr, ptr + segment_len)
            else:
                y_slice = slice(len(y_traj[0]))

            level1 = trajs[x_idx][0]
            level2 = trajs[y_idx][0]
            states_1 = x_traj[0][x_slice]
            states_2 = y_traj[0][y_slice]
            actions_1 = x_traj[1][x_slice]
            actions_2 = y_traj[1][y_slice]
            rewards_1 = x_traj[2][x_slice]
            rewards_2 = y_traj[2][y_slice]

            # expand to the same lenth
            seq_len_1 = len(x_traj[0])
            if seq_len_1 < segment_len:
                # start_idx_1 = 0
                states_1 = np.concatenate([states_1, states_1[-1] * np.ones((segment_len-seq_len_1, states_1.shape[-1]))])
                actions_1 = np.concatenate([actions_1, actions_1[-1] * np.ones((segment_len-seq_len_1, actions_1[0].shape[-1]))])
                rewards_1 = np.concatenate([rewards_1, np.zeros((segment_len-seq_len_1, ))])
            masks_1 = np.ones(segment_len)
            masks_1[seq_len_1:] *= 0

            seq_len_2 = len(y_traj[0])
            if seq_len_2 < segment_len:
                # start_idx_1 = 0
                states_2 = np.concatenate([states_2, states_2[-1] * np.ones((segment_len-seq_len_2, states_2.shape[-1]))])
                actions_2 = np.concatenate([actions_2, actions_2[-1] * np.ones((segment_len-seq_len_2, actions_2[0].shape[-1]))])
                rewards_2 = np.concatenate([rewards_2, np.zeros((segment_len-seq_len_2, ))])
            masks_2 = np.ones(segment_len)
            masks_2[seq_len_2:] *= 0

            # Done!
            returns_1 = rewards_1.sum()
            returns_2 = rewards_2.sum()
            if mode == 'train':
                pref_label = [1, 0] if level1 < level2 else [0, 1] # if noise level is small, then it is better traj.
                pref_noise = [level1, level2]
                if (returns_1 >= returns_2 and level1 <= level2) or (returns_1 <= returns_2 and level1 >= level2):
                    acc += 1
            else:
                pref_label = [1, 0] if returns_1 >= returns_2 else [0, 1] 
                if (x_idx < y_idx) == (returns_1 > returns_2):
                    acc += 1

            if task_type in ['baseline_lerp', 'baseline_ssrr', 'baseline_ssrr_irl', 'baseline_ssrr_bc']:
                pref_label = pref_noise
            else:
                pref_label = pref_noise

            if np.random.random() < 0.5:
                preference_dataset.add((states_1, states_2, actions_1, actions_2, rewards_1, rewards_2, masks_1, masks_2, pref_label))
            else:
                preference_dataset.add((states_2, states_1, actions_2, actions_1, rewards_2, rewards_1, masks_2, masks_1, 1 - np.array(pref_label)))
            pbar.update(1)
    
    accuracy = acc / sample_num
    print('Accuracy: %.4f'%(accuracy))

    return preference_dataset, accuracy

def collect_epsilon_dataset(file_paths, env, parameter, device, save_path):
    '''
        params:
            file_paths: list of file paths to saved models
            env: the environment object
            parameter: the parameters object containing configurations
            device: the device for computation (CPU/GPU)
            save_path: path to save the generated dataset
    '''
    min_length = 0
    num_trajs_per_eps = 10 # n_samples per epsilon
    
    '''load saved model'''
    policy_models = []
    for file_path in file_paths:
        distance = eval(file_path.split('_')[-1].replace('distance', '').replace('.pt', ''))
        epoch = eval(file_path.split('_')[-3].replace('epoch', ''))
        if epoch >= 50:
            continue
        policy_model = core.MLPActorCritic(
                                observation_space=env.observation_space, 
                                action_space=env.action_space,
                                k=parameter.sac_k,
                                hidden_sizes=parameter.ac_hidden_sizes,
                                device=device)
        policy_model.load_state_dict(torch.load(file_path))
        policy_models.append((epoch, distance, policy_model))

    '''epsilon sampling'''
    num_samples_total = 0
    noise_range_i = np.arange(0, 0.1, 0.1)
    for epoch, distance, model in policy_models:
        num_samples_total += len(noise_range_i) * num_trajs_per_eps
    
    with tqdm(total=num_samples_total) as pbar:
        pbar.set_description(f'Epsilon Sampling')
        
        trajs = []
        for epoch, distance, model in policy_models:
            for noise_level in noise_range_i:
                noise_policy = NoiseInjectedPolicy(
                                        env=env, 
                                        policy=model, 
                                        action_noise_type='epsilon', 
                                        noise_level=noise_level)
            
                agent_trajs = []

                assert (num_trajs_per_eps > 0 and min_length <= 0) or (min_length > 0 and num_trajs_per_eps <= 0)
                while len(agent_trajs) < num_trajs_per_eps:
                    obs, actions, rewards, dones = gen_traj(env,noise_policy, -1)
                    agent_trajs.append((obs, actions, rewards, dones))

                trajs.append((epoch, distance, noise_level, agent_trajs))
                pbar.update(num_trajs_per_eps)

    if not os.path.exists(os.path.abspath(os.path.dirname(save_path))):
        os.makedirs(os.path.abspath(os.path.dirname(save_path)))
    with open(save_path,'wb') as f:
        pickle.dump(trajs,f)

def collect_epsilon_dataset_for_baselines(expert_replay_buffer, parameter, device, gym_env, gail_file_path, save_path):
    if parameter.task_type in ['baseline_drex', 'baseline_lerp']:
        '''train BC'''
        expert_states = torch.from_numpy(expert_replay_buffer.states).to(device).float()
        expert_actions = torch.from_numpy(expert_replay_buffer.actions).to(device).float()
        expert_rewards = torch.from_numpy(expert_replay_buffer.rewards).to(device).float()
        expert_masks = torch.from_numpy(expert_replay_buffer.masks).to(device).float()
        policy_model = MLPBC(input_dim=expert_states.shape[-1],
                        hidden_size=256,
                        act_dim=expert_actions.shape[-1],
                        device=device,
                        env=gym_env
                        )
        optimizer = torch.optim.Adam(policy_model.parameters(), 
                                    lr=parameter.r_lr, weight_decay=parameter.r_weight_decay,
                                    betas=(parameter.r_momentum, 0.999))
        num_epochs = 2000
        with tqdm(total=num_epochs) as pbar:
            pbar.set_description('Training bc')
            for epoch in range(num_epochs):
                predict_actions = policy_model(expert_states)
                loss = ((predict_actions - expert_actions).sum(axis=-1) * expert_masks).square().mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix({'loss':loss.item()})
    elif parameter.task_type in ['baseline_gail', 'baseline_ssrr', 'baseline_ssrr_irl', 'baseline_ssrr_bc']: ## ssrr load model from airl directory
        '''load actor'''
        state_dict = torch.load(gail_file_path)
        policy_model = core.MLPActorCritic(observation_space=gym_env.observation_space, 
                                        action_space=gym_env.action_space,
                                        k=parameter.sac_k,
                                        hidden_sizes=parameter.ac_hidden_sizes,
                                        device=device)
        policy_model.load_state_dict(state_dict)

    '''epsilon sampling'''
    if parameter.task_type in ['baseline_lerp']:
        noise_range = np.arange(0.+ EPS, 1. + EPS, 0.05) # lerp assert noise > 0
    else:
        noise_range = np.arange(0., 1., 0.05) 
    
    # epsilon sampling from drex github
    min_length = 0
    num_trajs = 5 # from drex hyperparams
    with tqdm(total=noise_range.shape[-1]) as pbar:
        pbar.set_description(f'{parameter.task_type} epsilon sampling')
        
        trajs = []

        for noise_level in noise_range:
            noise_policy = NoiseInjectedPolicy(env=gym_env, 
                            policy=policy_model, 
                            action_noise_type='epsilon', 
                            noise_level=noise_level)
            agent_trajs = []

            assert (num_trajs > 0 and min_length <= 0) or (min_length > 0 and num_trajs <= 0)
            while (min_length > 0 and np.sum([len(obs) for obs,_,_,_ in agent_trajs])  < min_length) or\
                    (num_trajs > 0 and len(agent_trajs) < num_trajs):
                obs, actions, rewards, dones = gen_traj(gym_env,noise_policy,-1)
                # print(rewards.sum())
                agent_trajs.append((obs, actions, rewards, dones))

            trajs.append((noise_level, agent_trajs))

            pbar.update(1)
            
    if not os.path.exists(os.path.abspath(os.path.dirname(save_path))):
        os.makedirs(os.path.abspath(os.path.dirname(save_path)))        
    with open(save_path,'wb') as f:
        pickle.dump(trajs,f)


def make_preference_dataset(file_path, expert_states, expert_actions, max_ep_len, sample_num, segment_len, sub_level=8, mode="train"):
    preference_dataset = PreferenceBuffer(memory_size=sample_num)

    with open(file_path, 'rb') as f:
        trajs_ = pickle.load(f)

    expert_states = expert_states.squeeze()

    trajs = []
    expert_distances = []
    with tqdm(total=len(trajs_)) as pbar:
        pbar.set_description('preparing expert distance')

        for epoch, distance, noise, agent_trajs in trajs_:
            trajs.append((epoch, distance, noise, agent_trajs))
            # expert_distances.append((distance, noise, [expert_distance(agent_traj[0], expert_states) for agent_traj in agent_trajs]))
            pbar.update(1)

    noises = np.array([traj[2] for traj in trajs])
    epochs = np.array([traj[0] for traj in trajs])

    num_trajs_total = len(trajs)
    num_trajs_per_eps = len(trajs[0][-1])

    with tqdm(total=sample_num) as pbar:
        pbar.set_description('make preference dataset')

        acc = 0
        while preference_dataset.size() < sample_num:
            ''' select distance levels '''
            
            if np.random.random() > 0.2:
                idx_a = np.random.choice(num_trajs_total - 20)
                noise_a = noises[idx_a]
            else:
                available_idx = np.where(noises == 0)[0][:-1]
                idx_a = np.random.choice(available_idx)
                noise_a = 0

            epoch_filter = np.where(epochs > epochs[idx_a] + 10)
            if np.random.random() > 0.5:
                noise_filter = np.where(noises <= noise_a)[0]
            else:
                noise_filter = np.where(noises == 0)[0]
            try:
                idx_b = np.random.choice(np.intersect1d(epoch_filter, noise_filter))
            except:
                continue

            # Pick trajectory from each set
            jdx_a, jdx_b = np.random.choice(num_trajs_per_eps), np.random.choice(num_trajs_per_eps)
            x_traj = trajs[idx_a][-1][jdx_a]
            y_traj = trajs[idx_b][-1][jdx_b]
            states_1, actions_1, rewards_1, dones_1 = x_traj
            states_2, actions_2, rewards_2, dones_2 = y_traj
            total_distance_1 = expert_distance(states_1, expert_states)
            total_distance_2 = expert_distance(states_2, expert_states)
            # total_distance_1 = expert_distances[idx_a][2][jdx_a]
            # total_distance_2 = expert_distances[idx_b][2][jdx_b]
            if mode == "train" and (total_distance_1 - 0.1 < total_distance_2):
                continue
            
            '''padding'''
            seq_len_1 = actions_1.shape[0]
            if seq_len_1 < segment_len:
                start_idx_1 = 0
                states_1 = np.concatenate([states_1, np.zeros((segment_len-seq_len_1, states_1[0].shape[-1]))])
                states_1[seq_len_1:] = states_1[-1]
                actions_1 = np.concatenate([actions_1, np.zeros((segment_len-seq_len_1, actions_1[0].shape[-1]))])
                actions_1[seq_len_1:] = actions_1[-1]
                rewards_1 = np.concatenate([rewards_1, np.zeros((segment_len-seq_len_1, ))])
            else:
                start_idx_1 = np.random.randint(seq_len_1 - segment_len + 1)
                states_1 = states_1[start_idx_1: start_idx_1 + segment_len]
                actions_1 = actions_1[start_idx_1: start_idx_1 + segment_len]
                rewards_1 = rewards_1[start_idx_1: start_idx_1 + segment_len]
            masks_1 = np.ones(segment_len)
            masks_1[seq_len_1:] *= 0

            seq_len_2 = actions_2.shape[0]
            if seq_len_2 < segment_len:
                start_idx_2 = 0
                states_2 = np.concatenate([states_2, np.zeros((segment_len-seq_len_2, states_2[0].shape[-1]))])
                states_2[seq_len_2:] = states_2[-1]
                actions_2 = np.concatenate([actions_2, np.zeros((segment_len-seq_len_2, actions_2[0].shape[-1]))])
                actions_2[seq_len_2:] = actions_2[-1]
                rewards_2 = np.concatenate([rewards_2, np.zeros((segment_len-seq_len_2, ))])
            else:
                start_idx_2 = np.random.randint(seq_len_2 - segment_len + 1)
                states_2 = states_2[start_idx_2: start_idx_2 + segment_len]
                actions_2 = actions_2[start_idx_2: start_idx_2 + segment_len]
                rewards_2 = rewards_2[start_idx_2: start_idx_2 + segment_len]
            masks_2 = np.ones(segment_len)
            masks_2[seq_len_2:] *= 0

            label = [0, 1]
            if np.random.random() < 0.5:
                preference_dataset.add((states_1, states_2, actions_1, actions_2, rewards_1, rewards_2, masks_1, masks_2, label))
            else:
                preference_dataset.add((states_2, states_1, actions_2, actions_1, rewards_2, rewards_1, masks_2, masks_1, 1 - np.array(label)))

            if rewards_1.sum() < rewards_2.sum():
                acc += 1

            pbar.update(1)
    
    accuracy = acc / sample_num
    print('Accuracy: %.4f'%(accuracy))

    return preference_dataset, accuracy

def make_preference_dataset_no_noise(file_path, expert_states, expert_actions, max_ep_len, sample_num, 
                                     segment_len, sub_level=8, mode="train", num_epochs=200, 
                                     filter_distance=True, distance_threshold=0.1, filter_epoch=True, epoch_threshold=10, 
                                     distance_metric='ot'):
    '''
    filter_distance: sample based on w_distance between traj with demo
    filter_epoch : sample traj base on epoch, assure epoch b > epoch a
    '''
    assert mode == 'train' # test is not implemented

    if distance_metric == 'ot':
        f_dist = expert_distance
    elif distance_metric == 'mse':
        f_dist = expert_distance_mse

    preference_dataset = PreferenceBuffer(memory_size=sample_num)

    with open(file_path, 'rb') as f:
        trajs_ = pickle.load(f)

    expert_states = expert_states.squeeze()

    trajs = []

    # default: 全部采样
    epoch_max = 200 # need to change if epoch_total changed
    epoch_interval = epoch_max // num_epochs # 每隔多少epoch采样 
    epoch_available = np.arange(0, epoch_max, epoch_interval)

    expert_distances = []
    with tqdm(total=len(trajs_)) as pbar:
        pbar.set_description('preparing expert distance')

        for epoch, distance, noise, agent_trajs in trajs_:
            if epoch in epoch_available:
                trajs.append((epoch, distance, noise, agent_trajs))
                
                # if filter_distance:
                #     if noise == 0:
                #         expert_distances.append((epoch, distance, noise, [expert_distance(agent_traj[0], expert_states) for agent_traj in agent_trajs]))
                #     else:
                #         expert_distances.append((epoch, distance, noise, []))           

                pbar.update(1)

    noises = np.array([traj[2] for traj in trajs])
    epochs = np.array([traj[0] for traj in trajs])

    num_trajs_total = len(trajs)
    num_trajs_per_eps = len(trajs[0][-1])

    noise_filter = np.where(noises == 0)[0] # select only noise == 0

    with tqdm(total=sample_num) as pbar:
        pbar.set_description('make preference dataset')

        acc = 0
        while preference_dataset.size() < sample_num:

            available_idx = np.where(noises == 0)[0][:-1]
            idx_a = np.random.choice(available_idx)

            if filter_epoch:
                epoch_filter = np.where(epochs > epochs[idx_a] + epoch_threshold)
            else:
                epoch_filter = np.where(epochs > -1) # select all epoch

            try:
                idx_b = np.random.choice(np.intersect1d(epoch_filter, noise_filter))
            except:
                continue

            # Pick trajectory from each set
            jdx_a, jdx_b = np.random.choice(num_trajs_per_eps), np.random.choice(num_trajs_per_eps)
            x_traj = trajs[idx_a][-1][jdx_a]
            y_traj = trajs[idx_b][-1][jdx_b]
            states_1, actions_1, rewards_1, dones_1 = x_traj
            states_2, actions_2, rewards_2, dones_2 = y_traj

            if filter_distance:
                total_distance_1 = f_dist(states_1, expert_states)
                total_distance_2 = f_dist(states_2, expert_states)
                # total_distance_1 = expert_distances[idx_a][-1][jdx_a]
                # total_distance_2 = expert_distances[idx_b][-1][jdx_b]
                if mode == "train" and (total_distance_1 - distance_threshold < total_distance_2):
                    continue          

            '''padding'''
            seq_len_1 = actions_1.shape[0]
            if seq_len_1 < segment_len:
                start_idx_1 = 0
                states_1 = np.concatenate([states_1, np.zeros((segment_len-seq_len_1, states_1[0].shape[-1]))])
                states_1[seq_len_1:] = states_1[-1]
                actions_1 = np.concatenate([actions_1, np.zeros((segment_len-seq_len_1, actions_1[0].shape[-1]))])
                actions_1[seq_len_1:] = actions_1[-1]
                rewards_1 = np.concatenate([rewards_1, np.zeros((segment_len-seq_len_1, ))])
            else:
                start_idx_1 = np.random.randint(seq_len_1 - segment_len + 1)
                states_1 = states_1[start_idx_1: start_idx_1 + segment_len]
                actions_1 = actions_1[start_idx_1: start_idx_1 + segment_len]
                rewards_1 = rewards_1[start_idx_1: start_idx_1 + segment_len]
            masks_1 = np.ones(segment_len)
            masks_1[seq_len_1:] *= 0

            seq_len_2 = actions_2.shape[0]
            if seq_len_2 < segment_len:
                start_idx_2 = 0
                states_2 = np.concatenate([states_2, np.zeros((segment_len-seq_len_2, states_2[0].shape[-1]))])
                states_2[seq_len_2:] = states_2[-1]
                actions_2 = np.concatenate([actions_2, np.zeros((segment_len-seq_len_2, actions_2[0].shape[-1]))])
                actions_2[seq_len_2:] = actions_2[-1]
                rewards_2 = np.concatenate([rewards_2, np.zeros((segment_len-seq_len_2, ))])
            else:
                start_idx_2 = np.random.randint(seq_len_2 - segment_len + 1)
                states_2 = states_2[start_idx_2: start_idx_2 + segment_len]
                actions_2 = actions_2[start_idx_2: start_idx_2 + segment_len]
                rewards_2 = rewards_2[start_idx_2: start_idx_2 + segment_len]
            masks_2 = np.ones(segment_len)
            masks_2[seq_len_2:] *= 0

            label = [0, 1]
            if np.random.random() < 0.5:
                preference_dataset.add((states_1, states_2, actions_1, actions_2, rewards_1, rewards_2, masks_1, masks_2, label))
            else:
                preference_dataset.add((states_2, states_1, actions_2, actions_1, rewards_2, rewards_1, masks_2, masks_1, 1 - np.array(label)))

            if rewards_1.sum() < rewards_2.sum():
                acc += 1

            pbar.update(1)
    
    accuracy = acc / sample_num
    print('Accuracy: %.4f'%(accuracy))

    return preference_dataset, accuracy


def make_preference_dataset_no_noise_relative(file_path, expert_states, expert_actions, max_ep_len, sample_num, segment_len, sub_level=8, mode="train", num_epochs=50, filter_distance=True, distance_threshold=0.1, filter_epoch=True, epoch_threshold=10):
    '''
    filter_distance: sample based on w_distance between traj with demo
    filter_epoch : sample traj base on epoch, assure epoch b > epoch a
    '''
    assert mode == 'train' # test is not implemented
    preference_dataset = PreferenceBuffer(memory_size=sample_num)

    with open(file_path, 'rb') as f:
        trajs_ = pickle.load(f)

    expert_states = expert_states.squeeze()

    trajs = []
    epoch_max = 50 # need to change if epoch_total changed
    epoch_interval = epoch_max // num_epochs 
    epoch_available = np.arange(0, epoch_max, epoch_interval)

    expert_distances = []
    with tqdm(total=len(trajs_)) as pbar:
        pbar.set_description('preparing expert distance')

        for epoch, distance, noise, agent_trajs in trajs_:
            if epoch in epoch_available:
                trajs.append((epoch, distance, noise, agent_trajs))
                
                # if filter_distance:
                #     if noise == 0:
                #         expert_distances.append((epoch, distance, noise, [expert_distance(agent_traj[0], expert_states) for agent_traj in agent_trajs]))
                #     else:
                #         expert_distances.append((epoch, distance, noise, []))           

                pbar.update(1)

    noises = np.array([traj[2] for traj in trajs])
    epochs = np.array([traj[0] for traj in trajs])

    num_trajs_total = len(trajs)
    num_trajs_per_eps = len(trajs[0][-1])

    noise_filter = np.where(noises == 0)[0] # select only noise == 0

    with tqdm(total=sample_num) as pbar:
        pbar.set_description('make preference dataset')

        acc = 0
        while preference_dataset.size() < sample_num:

            available_idx = np.where(noises == 0)[0][:-1]
            idx_a = np.random.choice(available_idx)

            if filter_epoch:
                epoch_filter = np.where(epochs > epochs[idx_a] + epoch_threshold)
            else:
                epoch_filter = np.where(epochs > -1) # select all epoch

            try:
                idx_b = np.random.choice(np.intersect1d(epoch_filter, noise_filter))
            except:
                continue

            # Pick trajectory from each set
            jdx_a, jdx_b = np.random.choice(num_trajs_per_eps), np.random.choice(num_trajs_per_eps)
            x_traj = trajs[idx_a][-1][jdx_a]
            y_traj = trajs[idx_b][-1][jdx_b]
            states_1, actions_1, rewards_1, dones_1 = x_traj
            states_2, actions_2, rewards_2, dones_2 = y_traj

            if filter_distance:
                distance = expert_distance(states_1, states_2)
                if mode == "train" and distance < 0.1:
                    continue          

            '''padding'''
            seq_len_1 = actions_1.shape[0]
            if seq_len_1 < segment_len:
                start_idx_1 = 0
                states_1 = np.concatenate([states_1, np.zeros((segment_len-seq_len_1, states_1[0].shape[-1]))])
                states_1[seq_len_1:] = states_1[-1]
                actions_1 = np.concatenate([actions_1, np.zeros((segment_len-seq_len_1, actions_1[0].shape[-1]))])
                actions_1[seq_len_1:] = actions_1[-1]
                rewards_1 = np.concatenate([rewards_1, np.zeros((segment_len-seq_len_1, ))])
            else:
                start_idx_1 = np.random.randint(seq_len_1 - segment_len + 1)
                states_1 = states_1[start_idx_1: start_idx_1 + segment_len]
                actions_1 = actions_1[start_idx_1: start_idx_1 + segment_len]
                rewards_1 = rewards_1[start_idx_1: start_idx_1 + segment_len]
            masks_1 = np.ones(segment_len)
            masks_1[seq_len_1:] *= 0

            seq_len_2 = actions_2.shape[0]
            if seq_len_2 < segment_len:
                start_idx_2 = 0
                states_2 = np.concatenate([states_2, np.zeros((segment_len-seq_len_2, states_2[0].shape[-1]))])
                states_2[seq_len_2:] = states_2[-1]
                actions_2 = np.concatenate([actions_2, np.zeros((segment_len-seq_len_2, actions_2[0].shape[-1]))])
                actions_2[seq_len_2:] = actions_2[-1]
                rewards_2 = np.concatenate([rewards_2, np.zeros((segment_len-seq_len_2, ))])
            else:
                start_idx_2 = np.random.randint(seq_len_2 - segment_len + 1)
                states_2 = states_2[start_idx_2: start_idx_2 + segment_len]
                actions_2 = actions_2[start_idx_2: start_idx_2 + segment_len]
                rewards_2 = rewards_2[start_idx_2: start_idx_2 + segment_len]
            masks_2 = np.ones(segment_len)
            masks_2[seq_len_2:] *= 0

            label = [0, 1]
            if np.random.random() < 0.5:
                preference_dataset.add((states_1, states_2, actions_1, actions_2, rewards_1, rewards_2, masks_1, masks_2, label))
            else:
                preference_dataset.add((states_2, states_1, actions_2, actions_1, rewards_2, rewards_1, masks_2, masks_1, 1 - np.array(label)))

            if rewards_1.sum() < rewards_2.sum():
                acc += 1

            pbar.update(1)
    
    accuracy = acc / sample_num
    print('Accuracy: %.4f'%(accuracy))

    return preference_dataset, accuracy
