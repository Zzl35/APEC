from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import cv2
import sys
from collections import deque
import os
import ot
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import hydra
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'
from pathlib import Path

sys.path.append('..')
sys.path.append('.')


EPS = 1e-4


def optimal_transport_plan(X, Y, cost_matrix, method='sinkhorn', niter=500, epsilon=0.01):
    X_pot = np.ones(X.shape[0]) * (1 / X.shape[0])
    Y_pot = np.ones(Y.shape[0]) * (1 / Y.shape[0])
    transport_plan = ot.sinkhorn(X_pot, Y_pot, cost_matrix, epsilon, numItermax=niter)
    return transport_plan

def cosine_distance(x, y):
    ''' Compute the cosine distance between two sets of points (x, y) '''
    C = np.dot(x, y.T)
    x_norm = np.linalg.norm(x, ord=2, axis=1)
    y_norm = np.linalg.norm(y, ord=2, axis=1)
    x_n = np.expand_dims(x_norm, axis=1)
    y_n = np.expand_dims(y_norm, axis=1)
    norms = np.dot(x_n, y_n.T)
    C = (1 - C / norms)
    return C

def reshape_traj(traj):
    ''' Reshape trajectory to a 2D array if it is pixel-based (seqlen, channel, h, w) '''
    if len(traj.shape) == 4:  # Pixel-based data: (seqlen, channel, h, w)
        traj_flattened = traj.reshape(traj.shape[0], -1)  # Flatten the spatial dimensions (channel * h * w)
    else:  # Feature-based data: (seqlen, feature_dim)
        traj_flattened = traj
    return traj_flattened

def ot_distance(traj_1, traj_2):
    ''' Compute the optimal transport distance between two trajectories '''
    # Reshape trajectories if needed
    traj_1_flat = reshape_traj(traj_1 + EPS)
    traj_2_flat = reshape_traj(traj_2 + EPS)

    # Get the cost matrix based on cosine distance
    cost_matrix = cosine_distance(traj_1_flat, traj_2_flat)

    # Get the optimal transport plan using Sinkhorn's method
    transport_plan = optimal_transport_plan(traj_1_flat, traj_2_flat, cost_matrix, method='sinkhorn', niter=100).astype(np.float32)

    # Calculate and return the transport distance (mean of diagonal)
    distance = np.mean(np.diag(np.dot(transport_plan, cost_matrix.T)))
    return distance

def expert_distance(traj, expert_trajs, max_ep_len=1000):
    ''' Compute the distance between a trajectory and a set of expert trajectories '''
    # expert shape : n_samples * ...
    # Ensure expert trajectories are reshaped correctly (for pixel-based or feature-based)
    # expert_trajs_expand = expert_trajs.reshape(-1, expert_trajs.shape[-2], expert_trajs.shape[-1])

    # Calculate distances for each expert trajectory and return the mean distance
    distance = np.array([ot_distance(traj, expert_trajs[i]) * 1000 for i in range(expert_trajs.shape[0])]).mean()
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
    ''' Compute the optimal transport distance between two trajectories '''
    # Reshape trajectories if needed
    traj_1_flat = reshape_traj(traj_1 + EPS)
    traj_2_flat = reshape_traj(traj_2 + EPS)

    # Get the cost matrix based on cosine distance
    cost_matrix = mse_distance(traj_1_flat, traj_2_flat)

    # Get the optimal transport plan using Sinkhorn's method
    transport_plan = optimal_transport_plan(traj_1_flat, traj_2_flat, cost_matrix, method='sinkhorn', niter=100).astype(np.float32)

    # Calculate and return the transport distance (mean of diagonal)
    distance = np.mean(np.diag(np.dot(transport_plan, cost_matrix.T)))
    return distance

def expert_distance_mse(traj, expert_trajs, max_ep_len=1000):
    ''' Compute the distance between a trajectory and a set of expert trajectories '''
    # expert shape : n_samples * ...
    # Ensure expert trajectories are reshaped correctly (for pixel-based or feature-based)
    # expert_trajs_expand = expert_trajs.reshape(-1, expert_trajs.shape[-2], expert_trajs.shape[-1])

    # Calculate distances for each expert trajectory and return the mean distance
    distance = np.array([ot_distance_mse(traj, expert_trajs[i]) * 1000 for i in range(expert_trajs.shape[0])]).mean()
    return distance


def log_video(s1, s2, filepath):
    frames = np.concatenate([s1, s2], axis=-1).astype(np.uint8)[:, :3]

    frames = frames.transpose(0, 2, 3, 1)  
    n_frames, height, width, channels = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

    # 将每一帧写入视频文件
    for frame in frames:
        out.write(frame)  # frame 是一个 (height, width, channels) 的 3D 数组

    # 释放 VideoWriter 对象
    out.release()

    print(f"Video saved as {filepath}")


class PreferenceBuffer(object):
    def __init__(self, traj_path: str, idx_path: str, segment_len: int=500, device='cpu') -> None:
        self.traj_path = traj_path
        if idx_path is not None:
            with open(idx_path, 'rb') as f:
                self.indexes = pickle.load(f)
        else:
            self.indexes = []
        self.device = device
        self.segment_len = segment_len

    def __getitem__(self, index):
        idx_a, idx_b, jdx_a, jdx_b, pref_label = self.indexes[index]
        traj_1 = f'{self.traj_path}/{idx_a}/{jdx_a}.pkl'
        traj_2 = f'{self.traj_path}/{idx_b}/{jdx_b}.pkl'
        with open(traj_1, 'rb') as f:
            x_traj = pickle.load(f)
        with open(traj_2, 'rb') as f:
            y_traj = pickle.load(f)
        states_1, obs_1, actions_1, rewards_1 = x_traj
        states_2, obs_2, actions_2, rewards_2 = y_traj

        masks_1 = np.ones_like(rewards_1) 
        masks_2 = np.ones_like(rewards_2)  

        seq_len = states_1.shape[0]
        if self.segment_len < seq_len:
            start_idx = np.random.randint(seq_len - self.segment_len)
            states_1 = states_1[start_idx: start_idx+self.segment_len]
            states_2 = states_2[start_idx: start_idx+self.segment_len]
            actions_1 = actions_1[start_idx: start_idx+self.segment_len]
            actions_2 = actions_2[start_idx: start_idx+self.segment_len]
            rewards_1 = rewards_1[start_idx: start_idx+self.segment_len]
            rewards_2 = rewards_2[start_idx: start_idx+self.segment_len]
            masks_1 = masks_1[start_idx: start_idx+self.segment_len]
            masks_2 = masks_2[start_idx: start_idx+self.segment_len]
            pref_label = pref_label

        return states_1.astype(np.float32), states_2.astype(np.float32), \
                actions_1.astype(np.float32), actions_2.astype(np.float32), \
                rewards_1.astype(np.float32), rewards_2.astype(np.float32), \
                masks_1.astype(np.float32), masks_2.astype(np.float32), np.array(pref_label).astype(np.float32)

    def __len__(self):
        return len(self.indexes)

    def size(self):
        return self.__len__()
    
    def add(self, sample):
        self.indexes.append(sample)

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.indexes, f)

class PreferenceBufferFeature(object):
    def __init__(self, traj_path: str, idx_path: str, segment_len: int=500, device='cpu') -> None:
        self.traj_path = traj_path
        if idx_path is not None:
            with open(idx_path, 'rb') as f:
                self.indexes = pickle.load(f)
        else:
            self.indexes = []
        self.device = device
        self.segment_len = segment_len

    def __getitem__(self, index):
        idx_a, idx_b, jdx_a, jdx_b, pref_label = self.indexes[index]
        traj_1 = f'{self.traj_path}/{idx_a}/{jdx_a}.pkl'
        traj_2 = f'{self.traj_path}/{idx_b}/{jdx_b}.pkl'
        with open(traj_1, 'rb') as f:
            x_traj = pickle.load(f)
        with open(traj_2, 'rb') as f:
            y_traj = pickle.load(f)
        states_1, obs_1, actions_1, rewards_1 = x_traj
        states_2, obs_2, actions_2, rewards_2 = y_traj

        masks_1 = np.ones_like(rewards_1) 
        masks_2 = np.ones_like(rewards_2)  

        seq_len = obs_1.shape[0]
        if self.segment_len < seq_len:
            start_idx = np.random.randint(seq_len - self.segment_len)
            obs_1 = obs_1[start_idx: start_idx+self.segment_len]
            obs_2 = obs_2[start_idx: start_idx+self.segment_len]
            actions_1 = actions_1[start_idx: start_idx+self.segment_len]
            actions_2 = actions_2[start_idx: start_idx+self.segment_len]
            rewards_1 = rewards_1[start_idx: start_idx+self.segment_len]
            rewards_2 = rewards_2[start_idx: start_idx+self.segment_len]
            masks_1 = masks_1[start_idx: start_idx+self.segment_len]
            masks_2 = masks_2[start_idx: start_idx+self.segment_len]
            pref_label = pref_label

        return obs_1.astype(np.float32), obs_2.astype(np.float32), \
                actions_1.astype(np.float32), actions_2.astype(np.float32), \
                rewards_1.astype(np.float32), rewards_2.astype(np.float32), \
                masks_1.astype(np.float32), masks_2.astype(np.float32), np.array(pref_label).astype(np.float32)

    def __len__(self):
        return len(self.indexes)

    def size(self):
        return self.__len__()
    
    def add(self, sample):
        self.indexes.append(sample)

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.indexes, f)


def pad_and_slice(states, actions, rewards, segment_len):
    # deal with pixel-based and feature-based data
    seq_len = states.shape[0]

    shapes = list(states.shape)
    states_padded = np.zeros((segment_len, *shapes[1:]), dtype=np.uint8)
    actions_padded = np.zeros((segment_len, actions.shape[1]))
    rewards_padded = np.zeros(segment_len)

    if seq_len < segment_len:
        states_padded[:seq_len] = states
        actions_padded[:seq_len] = actions
        rewards_padded[:seq_len] = rewards
        states_padded[seq_len:] = states[-1]
        actions_padded[seq_len:] = actions[-1]
        rewards_padded[seq_len:] = rewards[-1]
    else :
        start_idx = np.random.randint(seq_len - segment_len + 1)
        states_padded = states[start_idx:start_idx + segment_len]
        actions_padded = actions[start_idx:start_idx + segment_len]
        rewards_padded = rewards[start_idx:start_idx + segment_len]

    masks = np.ones(segment_len)
    masks[seq_len:] = 0

    return states_padded, actions_padded, rewards_padded, masks


def plot_distance_and_return(trajs, distances, path):
    d = []
    for i, distance in enumerate(distances):
        d.append(np.mean(distance[1]))
    r = []
    for i, traj in enumerate(trajs):
        traj = traj[1]
        r.append(np.mean([[np.sum(x[3])] for x in traj]))

    r = np.array(r)
    d = np.array(d)
    r =(r-r.min()) / (r.max()-r.min())
    d =(d-d.min()) / (d.max()-d.min())

    plt.plot(np.arange(len(r)), r, c='blue')
    plt.plot(np.arange(len(d)), d, c='red')

    plt.savefig(path)


def make_preference_dataset(file_path, expert_states, sample_num, ckpt_num=50, mode="train", distance_threshold=0.1, epoch_threshold=0, obs_type='pixels', distance_type='w'):
    preference_dataset = PreferenceBuffer(traj_path=None, idx_path=None)
    expert_states = expert_states.squeeze()

    trajs = []
    for i in range(len(os.listdir(file_path))):
        trajs_ = []
        if i == 50:
            break
        for j in os.listdir(f'{file_path}/{i}'):
            with open(f'{file_path}/{i}/{j}', 'rb') as f:
                trajs_.append((int(j.split('.')[0]), pickle.load(f)))
        trajs.append(trajs_)

    expert_distances, distance_loaded = [], False
    file_path_list = file_path.split('/')
    post_fix = '' if distance_type == 'w' else f'_{distance_type}'
    if distance_type == 'w':
        f_dist = expert_distance 
    elif distance_type == 'mse':
        f_dist = expert_distance_mse
    else:
        print(f'distance type {distance_type} not implemented')
        raise NotImplementedError 
    distance_path = '/'.join(file_path_list[:-1]) + '/' + f'expert_distance_{file_path_list[-1]}{post_fix}.pkl'
    if os.path.exists(distance_path):
        with open(distance_path, 'rb') as f:
            expert_distances = pickle.load(f)
            distance_loaded = True
            print(f'load prebuild expert distance file at {distance_path}')
    else:
        print(f'Not found prebuild distance file at {distance_path}')
        expert_distances = []
        with tqdm(total=len(trajs)*len(trajs[0])) as pbar:
            pbar.set_description('preparing expert distance')
            for i, trajs_ in enumerate(trajs):
                if obs_type == 'pixels':
                    expert_distances.append((i * 20000, [f_dist(traj[1][0], expert_states) for traj in trajs_]))
                else:
                    expert_distances.append((i * 20000, [f_dist(traj[1][1], expert_states) for traj in trajs_]))
                pbar.update(len(trajs_))

    if not distance_loaded:
        print(f'distance file saved at {distance_path}')
        with open(distance_path, 'wb') as f:
            pickle.dump(expert_distances, f)

    num_trajs_per_eps = len(trajs[0])
    os.makedirs('outputs', exist_ok=True)
    available_ckpt = np.linspace(0, len(trajs)-1, ckpt_num).astype(int)
    with tqdm(total=sample_num) as pbar:
        pbar.set_description('make preference dataset')

        acc = 0
        while preference_dataset.size() < sample_num:

            idx_a = np.random.choice(available_ckpt[:-1])
            try:
                idx_b = np.random.choice([ckpt for ckpt in available_ckpt if ckpt > idx_a + epoch_threshold])
            except:
                continue

            # Pick trajectory from each set
            jdx_a, jdx_b = np.random.choice(num_trajs_per_eps), np.random.choice(num_trajs_per_eps)
            # print(idx_a, jdx_a, idx_b, jdx_b, available_ckpt[:-1], len(trajs[idx_a][-1]), len(trajs[idx_b][-1]))
            x_traj = trajs[idx_a][jdx_a]
            y_traj = trajs[idx_b][jdx_b]
            obs_1, states_1, actions_1, rewards_1 = x_traj[1]
            obs_2, states_2, actions_2, rewards_2 = y_traj[1]

            states_1 = obs_1 if obs_type == 'pixels' else states_1
            states_2 = obs_2 if obs_type == 'pixels' else states_2

            total_distance_1 = expert_distances[idx_a][-1][jdx_a]
            total_distance_2 = expert_distances[idx_b][-1][jdx_b]
            if mode != "test" and (total_distance_1 - distance_threshold < total_distance_2) and distance_threshold >= 0:
                continue         
            
            # padding 

            if mode != "test":
                label = [0, 1]
            else:
                label = [0, 1] if rewards_1.sum() < rewards_2.sum() else [1, 0]
            if np.random.random() < 0.5:
                new_sample = (idx_a, idx_b, x_traj[0], y_traj[0], label)
            else:
                new_sample = (idx_b, idx_a, y_traj[0], x_traj[0], 1 - np.array(label))

            preference_dataset.add(new_sample)

            if (rewards_1.sum() < rewards_2.sum()) == (label[0] < label[1]):
                acc += 1

            print(f'sample: {preference_dataset.size()} / {sample_num}')
            pbar.update(1)
    
    accuracy = acc / sample_num
    print('Accuracy: %.4f'%(accuracy))

    return preference_dataset, accuracy

@hydra.main(config_path='../cfgs', config_name='config_data')
def main(cfg):  
    root_dir = Path.cwd()
    mode=cfg.sample_mode  
    
    with open(cfg.expert_dataset, 'rb') as f:
        if cfg.obs_type == 'pixels':
            expert_states, _, _, _ = pickle.load(f)
        elif cfg.obs_type == 'features':
            _, expert_states, _, _ = pickle.load(f)
    
    buffer_path = cfg.buffer_path
    if mode == 'train':
        save_buffer_path = cfg.train_path
    elif mode =="valid":
        save_buffer_path = cfg.valid_path
    else:
        save_buffer_path = cfg.test_path
    train_buffer, acc = make_preference_dataset(file_path=buffer_path,
                                                expert_states=expert_states,
                                                sample_num=cfg.sample_num,
                                                ckpt_num=cfg.ckpt_num,
                                                mode=mode,
                                                distance_threshold=cfg.distance_threshold,
                                                epoch_threshold=int(cfg.epoch_threshold),
                                                obs_type=cfg.obs_type,
                                                distance_type=cfg.distance_type)
    
    train_buffer.save(save_buffer_path)
    print(f'Accuracy: {acc}, buffer saving at {save_buffer_path}')


if __name__ == '__main__':
    main()
