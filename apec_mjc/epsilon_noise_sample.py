'''
f-IRL: Extract policy/reward from specified expert samples
'''
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import gym

from parameter.Parameter import Parameter


import envs
from utils import system, collect, eval
from common_config.load_config import init_smart_logger
from smart_logger.util_logger.logger import Logger

from datetime import datetime
import pickle
import os


from tqdm import tqdm
from utils.replay_buffer import ExpertReplayBuffer
from utils.epsilon_utils import NoiseInjectedPolicy, gen_traj, MLPBC
import rl_alg.sac_agent as core


ENV2ENV_NAME = {"HopperFH-v0":"Hopper-v2", "HalfCheetahFH-v0":"HalfCheetah-v2", "Walker2dFH-v0":"Walker2d-v2",
                "AntFH-v0":"Ant-v2", "HumanoidFH-v0":"Humanoid-v2", 'dmc_quadruped_walk-v0':'quadruped_walk',
                'dmc_cheetah_run-v0':'cheetah_run', 'dmc_walker_walk-v0':'walker_walk'}




def load_expert(expert_traj_nums, env_name):
    ### load expert
    level2num = {"L":expert_traj_nums[0], "M":expert_traj_nums[1], "H":expert_traj_nums[2], "P":expert_traj_nums[3]}
    with open("./expert_data/{}_expert.pkl".format(env_name), 'rb') as f:
        total_data = pickle.load(f)
    expert_data_dict = {}
    for level in ["L", "M", "H", "P"]:
        level_traj_num = level2num[level]
        if level_traj_num>0:
            for key in total_data[level].keys():
                if expert_data_dict.get(key) is None:
                    expert_data_dict[key] = total_data[level][key][:level_traj_num, ...]
                else:
                    expert_data_dict[key] = np.concatenate((expert_data_dict[key], total_data[level][key][:level_traj_num, ...]), axis=0)
    expert_rewards = expert_data_dict["rewards"]
    print("Expert return: {}".format(np.sum(expert_rewards) / sum(expert_traj_nums)))
    ###

    ### add expert data into expert replay buffer
    expert_replay_buffer = ExpertReplayBuffer(states=expert_data_dict['states'],
                                              actions=expert_data_dict['actions'],
                                              rewards=expert_data_dict['rewards'],
                                              masks=expert_data_dict['masks'],
                                              init_num_trajs=sum(expert_traj_nums),
                                              device=device)
    
    return expert_replay_buffer

if __name__ == "__main__":
    parameter = Parameter()
    parameter.timestamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    parameter.save_config()

    task_type = parameter.task_type # drex:epsilon+bc, gail:epsilon+sac, lerp: drex with reward noise
    # common parameters
    alg_name = parameter.alg_name
    env_name = parameter.env_name
    seed = parameter.seed
    expert_traj_nums = [int(num) for num in parameter.expert_traj_nums.split("_")]
    save_dir = os.path.join('buffer', env_name, task_type, parameter.expert_traj_nums, str(seed))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_data_path = os.path.join(save_dir, f'buffer.pkl')

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid=os.getpid()

    '''make environment'''
    if "dmc" in env_name:
        env_fn = lambda : gym.make(env_name, seed=seed)
    else:
        env_fn = lambda : gym.make(env_name)
    gym_env = env_fn()

    '''load expert'''
    expert_replay_buffer = load_expert(expert_traj_nums, env_name)

    if task_type in ['baseline_drex', 'baseline_lerp']:
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
    elif task_type == 'gail':
        '''load actor'''
        file_path = f'logfile/train_irl/irl/{ENV2ENV_NAME[env_name]}/{parameter.expert_traj_nums}/{env_name}_maxentirl_sa_use_pref_False_seed_1_sac_epochs_5_sac_alpha_0.1_{parameter.expert_traj_nums}_last_n_samples-DEBUG-NEW/best_model/sac.pkl'
        # with open(file_path, 'rb') as f:
        #     state_dict = pickle.load(f)
        state_dict = torch.load(file_path)
        policy_model = core.MLPActorCritic(observation_space=gym_env.observation_space, 
                                        action_space=gym_env.action_space,
                                        k=parameter.sac_k,
                                        hidden_sizes=parameter.ac_hidden_sizes,
                                        device=device)
        policy_model.load_state_dict(state_dict)

    '''epsilon sampling'''
    noise_range = np.arange(0., 1., 0.05)
    min_length = 0
    num_trajs = 5
    with tqdm(total=noise_range.shape[-1]) as pbar:
        pbar.set_description(f'{task_type} epsilon sampling')
        
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
                agent_trajs.append((obs, actions, rewards, dones))

            trajs.append((noise_level, agent_trajs))

            pbar.update(1)
        
    with open(save_data_path,'wb') as f:
        pickle.dump(trajs,f)

            




