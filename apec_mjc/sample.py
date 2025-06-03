'''
f-IRL: Extract policy/reward from specified expert samples
'''
import warnings
warnings.filterwarnings('ignore')

import sys, os, time
import numpy as np
import torch
import pickle
import gym
from tqdm import tqdm
import envs

# from ruamel.yaml import YAML
sys.path.append('.')

from datetime import datetime
from parameter.Parameter import Parameter

from utils import system
from utils.replay_buffer import ExpertReplayBuffer, make_preference_dataset, epsilon_sample, collect_epsilon_dataset, collect_epsilon_dataset_for_baselines, make_preference_dataset_no_noise, make_preference_dataset_no_noise_relative
from utils.epsilon_utils import filter_paths
import rl_alg.sac_agent as core

ENV2ENV_NAME = {"HopperFH-v0":"Hopper-v2", "HalfCheetahFH-v0":"HalfCheetah-v2", "Walker2dFH-v0":"Walker2d-v2",
                "AntFH-v0":"Ant-v2", "HumanoidFH-v0":"Humanoid-v2", 'dmc_quadruped_walk-v0':'quadruped_walk',
                'dmc_cheetah_run-v0':'cheetah_run', 'dmc_walker_walk-v0':'walker_walk'}


def load_expert(expert_traj_nums, env_name, device):
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
    ###

    ### add expert data into expert replay buffer
    expert_replay_buffer = ExpertReplayBuffer(states=expert_data_dict['states'],
                                              actions=expert_data_dict['actions'],
                                              rewards=expert_data_dict['rewards'],
                                              masks=expert_data_dict['masks'],
                                              init_num_trajs=sum(expert_traj_nums),
                                              device=device)
    
    return expert_replay_buffer

SAMPLE_MAX_EPOCH = {
    'HopperFH-v0': 400,
    'HalfCheetahFH-v0':400,
    'Walker2dFH-v0':400,
    'AntFH-v0':400,
    'HumanoidFH-v0':400,

    'dmc_walker_walk-v0':100, 
    'dmc_quadruped_walk-v0':300,
    'dmc_cheetah_run-v0':300, 
}

if __name__ == "__main__":
    parameter = Parameter()
    parameter.timestamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    algorithm = 'pbrl'

    # common parameters
    alg_name = parameter.alg_name
    env_name = parameter.env_name
    seed = parameter.seed
    expert_traj_nums = [int(num) for num in parameter.expert_traj_nums.split("_")]

    # system: device, threads, seed, pid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid=os.getpid()

    # environment
    if "dmc" in env_name:
        env_fn = lambda : gym.make(env_name, seed=seed)
    else:
        env_fn = lambda : gym.make(env_name)
    gym_env = env_fn()
    
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    state_indices = list(range(state_size))
    action_indices = list(range(action_size))

    sub_level = 20 if parameter.expert_traj_nums == "0_0_0_1" else 10
    max_ep_len=1000         # default=1000
    sample_num=1000         # default=1000
    segment_len=1000        # default=1000

    expert_replay_buffer = load_expert(expert_traj_nums, env_name, device)
    os.makedirs(os.path.join('buffer', env_name, parameter.task_type, parameter.expert_traj_nums, str(seed)), exist_ok=True)
    save_replay_path = os.path.join('buffer', env_name, parameter.task_type, parameter.expert_traj_nums, str(seed), f'{parameter.sample_mode}_buffer_{parameter.task_type}.npz')
    if parameter.fix_task:
        # rebuttal: use nonoise buffer for ablation
        load_buffer_path = os.path.join('buffer', env_name, 'ablation_nonoise', parameter.expert_traj_nums, str(seed), 'buffer.pkl')
    else:
        load_buffer_path = os.path.join('buffer', env_name, parameter.task_type, parameter.expert_traj_nums, str(seed), 'buffer.pkl')
        
    # parameter_short_name = f'{env_name}_{parameter.alg_name}_use_pref_False_seed_{parameter.seed}_sac_epochs_5_sac_alpha_0.1_{parameter.expert_traj_nums}_last_n_samples-DEBUG-NEW'
    parameter_short_name = f'{env_name}_{parameter.alg_name}_use_pref_False_seed_0_sac_epochs_5_sac_alpha_0.1_{parameter.expert_traj_nums}_last_n_samples-DEBUG-NEW' # rebuttal: use seed=0
    if parameter.task_type in ['pbrl', 'supervised'] or parameter.task_type.startswith('ablation'):
        
        if parameter.collect_dataset:
            actor_dir = os.path.join('logfile', f'train_irl', f'irl', ENV2ENV_NAME[env_name], parameter.expert_traj_nums, parameter_short_name, 'best_model', 'actors')
            model_paths = filter_paths(file_dir=actor_dir, max_epoch=SAMPLE_MAX_EPOCH[env_name])
            collect_epsilon_dataset(model_paths, gym_env, parameter, device, load_buffer_path)
        if parameter.task_type == 'pbrl' or parameter.task_type.startswith('ablation'):
            load_buffer_path = os.path.join('buffer', env_name, 'ablation_nonoise', parameter.expert_traj_nums, str(parameter.seed), 'buffer.pkl')
            if parameter.task_type == 'pbrl':
                preference_buffer, accuracy = make_preference_dataset(load_buffer_path, 
                                                    expert_replay_buffer.states, 
                                                    expert_replay_buffer.actions,
                                                    max_ep_len=max_ep_len,
                                                    sample_num=sample_num,
                                                    segment_len=segment_len,
                                                    sub_level=sub_level,
                                                    mode=parameter.sample_mode)
            # elif parameter.task_type.endswith('epoch'): # ablation_*epoch
            #     n_epochs = int(parameter.task_type.split('_')[-1].replace('epoch', ''))
            #     preference_buffer, accuracy = make_preference_dataset_no_noise(load_buffer_path, 
            #                             expert_replay_buffer.states, 
            #                             expert_replay_buffer.actions,
            #                             max_ep_len=max_ep_len,
            #                             sample_num=sample_num,
            #                             segment_len=segment_len,
            #                             sub_level=sub_level,
            #                             mode=parameter.sample_mode,
            #                             num_epochs=n_epochs)
            elif parameter.task_type in ['ablation_sample_nodistance']:
                preference_buffer, accuracy = make_preference_dataset_no_noise(load_buffer_path, 
                                        expert_replay_buffer.states, 
                                        expert_replay_buffer.actions,
                                        max_ep_len=max_ep_len,
                                        sample_num=sample_num,
                                        segment_len=segment_len,
                                        sub_level=sub_level,
                                        mode=parameter.sample_mode,
                                        filter_distance=False)
            elif parameter.task_type in ['ablation_epochthresh20']:
                preference_buffer, accuracy = make_preference_dataset_no_noise(load_buffer_path, 
                                        expert_replay_buffer.states, 
                                        expert_replay_buffer.actions,
                                        max_ep_len=max_ep_len,
                                        sample_num=sample_num,
                                        segment_len=segment_len,
                                        sub_level=sub_level,
                                        mode=parameter.sample_mode,
                                        epoch_threshold=20)
            elif parameter.task_type in ['ablation_sample_noiter']: # only sample by distance 
                preference_buffer, accuracy = make_preference_dataset_no_noise(load_buffer_path, 
                                        expert_replay_buffer.states, 
                                        expert_replay_buffer.actions,
                                        max_ep_len=max_ep_len,
                                        sample_num=sample_num,
                                        segment_len=segment_len,
                                        sub_level=sub_level,
                                        mode=parameter.sample_mode,
                                        filter_epoch=False,
                                        distance_threshold=0)
            elif parameter.task_type in ['ablation_sample_distance2']: # only sample by distance 
                preference_buffer, accuracy = make_preference_dataset_no_noise(load_buffer_path, 
                                        expert_replay_buffer.states, 
                                        expert_replay_buffer.actions,
                                        max_ep_len=max_ep_len,
                                        sample_num=sample_num,
                                        segment_len=segment_len,
                                        sub_level=sub_level,
                                        mode=parameter.sample_mode,
                                        distance_threshold=0.2)
            elif parameter.task_type in ['ablation_sample_distance05']: # only sample by distance 
                preference_buffer, accuracy = make_preference_dataset_no_noise(load_buffer_path, 
                                        expert_replay_buffer.states, 
                                        expert_replay_buffer.actions,
                                        max_ep_len=max_ep_len,
                                        sample_num=sample_num,
                                        segment_len=segment_len,
                                        sub_level=sub_level,
                                        mode=parameter.sample_mode,
                                        distance_threshold=0.05)
            elif parameter.task_type in ['ablation_sample_relative']: # relative distance guidance 
                preference_buffer, accuracy = make_preference_dataset_no_noise_relative(load_buffer_path, 
                                        expert_replay_buffer.states, 
                                        expert_replay_buffer.actions,
                                        max_ep_len=max_ep_len,
                                        sample_num=sample_num,
                                        segment_len=segment_len,
                                        sub_level=sub_level,
                                        mode=parameter.sample_mode)
            else: 
                # parameter.task_type in ['ablation_nonoise'] or other cases 
                print('using default no_noise sample pipeline')
                preference_buffer, accuracy = make_preference_dataset_no_noise(load_buffer_path, 
                                        expert_replay_buffer.states, 
                                        expert_replay_buffer.actions,
                                        max_ep_len=max_ep_len,
                                        sample_num=sample_num,
                                        segment_len=segment_len,
                                        sub_level=sub_level,
                                        mode=parameter.sample_mode,
                                        distance_metric=parameter.distance_metric)
            
    elif parameter.task_type.startswith('baseline'):

        if parameter.task_type in ['baseline_ssrr']:
            gail_path = os.path.join('logfile', 'train_irl', 'irl', ENV2ENV_NAME[env_name], parameter.expert_traj_nums, parameter_short_name, 'best_model', 'sac_actor_last.pt') # ssrr use last model in noisy_airl
            collect_epsilon_dataset_for_baselines(expert_replay_buffer, parameter, device, gym_env, gail_path, load_buffer_path)    
        else:
            if parameter.collect_dataset:
                gail_path = os.path.join('logfile', 'train_irl', 'irl', ENV2ENV_NAME[env_name], parameter.expert_traj_nums, parameter_short_name, 'best_model', 'sac_actor_last.pt') # baseline_gail use last model in train_irl
                collect_epsilon_dataset_for_baselines(expert_replay_buffer, parameter, device, gym_env, gail_path, load_buffer_path)
        
        preference_buffer, accuracy = epsilon_sample(load_buffer_path, 
                                            expert_replay_buffer.states, 
                                            expert_replay_buffer.actions,
                                            max_ep_len=max_ep_len,
                                            sample_num=sample_num,
                                            segment_len=segment_len,
                                            sub_level=sub_level,
                                            mode=parameter.sample_mode,
                                            task_type=parameter.task_type)

    preference_buffer.save(save_replay_path)
