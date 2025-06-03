'''
f-IRL: Extract policy/reward from specified expert samples
'''
import sys, os, time
import numpy as np
import torch
import gym
import json
# from ruamel.yaml import YAML
sys.path.append('.')

from rl_alg.sac import ReplayBuffer, SAC
from parameter.Parameter import Parameter

import envs
from utils import system, collect, eval
from common_config.load_config import init_smart_logger
from smart_logger.util_logger.logger import Logger

from datetime import datetime
import pickle
import os
import torch.nn as nn

from utils.replay_buffer import ExpertReplayBuffer
import rl_alg.sac_agent as core
from epsilon_noise_sample import load_expert
from utils.plot.plot_utils import load_model, set_model_hypeparams

ENV2ENV_NAME = {"HopperFH-v0":"Hopper-v2", "HalfCheetahFH-v0":"HalfCheetah-v2", "Walker2dFH-v0":"Walker2d-v2",
                "AntFH-v0":"Ant-v2", "HumanoidFH-v0":"Humanoid-v2", 'dmc_quadruped_walk-v0':'quadruped_walk',
                'dmc_cheetah_run-v0':'cheetah_run', 'dmc_walker_walk-v0':'walker_walk'}

def try_evaluate(policy_type: str, obj=None):
    assert policy_type in ["Running"]
    update_time = parameter.r_gradient_step
    env_steps = parameter.sac_epochs * parameter.env_T

    # eval real reward
    real_return_sto, eval_states, eval_actions = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            parameter.irl_eval_episodes, parameter.env_T, False)

    eval_info = {
        "StoEpRetTest": real_return_sto,
        f"Discriminator {policy_type} Update Time": update_time,
        f"timestep": env_steps,
    }
    return eval_info

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
    logger.log("Expert return: {}".format(np.sum(expert_rewards) / sum(expert_traj_nums)))
    ###

    ### add expert data into expert replay buffer
    expert_replay_buffer = ExpertReplayBuffer(states=expert_data_dict['states'],
                                              actions=expert_data_dict['actions'],
                                              rewards=expert_data_dict['rewards'],
                                              masks=expert_data_dict['masks'],
                                              init_num_trajs=sum(expert_traj_nums),
                                              device=device)
    
    return expert_replay_buffer

def load_params(path):
    load_dict = {}
    with open(path, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)
    return load_dict


if __name__ == "__main__":
    init_smart_logger()
    parameter = Parameter()
    parameter.timestamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')

    task_type = parameter.task_type # irl for gail reward, drex for bc+epsilon reward, gail for ac+epsilon reward, pbrl for our reward  
    

    logger = Logger(log_name=parameter.short_name, log_signature=parameter.signature,
                    logger_category='eval_reward/{}/{}'.format(task_type, ENV2ENV_NAME[parameter.env_name]), backup_code=False)
    parameter.set_logger(logger)
    parameter.set_config_path(os.path.join(logger.output_dir, 'config'))
    parameter.save_config()
    logger.log(parameter)

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
    if alg_name in ['maxentirl_sa', 'gail']:
        reward_indices = list(range(state_size + action_size))
    elif alg_name in ['maxentirl']:
        reward_indices = list(range(state_size))

    algorithm, load_buffer_dirname, load_model_dirname = set_model_hypeparams(task_type)

    load_logfile_dir = os.path.join('logfile', f'train_{algorithm}', load_model_dirname, ENV2ENV_NAME[env_name], parameter.expert_traj_nums, parameter.short_name)
    load_reward_path = os.path.join(load_logfile_dir, 'best_model', 'reward_model_last.pkl')
    model_params = load_params(os.path.join(load_logfile_dir, 'config', 'parameter.json'))
    print(f"LOAD REWARD MODEL FROM {load_reward_path}")

    reward_func = load_model(task_type, model_params, reward_indices, load_reward_path)
    
    replay_buffer = ReplayBuffer(
                        state_size, 
                        action_size,
                        device=device,
                        size=parameter.sac_buffer_size,
                        env_T=parameter.env_T)
    
    # preload same buffer from SAC training
    if parameter.fix_seed:
        load_replay_path = os.path.join('buffer', env_name, 'ablation_nonoise', '0_0_1_0', '0', 'buffer.pkl')
    else:
        load_replay_path = os.path.join('buffer', env_name, 'ablation_nonoise', '0_0_1_0', str(parameter.seed), 'buffer.pkl')
    if env_name in ['AntFH-v0', 'Walker2dFH-v0', 'HalfCheetahFH-v0', 'HopperFH-v0']: 
        replay_buffer.load_epsilon_buffer(load_replay_path, n=10)

    ac_kwargs = {}
    ac_kwargs["hidden_sizes"] = parameter.ac_hidden_sizes
    sac_agent = SAC(env_fn=env_fn, replay_buffer=replay_buffer, ac_kwargs=ac_kwargs, obj=alg_name, k=parameter.sac_k, seed=seed, steps_per_epoch=parameter.env_T,
        epochs=parameter.sac_epochs, replay_size=parameter.sac_buffer_size, lr=parameter.sac_lr, alpha=parameter.sac_alpha, 
        batch_size=parameter.sac_batch_size, start_steps=parameter.env_T * parameter.sac_explore_episodes,
        update_after=parameter.env_T * parameter.sac_explore_episodes,
        num_test_episodes=parameter.num_test_episodes, max_ep_len=parameter.env_T,
        reward_indices=reward_indices, device=device, automatic_alpha_tuning=parameter.auto_alpha, reinitialize=parameter.sac_reinitialize
    )
    sac_agent.epochs = 3000
    sac_agent.test_fn = sac_agent.test_agent_ori_env 
    sac_agent.reset_train()
    sac_agent.expert_schedule = np.linspace(1, 0, sac_agent.epochs * sac_agent.steps_per_epoch)
    sac_agent.e_optimizer = torch.optim.Adam(sac_agent.ac.parameters(), lr=1e-4)
    sac_agent.expert_buffer = load_expert(expert_traj_nums, env_name)
    
    save_model_path = os.path.join(logger.output_dir, "best_model")
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    sac_agent.reward_function = reward_func.get_scalar_reward # only need to change reward in sac
    sac_test_rets, sac_alphas, sac_loss_qs, sac_loss_pis, sac_log_pis, sac_time_steps = sac_agent.learn_mujoco(logger=logger, 
                                                                                                               save_path=os.path.join(save_model_path, 'sac_actor_last.pkl'))
    log_info = try_evaluate("Running", alg_name)

    # max_real_return_sto = log_info['StoEpRetTest']
    # save_reward_dir = os.path.join(logger.output_dir, "best_model")
    # if not os.path.exists(save_reward_dir):
    #     os.makedirs(save_reward_dir)
    # torch.save(reward_func.state_dict(), os.path.join(save_reward_dir, 'reward_model_last.pkl'))

    logger.add_tabular_data(**log_info)
    logger.dump_tabular()

