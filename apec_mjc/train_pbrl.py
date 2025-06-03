'''
f-IRL: Extract policy/reward from specified expert samples
'''
import sys, os, time
import numpy as np
import torch
import gym
# from ruamel.yaml import YAML
sys.path.append('.')
from f_div import maxentirl_sa_loss, gail_loss, bt_loss, gradient_penalty
from model.reward import PrefRewardModel
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

from utils.replay_buffer import ExpertReplayBuffer, PreferenceBuffer
from utils.plot.plot_utils import set_model_hypeparams

ENV2ENV_NAME = {"HopperFH-v0":"Hopper-v2", "HalfCheetahFH-v0":"HalfCheetah-v2", "Walker2dFH-v0":"Walker2d-v2",
                "AntFH-v0":"Ant-v2", "HumanoidFH-v0":"Humanoid-v2", 'dmc_quadruped_walk-v0':'quadruped_walk',
                'dmc_cheetah_run-v0':'cheetah_run', 'dmc_walker_walk-v0':'walker_walk'}

def try_evaluate(alg_name, samples, reward_func):
    states1, states2, actions1, actions2, real_rewards1, real_rewards2, masks_1, masks_2, pref_label = samples
    inputs1 = torch.cat([states1, actions1], dim=-1) if alg_name in ['maxentirl_sa'] else states1
    inputs2 = torch.cat([states2, actions2], dim=-1) if alg_name in ['maxentirl_sa'] else states2
    pred_rewards1 = reward_func.r(inputs1).squeeze()
    pred_rewards2 = reward_func.r(inputs2).squeeze()

    pred_returns1 = torch.sum(pred_rewards1 * masks_1, dim=-1, keepdim=True)
    pred_returns2 = torch.sum(pred_rewards2 * masks_2, dim=-1, keepdim=True)
    real_returns1 = torch.sum(real_rewards1.squeeze() * masks_1, dim=-1, keepdim=True)
    real_returns2 = torch.sum(real_rewards2.squeeze() * masks_2, dim=-1, keepdim=True)
    
    p_returns = torch.cat([pred_returns1, pred_returns2], dim=-1).detach().cpu().numpy()
    r_returns = torch.cat([real_returns1, real_returns2], dim=-1).detach().cpu().numpy()
    pred_label = np.argmax(p_returns, axis=1)
    real_label = np.argmax(r_returns, axis=1)
    accuracy = np.mean(pred_label == real_label)
    
    masks = torch.cat([masks_1, masks_2], dim=0).detach().cpu().numpy().reshape(-1)
    real_rewards = torch.cat([real_rewards1, real_rewards2], dim=0).detach().cpu().numpy().reshape(-1)
    pred_rewards = torch.cat([pred_rewards1, pred_rewards2], dim=0).detach().cpu().numpy().reshape(-1)
    real_rewards = real_rewards[masks > 0]
    pred_rewards = pred_rewards[masks > 0]
    real_returns = torch.cat([real_returns1, real_returns2], dim=0).detach().cpu().numpy().reshape(-1)
    pred_returns = torch.cat([pred_returns1, pred_returns2], dim=0).detach().cpu().numpy().reshape(-1)

    max_gt_idx = np.argmax(real_returns)
    max_gt_return = real_returns[max_gt_idx]
    max_gt_idx_pred_return = pred_returns[max_gt_idx]

    max_pred_idx = np.argmax(pred_returns)
    max_pred_return = pred_returns[max_pred_idx]
    max_pred_idx_gt_return = real_returns[max_pred_idx]
    
    eval_info = {
        "ReturnCorr": np.corrcoef(pred_returns, real_returns)[0, 1],
        "RewardCorr": np.corrcoef(pred_rewards, real_rewards)[0, 1],
        "PrefLabelAcc": accuracy,
        'MaxGTReturn': max_gt_return,
        'MaxGTPredReturn': max_gt_idx_pred_return,
        'MaxPredReturn': max_pred_return,
        'MaxPredGTReturn': max_pred_idx_gt_return,
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

if __name__ == "__main__":
    init_smart_logger()
    parameter = Parameter()
    parameter.timestamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    logger = Logger(log_name=parameter.short_name, log_signature=parameter.signature,
                    logger_category='train_pbrl/{}/{}/{}'.format(parameter.task_type, ENV2ENV_NAME[parameter.env_name], parameter.expert_traj_nums), backup_code=False)
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

    sub_level = 20 if parameter.expert_traj_nums == "0_0_0_1" else 3
    expert_replay_buffer = load_expert(expert_traj_nums, env_name)

    _, buffer_dirname, _ = set_model_hypeparams(parameter.task_type)
    if not parameter.fix_seed:
        train_replay_path = os.path.join('buffer', env_name, buffer_dirname, parameter.expert_traj_nums, str(seed), f'train_buffer_{buffer_dirname}.npz')
    else:
        # for rebuttal
        train_replay_path = os.path.join('buffer', env_name, buffer_dirname, parameter.expert_traj_nums, '0', f'train_buffer_{buffer_dirname}.npz')
    preference_buffer = PreferenceBuffer(memory_size=10000)
    preference_buffer.load(train_replay_path)
    print(train_replay_path)

    states_1, states_2, actions_1, actions_2, rewards_1, rewards_2, masks_1, masks_2, pref_labels = preference_buffer.get_all(device=torch.device('cpu'))
    gt_label = (rewards_1.sum(axis=-1) < rewards_2.sum(axis=-1)).int()
    sample_label = torch.argmax(pref_labels, dim=-1).squeeze()
    accuracy = ((gt_label == sample_label).sum() / sample_label.shape[0]).item()

    if parameter.env_type == 'mujoco':
        test_replay_path = os.path.join('buffer', env_name, 'maxentirl_sa', '0_0_0_1', '1', 'test_pbrl_buffer.npz')
    elif parameter.env_type == 'dmc':
        test_replay_path = os.path.join('buffer', env_name, 'ablation_nonoise', '0_0_1_0', '1', 'test_buffer_ablation_nonoise.npz')

    save_test_replay_path = os.path.join('buffer', env_name, 'maxentirl', parameter.expert_traj_nums, str(seed), 'test_pbrl_buffer.npz')
    test_buffer = PreferenceBuffer(memory_size=10000)
    test_buffer.load(test_replay_path)
    
    reward_func = PrefRewardModel(input_dim=len(reward_indices), hidden_sizes=parameter.r_hidden_sizes,
                                hid_act=parameter.r_hid_act, use_bn=parameter.r_use_bn, residual=parameter.r_residual,
                                clamp_magnitude=parameter.r_clamp_magnitude, device=device).to(device)
    reward_optimizer = torch.optim.Adam(reward_func.parameters(), lr=parameter.r_lr, weight_decay=parameter.r_weight_decay,
                                        betas=(parameter.r_momentum, 0.999))
    
    pref_losses = []
    loss = nn.BCELoss()
    max_accuracy, max_corr = -np.inf, -np.inf
    for itr in range(parameter.pbrl_n_iters):
        train_samples = preference_buffer.get_samples(batch_size=1000, device=device)
        test_samples = test_buffer.get_samples(batch_size=1000, device=device)
        pref_loss = bt_loss(alg_name, train_samples, reward_func, segment_len=parameter.segment_len)

        pref_losses.append(pref_loss.item())
        reward_optimizer.zero_grad()
        pref_loss.backward()
        reward_optimizer.step()
    
        if (itr + 1) % 10 == 0:
            log_info_train = try_evaluate(alg_name, train_samples, reward_func)
            log_info_test = try_evaluate(alg_name, test_samples, reward_func)
            log_info = {}
            for key in log_info_train.keys():
                log_info[f'train_{key}'] = log_info_train[key]
                log_info[f'test_{key}'] = log_info_test[key]
            save_reward_dir = os.path.join(logger.output_dir, "best_model")
            os.makedirs(save_reward_dir, exist_ok=True)
            torch.save(reward_func.state_dict(), os.path.join(save_reward_dir, 'reward_model_last.pkl'))
            log_info['gen_accuracy'] = accuracy
            log_info["Itration"] = itr
            log_info["pref_loss"] = np.mean(pref_losses)
            logger.add_tabular_data(**log_info)
            logger.dump_tabular()
    print('Training reward done.')