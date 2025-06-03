import sys, os, time
from datetime import datetime
from utils import system
from common_config.load_config import init_smart_logger
from smart_logger.util_logger.logger import Logger
from parameter.Parameter import Parameter

import gym
import numpy as np 
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pickle

import envs
from rl_alg.sac import ReplayBuffer, SAC


class ExpertBuffer:
    def __init__(self):
        self.data_dict = None

    def add(self, states, actions, next_states, rewards, dones, masks):
        if self.data_dict is None:
            self.data_dict = {}
            self.data_dict["states"] = states
            self.data_dict["actions"] = actions
            self.data_dict["next_states"] = next_states
            self.data_dict["rewards"] = rewards
            self.data_dict["dones"] = dones
            self.data_dict["masks"] = masks
        else:
            self.data_dict["states"] = np.concatenate((self.data_dict["states"], states), axis=0)
            self.data_dict["actions"] = np.concatenate((self.data_dict["actions"], actions), axis=0)
            self.data_dict["next_states"] = np.concatenate((self.data_dict["next_states"], next_states), axis=0)
            self.data_dict["rewards"] = np.concatenate((self.data_dict["rewards"], rewards), axis=0)
            self.data_dict["dones"] = np.concatenate((self.data_dict["dones"], dones), axis=0)
            self.data_dict["masks"] = np.concatenate((self.data_dict["masks"], masks), axis=0)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.data_dict, f)



def train_policy(EnvCls, parameter, logger):
    env = EnvCls()
    
    replay_buffer = ReplayBuffer(
        env.observation_space.shape[0], 
        env.action_space.shape[0],
        device=device,
        size=parameter.sac_buffer_size)

    replay_buffer_expert = ExpertBuffer()
    
    ac_kwargs = {}
    ac_kwargs["hidden_sizes"] = parameter.ac_hidden_sizes
    sac_agent = SAC(env_fn=EnvCls, replay_buffer=replay_buffer, ac_kwargs=ac_kwargs, seed=parameter.seed, steps_per_epoch=env_T,
                    epochs=parameter.sac_epochs, replay_size=parameter.sac_buffer_size, lr=parameter.sac_lr, alpha=parameter.sac_alpha,
                    batch_size=parameter.sac_batch_size, start_steps=env_T * parameter.sac_explore_episodes,
                    update_after=env_T * parameter.sac_explore_episodes,
                    num_test_episodes=parameter.num_test_episodes, max_ep_len=env_T, 
                    automatic_alpha_tuning=parameter.auto_alpha, device=device
        )
    assert sac_agent.reinitialize == True

    sac_agent.epochs = 1000
    sac_agent.test_fn = sac_agent.test_agent_ori_env
    sac_agent.reset_train()
    sac_agent.learn_mujoco(print_out=True, logger=logger, replay_buffer_expert=replay_buffer_expert,
                           buffer_save_path="../imitation_data_total_expert/{}_replay_buffer.pkl".format(parameter.env_name),
                           learn_grid=(parameter.env_name in ["CustomizedGridWorldEasy-v0", "CustomizedGridWorld-v0"]))

    return sac_agent.get_action


if __name__ == "__main__":
    init_smart_logger()
    parameter = Parameter()
    parameter.timestamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    logger = Logger(log_name=parameter.short_name, log_signature=parameter.signature,
                            logger_category='train_expert/{}'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')), backup_code=False)
    parameter.set_logger(logger)
    parameter.set_config_path(os.path.join(logger.output_dir, 'config'))
    parameter.save_config()
    logger.log(parameter)

    # common parameters
    env_name, env_T = parameter.env_name, parameter.env_T
    seed = parameter.seed
    sac_epochs = parameter.sac_epochs

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid=os.getpid()

    ## initialize draw
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    if "dmc" in env_name:
        EnvCls = lambda : gym.make(env_name, seed=seed)
    else:
        EnvCls = lambda : gym.make(env_name, T=env_T, seed=seed)
    print(f"training Expert on {env_name}")
    policy = train_policy(EnvCls, parameter, logger)

    # env = EnvCls()

    # expert_states_sto, expert_actions_sto, expert_returns, expert_rewards_sto = evaluate_policy(policy, env, parameter.expert_samples_episode)
    # return_info = f'Expert(Sto) Return Avg: {expert_returns.mean():.2f}, std: {expert_returns.std():.2f}'
    # print(return_info)

    # os.makedirs('expert_data/meta/', exist_ok=True)
    # log_txt = open(f"expert_data/meta/{env_name}_{seed}_{sac_epochs}.txt", 'w')
    # log_txt.write(return_info + '\n')
    # log_txt.write(repr(expert_returns)+'\n')

    # sns.violinplot(data=expert_returns, ax=axs[1])
    # axs[1].set_title("violin plot of expert(sto) return")

    # expert_states_det, expert_actions_det, expert_returns, expert_rewards_det = evaluate_policy(policy, env, parameter.expert_samples_episode, True)
    # sns.violinplot(data=expert_returns, ax=axs[0])
    # axs[0].set_title("violin plot of expert(det) return")
    # plt.savefig(os.path.join(f'expert_data/meta/{env_name}_{seed}_{sac_epochs}.png')) 

    # return_info = f'Expert(Det) Return Avg: {expert_returns.mean():.2f}, std: {expert_returns.std():.2f}'
    # print(return_info)
    # log_txt.write(return_info + '\n')
    # log_txt.write(repr(expert_returns)+'\n')

    # os.makedirs('expert_data/states/', exist_ok=True)
    # os.makedirs('expert_data/actions/', exist_ok=True)
    # os.makedirs('expert_data/rewards/', exist_ok=True)
    # torch.save(expert_states_sto, f'expert_data/states/{env_name}_seed_{seed}_epoch_{sac_epochs}_sto.pt')
    # torch.save(expert_states_det, f'expert_data/states/{env_name}_seed_{seed}_epoch_{sac_epochs}_det.pt')
    # torch.save(expert_actions_sto, f'expert_data/actions/{env_name}_seed_{seed}_epoch_{sac_epochs}_sto.pt')
    # torch.save(expert_actions_det, f'expert_data/actions/{env_name}_seed_{seed}_epoch_{sac_epochs}_det.pt')
    # torch.save(expert_rewards_sto, f'expert_data/rewards/{env_name}_seed_{seed}_epoch_{sac_epochs}_sto.pt')
    # torch.save(expert_rewards_det, f'expert_data/rewards/{env_name}_seed_{seed}_epoch_{sac_epochs}_det.pt')