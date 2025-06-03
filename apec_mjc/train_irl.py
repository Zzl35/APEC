'''
f-IRL: Extract policy/reward from specified expert samples
'''
import sys, os, time
import numpy as np
import torch
import gym
# from ruamel.yaml import YAML


from f_div import maxentirl_sa_loss, gail_loss, maxentirl_loss
from model.reward import MLPReward, GAILMLPReward
from rl_alg.sac import ReplayBuffer, SAC
from parameter.Parameter import Parameter
from model.pref_reward import PrefRewardModel

import envs
from utils import system, collect, eval
from common_config.load_config import init_smart_logger
from smart_logger.util_logger.logger import Logger

from datetime import datetime
import dateutil.tz
import json, copy
import pickle
import os

from utils.replay_buffer import ExpertReplayBuffer, expert_distance

ENV2ENV_NAME = {"HopperFH-v0":"Hopper-v2", "HalfCheetahFH-v0":"HalfCheetah-v2", "Walker2dFH-v0":"Walker2d-v2",
                "AntFH-v0":"Ant-v2", "HumanoidFH-v0":"Humanoid-v2", 'dmc_quadruped_walk-v0':'quadruped_walk',
                'dmc_cheetah_run-v0':'cheetah_run', 'dmc_walker_walk-v0':'walker_walk'}


def try_evaluate(itr: int, policy_type: str, obj=None, expert_trajs=None):
    assert policy_type in ["Running"]
    update_time = itr * parameter.r_gradient_step
    env_steps = itr * parameter.sac_epochs * parameter.env_T

    # eval real reward
    # real_return_det = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
    #                                         parameter.irl_eval_episodes, parameter.env_T, True)
    real_return_sto, eval_states, eval_actions = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            parameter.irl_eval_episodes, parameter.env_T, False)
    
    distance = np.mean([expert_distance(traj=np.concatenate([eval_states[i], eval_actions[i]], axis=-1).squeeze(), 
                         expert_trajs=expert_trajs) for i in range(len(eval_states))])

    eval_info = {
        # "EpRetTest": round(real_return_det, 2),
        "StoEpRetTest": real_return_sto,
        "ExpertDistance" : distance,
        f"Discriminator {policy_type} Update Time": update_time,
        f"timestep": env_steps,
    }

    return eval_info

if __name__ == "__main__":
    init_smart_logger()
    parameter = Parameter()
    parameter.timestamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    algorithm = parameter.task_type
    
    logger = Logger(log_name=parameter.short_name, log_signature=parameter.signature,
                            logger_category='train_irl/{}/{}/{}'.format(algorithm, ENV2ENV_NAME[parameter.env_name], parameter.expert_traj_nums), backup_code=False)
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
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid=os.getpid()
    
    # assumptions
    assert alg_name in ['maxentirl_sa', 'maxentirl', 'gail']

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

    ### bottleneck
    if parameter.use_best_traj:
        expert_returns = np.sum(expert_rewards, axis=-1)
        best_traj_index = sorted(list(range(len(expert_returns))), key=lambda x: expert_returns[x])[-1]
        for key in expert_data_dict.keys():
            expert_data_dict[key] = expert_data_dict[key][best_traj_index:best_traj_index+1, ...]
        expert_rewards = expert_data_dict["rewards"]
        logger.log("Expert return: {}".format(np.sum(expert_rewards)))
    ###

    ### add expert data into expert replay buffer
    expert_replay_buffer = ExpertReplayBuffer(states=expert_data_dict['states'],
                                              actions=expert_data_dict['actions'],
                                              rewards=expert_data_dict['rewards'],
                                              masks=expert_data_dict['masks'],
                                              init_num_trajs=sum(expert_traj_nums),
                                              device=device)
    ###

    if alg_name in ['maxentirl_sa']:
        # Initilialize reward as a neural network
        reward_func = MLPReward(input_dim=len(state_indices)+len(action_indices), hidden_sizes=parameter.r_hidden_sizes,
                                hid_act=parameter.r_hid_act, use_bn=parameter.r_use_bn, residual=parameter.r_residual,
                                clamp_magnitude=parameter.r_clamp_magnitude, device=device).to(device)
        reward_optimizer = torch.optim.Adam(reward_func.parameters(), lr=parameter.r_lr, weight_decay=parameter.r_weight_decay,
                                            betas=(parameter.r_momentum, 0.999))
    elif alg_name in ['maxentirl']:
        reward_func = MLPReward(input_dim=len(state_indices), hidden_sizes=parameter.r_hidden_sizes,
                                hid_act=parameter.r_hid_act, use_bn=parameter.r_use_bn, residual=parameter.r_residual,
                                clamp_magnitude=parameter.r_clamp_magnitude, device=device).to(device)
        reward_optimizer = torch.optim.Adam(reward_func.parameters(), lr=parameter.r_lr, weight_decay=parameter.r_weight_decay,
                                            betas=(parameter.r_momentum, 0.999))
    elif alg_name in ['gail']:
        # Initilialize reward as a neural network
        reward_func = GAILMLPReward(input_dim=len(state_indices)+len(action_indices), hidden_sizes=parameter.r_hidden_sizes,
                                    hid_act=parameter.r_hid_act, use_bn=parameter.r_use_bn, residual=parameter.r_residual,
                                    clamp_magnitude=parameter.r_clamp_magnitude, device=device).to(device)
        reward_optimizer = torch.optim.Adam(reward_func.parameters(), lr=parameter.r_lr, weight_decay=parameter.r_weight_decay,
                                            betas=(parameter.r_momentum, 0.999))

    ### initial reward
    if parameter.use_pref:
        total_pref_queries = 0
        assert alg_name in ['maxentirl_sa', 'gail', 'maxentirl']
        pref_reward = PrefRewardModel(ds=len(state_indices), da=len(action_indices), device=device,
                                        size_segment=parameter.pref_size_segment, teacher_eps_mistake=parameter.teacher_eps_mistake,
                                        teacher_beta=parameter.teacher_beta, teacher_gamma=parameter.teacher_gamma,
                                        teacher_eps_skip=parameter.teacher_eps_skip, teacher_eps_equal=parameter.teacher_eps_equal)
        margin = np.mean(expert_rewards) * parameter.pref_size_segment
        pref_reward.set_teacher_thres_skip(margin)
        pref_reward.set_teacher_thres_equal(margin)
        pref_reward.add_data_batch(expert_data_dict["states"], expert_data_dict["actions"], expert_data_dict["rewards"], expert_data_dict["masks"])
        if parameter.expert_pref_size>0:
            labeled_queries = pref_reward.uniform_sampling(mb_size=parameter.expert_pref_size)
            total_pref_queries += parameter.expert_pref_size
            ### train pref reward and update weight in buffer
            for step in range(200):
                train_acc = pref_reward.train_reward()
                total_acc = np.mean(train_acc)
                if total_acc > 0.98:
                    break
            pref_reward_info = {
                'TotalPrefQueries': total_pref_queries,
                'PrefModelAccuracy': total_acc
            }
            logger.add_tabular_data(tb_prefix='pref_reward', **pref_reward_info)
        expert_replay_buffer.update_trajs_weight(pref_reward, parameter.pref_beta)
        pic_path = os.path.join(logger.output_dir, "pics")
        expert_replay_buffer.draw(dir=pic_path, name="init")
    ###

    max_real_return_det, max_real_return_sto = -np.inf, -np.inf
    for itr in range(parameter.irl_n_iters):
        if parameter.sac_reinitialize or itr == 0:
            # Reset SAC agent with old policy, new environment, and new replay buffer
            print("========== Reinitializing sac ==========")
            replay_buffer = ReplayBuffer(
                state_size, 
                action_size,
                device=device,
                size=parameter.sac_buffer_size,
                env_T=parameter.env_T)
                
            ac_kwargs = {}
            ac_kwargs["hidden_sizes"] = parameter.ac_hidden_sizes
            sac_agent = SAC(env_fn=env_fn, replay_buffer=replay_buffer, ac_kwargs=ac_kwargs, obj=alg_name, k=parameter.sac_k, seed=seed, steps_per_epoch=parameter.env_T,
                epochs=parameter.sac_epochs, replay_size=parameter.sac_buffer_size, lr=parameter.sac_lr, alpha=parameter.sac_alpha, 
                batch_size=parameter.sac_batch_size, start_steps=parameter.env_T * parameter.sac_explore_episodes,
                update_after=parameter.env_T * parameter.sac_epochs, update_every=parameter.sac_update_every,
                num_test_episodes=parameter.num_test_episodes, max_ep_len=parameter.env_T, log_step_interval=parameter.sac_log_step_interval,
                reward_indices=reward_indices, device=device, automatic_alpha_tuning=parameter.auto_alpha, reinitialize=parameter.sac_reinitialize
            )

            sac_agent.reset_train()
        
        sac_agent.reward_function = reward_func.get_scalar_reward # only need to change reward in sac
        save_model_path = os.path.join(logger.output_dir, "best_model")
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        sac_test_rets, sac_alphas, sac_loss_qs, sac_loss_pis, sac_log_pis, sac_time_steps = sac_agent.learn_mujoco(save_path=os.path.join(save_model_path, 'sac_actor_last.pt'))

        ### sample not enough, skip train until sample enough
        if sac_agent.replay_buffer.size < parameter.env_T * parameter.sac_epochs:
            continue
        ###
        
        ## save buffer
        if itr % 10 == 0:
            # save_path = os.path.join('buffer', env_name, alg_name, parameter.expert_traj_nums, str(seed))
            # os.makedirs(save_path, exist_ok=True)
            # replay_buffer.save(os.path.join(save_path, 'buffer.npz'))
            save_buffer_path = os.path.join(logger.output_dir, 'buffer')
            if not os.path.exists(save_buffer_path):
                os.makedirs(save_buffer_path)
            replay_buffer.save(os.path.join(save_buffer_path, 'buffer.npz'))

        ## log sac
        sac_info = {
            "alpha": np.mean(sac_alphas),
            "loss_q": np.mean(sac_loss_qs),
            "loss_pi": np.mean(sac_loss_pis),
            "log_pi": np.mean(sac_log_pis),
        }
        logger.add_tabular_data(tb_prefix='sac', **sac_info)

        print(f"+ Preparing agent samples for discriminator")
        start = time.time()
        if parameter.disc_agent_sample_mode=="last_n_samples":
            agent_samples = sac_agent.replay_buffer.last_n_samples_sa(n=1000)
        elif parameter.disc_agent_sample_mode=="sample_from_replay_buffer":
            agent_samples = sac_agent.replay_buffer.sample_batch_sa(batch_size=1000)
        else:
            assert False
        print(f'- Preparing agent samples for discriminator. End: time {time.time() - start:.0f}s')

        ### add and train preference
        if parameter.use_pref and parameter.expand_expert and itr >= parameter.expand_expert_warm_up:
            if (itr - parameter.expand_expert_warm_up) % parameter.expand_expert_interval == 0:
                print(f"+ Add traj into expert replay buffer and update pref reward")
                tmp_state, tmp_action, tmp_reward, tmp_mask = replay_buffer.last_n_trajs(10)
                # tmp_state, tmp_action, tmp_mask = replay_buffer.last_n_trajs(max(1, parameter.expand_expert_interval//2))
                tmp_trajs = np.concatenate([tmp_state, tmp_action], axis=-1)
                with torch.no_grad():
                    tmp_D_logits = pref_reward.predict_r_hat_torch(torch.FloatTensor(tmp_trajs).to(device)).squeeze(-1)
                tmp_D_logits *= torch.FloatTensor(tmp_mask).to(device)
                if parameter.pref_beta > 0:
                    tmp_traj_weight = torch.exp(torch.clamp(torch.sum(tmp_D_logits, dim=1) / parameter.pref_beta, -float("inf"), 85))
                else:
                    tmp_traj_weight = torch.exp(torch.sum(tmp_D_logits, dim=1) * 0)
                tmp_traj_weight = (tmp_traj_weight / torch.sum(tmp_traj_weight)).cpu().numpy()
                ## 选择reward高的扩展
                # best_indexes = np.random.choice(a=np.array(list(range(tmp_traj_weight.shape[0]))), size=parameter.expand_expert_once, replace=False, p=tmp_traj_weight)
                best_indexes = sorted(list(range(tmp_traj_weight.shape[0])), key=lambda x: tmp_traj_weight[x], reverse=True)[:parameter.expand_expert_once]
                ##
                best_indexes_valid = []
                for best_index in best_indexes:
                    if np.sum(tmp_mask[best_index])>=parameter.pref_size_segment:
                        best_indexes_valid.append(best_index)
                ## 将扩展的轨迹加入reward buffer和expert buffer
                pref_reward.add_data_batch(tmp_state[best_indexes_valid],
                                           tmp_action[best_indexes_valid],
                                           tmp_reward[best_indexes_valid],
                                           tmp_mask[best_indexes_valid])
                expert_replay_buffer.store_batch(states=tmp_state[best_indexes_valid],
                                                 actions=tmp_action[best_indexes_valid],
                                                 rewards=tmp_reward[best_indexes_valid],
                                                 masks=tmp_mask[best_indexes_valid])
                ##
                if parameter.expand_pref_once>0:
                    if total_pref_queries<parameter.max_pref_size:
                        # labeled_queries = pref_reward.uniform_sampling(mb_size=parameter.expand_pref_once)
                        labeled_queries = pref_reward.disagreement_sampling(mb_size=parameter.expand_pref_once)
                        total_pref_queries += parameter.expand_pref_once
                    ## train pref reward and update weight in buffer
                    for step in range(parameter.pref_model_step):
                        train_acc = pref_reward.train_reward()
                        total_acc = np.mean(train_acc)
                        if total_acc > 0.98:
                            break
                    pref_reward_info = {
                        'TotalPrefQueries': total_pref_queries,
                        'PrefModelAccuracy': total_acc
                    }
                    logger.add_tabular_data(tb_prefix='pref_reward', **pref_reward_info)
                expert_replay_buffer.update_trajs_weight(pref_reward, parameter.pref_beta)
                expert_replay_buffer.draw(dir=pic_path, name=str(itr))
                ## 保存pref_reward和reward_buffer
                reward_model_path = os.path.join(logger.output_dir, "reward_model")
                if not os.path.exists(reward_model_path):
                    os.mkdir(reward_model_path)
                pref_reward.save(reward_model_path, "default")
                ##

            print(f"- Add traj into expert replay buffer and update pref reward")
        ###

        print(f"+ Training discriminator for {parameter.r_gradient_step} times")
        start = time.time()
        ### Preparing expert samples for discriminator from expert buffer
        if parameter.use_pref:
            expert_samples = expert_replay_buffer.sample_with_pref()
        else:
            expert_samples = expert_replay_buffer.sample_all_without_pref()
        # assert expert_samples[0].shape[0]==sum(expert_traj_nums)
        ###
        
        # optimization w.r.t. reward
        reward_losses = []
        for _ in range(parameter.r_gradient_step):
            if alg_name == 'maxentirl_sa':
                loss = maxentirl_sa_loss(alg_name, agent_samples=agent_samples, expert_samples=expert_samples, reward_func=reward_func, device=device)
            elif alg_name == 'gail':
                loss = gail_loss(alg_name, agent_samples=agent_samples, expert_samples=expert_samples, reward_func=reward_func, device=device)
            elif alg_name == 'maxentirl':
                loss = maxentirl_loss(alg_name, agent_samples=agent_samples, expert_samples=expert_samples, reward_func=reward_func, device=device)
            else:
                assert False
                
            reward_losses.append(loss.item())
            reward_optimizer.zero_grad()
            loss.backward()
            reward_optimizer.step()
        print(f'- Training discriminator End: time {time.time() - start:.0f}s')

        # evaluating the learned reward
        log_info = try_evaluate(itr, "Running", alg_name, expert_trajs=np.concatenate([expert_replay_buffer.states, expert_replay_buffer.actions], axis=-1).squeeze())

        if log_info['StoEpRetTest'] > max_real_return_sto:
            max_real_return_sto = log_info['StoEpRetTest']
            save_reward_dir = os.path.join(logger.output_dir, "best_model")
            if not os.path.exists(save_reward_dir):
                os.makedirs(save_reward_dir)
            torch.save(reward_func.state_dict(), os.path.join(save_reward_dir, 'reward_model_best.pkl'))
        
        save_actor_dir = os.path.join(logger.output_dir, "best_model", "actors")
        if not os.path.exists(save_actor_dir):
            os.makedirs(save_actor_dir)
        torch.save(reward_func.state_dict(), os.path.join(save_reward_dir, 'reward_model_last.pkl'))
        torch.save(sac_agent.ac.state_dict(), os.path.join(save_actor_dir, f"sac_actor_epoch{itr}_return{log_info['StoEpRetTest']:.4f}_distance{log_info['ExpertDistance']:.4f}.pt"))
        log_info["Itration"] = itr
        log_info["discriminator_loss"] = np.mean(reward_losses)
        logger.add_tabular_data(**log_info)
        logger.dump_tabular()