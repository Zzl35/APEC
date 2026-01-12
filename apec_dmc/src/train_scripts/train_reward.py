#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import sys
sys.path.append('..')
sys.path.append('.')

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
from dm_env import specs, StepType
from matplotlib import pyplot as plt 
from plot.plot_utils import COLORS

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

from reward.pref_reward import PrefRewardModel
from data_scripts.make_pref_dataset import PreferenceBuffer

warnings.simplefilter("always", RuntimeWarning)


def to_torch(batch, device):
    for i in range(len(batch)):
        batch[i] = batch[i].to(device)
    return batch


def eval_single_buffer(buffer, reward_func, device=torch.device('cpu')):
    pred_returns_all, real_returns_all, pred_rewards_all, real_rewards_all = [], [], [], []
    accuracy, syn_accuracy = 0, 0
    max_gt_return_global = -1e6
    max_gt_pred_return_global = -1e6
    max_pred_return_global = -1e6
    max_pred_gt_return_global = -1e6
    total_size = 0
    for batch in buffer:
        batch = to_torch(batch, device)
        states1, states2, actions1, actions2, real_rewards1, real_rewards2, masks_1, masks_2, pref_label = batch

        # Predicted rewards and returns
        bs, seq_len, c, h, w = states1.shape
        states1 = states1.reshape(-1, c, h, w)
        states2 = states2.reshape(-1, c, h, w)

        with torch.no_grad():
            pred_rewards1 = reward_func(states1, actions1.reshape(-1, actions1.shape[-1])).reshape(bs, -1)
            pred_rewards2 = reward_func(states2, actions2.reshape(-1, actions2.shape[-1])).reshape(bs, -1)

        pred_returns1 = torch.sum(pred_rewards1 * masks_1, dim=-1, keepdim=True)
        pred_returns2 = torch.sum(pred_rewards2 * masks_2, dim=-1, keepdim=True) 
        real_returns1 = torch.sum(real_rewards1 * masks_1, dim=-1, keepdim=True)
        real_returns2 = torch.sum(real_rewards2 * masks_2, dim=-1, keepdim=True) 

        # Preference labels
        p_returns = torch.cat([pred_returns1, pred_returns2], dim=-1).detach().cpu().numpy()
        r_returns = torch.cat([real_returns1, real_returns2], dim=-1).detach().cpu().numpy()
        pred_label = np.argmax(p_returns, axis=1)
        real_label = np.argmax(r_returns, axis=1)
        accuracy += np.sum(pred_label == real_label)
        syn_accuracy += np.sum(pred_label == np.argmax(pref_label.detach().cpu().numpy(), axis=1))

        # Flatten rewards
        masks = torch.cat([masks_1, masks_2], dim=0).detach().cpu().numpy().reshape(-1)
        real_rewards = torch.cat([real_rewards1, real_rewards2], dim=0).detach().cpu().numpy().reshape(-1)
        pred_rewards = torch.cat([pred_rewards1, pred_rewards2], dim=0).detach().cpu().numpy().reshape(-1)
        real_rewards = real_rewards[masks > 0]
        pred_rewards = pred_rewards[masks > 0]
        real_returns = torch.cat([real_returns1, real_returns2], dim=0).detach().cpu().numpy().reshape(-1)
        pred_returns = torch.cat([pred_returns1, pred_returns2], dim=0).detach().cpu().numpy().reshape(-1)

        # Maximum returns
        max_gt_idx = np.argmax(real_returns)
        max_gt_return = real_returns[np.argmax(real_returns)]
        max_gt_idx_pred_return = pred_returns[max_gt_idx]
        if max_gt_return > max_gt_return_global:
            max_gt_return_global = max_gt_return
            max_gt_pred_return_global = max_gt_idx_pred_return

        max_pred_idx = np.argmax(pred_returns)
        max_pred_return = pred_returns[max_pred_idx]
        max_pred_idx_gt_return = real_returns[max_pred_idx]
        if max_pred_return > max_pred_return_global:
            max_pred_return_global = max_pred_return
            max_pred_gt_return_global = max_pred_idx_gt_return

        pred_returns_all.append(pred_returns)
        real_returns_all.append(real_returns)
        pred_rewards_all.append(pred_rewards)
        real_rewards_all.append(real_rewards)

        total_size += bs
        if total_size > 500:
            break

    # Aggregate results
    pred_returns_all = np.concatenate(pred_returns_all, axis=0)
    real_returns_all = np.concatenate(real_returns_all, axis=0)
    pred_rewards_all = np.concatenate(pred_rewards_all, axis=0)
    real_rewards_all = np.concatenate(real_rewards_all, axis=0)
    accuracy /= total_size
    syn_accuracy /= total_size

    # Update evaluation results
    eval_results = {}
    eval_results["ReturnCorr"] = np.corrcoef(pred_returns_all, real_returns_all)[0, 1]
    eval_results["RewardCorr"] = np.corrcoef(pred_rewards_all, real_rewards_all)[0, 1]
    eval_results["PrefLabelAcc"] = accuracy
    eval_results["SyntheticLabelAcc"] = syn_accuracy
    eval_results['MaxGTReturn'] = max_gt_return_global
    eval_results['MaxGTPredReturn'] = max_gt_pred_return_global
    eval_results['MaxPredReturn'] = max_pred_return_global
    eval_results['MaxPredGTReturn'] = max_pred_gt_return_global

    return eval_results, real_rewards_all, pred_rewards_all, real_returns_all, pred_returns_all


def evaluate_buffer(train_buffer, test_buffer, reward_func, device=torch.device('cpu'), savepath=None):
    train_eval_results, train_real_rewards, train_pred_rewards, train_real_returns, train_pred_returns = eval_single_buffer(train_buffer, reward_func, device)
    test_eval_results, test_real_rewards, test_pred_rewards, test_real_returns, test_pred_returns = eval_single_buffer(test_buffer, reward_func, device)
    FONTSIZE = 7
    if savepath is not None:
        # plot reward corr
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        real_rewards = np.concatenate([train_real_rewards, test_real_rewards])
        pred_rewards = np.concatenate([train_pred_rewards, test_pred_rewards])
        real_returns = np.concatenate([train_real_returns, test_real_returns])
        pred_returns = np.concatenate([train_pred_returns, test_pred_returns])
        
        real_min, real_max = real_rewards.min(), real_rewards.max()
        pred_min, pred_max = pred_rewards.min(), pred_rewards.max()
        indices = np.random.choice(real_rewards.shape[0], size=2000, replace=False)
        axes[0].scatter(x=real_rewards[indices], y=pred_rewards[indices], marker='.', color=COLORS[0])
        axes[0].plot([real_min, real_max], [pred_min, pred_max], linestyle='--', color='black')
        axes[0].set_xlabel('Ground Truth Reward', fontsize=FONTSIZE+1)
        axes[0].set_ylabel('Predicted Reward', fontsize=FONTSIZE+1)
        axes[0].tick_params(labelsize=FONTSIZE)

        real_min, real_max = real_returns.min(), real_returns.max()
        pred_min, pred_max = pred_returns.min(), pred_returns.max()
        train_indices = np.arange(len(train_real_returns))
        test_indices = np.arange(len(train_real_returns), len(real_returns))
        axes[1].scatter(x=real_returns[train_indices], y=pred_returns[train_indices], marker='.', color=COLORS[0])
        axes[1].scatter(x=real_returns[test_indices], y=pred_returns[test_indices], marker='.', color=COLORS[1])
        axes[1].plot([real_min, real_max], [pred_min, pred_max], linestyle='--', color='black')
        axes[1].set_xlabel('Ground Truth Return', fontsize=FONTSIZE+1)
        axes[1].set_ylabel('Predicted Return', fontsize=FONTSIZE+1)
        axes[1].tick_params(labelsize=FONTSIZE)

        if not os.path.exists(os.path.abspath(os.path.dirname(savepath))):
            os.makedirs(os.path.abspath(os.path.dirname(savepath)))
        abspath = os.path.abspath(savepath)
        print(f'correlation saved at {abspath}')
        plt.savefig(savepath)
        plt.close()

    return train_eval_results, test_eval_results

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec[cfg.obs_type].shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class WorkspacePBRL:
    def __init__(self, cfg, work_dir):
        self.work_dir = Path(work_dir)
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.reward_model = PrefRewardModel(
                obs_shape=self.train_env.observation_spec()[cfg.obs_type].shape,
                action_dim=self.train_env.action_spec().shape,
                use_action=cfg.use_action,
                feature_dim=cfg.agent.feature_dim,
                hidden_sizes=(512, 512),
                device=self.device,
            ).to(self.device)
        self.shift_aug = utils.RandomShiftsAug(pad=4)
        print('Loading buffer')

        train_buffer = PreferenceBuffer(self.cfg.train_buffer_path, self.cfg.train_idx_path, self.cfg.segment_len, device=self.device)
        self.train_reward_buffer = torch.utils.data.DataLoader(train_buffer, batch_size=self.cfg.micro_bs, shuffle=True, num_workers=6)

        valid_buffer = PreferenceBuffer(self.cfg.train_buffer_path, self.cfg.valid_idx_path, 500, device=self.device)
        self.valid_reward_buffer = torch.utils.data.DataLoader(valid_buffer, batch_size=8, shuffle=False, num_workers=6)

        test_reward_buffer = PreferenceBuffer(self.cfg.test_buffer_path, self.cfg.test_idx_path, 500, device=self.device)
        self.test_reward_buffer = torch.utils.data.DataLoader(test_reward_buffer, batch_size=8, shuffle=False, num_workers=6)
        # self.test_reward_buffer.load(self.cfg.train_path)
        print('Load buffer done.')

        self.reward_opt = torch.optim.AdamW(self.reward_model.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
            
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        
    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, task_type='train_pbrl')
        # create envs
        self.train_env = hydra.utils.call(self.cfg.suite.task_make_fn)
        self.eval_env = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create replay buffer
        data_specs = [
            self.train_env.observation_spec()[self.cfg.obs_type],
            self.train_env.action_spec(),
            specs.Array((1, ), np.float32, 'reward'),
            specs.Array((1, ), np.float32, 'discount')
        ]

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.suite.save_snapshot, self.cfg.nstep, self.cfg.suite.discount)

        self._replay_iter = None
        self.expert_replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def bt_update(self, optim, macro_bs=64):
        total_loss = 0.0
        
        size = 0
        for train_samples_batch in self.train_reward_buffer:
            train_samples_batch = to_torch(train_samples_batch, self.device)
            states1, states2, actions1, actions2, rewards1, rewards2, masks_1, masks_2, pref_label = train_samples_batch

            bs, seq_len, c, h, w = states1.shape
            if self.cfg.use_shift_aug:
                states1 = self.shift_aug(states1.reshape(-1, c, h, w))
                states2 = self.shift_aug(states2.reshape(-1, c, h, w))
            else:
                states1 = states1.reshape(-1, c, h, w)
                states2 = states2.reshape(-1, c, h, w)
            actions1 = actions1.reshape(-1, actions1.shape[-1])
            actions2 = actions2.reshape(-1, actions2.shape[-1])

            # Compute predicted rewards and reshape
            pred_rewards1 = self.reward_model(states1, actions1).reshape(bs, -1)
            pred_rewards2 = self.reward_model(states2, actions2).reshape(bs, -1)

            # Compute predicted returns
            pred_returns1 = torch.sum(pred_rewards1 * masks_1, dim=-1, keepdim=True)
            pred_returns2 = torch.sum(pred_rewards2 * masks_2, dim=-1, keepdim=True)

            # Compute probabilities using Bradley-Terry model
            prob = torch.sigmoid(pred_returns2 - pred_returns1)
            batch_loss = F.binary_cross_entropy(
                prob, torch.argmax(pref_label, dim=-1, keepdim=True).float(), reduction='sum'
            )
            if size == 0:
                optim.zero_grad()
            batch_loss.backward()
            size += bs
            if size == macro_bs:
                optim.step()
                size = 0

            # Accumulate loss
            total_loss += batch_loss.item()

        return total_loss

    def train_pbrl(self):
        save_reward_dir = os.path.join(self.work_dir, "best_model")
        save_corr_dir = os.path.join(self.work_dir, 'correlations')
        os.makedirs(save_reward_dir, exist_ok=True)

        for itr in range(50):
            pref_loss = self.bt_update(optim=self.reward_opt, macro_bs=self.cfg.macro_bs)
            torch.save(self.reward_model.state_dict(), os.path.join(save_reward_dir, f'reward_model_{itr}.pkl'))
            if itr % 1 == 0:
                log_info_train_accum = {}
                log_info_test_accum = {}
                log_info_train, log_info_test = evaluate_buffer(self.valid_reward_buffer, self.test_reward_buffer, self.reward_model, self.device, savepath=os.path.join(save_corr_dir, f'corr_{itr}.pdf'))

                for key in log_info_train.keys():
                    if key not in log_info_train_accum:
                        log_info_train_accum[key] = log_info_train[key]
                        log_info_test_accum[key] = log_info_test[key]
                    else:
                        log_info_train_accum[key] += log_info_train[key]
                        log_info_test_accum[key] += log_info_test[key]

                log_info = {}
                log_info['Itration'] = itr
                log_info['pref_loss'] = pref_loss
                for key in log_info_train_accum.keys():
                    log_info[f'train_{key}'] = log_info_train_accum[key]
                    log_info[f'test_{key}'] = log_info_test_accum[key]

                torch.save(self.reward_model.state_dict(), os.path.join(save_reward_dir, f'reward_model_last.pkl')) 
                with self.logger.log_and_dump_ctx(itr, ty='train') as log:
                    for key in log_info.keys():
                        log(key, log_info[key])


@hydra.main(config_path='../cfgs', config_name='config_pbrl')
def main(cfg):
    workspace = WorkspacePBRL(cfg, work_dir=f'train_pbrl/{cfg.suite.name}_{cfg.num_demos}/{cfg.task_name}_{cfg.exp_id}/{cfg.seed}')
    workspace.train_pbrl()


if __name__ == '__main__':
    main()
