#!/usr/bin/env python3

import warnings
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'
from pathlib import Path

import sys
sys.path.append('..')
sys.path.append('.')

import hydra
import numpy as np
import torch
from dm_env import specs

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_expert_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import pickle

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

torch.backends.cudnn.benchmark = True
from reward.pref_reward import PrefRewardModel


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec[cfg.obs_type].shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class WorkspaceIL:
    def __init__(self, cfg, work_dir):
        self.work_dir = Path(work_dir)
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        # self.convert_for_preload(f'{cfg.buffer_dir}/buffer_{cfg.buffer_type}.pkl', self.work_dir / 'buffer')
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), cfg.agent)

        if repr(self.agent) == 'drqv2':
            self.cfg.suite.num_train_frames = self.cfg.num_train_frames_drq
        if repr(self.agent) == 'bc':
            self.cfg.suite.num_train_frames = self.cfg.num_train_frames_bc
            self.cfg.suite.num_seed_frames = 0

        # Load weights
        if cfg.bc_regularize:
            snapshot = Path(cfg.bc_weight)
            if snapshot.exists():
                print(f'resuming bc: {snapshot}')
                self.agent.bc_actor = self.load_snapshot_not_inplace(snapshot)

        self.expert_replay_loader = make_expert_replay_loader(
            self.cfg.expert_dataset, self.cfg.batch_size // 2, self.cfg.num_demos, self.cfg.obs_type)
        self.expert_replay_iter = iter(self.expert_replay_loader)
            
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        with open(self.cfg.expert_dataset, 'rb') as f:
            if self.cfg.obs_type == 'pixels':
                self.expert_demo, _, _, self.expert_reward = pickle.load(f)
            elif self.cfg.obs_type == 'features':
                _, self.expert_demo, _, self.expert_reward = pickle.load(f)
        self.expert_demo = self.expert_demo[:self.cfg.num_demos]
        self.expert_reward = np.mean(self.expert_reward[:self.cfg.num_demos])

        self.pbrl_reward_model = PrefRewardModel(
                obs_shape=self.train_env.observation_spec()[cfg.obs_type].shape,
                action_dim=self.train_env.action_spec().shape,
                use_action=cfg.use_action,
                feature_dim=cfg.agent.feature_dim,
                hidden_sizes=(512, 512),
                device=self.device,).to(self.device)
        self.pbrl_reward_model.load_state_dict(torch.load(cfg.reward_path))
        print(f'Load reward model from {cfg.reward_path}')
        
    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, task_type='eval_reward')
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
                                                  self.work_dir / 'buffer',
                                                  Path(self.cfg.load_dir))

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
    @torch.no_grad()
    def eval(self):
        step, episode, total_reward, total_reward_our = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)

        if self.cfg.suite.name == 'openaigym' or self.cfg.suite.name == 'metaworld':
            paths = []
        while eval_until_episode(episode):
            observations, actions = [], []
            if self.cfg.suite.name == 'metaworld':
                path = []
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation[self.cfg.obs_type],
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                if self.cfg.suite.name == 'metaworld':
                    path.append(time_step.observation['goal_achieved'])
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                observations.append(time_step.observation[self.cfg.obs_type])
                actions.append(time_step.action)

                step += 1

            observations = torch.from_numpy(np.stack(observations)).to(self.device) # seq_len * c * h * w
            seq_len, c, h, w = observations.shape
            actions = torch.from_numpy(np.stack(actions)).to(self.device)
            total_reward_our += self.pbrl_reward_model(observations, actions).sum()

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')
            if self.cfg.suite.name == 'openaigym':
                paths.append(time_step.observation['goal_achieved'])
            elif self.cfg.suite.name == 'metaworld':
                paths.append(1 if np.sum(path)>10 else 0)
        
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward_real', total_reward / episode)
            log('episode_reward_our', total_reward_our / episode)
            log('episode_length', step * self.cfg.suite.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            if repr(self.agent) != 'drqv2':
                log('expert_reward', self.expert_reward)
            if self.cfg.suite.name == 'openaigym' or self.cfg.suite.name == 'metaworld':
                log("success_percentage", np.mean(paths))


    def train_il(self):
        # predicates
        train_until_step = utils.Until(self.cfg.suite.num_train_frames,
                                       self.cfg.suite.action_repeat)
        seed_until_step = utils.Until(self.cfg.suite.num_seed_frames,
                                      self.cfg.suite.action_repeat)
        eval_every_step = utils.Every(self.cfg.suite.eval_every_frames,
                                      self.cfg.suite.action_repeat)

        episode_step, episode_reward = 0, 0

        time_steps = list()
        observations = list()
        actions = list()

        time_step = self.train_env.reset()
        time_steps.append(time_step)
        observations.append(time_step.observation[self.cfg.obs_type])
        actions.append(time_step.action)
        
        if repr(self.agent) == 'potil':
            if self.agent.auto_rew_scale:
                self.agent.sinkhorn_rew_scale = 1.  # Set after first episode

        self.train_video_recorder.init(time_step.observation[self.cfg.obs_type])
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                if self._global_episode % 10 == 0:
                    self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated

                observations = torch.from_numpy(np.stack(observations)).to(self.device) # seq_len * c * h * w
                actions = torch.from_numpy(np.stack(actions)).to(self.device)
                new_rewards = self.pbrl_reward_model(observations, actions).detach().cpu().numpy().squeeze()

                for i, elt in enumerate(time_steps):
                    elt = elt._replace(
                        observation=time_steps[i].observation[self.cfg.obs_type])
                    elt = elt._replace(reward=new_rewards[i]) # 注释该行则使用默认reward
                    self.replay_storage.add(elt)

                # reset env
                time_steps = list()
                observations = list()
                actions = list()

                time_step = self.train_env.reset()
                time_steps.append(time_step)
                observations.append(time_step.observation[self.cfg.obs_type])
                actions.append(time_step.action)
                self.train_video_recorder.init(time_step.observation[self.cfg.obs_type])
                # try to save snapshot
                if self.cfg.suite.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()
                self.save_snapshot_iter(self.global_frame)
                
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation[self.cfg.obs_type],
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                # Update
                metrics = self.agent.update(self.replay_iter, self.expert_replay_iter, 
                                            self.global_step, self.cfg.bc_regularize)
                if metrics is not None and self.global_step % int(1e3) == 0:
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('train_total_time', self.timer.total_time())
                        for key, value in metrics.items():
                            log(key, value)
                elif self.global_step % int(1e2) == 0:
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward

            time_steps.append(time_step)
            observations.append(time_step.observation[self.cfg.obs_type])
            actions.append(time_step.action)

            self.train_video_recorder.record(time_step.observation[self.cfg.obs_type])
            episode_step += 1
            self._global_step += 1

    def convert_for_preload(self, pkl_file_path, output_dir):
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载 pkl 文件数据
        with open(pkl_file_path, 'rb') as f:
            buffer_data = pickle.load(f)
        
        # buffer_data 格式假设为 [(iteration, [list of trajectories]), ...]
        traj_idx = 0
        for iter_idx, trajectories in buffer_data:
            for _, (obs, states, actions, rewards) in enumerate(trajectories):
                # 构造文件名
                npz_file_path = os.path.join(
                    output_dir, 
                    f"preload_{traj_idx}_{len(obs)}.npz"
                )
                # 保存 .npz 文件
                np.savez_compressed(
                    npz_file_path,
                    observation=np.array(obs, dtype=np.float32),
                    state=np.array(states, dtype=np.float32),
                    action=np.array(actions, dtype=np.float32),
                    reward=np.array(rewards, dtype=np.float32),
                    discount=np.ones((rewards.shape), dtype=np.float32)
                )
                print(f"Saved iteration {iter_idx}, trajectory {traj_idx} with {len(obs)} transitions to {npz_file_path}")
                traj_idx += 1

        print(f"All trajectories have been converted and saved to {output_dir}.")

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open('wb') as f:
            torch.save(payload, f)
    
    def save_snapshot_iter(self, iter):
        snapdir = self.work_dir / 'snapshots'
        snapdir.mkdir(exist_ok=True)
        snapshot = snapdir / f'snapshot_{iter}.pt'
        keys_to_save = ['timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        self.agent.load_snapshot(agent_payload)
    
    def load_snapshot_not_inplace(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        agent_payload = {}
        agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), self.cfg.agent)
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        agent.load_snapshot(agent_payload)
        return agent


@hydra.main(config_path='../cfgs', config_name='config_eval')
def main(cfg):
    workspace = WorkspaceIL(cfg, work_dir=f'eval_reward/{cfg.suite.name}_{cfg.num_demos}/{cfg.task_name}_{cfg.exp_id}/{cfg.seed}')
    # workspace.load_snapshot(Path(cfg.bc_weight))
    workspace.train_il()


if __name__ == '__main__':
    main()
