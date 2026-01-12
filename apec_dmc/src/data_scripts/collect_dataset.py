import warnings

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
from dm_env import specs

import utils
from logger import Logger
from video import TrainVideoRecorder, VideoRecorder
import pickle
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec[cfg.obs_type].shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg, work_dir):
        self.work_dir = Path(work_dir)
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create envs
        self.train_env = hydra.utils.call(self.cfg.suite.task_make_fn)
        self.eval_env = hydra.utils.call(self.cfg.suite.task_make_fn)

        self.video_recorder = VideoRecorder(None)
        self.train_video_recorder = TrainVideoRecorder(None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def generate_buffer(self, iter):
        step, episode, total_reward = 0, 0, 0
        generate_until_episode = utils.Until(self.cfg.gen_num_demos)
        observations_list = list()
        states_list = list()
        actions_list = list()
        rewards_list = list()
        while generate_until_episode(episode):
            observations = list()
            states = list()
            actions = list()
            rewards = list()
            episode_reward = 0

            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env)
            while not time_step.last():
                observations.append(time_step.observation[self.cfg.obs_type])
                states.append(time_step.observation['state']) # 0114: 由features改为state，features不包含直观速度和位移等取值
                rewards.append(time_step.reward)
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        time_step.observation[self.cfg.obs_type],
                        self.global_step,
                        eval_mode=True)
                time_step = self.eval_env.step(action)
                actions.append(time_step.action)
                self.video_recorder.record(self.eval_env)
                episode_reward += time_step.reward
                step += 1

            # print(episode, episode_reward)
            # if episode_reward > 100:
            episode += 1
            self.video_recorder.save(f'{iter}_{episode}_eval.mp4')
            rewards_list.append(np.array(rewards))
            observations_list.append(np.stack(observations, 0))
            states_list.append(np.stack(states, 0))
            actions_list.append(np.stack(actions, 0))
                
        # Make np arrays
        observations_list = np.array(observations_list, dtype=np.uint8)
        states_list = np.array(states_list)
        actions_list = np.array(actions_list)
        rewards_list = np.array(rewards_list)

        # Save demo in pickle file
        payload = [
            observations_list, states_list, actions_list, rewards_list
        ]

        return payload        


    def load_snapshot(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        self.agent.load_snapshot(agent_payload)


@hydra.main(config_path='../cfgs', config_name='config_collect')
def main(cfg):
    buffer_type = cfg.buffer_type
    workspace = Workspace(cfg, work_dir=f'collect_buffer/{cfg.suite.name}_{cfg.num_demos}/{cfg.task_name}/{cfg.seed}/buffer/{buffer_type}')

    save_dir = cfg.buffer_dir
    if not os.path.exists(os.path.abspath(os.path.dirname(save_dir))):
        print(f'Making directory:{save_dir}')
        os.makedirs(os.path.abspath(os.path.dirname(save_dir)))
        assert os.path.exists(os.path.abspath(os.path.dirname(save_dir)))
    print(f'buffer will be saved at {save_dir}')
    samples = []
    # Load weights
    snapshot_dir = Path(cfg.weight)
    weight_filenames = sorted(os.listdir(snapshot_dir), key=lambda x: int(x.split('_')[-1].replace('.pt', '')))

    with tqdm(total=cfg.num_demos*len(weight_filenames)) as pbar:
        pbar.set_description('collecting buffer')
        
        for i, weight_name in enumerate(weight_filenames):
            snapshot_path = os.path.join(snapshot_dir, weight_name)
            iteration = int(weight_name.split('_')[-1].replace('.pt','')) 
            # if iteration >= 1000000:
            #     continue

            snapshot_path = Path(snapshot_path)
            workspace.load_snapshot(snapshot_path)
            
            obs, states, actions, rewards = workspace.generate_buffer(iteration) # n_samples * seq_len * feature_dim
            print(weight_name, rewards.sum(axis=1).mean(), rewards.sum(axis=1).std())
            os.makedirs(f'{save_dir}/{i}', exist_ok=True)
            for j in range(obs.shape[0]):
                with open(f'{save_dir}/{i}/{j}.pkl', 'wb') as f:
                    pickle.dump((obs[j], states[j], actions[j], rewards[j]), f)

            pbar.update(obs.shape[0])

            sizeofsample = sys.getsizeof(samples) / (1024**3)
            print(len(samples))
            pbar.set_postfix_str(f'sample size:{sizeofsample}G')
    
    print(f'buffer saved at {save_dir}')


if __name__ == '__main__':
    main()
