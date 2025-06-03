import math

import gym
from gym.spaces import Box
import numpy as np
from typing import Tuple, Union, Dict, Optional
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
except ModuleNotFoundError:
    print(f'Matplotlib import fail, render is not available.')


class CustomizedGrid(gym.Env):
    WIDTH = 2.0
    HEIGHT = 2.0
    RADIUS = 0.75
    MAX_STEP = 100
    MAX_STEP_SIZE = 0.05
    INIT_WIDTH = 0.1
    INIT_HEIGHT = 0.1
    def __init__(self, T=100, seed=None):
        # obs: x, y, target_x, target_y, x - target_x, y - target_y, l2_norm_pos, l2_norm_target
        self.position_minium = np.array([-self.WIDTH / 2, -self.HEIGHT / 2])
        self.position_maximum = np.array([self.WIDTH / 2, self.HEIGHT / 2])
        self.observation_space = Box(low=np.array([self.position_minium[0], self.position_minium[1], self.position_minium[0], self.position_minium[1]]),
                                    high=np.array([self.position_maximum[0], self.position_maximum[1], self.position_maximum[0], self.position_maximum[1]]),
                                    dtype=np.float64)
        self.action_minimum = np.array([-self.MAX_STEP_SIZE, -self.MAX_STEP_SIZE])
        self.action_maximum = np.array([self.MAX_STEP_SIZE, self.MAX_STEP_SIZE])
        self.action_space = Box(low=self.action_minimum,
                                high=self.action_maximum,
                                dtype=np.float64)
        self.rnd_seed = np.random.RandomState(seed=seed)
        self.seed(seed)

        self.position = np.array([
            (self.rnd_seed.rand() * 2 - 1) * self.INIT_WIDTH / 2,
            (self.rnd_seed.rand() * 2 - 1) * self.INIT_HEIGHT / 2,
                                  ])
        self.target = np.array([
            (self.rnd_seed.rand() * 2 - 1) * self.WIDTH / 2,
            (self.rnd_seed.rand() * 2 - 1) * self.HEIGHT / 2,
        ])
        self.step_cnt = 0

    def seed(self, seed=None):
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
            self.rnd_seed.seed(seed)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Optional[Dict]]:
        self.step_cnt += 1
        if isinstance(action, list):
            action = np.array(action)
        action = action.reshape((-1,))
        action = np.clip(action, a_max=self.action_maximum, a_min=self.action_minimum)
        self.position = self.position + action
        reward = self._reward(self.position, self.target, radius=self.RADIUS)
        self.position = np.clip(self.position, a_max=self.position_maximum, a_min=self.position_minium)

        done = self.step_cnt >= self.MAX_STEP

        next_obs = self._get_obs()
        return next_obs, reward, done, {}

    def reset(self) -> np.ndarray:
        self.position = np.array([
            (self.rnd_seed.rand() * 2 - 1) * self.INIT_WIDTH / 2,
            (self.rnd_seed.rand() * 2 - 1) * self.INIT_HEIGHT / 2,
        ])
        self.target = np.array([
            (self.rnd_seed.rand() * 2 - 1) * self.WIDTH / 2,
            (self.rnd_seed.rand() * 2 - 1) * self.HEIGHT / 2,
        ])
        self.step_cnt = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.position[0],
            self.position[1],
            self.target[0],
            self.target[1]
        ])

    def render(self, mode="human") -> None:
        if mode == "human":
            if not hasattr(self, 'figure') or not hasattr(self, 'ax'):
                self.figure, self.ax = plt.subplots()
                self.line, = self.ax.plot([], [], 'ro', markersize=10)  # 代表玩家的点
                self.target_plot, = self.ax.plot([], [], 'bo', markersize=10)  # 代表目标的点
                self.ax.set_xlim(-self.WIDTH / 2 * 1.04, self.WIDTH / 2 * 1.04)
                self.ax.set_ylim(-self.HEIGHT / 2 * 1.04, self.HEIGHT / 2 * 1.04)
                self.ax.set_aspect('equal', adjustable='box')
                self.ax.grid()
                moveable_area = Circle((0, 0), self.RADIUS, color='green', fill=False)
                self.ax.add_patch(moveable_area)
            self.line.set_data(self.position[0], self.position[1])
            self.target_plot.set_data(self.target[0], self.target[1])
            plt.pause(1 / 30)  # 短暂暂停以更新图像
    @staticmethod
    def _reward(position: np.ndarray, target: np.ndarray, radius: float) -> Union[float, np.ndarray]:
        max_distance = 2 * math.sqrt(2)
        position_norm = np.linalg.norm(position, axis=-1)
        target_norm = np.linalg.norm(target, axis=-1)
        distance_norm = np.linalg.norm(position - target, axis=-1)
        PENALTY_RATIO = 10.0
        REWARD_BASELINE_RATIO = 0.2
        # single reward computation
        if position_norm > radius:
            reward = -PENALTY_RATIO * max_distance
        else:
            reward = max_distance * REWARD_BASELINE_RATIO - distance_norm
            if target_norm <= radius and distance_norm < 0.01 * max_distance:
                reward = max_distance - distance_norm
            if target_norm > radius and distance_norm < target_norm - 0.75 + 0.1 * max_distance:
                reward = max_distance
        return reward
    # def _reward(position: np.ndarray, target: np.ndarray, radius: float) -> Union[float, np.ndarray]:
    #     max_distance = 2 * math.sqrt(2)
    #     position_norm = np.linalg.norm(position, axis=-1)
    #     distance_norm = np.linalg.norm(position - target, axis=-1)
    #     PENALTY_RATIO = 10.0
    #     REWARD_BASELINE_RATIO = 0.2
    #     if np.isscalar(position_norm):
    #         # single reward computation
    #         if position_norm > radius:
    #             reward = -PENALTY_RATIO * max_distance
    #         else:
    #             reward = max_distance * REWARD_BASELINE_RATIO - distance_norm
    #             if distance_norm < 0.01 * max_distance:
    #                 reward += max_distance
    #     else:
    #         # batch reward computation
    #         reward = np.zeros_like(position_norm)
    #         position_exceeding_mask = position_norm > radius
    #         reward[position_exceeding_mask] = -PENALTY_RATIO * max_distance
    #         reward[~position_exceeding_mask] = max_distance * REWARD_BASELINE_RATIO - distance_norm[~position_exceeding_mask]
    #     return reward

    def _get_obs_batch(self) -> np.ndarray:
        return np.concatenate([self.position, self.target], axis=-1)

    # position: batch x 2
    # target: batch x 2
    def reset_batch(self, batch) -> np.ndarray:
        pos_w = self.rnd_seed.rand(batch, 1) * 2 - 1
        pos_h = self.rnd_seed.rand(batch, 1) * 2 - 1
        pos_w = pos_w * self.INIT_WIDTH / 2
        pos_h = pos_h * self.INIT_HEIGHT / 2
        self.position = np.concatenate((pos_w, pos_h), axis=-1)
        
        tar_w = self.rnd_seed.rand(batch, 1) * 2 - 1
        tar_h = self.rnd_seed.rand(batch, 1) * 2 - 1
        tar_w = pos_w * self.WIDTH / 2
        tar_h = pos_h * self.HEIGHT / 2
        self.target = np.concatenate((tar_w, tar_h), axis=-1)
        return self._get_obs_batch()

    def step_batch(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Optional[Dict]]:
        self.step_cnt += 1
        if isinstance(action, list):
            action = np.array(action)
        action = action.reshape((-1, 2))
        action = np.clip(action, a_max=self.action_maximum, a_min=self.action_minimum)
        self.position = self.position + action
        self.position = np.clip(self.position, a_max=self.position_maximum, a_min=self.position_minium)

        done = self.step_cnt >= self.MAX_STEP

        next_obs = self._get_obs_batch()
        return next_obs, None, done, {}



gym.register(id='CustomizedGridWorld-v0',
             entry_point=CustomizedGrid)

def main():
    # env = gym.make('CustomizedGridWorld-v0')
    # obs = env.reset()
    # done = False
    # cum_ret = 0
    # while not done:
    #     next_obs, reward, done, info = env.step(env.action_space.sample())
    #     # print(next_obs)
    #     env.render()
    #     cum_ret += reward
    # print(f'cum_ret: {cum_ret}')

    env = gym.make('CustomizedGridWorld-v0')
    for target, state in [[(-0.9, -0.9), "out_of_circle"], [(-0.4, -0.4), "in_circle"]]:
        env.target = np.array(target)
        x_heap, y_heap, z_heap = [], [], []
        step_size = 0.01
        for i in np.arange(-1, 1+step_size, step_size):
            tmp_x = []
            tmp_y = []
            tmp_z = []
            for j in np.arange(-1, 1+step_size, step_size):
                tmp_x.append(i)
                tmp_y.append(j)
                pos = np.array([i, j])
                tmp_z.append(env._reward(pos, env.target, 0.75))
            x_heap.append(tmp_x)
            y_heap.append(tmp_y)
            z_heap.append(tmp_z)

        WIDTH = HEIGHT = 2.0
        RADIUS = 0.75
        figure, ax = plt.subplots()
        ax.set_xlim(-WIDTH / 2 * 1.04, WIDTH / 2 * 1.04)
        ax.set_ylim(-HEIGHT / 2 * 1.04, HEIGHT / 2 * 1.04)
        ax.set_aspect('equal', adjustable='box')
        ax.grid()
        c = ax.pcolormesh(x_heap, y_heap, z_heap, cmap='viridis_r', shading='gouraud')
        plt.colorbar(c, label='AUPR')
        moveable_area = Circle((0, 0), RADIUS, color='green', fill=False)
        ax.add_patch(moveable_area)
        plt.savefig("./CustomizedGridWorld_reward_{}.png".format(state))
        plt.close()


if __name__ == '__main__':
    main()