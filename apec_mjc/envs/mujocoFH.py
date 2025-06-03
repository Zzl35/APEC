# Fixed Horizon wrapper of mujoco environments
import gym
import numpy as np

# 
class MujocoFH(gym.Env):
    def __init__(self, env_name, T=1000, r=None, obs_mean=None, obs_std=None, seed=1, state_only_reward=True, action_dependent=False):
        self.env = gym.make(env_name)
        self.T = T
        self.r = r
        assert (obs_mean is None and obs_std is None) or (obs_mean is not None and obs_std is not None)
        self.obs_mean, self.obs_std = obs_mean, obs_std

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        self.state_only_reward = state_only_reward
        self.action_dependent = action_dependent

        self.seed(seed)
        
    def seed(self, seed):
        self.env.seed(seed)
        self.env.action_space.seed(seed)

    def reset(self):
        self.t = 0
        self.terminated = False
        self.terminal_state = None

        self.obs = self.env.reset()
        self.obs = self.normalize_obs(self.obs)
        return self.obs.copy()
    
    def step(self, action):
        self.t += 1

        if self.terminated:
            return self.terminal_state, 0, self.t == self.T, True
        else:
            prev_obs = self.obs.copy()
            self.obs, r, done, info = self.env.step(action)
            if self.action_dependent:
                r = r - np.square(action).sum() * 3.
            self.obs = self.normalize_obs(self.obs)
            
            if self.r is not None:  # from irl model
                if self.state_only_reward:
                    r = self.r(prev_obs)
                else:
                    sa = np.concatenate((prev_obs, action), axis=0)
                    r = self.r(sa)
            
            if done:
                self.terminated = True
                self.terminal_state = self.obs
            
            return self.obs.copy(), r, done, done
    
    def normalize_obs(self, obs):
        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / self.obs_std
        return obs
