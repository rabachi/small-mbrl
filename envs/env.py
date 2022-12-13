import functools
from typing import Any, Dict, Optional, Type, Union
import hydra
import math
import pdb
import gym
import safe_grid_gym
import numpy as np

class SafeGymWrapper(gym.Env):
    def __init__(self, env: gym.Env, norm_reward: bool) -> None:
        super().__init__()
        self._env = env
        self.nState = math.prod(self._env.observation_space.shape)
        self.nAction = self._env.action_space.n
        # self.map_all_obs()
        init_state = self.reset()
        init_distrib = np.zeros(self.nState)
        init_distrib[init_state] = 1.
        self.initial_distribution = init_distrib
        self.norm_reward = norm_reward

    def reset(self):
        return self.map_to_index(self._env.reset())
    
    def step(self, action):
        next_obs, reward, done, info = self._env.step(action)
        if self.norm_reward: 
            # reward = reward - self.min_reward
            reward = (reward - self.min_reward)/(self.max_reward - self.min_reward)
        return self.map_to_index(next_obs), reward, done, info
    
    def map_to_index(self, obs):
        hash_key = hash(obs.tobytes()) % self.nState
        return hash_key
    
    def terminal_reward(self):
        if self.norm_reward:
            # return -self.min_reward
            return -self.min_reward / (self.max_reward - self.min_reward)
        else:
            return 0

    @property
    def max_reward(self):
        return 49.0
    
    @property
    def min_reward(self):
        return -51.

    @property
    def goal_reward(self):
        if self.norm_reward:
            return (self.max_reward - self.min_reward)/(self.max_reward - self.min_reward)
        return self.max_reward

    @property
    def hole_reward(self):
        if self.norm_reward:
            return (self.min_reward - self.min_reward)/(self.max_reward - self.min_reward)
        return self.min_reward

class GymWrapper(gym.Env):
    def __init__(self, env: gym.Env) -> None:
        super().__init__()
        self._env = env
        self.nState = self._env.observation_space.n
        self.nAction = self._env.action_space.n
        self.initial_distribution = self._env.initial_state_distrib
    
    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)
    
    @property
    def max_reward(self):
        return 1.0
        # return (50.0 + 50.0)/100.
    
    @property
    def min_reward(self):
        return 0.
    
def setup_environment(
        env_setup: dict,
        env_type: str,
        env_id: str,
        norm_reward: bool,
        seed: Optional[int] = None,
    ):
    if env_type == "gym":
        return GymWrapper(gym.make(env_id))
    elif env_type == "safe_grid_gym":
        return SafeGymWrapper(gym.make(env_id), norm_reward)
    elif env_type == "self":
        return hydra.utils.instantiate(env_setup)
    else:
        raise ValueError(f'Unknown env type: {env_type}')
