import functools
from typing import Any, Dict, Optional, Type, Union
import hydra
import math
import safe_grid_gym
import gym
import numpy as np

class SafeGymWrapper(gym.Env):
    def __init__(self, env: gym.Env) -> None:
        super().__init__()
        self._env = env
        self.nState = math.prod(self._env.observation_space.shape)
        self.nAction = self._env.action_space.n
        # self.map_all_obs()
        init_state = self.reset()
        init_distrib = np.zeros(self.nState)
        init_distrib[init_state] = 1.
        self.initial_distribution = init_distrib

    def reset(self):
        return self.map_to_index(self._env.reset())
    
    def step(self, action):
        next_obs, reward, done, info = self._env.step(action)
        return self.map_to_index(next_obs), reward, done, info
    
    def map_to_index(self, obs):
        hash_key = hash(obs.tobytes()) % self.nState
        return hash_key

    # def map_all_obs(self):
    #     obs_mapping = {}
    #     index = 0
    #     obs = self._env.reset()
    #     while index < self.nState:
    #         hash_key = obs.tobytes()
    #         if hash_key not in obs_mapping.keys():
    #             obs_mapping[hash_key] = index
    #             index += 1
    #         obs = self.get_next_obs(obs)
    #     self.obs_mapping = obs_mapping

    # def get_next_obs(self, obs):
    #     next_obs = obs.flatten()
    #     for i in next_obs:
    #         if i == 2:
    #             next_obs[i] = 
    #     return next_obs

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
    
def setup_environment(
        env_setup: dict,
        env_type: str,
        env_id: str,
        seed: Optional[int] = None,
    ):
    if env_type == "gym":
        return GymWrapper(gym.make(env_id))
    elif env_type == "safe_grid_gym":
        return SafeGymWrapper(gym.make(env_id))
    elif env_type == "self":
        return hydra.utils.instantiate(env_setup)
    else:
        raise ValueError(f'Unknown env type: {env_type}')