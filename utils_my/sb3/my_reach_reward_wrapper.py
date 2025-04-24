from gymnasium import RewardWrapper,ObservationWrapper, ActionWrapper, Env, spaces
from utils_my.env_utils.register_env import register_all_with_default_dense_params, register_all_with_default_sparse_params
from panda_gym.utils import distance
import numpy as np
from typing import Any, Dict
register_all_with_default_dense_params() 


# class PowerRewardWrapper(RewardWrapper):
#     def __init__(self, env, b=1.0):
#         super().__init__(env)
#         self.b = b
    
#     def step(self, action):
#         observation, reward, terminated, truncated, info = self.env.step(action)
#         modified_reward = -np.power(np.abs(reward), self.b)
#         return observation, modified_reward, terminated, truncated, info

#     def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
#         d = distance(achieved_goal, desired_goal)
#         d = np.power(np.abs(d), self.b)
#         return -d.astype(np.float32)


class PowerRewardWrapper(RewardWrapper):
    def __init__(self, env, b=1.0):
        super().__init__(env)
        self.b = b

    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(original_reward), terminated, truncated, info

    def reward(self, reward):
        return -np.power(np.abs(reward), self.b)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        original_reward = self.unwrapped.compute_reward(achieved_goal, desired_goal, info)
        return self.reward(original_reward)