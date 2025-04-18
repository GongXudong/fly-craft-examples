from typing import Dict, Any
import numpy as np
from gymnasium import RewardWrapper
from utils_my.env_utils.register_env import register_all_with_default_dense_params,register_nsubsteps_all_with_sparse_params
register_all_with_default_dense_params() 
register_nsubsteps_all_with_sparse_params()
class PowerRewardWrapper(RewardWrapper):
    def __init__(self, env, b=1.0,reward_type="dense",distance_threshold = 0.01 ):
        super().__init__(env)
        self.b = b
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
    
    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(original_reward), terminated, truncated, info

    def reward(self, reward):
        if self.reward_type == "sparse":
            return reward 
        else:
            return -np.power(np.abs(reward), self.b)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        original_reward = self.unwrapped.compute_reward(achieved_goal, desired_goal, info)
        return self.reward(original_reward)