from typing import Dict, Any
import numpy as np
from gymnasium import RewardWrapper


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