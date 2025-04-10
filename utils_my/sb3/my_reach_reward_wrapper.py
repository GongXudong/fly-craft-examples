from gymnasium import RewardWrapper,ObservationWrapper, ActionWrapper, Env, spaces
from utils_my.env_utils.register_env import register_all_with_default_dense_params, register_all_with_default_sparse_params
import numpy as np
register_all_with_default_dense_params() 


class PowerRewardWrapper(RewardWrapper):
    def __init__(self, env, b=1.0):
        super().__init__(env)
        self.b = b
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        modified_reward = -np.power(np.abs(reward), self.b)
        return observation, modified_reward, terminated, truncated, info