import gymnasium as gym
import numpy as np
from gymnasium import Wrapper


class PowerRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, b=1.0):
        super().__init__(env)
        self.b = b

    # --- 1. 核心计算逻辑 (抽取出来) ---
    def _apply_power(self, reward):
        
        #reward = np.array(reward, dtype=np.float64)
        # 处理标量或数组 (compute_reward 可能会传入数组)
        return np.sign(reward) * (np.abs(reward) ** self.b)

    # --- 2. 在线交互用的 (覆盖 RewardWrapper 的接口) ---
    def reward(self, reward):
        # step() 会自动调用这个
        # print(f"original reward= {reward}")
        # print(f"wrapper reward= {self._apply_power(reward)}")
        return self._apply_power(reward)

    # --- 3. 离线重算用的 (覆盖 GoalEnv 的接口) ---
    def compute_reward(self, achieved_goal, desired_goal, info):
        # 第一步：调用原始环境计算 原始奖励
        # 注意：这里要调用 self.env.compute_reward，而不是 super()
        # 因为 Wrapper 链条中，compute_reward 通常直接透传给底层环境
        base_reward = self.unwrapped.compute_reward(achieved_goal, desired_goal, info)
        
        # 第二步：套用同样的变换逻辑
        return self._apply_power(base_reward)
    
