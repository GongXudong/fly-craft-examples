import sys
from pathlib import Path
from abc import ABC
from typing import Tuple, List
from copy import deepcopy

import numpy as np
import gymnasium as gym
import torch as th

from stable_baselines3.sac import SAC, MultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.distributions import kl_divergence
from torch.cuda import is_available

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledObservationWrapper
from train_scripts.disc.utils.reset_env_utils import (
    get_lower_bound_of_desired_goal,
    get_upper_bound_of_desired_goal,
)

class AttackerBase(ABC):

    def __init__(
        self,
        policy: MultiInputPolicy,
        env: gym.Env,
        epsilon: np.ndarray,
        max_random_limit_when_get_achievable_goal: int=100,
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
    ):
        self.policy = policy
        self.env = env
        self.epsilon = epsilon
        self.max_random_limit_when_get_achievable_goal = max_random_limit_when_get_achievable_goal
        self.device = device

    def get_an_achievable_desired_goal(self) -> Tuple[bool, np.ndarray, float, List, List]:
        """获得一个能够到达的goal。

        Returns:
            Tuple[bool, np.ndarray, float, List, List]: 是否找到了一个能够achieve的goal，这个goal，完成这个goal获得的累计奖励，完成这个goal过程中的obs list，完成这个goal过程中的action对应的action distribution
        """
        cnt = 0
        self.policy.set_training_mode(False)

        while True:
            cnt += 1

            obs, info = self.env.reset()
            desired_goal = obs["desired_goal"]
            cumulative_reward, is_success,  = 0.0, False
            obs_history, action_distribution_list = [], []

            # 计算完成desired_goal过程中的观测序列obs_history，动作分布序列action_distribution_of_desired_goal
            while True:
                tmp_action, _ = self.policy.predict(observation=obs, deterministic=True)
                tmp_action_dist = self.policy.actor.action_dist
                obs_history.append(obs)
                action_distribution_list.append(
                    deepcopy(tmp_action_dist)
                )

                new_obs, reward, terminated, truncated, info = self.env.step(tmp_action)

                cumulative_reward += reward
                is_success = info["is_success"] if "is_success" in info else False

                if terminated or truncated:
                    if is_success:
                        print(f"find an achievable goal with trying num: {cnt}.")
                        return True, desired_goal, cumulative_reward, obs_history, action_distribution_list
                    else:
                        break

                obs = new_obs
            
            # 防止因policy没有任何任务完成能力而导致的死循环！！！
            if cnt == self.max_random_limit_when_get_achievable_goal:
                return False, None, None, None, None

    def attack(self, desired_goal: np.ndarray, **kwargs):
        """从desired_goal周围（以desired_goal为中心，正负epsilon范围）找一个noised_desired_goal，
        使D_KL (pi(.|s, desired_goal), pi(.|s, noised_desired_goal))尽量大
        """
        pass