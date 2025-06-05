import sys
from pathlib import Path
from abc import ABC
from typing import Tuple, List
from copy import deepcopy

import numpy as np
import gymnasium as gym
import torch as th

from stable_baselines3.ppo import PPO, MultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.distributions import kl_divergence

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledObservationWrapper


# 注：Stable Baselines3中PPO与SAC不共享MultiInputPolicy！！！！！
# PPO: BasePolicy -> ActorCriticPolicy -> MultiInputPolicy (在ppo中与MultiInputActorCriticPolicy同名)
# SAC: BasePolicy -> SACPolicy -> MultiInputPolicy


class AttackerBase(ABC):

    def __init__(
        self,
        policy: MultiInputPolicy,
        env: gym.Env,
        epsilon: np.ndarray,
        max_random_limit_when_get_achievable_goal: int=100,
        device = th.device("cuda" if th.cuda.is_available() else "cpu"),
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

        with th.no_grad():
            while True:
                cnt += 1

                obs, info = self.env.reset()
                desired_goal = obs["desired_goal"]
                cumulative_reward, is_success,  = 0.0, False
                obs_history, action_distribution_list = [], []

                # 计算完成desired_goal过程中的观测序列obs_history，动作分布序列action_distribution_of_desired_goal
                while True:
                    # TODO: 待优化，获得action与获得action distribution，网络forward了两次，可优化成1次！！！
                    tmp_action, _ = self.policy.predict(observation=obs, deterministic=True)
                    
                    obs_pth, _ = self.policy.obs_to_tensor(obs)
                    tmp_action_dist = self.policy.get_distribution(obs=obs_pth)
                    
                    obs_history.append(obs)
                    # print(f":--------------------", type(tmp_action_dist))
                    action_distribution_list.append(
                        deepcopy(tmp_action_dist.distribution)
                        # tmp_action_dist
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
