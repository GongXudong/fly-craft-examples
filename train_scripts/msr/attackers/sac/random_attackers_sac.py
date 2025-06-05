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
from train_scripts.msr.utils.reset_env_utils import (
    get_lower_bound_of_desired_goal,
    get_upper_bound_of_desired_goal,
)
from train_scripts.msr.attackers.sac.base_attackers_sac import AttackerBase


class RandomAttacker(AttackerBase):
    
    def __init__(
        self, 
        policy: MultiInputPolicy, 
        env: gym.Env, 
        epsilon: np.ndarray,
        max_random_limit_when_get_achievable_goal: int=100,
        device = th.device("cuda" if th.cuda.is_available() else "cpu"),
    ):
        super().__init__(policy, env, epsilon, max_random_limit_when_get_achievable_goal, device)
    
    def attack(self, desired_goal: np.ndarray, **kwargs):

        assert "observation_history" in kwargs, "must pass key: observation_history"
        assert "action_distribution_list" in kwargs, "must pass key: action_distribution_list"

        observation_history: np.ndarray = kwargs.get("observation_history")
        action_distribution_list: np.ndarray = kwargs.get("action_distribution_list")
        random_noise_num: int = kwargs.get("random_noise_num", 10)

        noised_desired_goals = [
            desired_goal.copy() + 2 * (np.random.random(self.epsilon.shape) - 0.5) * self.epsilon
            for _ in range(random_noise_num)
        ]

        action_distribution_of_noised_desired_goals = []  # 2D array, 存储每个noised_desired_goal对应的action_distribution list
        
        # 计算每个noised_desired_goal在obs_history上的action_dist
        for noised_desired_goal in noised_desired_goals:

            tmp_action_dist_list = []

            for obs in observation_history:
                tmp_obs = deepcopy(obs)

                # 由于实际使用时，会给FlyCraftEnv套上ScaledObservationWrapper，所以需要根据使用wrapper的情况对desired_goal进行设置！！！
                tmp_env = self.env
                while True:
                    if isinstance(tmp_env, ScaledObservationWrapper):
                        tmp_obs = tmp_env.inverse_scale_state(tmp_obs)
                        tmp_obs["desired_goal"] = noised_desired_goal
                        tmp_obs = tmp_env.scale_state(tmp_obs)
                        break

                    if not isinstance(tmp_env, gym.Wrapper):
                        tmp_obs["desired_goal"] = noised_desired_goal
                        break
                    
                    tmp_env = tmp_env.env

                tmp_action, _ = self.policy.predict(observation=tmp_obs, deterministic=True)
                tmp_action_dist = self.policy.actor.action_dist
                # print(tmp_action_dist.distribution.mean)
                tmp_action_dist_list.append(deepcopy(tmp_action_dist))
            
            action_distribution_of_noised_desired_goals.append(tmp_action_dist_list)
        
        # 找action distribution差异最大的noised_desired_goal
        maximum_noised_disred_goal, max_discrepency = desired_goal, -1.0
        
        for tmp_goal, tmp_action_dist_list in zip(noised_desired_goals, action_distribution_of_noised_desired_goals):
            tmp_KLs = []
            for aa, bb in zip(action_distribution_list, tmp_action_dist_list):
                # print(aa.distribution.mean, bb.distribution.mean)
                
                tmp_KL = kl_divergence(aa, bb).sum(axis=-1)
                tmp_KLs.append(tmp_KL.cpu().numpy())
            
            tmp = np.array(tmp_KLs).mean()
            # print(f"tmp KL, goal: {tmp_goal}, delta: {np.array(tmp_goal) - np.array(desired_goal)}, kl: {tmp}")
            if tmp > max_discrepency:
                max_discrepency = tmp
                maximum_noised_disred_goal = deepcopy(tmp_goal)
        
        return maximum_noised_disred_goal, max_discrepency
