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
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.distributions import Distribution
from torch.cuda import is_available

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
from train_scripts.disc.utils.reset_env_utils import (
    get_lower_bound_of_desired_goal,
    get_upper_bound_of_desired_goal,
)
from train_scripts.disc.attackers.ppo.base_attackers_ppo import AttackerBase


class GradientAscentAttacker(AttackerBase):

    def __init__(
        self, 
        policy: MultiInputPolicy, 
        env: gym.Env, 
        epsilon: np.ndarray, 
        max_random_limit_when_get_achievable_goal: int = 100, 
        device=th.device("cuda" if th.cuda.is_available() else "cpu"),
        policy_distance_measure_func: str = "KL",    
    ):
        super().__init__(policy, env, epsilon, max_random_limit_when_get_achievable_goal, device)

        self.policy_distance_measure_func = policy_distance_measure_func
    
    def _init_noise(self):
        """在调用这个函数前，需先调用self._calc_noise_min_max(desired_goal)，保证self.noise_min和self.noise_max已经被计算出来。

        Returns:
            _type_: _description_
        """
        # 直接生成[-epsilon, epsilon]范围内的噪声
        # return 2 * (th.rand(1, *self.epsilon.shape, device=self.device) - 0.5) * self.epsilon

        # 生成[noise_min, noise_max]范围内的噪声, rand * (noise_max - noise_min) + (noise_min + noise_max) / 2
        return (th.rand(1, *self.epsilon.shape, device=self.device) - 0.5) * (self.noise_max - self. noise_min) + (self.noise_min + self.noise_max) / 2

    def _calc_noise_min_max(self, desired_goal: np.ndarray):
        """给定desired_goal时，计算能在这个desired_goal施加的noise的最小最大值。
        最小值：max(desired_goal_min - desired_goal, -epsilon), 最大值：min(desired_goal_max - desired_goal, epsilon).
        
        注意：desired_goal的范围是原始环境的范围！！！
        """
        self.noise_min = th.maximum(
            th.tensor(-self.epsilon, device=self.device, requires_grad=False),
            th.tensor(get_lower_bound_of_desired_goal(self.env) - desired_goal, device=self.device, requires_grad=False)
        )

        self.noise_max = th.minimum(
            th.tensor(self.epsilon, device=self.device, requires_grad=False),
            th.tensor(get_upper_bound_of_desired_goal(self.env) - desired_goal, device=self.device, requires_grad=False)
        )

        tmp_env = self.env
        while True:
            if isinstance(tmp_env, ScaledObservationWrapper):
                self.noise_min = th.tensor(tmp_env.goal_scalar.transform(self.noise_min.reshape((1,-1))).reshape((-1)) - np.array([0., 0.5, 0.5]), device=self.device, requires_grad=False)
                self.noise_max = th.tensor(tmp_env.goal_scalar.transform(self.noise_max.reshape((1,-1))).reshape((-1)) - np.array([0., 0.5, 0.5]), device=self.device, requires_grad=False)
                break
            
            if not isinstance(tmp_env, gym.Wrapper):
                break
            
            tmp_env = tmp_env.env


    def _calc_loss(self, obs_history: List, noise: th.Tensor, action_distribution_list: List):
        """_summary_

        Args:
            obs_history (PyTorchObs): 
            noise (th.Tensor): shape: desired_goal_shape
            action_distribution_list (np.ndarray): _description_
        """
        # 手动准备obs的tensor，并调用policy._predict()，并根据action_distribution计算loss

        # 定义loss function
        def _f(obs: PyTorchObs, reference_action_distribution: Distribution):
            """先obs -> distribution，再根据两个distribution计算loss (JS散度)

            Args:
                obs (PyTorchObs): 向desired_goal加过噪声的obs
                reference_action_distribution_list (np.ndarray): 在原obs上的action_distribution_list
            """

            tmp_action_dist = self.policy.get_distribution(obs=obs)
            
            if self.policy_distance_measure_func == "KL":
                distance = th.distributions.kl_divergence(reference_action_distribution, tmp_action_dist.distribution).sum(axis=-1)
            elif self.policy_distance_measure_func == "JS":
                distance = th.distributions.kl_divergence(reference_action_distribution, tmp_action_dist.distribution).sum(axis=-1) + th.distributions.kl_divergence(tmp_action_dist.distribution, reference_action_distribution).sum(axis=-1)
            else:
                raise ValueError("policy_distance_measure_func must be either KL or JS!")
            
            return - distance
            
            # JS_distance = th.distributions.kl_divergence(reference_action_distribution, tmp_action_dist.distribution).sum(axis=-1) + th.distributions.kl_divergence(tmp_action_dist.distribution, reference_action_distribution).sum(axis=-1)
            # return - JS_distance

        # 把np.ndarray类型的obs_history转换成PyTorchObs类型， { "observation": tensor(shape: batch_size, observation_shape), "desired_goal": tensor(shape: batch_size, desired_goal_shape), "achieved_goal": tensor(shape: batch_size, achieved_goal_shape)}
        
        loss_list = []

        for obs, action_dist in zip(obs_history, action_distribution_list):
            obs_tensor, _ = self.policy.obs_to_tensor(obs)
            # 把noise加到desired_goal中
            obs_tensor["desired_goal"] = obs_tensor["desired_goal"] + noise
            loss_list.append(
                _f(obs=obs_tensor, reference_action_distribution=action_dist)
            )

        return th.mean(th.concat(loss_list))        
        # obs_tensor, _ = self.policy.obs_to_tensor(obs_history)
        
        # # 把noise加到desired_goal中
        # extended_noise = noise.reshape((1, -1)).repeat((len(obs_history), 1))
        # obs_tensor["desired_goal"] = obs_tensor["desired_goal"] + extended_noise


    def attack(self, desired_goal: np.ndarray, **kwargs):
        """
        注意1：desired_goal的范围是原始环境的范围！！！
        
        注意2：在函数内部，noise的计算是在归一化的范围！！！

        Args:
            desired_goal (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        assert "observation_history" in kwargs, "must pass key: observation_history"
        assert "action_distribution_list" in kwargs, "must pass key: action_distribution_list"
        assert "lr" in kwargs, "must pass key: lr"
        assert "optimize_steps" in kwargs, "must pass key: optimize_steps"

        observation_history: np.ndarray = kwargs.get("observation_history")
        action_distribution_list: np.ndarray = kwargs.get("action_distribution_list")
        lr: float = kwargs.get("lr", 1e-4)
        optimize_steps: int = kwargs.get("optimize_steps", 10)

        # 计算噪声的最大值、最小值
        self._calc_noise_min_max(desired_goal=desired_goal)
        print(f"check min max: {self.noise_min}, {self.noise_max}")

        # 生成一个noise, 并设置成可训练的参数
        noise_para = self._init_noise()
        print(f"init noise: {noise_para}, noise_min: {self.noise_min}, noise_max: {self.noise_max}")
        for step_cnt in range(optimize_steps):
            
            print(f"In optimize noise para, step_cnt: {step_cnt}, para: {noise_para}")

            noise_para = th.nn.Parameter(noise_para.clone(), requires_grad=True)
            optimizer = th.optim.Adam([noise_para], lr=lr) 

            # 在observation_history上根据D_JS计算loss
            loss = self._calc_loss(
                obs_history=observation_history,
                noise=noise_para,
                action_distribution_list=action_distribution_list
            )

            print(f"check loss: {loss}")

            # 更新noise，并保证noise在合法的desired_goal范围
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            # para = torch.clamp(para, -eps, eps).detach()
            noise_para = th.maximum(th.minimum(noise_para, self.noise_max), self.noise_min).detach()
        
        noised_desired_goal = observation_history[0]["desired_goal"] + noise_para.cpu().numpy()
        noised_desired_goal = self.env.env.goal_scalar.inverse_transform(noised_desired_goal.reshape((1, -1))).reshape((-1))

        return noised_desired_goal, -loss.cpu().detach().numpy()
