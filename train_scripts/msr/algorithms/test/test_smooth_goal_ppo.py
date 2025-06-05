import unittest
import numpy as np
from pathlib import Path
import sys
import gymnasium as gym
from collections import namedtuple
from copy import deepcopy

from stable_baselines3.ppo import PPO, MultiInputPolicy
from stable_baselines3.common.logger import Logger, configure
import torch as th
import flycraft

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
from train_scripts.msr.attackers.ppo.gradient_ascent_attackers_ppo import GradientAscentAttacker
from train_scripts.msr.utils.evaluation import my_evaluate_with_customized_dg
from train_scripts.msr.utils.reset_env_utils import (
    get_lower_bound_of_desired_goal,
    get_upper_bound_of_desired_goal,
)
from train_scripts.msr.algorithms.smooth_goal_ppo import SmoothGoalPPO

gym.register_envs(flycraft)


class SmoothGoalSACTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        env_id = "FlyCraft-v0"

        self.env_used_in_algo = gym.make(
            env_id,
            config_file=PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_ppo_easy.json"
        )
        self.env_used_in_algo = ScaledActionWrapper(ScaledObservationWrapper(self.env_used_in_algo))

        self.env_used_in_attacker = gym.make(
            env_id,
            config_file=PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_ppo_easy.json"
        )
        self.env_used_in_attacker = ScaledActionWrapper(ScaledObservationWrapper(self.env_used_in_attacker))

        sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "train_scripts" / "disc" / "algorithms" / "test" / "test_noised_goal_ppo").absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

        self.ppo_algo = SmoothGoalPPO(
            policy=MultiInputPolicy,
            env=self.env_used_in_algo,
            goal_noise_epsilon=np.array([10., 3., 3.]),
            env_used_in_attacker=self.env_used_in_attacker
        )
        self.ppo_algo.set_logger(sb3_logger)
    
    def test_sample_goal_noise_1(self):
        print("In test sample goal noise.............")
        for i in range(10):
            obs, _ = self.env_used_in_attacker.reset()
            tmp_noise = self.ppo_algo.sample_a_goal_noise(scaled_desired_goal=obs["desired_goal"])
            print(f"iter {i}, {tmp_noise}")
            self.assertTrue(th.all(th.less_equal(tmp_noise, self.ppo_algo.ppo_ga_attacker.noise_max)))
            self.assertTrue(th.all(th.less_equal(self.ppo_algo.ppo_ga_attacker.noise_min, tmp_noise)))
    
    def test_train(self):
        self.ppo_algo.learn(total_timesteps=100000)


if __name__ == "__main__":
    unittest.main()
