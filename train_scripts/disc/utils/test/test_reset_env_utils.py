import unittest
import numpy as np
from pathlib import Path
import sys
import gymnasium as gym
from collections import namedtuple

from stable_baselines3.sac import SAC

import flycraft

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.disc.utils.reset_env_utils import (
    get_lower_bound_of_desired_goal, 
    get_upper_bound_of_desired_goal, 
    get_validate_part_of_noised_desired_goal,
    sample_a_noised_desired_goal_by_random,
    reset_env_with_desired_goal,
)

gym.register_envs(flycraft)


class ResetEnvUtilsTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.env = gym.make(
            "FlyCraft-v0",
            config_file=PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_sac.json"
        )

    def test_get_goal_bound(self):
        print(get_lower_bound_of_desired_goal(self.env))
        print(get_upper_bound_of_desired_goal(self.env))
    
    def test_get_validate_part_of_noised_desired_goal(self):

        desired_goals = np.array([
            # 测试下边界
            [150.0, -10, -30],
            [151, -9.9, -29.9],
            # 测试上边界
            [250, 10, 30],
            [249, 9.9, 29.9],
            # 测试正常范围
            [200, 3, 18],
        ])

        noises = np.array([
            # 测试下边界
            [-1.0, -1.0, -1.0],
            [-2.0, -1.0, -2.0],
            # 测试上边界
            [1.0, 1.0, 1.0],
            [2.0, 3.0, 3.0],
            # 测试正常范围
            [10., 2., 3.],
        ])

        for dg, noise in zip(desired_goals, noises):
            
            noised_dg = get_validate_part_of_noised_desired_goal(
                env=self.env,
                desired_goal=dg,
                noise=noise
            )

            self.assertTrue(
                np.all(get_lower_bound_of_desired_goal(self.env) <= noised_dg)
                and 
                np.all(noised_dg <= get_upper_bound_of_desired_goal(self.env))
            )

    def test_sample_a_noised_desired_goal_by_random(self):
        
        TEST_NUM = 1000
        NOISE_MAX_RANGE = np.array([10.0, 3.0, 3.0])  # v, mu, chi的最大噪声值

        for _ in range(TEST_NUM):
            random_desired_goal_dict = self.env.unwrapped.task.goal_sampler.sample_goal()
            random_desired_goal = np.array([random_desired_goal_dict['v'], random_desired_goal_dict['mu'], random_desired_goal_dict['chi']])

            noised_dg = sample_a_noised_desired_goal_by_random(
                env=self.env,
                desired_goal=random_desired_goal,
                epsilon=NOISE_MAX_RANGE,
            )

            self.assertTrue(
                np.all(get_lower_bound_of_desired_goal(self.env) <= noised_dg)
                and 
                np.all(noised_dg <= get_upper_bound_of_desired_goal(self.env))
            )

    def test_reset_env_with_desired_goal(self):

        desired_goals = np.array([
            [150, 10, 20],
            [200, 8, -15],
            [250, -5, -29],
        ])

        for dg in desired_goals:

            # 注意：调用此函数后，需要手动将env.unwrapped.task.goal_sampler.use_fixed_goal恢复成原来的值！！！
            tmp_env_sample_fixed_goal = self.env.unwrapped.task.goal_sampler.use_fixed_goal
            obs, info = reset_env_with_desired_goal(self.env, dg, validate_desired_goal_bound=True)
            self.env.unwrapped.task.goal_sampler.use_fixed_goal = tmp_env_sample_fixed_goal

            for _ in range(10):
                
                self.assertTrue(np.all(dg == obs["desired_goal"]))

                action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                if terminated or truncated:
                    break

if __name__ == "__main__":
    unittest.main()
