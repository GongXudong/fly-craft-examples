import unittest
import numpy as np
from pathlib import Path
import sys
import gymnasium as gym
from collections import namedtuple
from copy import deepcopy

from stable_baselines3.sac import SAC
import torch as th
import flycraft

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
from train_scripts.disc.attackers.sac.gradient_ascent_attackers_sac import GradientAscentAttacker
from train_scripts.disc.utils.evaluation import my_evaluate_with_customized_dg
from train_scripts.disc.utils.reset_env_utils import (
    get_lower_bound_of_desired_goal,
    get_upper_bound_of_desired_goal,
)

gym.register_envs(flycraft)


class GradientAscentAttackerTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        env_id = "FlyCraft-v0"
        self.original_env = gym.make(
            env_id,
            config_file=PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_sac.json"
        )
        self.scaled_obs_env = ScaledObservationWrapper(self.original_env)
        self.scaled_act_obs_env = ScaledActionWrapper(self.scaled_obs_env)

        exp_name = "sac_her_easy_10hz_128_128_1e6steps_loss_1_singleRL"
        model_name = "best_model"
        policy_dir = PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / exp_name / model_name
        sac_algo = SAC.load(
            policy_dir,
            env=self.scaled_act_obs_env
        )

        self.GAAttacker = GradientAscentAttacker(
            policy=sac_algo.policy,
            env=self.scaled_act_obs_env,
            epsilon=np.array([10., 3., 3.]),
            max_random_limit_when_get_achievable_goal=100,
        )
    
    def test_init_noise_1(self):
        print("In test init noise.............")
        dg_1 = np.array([249, 9, 0])
        self.GAAttacker._calc_noise_min_max(desired_goal=dg_1)
        for i in range(10):
            tmp_noise = self.GAAttacker._init_noise()
            print(f"iter {i}, {tmp_noise}")
            self.assertTrue(th.all(th.less_equal(tmp_noise, self.GAAttacker.noise_max)))
            self.assertTrue(th.all(th.less_equal(self.GAAttacker.noise_min, tmp_noise)))
    
    def test_noise_minmax_1(self):
        print("In test noise min max 1.............")
        dg_1 = np.array([242, 9, 23])
        self.GAAttacker._calc_noise_min_max(desired_goal=dg_1)
        # print("env max: ", get_upper_bound_of_desired_goal(self.GAAttacker.env))
        print(f"noise max: {self.GAAttacker.noise_min, self.GAAttacker.noise_max}")
        self.assertTrue(
            th.equal(
                self.GAAttacker.noise_max, 
                th.tensor(
                    self.scaled_obs_env.goal_scalar.transform(np.array([8, 1, 3]).reshape((1,-1))).reshape((-1)) - np.array([0., 0.5, 0.5]), 
                    device=self.GAAttacker.device, 
                    requires_grad=False
                )
            )
        )
        self.assertTrue(
            th.equal(
                self.GAAttacker.noise_min, 
                th.tensor(
                    self.scaled_obs_env.goal_scalar.transform(np.array([-10, -3, -3]).reshape((1,-1))).reshape((-1)) - np.array([0., 0.5, 0.5]), 
                    device=self.GAAttacker.device, 
                    requires_grad=False
                )
            )
        )

    def test_noise_minmax_2(self):
        print("In test noise min max 2.............")
        dg_1 = np.array([222, 0, 23])
        self.GAAttacker._calc_noise_min_max(desired_goal=dg_1)

        # print(self.scaled_obs_env.goal_scalar.data_min_, self.scaled_obs_env.goal_scalar.data_max_)

        # print(self.scaled_obs_env.goal_scalar.transform(np.array([10, 3, 3]).reshape((1,-1))).reshape((-1)) - np.array([0., 0.5, 0.5]))
        # print(self.GAAttacker.noise_max)
        self.assertTrue(
            th.equal(
                self.GAAttacker.noise_max, 
                th.tensor(
                    self.scaled_obs_env.goal_scalar.transform(np.array([10, 3, 3]).reshape((1,-1))).reshape((-1)) - np.array([0., 0.5, 0.5]),
                    device=self.GAAttacker.device, 
                    requires_grad=False
                )
            )
        )

        # print(self.scaled_obs_env.goal_scalar.transform(np.array([-10, -3, -3]).reshape((1,-1))).reshape((-1)) - np.array([0., 0.5, 0.5]))
        # print(self.GAAttacker.noise_min)
        self.assertTrue(
            th.equal(
                self.GAAttacker.noise_min, 
                th.tensor(
                    self.scaled_obs_env.goal_scalar.transform(np.array([-10, -3, -3]).reshape((1,-1))).reshape((-1)) - np.array([0., 0.5, 0.5]),
                    device=self.GAAttacker.device, 
                    requires_grad=False
                )
            )
        )

    def test_noise_minmax_3(self):
        print("In test noise min max 3.............")
        dg_1 = np.array([151, -9, 0])
        self.GAAttacker._calc_noise_min_max(desired_goal=dg_1)
        self.assertTrue(
            th.equal(
                self.GAAttacker.noise_max, 
                th.tensor(
                    self.scaled_obs_env.goal_scalar.transform(np.array([10, 3, 3]).reshape((1,-1))).reshape((-1)) - np.array([0., 0.5, 0.5]),
                    device=self.GAAttacker.device, 
                    requires_grad=False
                )
            )
        )
        self.assertTrue(
            th.equal(
                self.GAAttacker.noise_min, 
                th.tensor(
                    self.scaled_obs_env.goal_scalar.transform(np.array([-1, -1, -3]).reshape((1,-1))).reshape((-1)) - np.array([0., 0.5, 0.5]),
                    device=self.GAAttacker.device, 
                    requires_grad=False
                )
            )
        )
    
    def test_calc_loss_1(self):
        print("test calc loss 1...................... ")
        flag, dg, cumulative_rewards, obs_list, action_dist_list = self.GAAttacker.get_an_achievable_desired_goal()
        dg = self.scaled_obs_env.goal_scalar.inverse_transform(dg.reshape((1, -1))).reshape((-1))
        print(dg)

        noised_desired_goal, max_discrepency = self.GAAttacker.attack(
            desired_goal=dg,
            observation_history=obs_list,
            action_distribution_list=action_dist_list,
            lr=1e-3,
            optimize_steps=20,
        )

        print(f"noised_dg: {noised_desired_goal}, max discrepency: {max_discrepency}")

if __name__ == "__main__":
    unittest.main()
