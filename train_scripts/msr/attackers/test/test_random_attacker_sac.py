import unittest
import numpy as np
from pathlib import Path
import sys
import gymnasium as gym
from collections import namedtuple
from copy import deepcopy

from stable_baselines3.sac import SAC

import flycraft

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
from train_scripts.msr.attackers.sac.random_attackers_sac import RandomAttacker
from train_scripts.msr.utils.evaluation import my_evaluate_with_customized_dg

gym.register_envs(flycraft)


class RandomAttackerTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        env_id = "FlyCraft-v0"
        env = gym.make(
            env_id,
            config_file=PROJECT_ROOT_DIR / "configs" / "env" / "VVCGym" / "env_config_for_sac.json"
        )
        env = ScaledActionWrapper(ScaledObservationWrapper(env))

        exp_name = "sac_her_easy_10hz_128_128_1e6steps_loss_1_singleRL"
        model_name = "best_model"
        policy_dir = PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / exp_name / model_name
        sac_algo = SAC.load(
            policy_dir,
            env=env
        )

        self.randomAttacker = RandomAttacker(
            policy=sac_algo.policy,
            env=env,
            epsilon=np.array([10, 3, 3]),
            max_random_limit_when_get_achievable_goal=100,
        )

    def test_get_an_achievable_desired_goal(self):
        print("test get an achievable desired goal...............")
        for i in range(10):
            flag, desired_goal, cumulative_reward, obs_list, action_dist_list = self.randomAttacker.get_an_achievable_desired_goal()

            unscaled_desired_goal = self.randomAttacker.env.env.inverse_scale_state(obs_list[0])["desired_goal"]

            # print(flag, desired_goal)
            is_success, desired_goal_val, cumulative_reward_val = my_evaluate_with_customized_dg(
                policy=self.randomAttacker.policy,
                env=self.randomAttacker.env,
                desired_goal=unscaled_desired_goal
            )

            tmp_obs = deepcopy(obs_list[0])
            tmp_obs["desired_goal"] = desired_goal_val
            unscaled_desired_goal_val = self.randomAttacker.env.env.inverse_scale_state(tmp_obs)["desired_goal"]

            # print(f"check: {unscaled_desired_goal}, {unscaled_desired_goal_val}")

            self.assertTrue(np.allclose(desired_goal, desired_goal_val))
            self.assertTrue(np.allclose(unscaled_desired_goal, unscaled_desired_goal_val))
            self.assertEqual(flag, is_success, "完成状态不一致！")
            self.assertAlmostEqual(cumulative_reward, cumulative_reward_val, delta=1e-4, msg="累计奖励不一致！")

    def test_attack(self):
        print("test attack........................")
        for _ in range(5):

            flag, desired_goal, cumulative_reward, obs_list, action_dist_list = self.randomAttacker.get_an_achievable_desired_goal()
            # 注意：套了ScaledObservationWrapper后，需要对desired_goal进行inverse_scale!!!!!
            unscaled_desired_goal = self.randomAttacker.env.env.inverse_scale_state(obs_list[0])["desired_goal"]

            noised_goal, discrepency = self.randomAttacker.attack(
                desired_goal=unscaled_desired_goal,
                observation_history=obs_list,
                action_distribution_list=action_dist_list,
                random_noise_num=10,
            )
            print(f"desired goal: {unscaled_desired_goal}")
            print(f"noised goal: {noised_goal}")
            print(f"discrepency: {discrepency}")

            my_evaluate_with_customized_dg(
                policy=self.randomAttacker.policy,
                env=self.randomAttacker.env,
                desired_goal=noised_goal
            )
    

if __name__ == "__main__":
    unittest.main()
