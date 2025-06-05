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

from utils_my.sb3.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper

from train_scripts.msr.utils.reset_env_utils import (
    get_lower_bound_of_desired_goal, 
    get_upper_bound_of_desired_goal, 
    get_validate_part_of_noised_desired_goal,
    sample_a_noised_desired_goal_by_random,
    reset_env_with_desired_goal,
)

from train_scripts.msr.utils.evaluation import (
    my_evaluate_with_original_dg,
    my_evaluate_with_customized_dg,
)

gym.register_envs(flycraft)


class EvaluationTest(unittest.TestCase):
    
    def setUp(self) -> None:
        super().setUp()
        env = gym.make(
            "FlyCraft-v0",
            config_file=PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_sac.json"
        )

        self.env = ScaledActionWrapper(ScaledObservationWrapper(env))

        exp_name = "sac_her_10hz_128_128_1e6steps_loss_2_singleRL"
        model_name = "best_model"
        policy_dir = PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / exp_name / model_name
        self.sac_algo = SAC.load(
            policy_dir,
            env=self.env
        )

    def test_my_evaluate_with_original_dg(self):
        print("test evaluate on random goal.")
        for _ in range(5):
            print(
                my_evaluate_with_original_dg(
                    policy=self.sac_algo.policy,
                    env=self.env
                )
            )

    def test_my_evaluate_with_customized_dg(self):
        print("test evaluate on fixed goal.")

        desired_goals = np.array([
            [150, 10, 20],
            [200, 8, -15],
            [250, -5, -29],
        ])

        for dg in desired_goals:
            noised_dg = get_validate_part_of_noised_desired_goal(
                env=self.env, 
                desired_goal=dg,
            )

            print(
                my_evaluate_with_customized_dg(
                    policy=self.sac_algo.policy,
                    env=self.env,
                    desired_goal=noised_dg
                )
            )


if __name__ == "__main__":
    unittest.main()