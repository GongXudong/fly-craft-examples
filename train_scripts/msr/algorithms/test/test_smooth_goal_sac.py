import unittest
import numpy as np
from pathlib import Path
import sys
import gymnasium as gym
from collections import namedtuple
from copy import deepcopy

from stable_baselines3.sac import SAC, MultiInputPolicy
from stable_baselines3.common.logger import Logger, configure
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
from train_scripts.disc.algorithms.smooth_goal_sac import SmoothGoalSAC

gym.register_envs(flycraft)


class SmoothGoalSACTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        env_id = "FlyCraft-v0"

        self.env_used_in_algo = gym.make(
            env_id,
            config_file=PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_sac.json"
        )
        self.env_used_in_algo = ScaledActionWrapper(ScaledObservationWrapper(self.env_used_in_algo))

        self.helper_env = gym.make(
            env_id,
            config_file=PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_sac.json"
        )
        self.helper_env = ScaledActionWrapper(ScaledObservationWrapper(self.helper_env))

        sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "train_scripts" / "disc" / "algorithms" / "test" / "test_noised_goal_sac").absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

        self.sac_algo = SmoothGoalSAC(
            policy=MultiInputPolicy,
            env=self.env_used_in_algo,
            goal_noise_epsilon=np.array([10., 3., 3.]),
        )
        self.sac_algo.set_logger(sb3_logger)
    
    def test_init_desired_goal_params(self):
        print("In test init desired goal params.............")
        self.sac_algo.init_desired_goal_params(self.helper_env)
        
        print(self.sac_algo.desired_goal_max)
        print(self.sac_algo.desired_goal_min)
        print(self.sac_algo.noise_max)
        print(self.sac_algo.noise_min)
    
    def test_train(self):
        self.sac_algo.init_desired_goal_params(self.helper_env)
        self.sac_algo.learn(total_timesteps=10000)


if __name__ == "__main__":
    unittest.main()
