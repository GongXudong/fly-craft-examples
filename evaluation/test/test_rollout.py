import unittest
import numpy as np
from pathlib import Path
import sys
import gymnasium as gym
from collections import namedtuple
from copy import deepcopy

from stable_baselines3.ppo import PPO

import flycraft

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
from evaluation.rollout import Rollout

gym.register_envs(flycraft)


class RolloutTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        env_id = "FlyCraft-v0"
        env = gym.make(
            env_id,
            config_file=PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_ppo_easy.json"
        )
        env = ScaledActionWrapper(ScaledObservationWrapper(env))

        exp_name = "PPO/ppo_10hz_128_128_2e8steps_easy_1_singleRL"
        model_name = "best_model"
        policy_dir = PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / exp_name / model_name
        ppo_algo = PPO.load(
            policy_dir,
            env=env
        )

        self.rollout = Rollout(
            env=env,
            algo=ppo_algo,
            debug_mode=True,
        )

    def show(self):
        tmp = {
            'step': 1, 
            'is_success': False, 
            'rewards': {
                '<rewards.dense_reward_based_on_angle_and_velocity.DenseRewardBasedOnAngleAndVelocity object at 0x7f0908cdab40>': -0.3995688407121351
            }, 
            'action': {
                'p': 180.0, 
                'nz': -4.0, 
                'pla': 1.0, 
                'rud': 0.0
            }, 
            'desired_goal': [220.85320159701126, -2.0298400597549104, -21.853603357265627], 
            'plane_state': {
                'lef': 0.0, 
                'npos': 20.000000000000004, 
                'epos': 0.0, 
                'h': 5000.0, 
                'alpha': 2.5016133770396713, 
                'beta': -0.021475745218794213, 
                'phi': 0.0, 
                'theta': 2.6132205230940393, 
                'psi': 0.0, 
                'p': 74.3690767847601, 
                'q': -9.385704462768931, 
                'r': 3.113576280240488, 
                'v': 200.60434088883915, 
                'vn': 200.60394621462078, 
                've': -0.07519101146158212, 
                'vh': 0.3907593619091134, 
                'nx': 0.6791050542252308, 
                'ny': -0.07641640436443489, 
                'nz': 1.3675872794305908, 
                'ele': 7.945892300670966, 
                'ail': -21.5, 
                'rud': 0.0, 
                'thrust': 1.0, 
                'lon': 122.425, 
                'lat': 31.425180164618112, 
                'mu': 5.073556701235858e-16, 
                'chi': 0.0
            }, 
            'plane_next_state': {
                'lef': 0.0, 
                'npos': 20.000000000000004, 
                'epos': 0.0, 
                'h': 5000.0, 
                'alpha': 2.5016133770396713, 
                'beta': -0.021475745218794213, 
                'phi': 0.0, 
                'theta': 2.6132205230940393, 
                'psi': 0.0, 
                'p': 74.3690767847601, 
                'q': -9.385704462768931, 
                'r': 3.113576280240488, 
                'v': 200.60434088883915, 
                'vn': 200.60394621462078, 
                've': -0.07519101146158212, 
                'vh': 0.3907593619091134, 
                'nx': 0.6791050542252308, 
                'ny': -0.07641640436443489, 
                'nz': 1.3675872794305908, 
                'ele': 7.945892300670966, 
                'ail': -21.5, 
                'rud': 0.0, 
                'thrust': 1.0, 
                'lon': 122.425, 
                'lat': 31.425180164618112, 
                'mu': 5.073556701235858e-16, 
                'chi': 0.0
            }
        }

    def test_rollout_with_random_desired_goal(self):
        print("test rollout with random desired goal...............")
        
        self.rollout.env.unwrapped.task.goal_sampler.use_fixed_goal = False
        
        for i in range(3):
            self.rollout.rollout_one_trajectory(save_acmi=True, save_dir=PROJECT_ROOT_DIR / "evaluation" / "test" / "cache")
    
    def test_rollout_with_customized_desired_goal(self):
        print("test rollout with customized desired goal...............")
        
        self.rollout.env.unwrapped.task.goal_sampler.use_fixed_goal = True
        self.rollout.env.unwrapped.task.goal_sampler.goal_v = 220.0
        self.rollout.env.unwrapped.task.goal_sampler.goal_mu = 5.0
        self.rollout.env.unwrapped.task.goal_sampler.goal_chi = 10.0
        
        self.rollout.rollout_one_trajectory(save_acmi=True, save_dir=PROJECT_ROOT_DIR / "evaluation" / "test" / "cache")

if __name__ == "__main__":
    unittest.main()
