import unittest
import sys
from pathlib import Path
import numpy as np
from time import time
from tqdm import tqdm
import gymnasium as gym
import flycraft
from stable_baselines3.ppo import PPO, MultiInputPolicy

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.goal_adversary.adversary_wrappers.anti_goal_adversary_vec_wrapper import VecAntiGoalAdversaryWrapper
from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from train_scripts.goal_adversary.utils.random_policy import RandomPolicy
from train_scripts.msr.evaluate.evaluate_policy_by_success_rate import evaluate_policy_with_success_rate
from utils_my.sb3.vec_env_helper import get_vec_env

class TestVecAntiGoalAdversaryWrapper(unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.process_num = 4

        self.env_config_file = PROJECT_ROOT_DIR / "configs" / "env" / "goal_adversary" / "env_config_for_ppo_10hz_medium_b_05.json"

        self.origin_vec_env = get_vec_env(
            num_process=self.process_num,
            seed=26873,
            config_file=self.env_config_file,
            custom_config={
                "debug_mode": True,
            },
        )

        goal_adversary_algo_path = PROJECT_ROOT_DIR / "train_scripts/goal_adversary/adversary_wrappers/test/goal_adversary_anti_iter_1_bc_best_model.zip"
        goal_adversary_algo = PPO.load(goal_adversary_algo_path)
        noise_min = -np.array([0.0100, 0.0167, 0.0083]) * 1
        noise_max = np.array([0.0100, 0.0167, 0.0083]) * 1

        self.anti_goal_adversary_wrappered_vec_env = VecAntiGoalAdversaryWrapper(
            venv=self.origin_vec_env,
            goal_adversary_policy=goal_adversary_algo.policy,
            noise_min=noise_min,
            noise_max=noise_max,
            env_config=self.env_config_file,
        )
    
    # def test_init(self):
    #     print("In test init.............")
        
    #     print(self.anti_goal_adversary_wrappered_vec_env.observation_space)
    #     print(self.anti_goal_adversary_wrappered_vec_env.action_space)

    # def test_reset(self):
    #     print("In test reset.............")
        
    #     obs = self.anti_goal_adversary_wrappered_vec_env.reset()
        
    #     print(f"Reset observation: {obs}")
    #     self.assertEqual(type(obs), dict)

    #     print(f"{type(obs)}, {obs['observation'].shape}")
    #     self.assertTupleEqual(obs['observation'].shape, (self.process_num, self.origin_vec_env.observation_space['observation'].shape[0]))
    #     self.assertTupleEqual(obs['desired_goal'].shape, (self.process_num, self.origin_vec_env.observation_space['desired_goal'].shape[0]))
    #     self.assertTupleEqual(obs['achieved_goal'].shape, (self.process_num, self.origin_vec_env.observation_space['achieved_goal'].shape[0]))

    # def test_step(self):
    #     print("In test step.............")
        
    #     obs = self.anti_goal_adversary_wrappered_vec_env.reset()
    #     print(f"Reset observation: {obs}")

    #     for _ in tqdm(range(10)):
    #         action = [self.anti_goal_adversary_wrappered_vec_env.action_space.sample() for i in range(self.process_num)]
    #         action = np.array(action)
            
    #         obs, reward, done, info = self.anti_goal_adversary_wrappered_vec_env.step(action)
    #         print(f"Step observation: {obs}, reward: {reward}, done: {done}, info: {info}")
        
    #     print("Testing step completed.")

    # def test_evaluate_with_random_policy(self):
    #     print("In test evaluate with random policy.............")
        
    #     evaluate_num = 100

    #     random_policy = RandomPolicy(
    #         observation_space=self.anti_goal_adversary_wrappered_vec_env.observation_space,
    #         action_space=self.anti_goal_adversary_wrappered_vec_env.action_space,
    #     )

    #     res = evaluate_policy_with_success_rate(
    #         model=random_policy,
    #         env=self.anti_goal_adversary_wrappered_vec_env,
    #         n_eval_episodes=evaluate_num,
    #         deterministic=True,
    #     )

    #     print(f"Policy evaluation result: {res}")
    
    def test_evaluate_with_trained_policy(self):
        print("In test evaluate with trained policy.............")

        begin_time = time()

        evaluate_num = 100

        # trained_policy_path = PROJECT_ROOT_DIR / "train_scripts/goal_adversary/adversary_wrappers/test/iter_1_bc_checkpoint.zip"
        # trained_policy_path = PROJECT_ROOT_DIR / "train_scripts/goal_adversary/adversary_wrappers/test/iter_4_rl_bc_best_model.zip"
        trained_policy_path = PROJECT_ROOT_DIR / "checkpoints/goal_adversary/medium_10hz/iter_1_bc/anti_goal_adversary_128_128_1e7steps_seed_1/best_model.zip"

        trained_policy = PPO.load(trained_policy_path)

        res = evaluate_policy_with_success_rate(
            model=trained_policy,
            env=self.anti_goal_adversary_wrappered_vec_env,
            n_eval_episodes=evaluate_num,
            deterministic=True,
        )

        print(f"Policy evaluation result: {res}")
        print(f"Time taken for evaluation: {time() - begin_time:.2f} seconds")

if __name__ == "__main__":
    unittest.main()