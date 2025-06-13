import sys
from pathlib import Path
import unittest
from time import time
import numpy as np
from stable_baselines3.ppo import PPO, MultiInputPolicy

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.goal_adversary.utils.vec_env_helper import get_goal_adversary_vec_env, get_goal_adversary_efficient_vec_env
from train_scripts.msr.evaluate.evaluate_policy_by_success_rate import evaluate_policy_with_success_rate
from train_scripts.goal_adversary.utils.random_policy import RandomPolicy

class TestVecEnvHelper(unittest.TestCase):
    def setUp(self):
        algo_path = PROJECT_ROOT_DIR / "train_scripts/goal_adversary/adversary_wrappers/test/iter_1_bc_checkpoint.zip"
        # algo_path = PROJECT_ROOT_DIR / "train_scripts/goal_adversary/adversary_wrappers/test/iter_4_rl_bc_best_model.zip"
        algo_class = PPO
        noise_min = -np.array([0.0100, 0.0167, 0.0083]) * 1
        noise_max = np.array([0.0100, 0.0167, 0.0083]) * 1

        env_config_file = PROJECT_ROOT_DIR / "configs" / "env" / "goal_adversary" / "env_config_for_ppo_10hz_medium_b_05.json"

        self.vec_env = get_goal_adversary_vec_env(
            algo_path=algo_path,
            algo_class=algo_class,
            goal_noise_min=noise_min,
            goal_noise_max=noise_max,
            num_process=4,
            config_file=env_config_file,
        )

        self.efficient_vec_env = get_goal_adversary_efficient_vec_env(
            algo_class=algo_class,
            algo_path=algo_path,
            goal_noise_min=noise_min,
            goal_noise_max=noise_max,
            num_process=4,
            config_file=env_config_file,
        )

    # def test_evaluate_with_random_policy_efficient_version(self):
    #     print("In test evaluate with random policy.............")

    #     begin_time = time()

    #     evaluate_num = 100

    #     random_policy = RandomPolicy(
    #         observation_space=self.efficient_vec_env.observation_space,
    #         action_space=self.efficient_vec_env.action_space,
    #     )

    #     res = evaluate_policy_with_success_rate(
    #         model=random_policy,
    #         env=self.efficient_vec_env,
    #         n_eval_episodes=evaluate_num,
    #         deterministic=True,
    #     )

    #     print(f"Policy evaluation result: {res}")
    #     print(f"Time taken for evaluation: {time() - begin_time:.2f} seconds")

    def test_evaluate_with_trained_policy(self):
        print("In test evaluate.............")

        begin_time = time()

        evaluate_num = 100

        random_policy = PPO.load(PROJECT_ROOT_DIR / "checkpoints/goal_adversary/medium_10hz/baseline/ppo/128_128_2e8steps_seed_1/best_model.zip")

        res = evaluate_policy_with_success_rate(
            model=random_policy,
            env=self.efficient_vec_env,
            n_eval_episodes=evaluate_num,
            deterministic=True,
        )

        print(f"Policy evaluation result: {res}")
        print(f"Time taken for evaluation: {time() - begin_time:.2f} seconds")

if __name__ == "__main__":
    unittest.main()
