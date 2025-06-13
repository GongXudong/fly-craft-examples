import sys
from pathlib import Path
import unittest
from time import time
import numpy as np
from stable_baselines3.ppo import PPO, MultiInputPolicy

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.goal_adversary.utils.vec_env_helper import get_goal_adversary_vec_env, get_goal_adversary_efficient_vec_env, get_anti_goal_adversary_efficient_vec_env
from train_scripts.msr.evaluate.evaluate_policy_by_success_rate import evaluate_policy_with_success_rate
from train_scripts.goal_adversary.utils.random_policy import RandomPolicy

class TestVecEnvHelper(unittest.TestCase):
    def setUp(self):
        algo_path = PROJECT_ROOT_DIR / "train_scripts/goal_adversary/adversary_wrappers/test/iter_1_bc_checkpoint.zip"
        # algo_path = PROJECT_ROOT_DIR / "train_scripts/goal_adversary/adversary_wrappers/test/iter_4_rl_bc_best_model.zip"
        algo_class = PPO
        self.noise_min = -np.array([0.0100, 0.0167, 0.0083]) * 5
        self.noise_max = np.array([0.0100, 0.0167, 0.0083]) * 5

        self.env_config_file = PROJECT_ROOT_DIR / "configs" / "env" / "goal_adversary" / "env_config_for_ppo_10hz_medium_b_05.json"

        self.vec_env = get_goal_adversary_vec_env(
            algo_path=algo_path,
            algo_class=algo_class,
            goal_noise_min=self.noise_min,
            goal_noise_max=self.noise_max,
            num_process=4,
            config_file=self.env_config_file,
        )

        self.efficient_vec_env = get_goal_adversary_efficient_vec_env(
            algo_class=algo_class,
            algo_path=algo_path,
            goal_noise_min=self.noise_min,
            goal_noise_max=self.noise_max,
            num_process=4,
            config_file=self.env_config_file,
        )

    # def test_init(self):
    #     print("In test init.............")
    #     print("Observation space: ", self.vec_env.observation_space)
    #     print("Action space: ", self.vec_env.action_space)

    # def test_evaluate_with_initialized_policy(self):
    #     print("In test evaluate with initialized policy.............")

    #     evaluate_num = 100

    #     goal_adversary_algo = PPO(
    #         policy=MultiInputPolicy,
    #         env=self.vec_env,
    #     )

    #     print(goal_adversary_algo.observation_space)
    #     print(goal_adversary_algo.action_space)

    #     res = evaluate_policy_with_success_rate(
    #         model=goal_adversary_algo.policy,
    #         env=self.vec_env,
    #         n_eval_episodes=evaluate_num,
    #         deterministic=True,
    #     )

    #     print(f"Policy evaluation result: {res}")

    # def test_evaluate_with_random_policy(self):
    #     print("In test evaluate with initialized policy.............")

    #     begin_time = time()

    #     evaluate_num = 100

    #     random_policy = RandomPolicy(
    #         observation_space=self.vec_env.observation_space,
    #         action_space=self.vec_env.action_space,
    #     )

    #     res = evaluate_policy_with_success_rate(
    #         model=random_policy,
    #         env=self.vec_env,
    #         n_eval_episodes=evaluate_num,
    #         deterministic=True,
    #     )

    #     print(f"Policy evaluation result: {res}")
    #     print(f"Time taken for evaluation: {time() - begin_time:.2f} seconds")

    # def test_init_efficient_version(self):
    #     print("In test init (efficient get vec_env version).............")
    #     print("Observation space: ", self.efficient_vec_env.observation_space)
    #     print("Action space: ", self.efficient_vec_env.action_space)

    # def test_evaluate_with_random_policy_efficient_version(self):
    #     print("In test evaluate with initialized policy (efficient get vec_env version).............")

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
    
    # def test_evaluate_get_anti_goal_adversary_vec_wrapper_with_random_policy_efficient_version(self):
    #     print("In test evaluate get_anti_goal_adversary_vec_wrapper with random policy (efficient get vec_env version).............")

    #     self.efficient_anti_goal_adversary_vec_env = get_anti_goal_adversary_efficient_vec_env(
    #         goal_adversary_algo_class=PPO,
    #         goal_adversary_algo_path=PROJECT_ROOT_DIR / "train_scripts/goal_adversary/adversary_wrappers/test/goal_adversary_anti_iter_1_bc_best_model",
    #         goal_noise_min=self.noise_min,
    #         goal_noise_max=self.noise_max,
    #         num_process=4,
    #         config_file=self.env_config_file,
    #     )

    #     begin_time = time()

    #     evaluate_num = 100

    #     random_policy = RandomPolicy(
    #         observation_space=self.efficient_anti_goal_adversary_vec_env.observation_space,
    #         action_space=self.efficient_anti_goal_adversary_vec_env.action_space,
    #     )

    #     res = evaluate_policy_with_success_rate(
    #         model=random_policy,
    #         env=self.efficient_anti_goal_adversary_vec_env,
    #         n_eval_episodes=evaluate_num,
    #         deterministic=True,
    #     )

    #     print(f"Policy evaluation result: {res}")
    #     print(f"Time taken for evaluation: {time() - begin_time:.2f} seconds")

    def test_evaluate_get_anti_goal_adversary_vec_wrapper_with_trained_policy_efficient_version(self):
        print("In test evaluate get_anti_goal_adversary_vec_wrapper with trained policy (efficient get vec_env version).............")

        self.efficient_anti_goal_adversary_vec_env = get_anti_goal_adversary_efficient_vec_env(
            goal_adversary_algo_class=PPO,
            goal_adversary_algo_path=PROJECT_ROOT_DIR / "train_scripts/goal_adversary/adversary_wrappers/test/goal_adversary_anti_iter_1_bc_best_model",
            goal_noise_min=self.noise_min,
            goal_noise_max=self.noise_max,
            num_process=4,
            config_file=self.env_config_file,
        )

        begin_time = time()

        evaluate_num = 100

        algo = PPO.load(
            # path=PROJECT_ROOT_DIR / "train_scripts/goal_adversary/adversary_wrappers/test/iter_1_bc_checkpoint",
            path=PROJECT_ROOT_DIR / "train_scripts/goal_adversary/adversary_wrappers/test/iter_4_rl_bc_best_model",
            env=self.efficient_anti_goal_adversary_vec_env,
        )

        res = evaluate_policy_with_success_rate(
            model=algo.policy,
            env=self.efficient_anti_goal_adversary_vec_env,
            n_eval_episodes=evaluate_num,
            deterministic=True,
        )

        print(f"Policy evaluation result: {res}")
        print(f"Time taken for evaluation: {time() - begin_time:.2f} seconds")

if __name__ == "__main__":
    unittest.main()
