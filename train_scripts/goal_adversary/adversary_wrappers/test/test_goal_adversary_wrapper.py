import unittest
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import flycraft
from stable_baselines3.ppo import PPO, MultiInputPolicy

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.goal_adversary.adversary_wrappers.goal_adversary_wrapper import GoalAdversaryWrapper
from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from train_scripts.goal_adversary.utils.random_policy import RandomPolicy
from train_scripts.msr.evaluate.evaluate_policy_by_success_rate import evaluate_policy_with_success_rate


class TestGoalAdversaryWrapper(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.original_env = gym.make(
            "FlyCraft-v0",
            config_file = PROJECT_ROOT_DIR / "configs" / "env" / "goal_adversary" / "env_config_for_ppo_10hz_medium_b_05.json",
            custom_config = {
                "debug_mode": True,
            },
        )
        self.original_env = ScaledActionWrapper(ScaledObservationWrapper(self.original_env))

        algo_dir = PROJECT_ROOT_DIR / "train_scripts/goal_adversary/adversary_wrappers/test/iter_1_bc_checkpoint.zip"
        # algo_dir = PROJECT_ROOT_DIR / "train_scripts/goal_adversary/adversary_wrappers/test/iter_4_rl_bc_best_model.zip"

        self.algo = PPO.load(algo_dir)

        self.env = GoalAdversaryWrapper(
            env=self.original_env, 
            policy=self.algo.policy,
            noise_min=-np.array([0.0100, 0.0167, 0.0083])*10,
            noise_max=np.array([0.0100, 0.0167, 0.0083])*10,
        )

    # def test_init(self):
    #     print("In test init.............")
    #     self.assertIsNotNone(self.env)
    
    # def test_policy(self):
    #     print("In test policy.............")
    #     res = evaluate_policy_with_success_rate(
    #         model=self.algo.policy,
    #         env=self.original_env,
    #         n_eval_episodes=100,
    #         deterministic=True,
    #         render=False,
    #     )
    #     print(f"Policy evaluation result: {res}")

    # def test_reset(self):
    #     print("In test reset.............")
    #     obs, info = self.env.reset()
    #     print(self.env.observation_space)
    #     print(self.env.action_space)
    #     print(obs)
    #     print(info)
    
    # def test_step(self):
    #     print("In test step.............")
    #     obs, info = self.env.reset()
    #     while True:
    #         print("\nnew step")
    #         action = self.env.action_space.sample()
    #         obs, reward, terminated, truncated, info = self.env.step(action)

    #         print(f"Action: {action}, \nObs: {obs}, \nReward: {reward}, \nTerminated: {terminated}, \nTruncated: {truncated}, \nInfo: {info}")

    #         if terminated or truncated:
    #             print("Episode ended.")
    #             break

    # def test_evaluate_with_random_policy1(self):
    #     print("In test evaluate with random policy 1.............")
    #     evaluate_num = 100
    #     success_episodes = 0
    #     total_reward, episode_reward_sum = 0.0, 0.0

    #     for i in tqdm(range(evaluate_num)):
    #         obs, info = self.env.reset()
    #         episode_reward_sum = 0.0
    #         while True:
    #             action = self.env.action_space.sample()
    #             obs, reward, terminated, truncated, info = self.env.step(action)
    #             episode_reward_sum += reward
    #             if terminated or truncated:
    #                 if info["is_success"]:
    #                     success_episodes += 1
                    
    #                 print(episode_reward_sum)
    #                 total_reward += episode_reward_sum
    #                 break
        
    #     print(f"success rate: {success_episodes / evaluate_num}, return: {total_reward / evaluate_num}")

    def test_evaluate_with_random_policy2(self):
        print("In test evaluate with random policy 2.............")
        evaluate_num = 100

        policy = RandomPolicy(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        )

        res = evaluate_policy_with_success_rate(
            model=policy,
            env=self.env,
            n_eval_episodes=evaluate_num,
            deterministic=True,
        )
        print(f"Policy evaluation result: {res}")


    # def test_evaluate_with_initialized_policy(self):
    #     print("In test evaluate with initialized policy.............")
    #     evaluate_num = 100
    #     success_episodes = 0
    #     total_reward, episode_reward_sum = 0.0, 0.0

    #     algo = PPO(
    #         policy=MultiInputPolicy,
    #         env=self.env,
    #     )

    #     for i in tqdm(range(evaluate_num)):
    #         obs, info = self.env.reset()
    #         episode_reward_sum = 0.0
    #         while True:
    #             action, _ = algo.policy.predict(obs, deterministic=True)
    #             # print(action)  # 初始化的policy，输出的动作在0附近
    #             obs, reward, terminated, truncated, info = self.env.step(action)
    #             episode_reward_sum += reward
    #             if terminated or truncated:
    #                 if info["is_success"]:
    #                     success_episodes += 1
    #                     total_reward += episode_reward_sum
    #                 break

    #     # 注意：初始化的policy，输出的动作在0附近，所以得到的成功率类似不添加噪声的情况！！！
    #     print(f"success rate: {success_episodes / evaluate_num}, return: {total_reward / evaluate_num}")
        

if __name__ == '__main__':
    unittest.main()