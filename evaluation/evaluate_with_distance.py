import sys
import os
from pathlib import Path
from copy import deepcopy
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import pandas as pd
import argparse
from typing import List

import torch as th
from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import MultiInputPolicy as SACMultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy


PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))


# from utils_my.env_utils.register_env import register_all_with_default_dense_params, register_all_with_default_sparse_params

# register_all_with_default_dense_params()  # 注意此处：max_episode_steps, 根据环境文件的配置修改此值！！！！
# register_all_with_default_sparse_params()
from utils_my.sb3.my_reach_reward_wrapper import PowerRewardWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

def work(args):

    print(f"seed in eval: {args.env_random_seed}")
    # vec_env = make_vec_env(
    #     env_id=args.env,
    #     n_envs=args.n_envs,
    #     seed=args.env_random_seed,
    #     vec_env_cls=SubprocVecEnv, 
    #     wrapper_class=PowerRewardWrapper, 
    #     wrapper_kwargs={"b": args.env_beta},
    # )

    env_id = args.env_id
    env_beta = args.env_beta
    env = gym.make(env_id)
    env = PowerRewardWrapper(env,b = env_beta)

    algo = SAC.load(str(args.algo_ckpt_dir +'/'+ args.algo_ckpt_model_name),
                    env = env)
    algo.policy.set_training_mode(False)

    # mean_reward, mean_episode_length, success_rate = evaluate_policy_with_success_rate(
    #     model=algo_ppo.policy,
    #     env=vec_env,
    #     n_eval_episodes=n_eval_episodes
    # )
    initial_positions = []
    final_positions = []
    desired_goals = []
    goal_distances = []
    for episode in range(args.n_eval_episodes):
        obs, info = env.reset()
        initial_positions.append(obs["observation"])  # 初始位置
        desired_goals.append(obs["desired_goal"])  # 目标位置

        done = False
        while not done:
            action, _ = algo.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        final_positions.append(obs["observation"])  # 终止位置
        goal_distances.append(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))  # 终止位置和目标位置的距离

        if episode % 100 == 0:
            print(f"Episode {episode + 1}/1000 completed")


    data = {
        "initial_position": initial_positions,
        "final_position": final_positions,
        "desired_goal": desired_goals,
        "goal_distance": goal_distances
    }
    save_path: Path = PROJECT_ROOT_DIR / args.res_file_save_name
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)    
    

    
    
    if not save_path.parent.exists():
        os.makedirs(save_path.parent)

    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pass configurations")
    # environment
    parser.add_argument("--env-id", type=str, default="my-reach", help="environment ID")
    parser.add_argument("--env-beta", type=float, default="1.0", help="beta b")
    parser.add_argument("--n-envs", type=int, default=8, help="the number of environments used in this evaluation")
    parser.add_argument("--n-eval-episodes", type=int, default=1000, help="the number of episodes used in this evaluation")
    parser.add_argument("--env-random-seed", type=int, default=2426, help="environment random seed")
    # algorithm
    parser.add_argument("--algo-class", type=str, default="SAC", help="algorithm class, can be one of: SAC")
    parser.add_argument("--algo-ckpt-dir", type=str, default="checkpoints/D2D/panda_reach_dense/distance_threshold_0_01/her/two_stage_relative_hard_b_2_b_05/sac_her_10hz_128_128_b_2_5e4steps_seed_1_singleRL", help="algorithm checkpoint file")
    parser.add_argument("--algo-ckpt-model-name", type=str, default="best_model", help="algorithm checkpoint model name")
    

    # save res file
    parser.add_argument("--res-file-save-name", type=str, default="logs/D2D/panda_reach_dense/distance_threshold_0_01/her/two_stage_relative_hard_b_2_b_05/sac_her_10hz_128_128_b_2_5e4steps_seed_1_singleRL/evaluate_with_distance.csv", help="result file save name")


    args = parser.parse_args()

    work(args)
