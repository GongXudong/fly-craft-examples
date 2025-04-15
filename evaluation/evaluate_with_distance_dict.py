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

# 添加项目根目录到 sys.path
PROJECT_ROOT_DIR = Path().absolute().parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

# 导入自定义模块
from train_scripts.D2D.utils.wrappers import PowerRewardWrapper

def evaluate_distance(args):
    env_id = args['env_id']
    env_beta = args['env_beta']
    n_eval_episodes = args['n_eval_episodes']
    algo_ckpt_dir = args['algo_ckpt_dir']
    algo_ckpt_model_name = args['algo_ckpt_model_name']
    res_file_save_name = args['res_file_save_name']

    env = gym.make(env_id)
    env = PowerRewardWrapper(env, b=env_beta)

    algo = SAC.load(str(algo_ckpt_dir + '/' + algo_ckpt_model_name), env=env)
    algo.policy.set_training_mode(False)

    initial_positions = []
    final_positions = []
    desired_goals = []
    goal_distances = []

    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        initial_positions.append(obs["observation"])  # 初始位置
        desired_goals.append(obs["desired_goal"])  # 目标位置

        done = False
        terminated = False
        truncated = False
        while not done:
            action, _ = algo.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if terminated:
            initial_positions.pop()
            desired_goals.pop()
        elif truncated:
            final_positions.append(obs["observation"])  # 终止位置
            goal_distances.append(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))  # 终止位置和目标位置的距离

        if episode % 100 == 0:
            print(f"Episode {episode + 1}/{n_eval_episodes} completed")

    print(f"Evaluation on {algo_ckpt_dir}")
    success_rate = (n_eval_episodes - len(initial_positions)) / n_eval_episodes
    average_distance = sum(goal_distances) / n_eval_episodes if goal_distances else 0
    total_distance = sum(goal_distances)
    print(f"Success rate = {success_rate}, Fail episodes number = {len(initial_positions)}, Average distance = {average_distance}, Total distance = {total_distance}")

    data = {
        "initial_position": initial_positions,
        "final_position": final_positions,
        "desired_goal": desired_goals,
        "goal_distance": goal_distances
    }
    save_path = PROJECT_ROOT_DIR / res_file_save_name / 'evaluate_with_distance.csv'
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)

    if not save_path.parent.exists():
        os.makedirs(save_path.parent)

    df.to_csv(save_path, index=False)

# # 定义参数字典
# args = {
#     "env_id": "my-reach",
#     "env_beta": 2.0,
#     "n_eval_episodes": 1000,
#     "algo_ckpt_dir": "checkpoints/D2D/panda_reach_dense/distance_threshold_0_01/her/two_stage_relative_hard_b_2_b_05/sac_her_10hz_128_128_b_2_5e4steps_seed_1_singleRL",
#     "algo_ckpt_model_name": "best_model",
#     "res_file_save_name": "logs/D2D/panda_reach_dense/distance_threshold_0_01/her/two_stage_relative_hard_b_2_b_05/sac_her_10hz_128_128_b_2_5e4steps_seed_1_singleRL"
# }

# # 调用 evaluate_distance 函数
# evaluate_distance(args)