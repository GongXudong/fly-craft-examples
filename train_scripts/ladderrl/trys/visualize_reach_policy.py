import gymnasium as gym
import numpy as np
from pathlib import Path
import os
import sys
import torch as th
import argparse

from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
import flycraft
from flycraft.utils.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.vec_env_helper import get_vec_env
from utils_my.sb3.my_eval_callback import MyEvalCallback
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate
from train_scripts.ladderrl.utils.load_data_from_csv import load_random_trajectories_from_csv_files

import warnings
warnings.filterwarnings("ignore")  # 过滤Gymnasium的UserWarning
# gym.register_envs(flycraft)


# from utils_my.sb3.my_reach_reward_wrapper import PowerRewardWrapper
from train_scripts.ladderrl.utils.wrappers import PowerRewardWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv



import time

def visualize_sac_policy(model_path):
    # 1. 创建环境
    # render_mode='human' 让您能看到 3D 动画窗口
    env = gym.make("my-reach-08", render_mode='human')

    # 2. 加载您的 SAC 模型
    # 注意：请确保 'your_model_name' 是您实际保存的文件路径（不需要加 .zip 后缀）
    print(f"正在加载 SAC 模型: {model_path}...")
    
    try:
        model = SAC.load(model_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{model_path}.zip'。请确认文件名和路径是否正确。")
        return

    # 3. 初始化环境
    observation, info = env.reset()
    
    print("开始演示 SAC 策略... (按 Ctrl+C 停止)")
    
    try:
        # 运行 1000 个时间步
        for _ in range(10000):
            # -------------------------------------------------------
            # 关键点：deterministic=True
            # SAC 训练时是随机策略，测试时必须设为 True 以获取最优表现
            # -------------------------------------------------------
            action, _states = model.predict(observation, deterministic=True)
            
            # 执行动作
            observation, reward, terminated, truncated, info = env.step(action)
            
            # 稍微暂停一下，否则动作太快看不清
            time.sleep(0.05)

            # 如果任务完成或超时，重置环境
            if terminated or truncated:
                print("回合结束，重置环境")
                observation, info = env.reset()
                time.sleep(0.5) # 回合间稍微停顿

    except KeyboardInterrupt:
        print("\n演示结束")
    finally:
        env.close()

if __name__ == "__main__":
    
    visualize_sac_policy("/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/distance_threshold_0.01_to_0_005/sac_without_reset_relable_goal_range_05_to_08/b_0_5/sac_10hz_128_128_b_0_5_1e5steps_threshold_0_01_0_005_range_05_08_seed_1_singleRL/best_model")

# if __name__ == '__main__':
#     vec_env = make_vec_env(
#         env_id="my-reach",
#         n_envs=4,
#         seed=102,
#         vec_env_cls=SubprocVecEnv, 
#         wrapper_class=PowerRewardWrapper, 
#         wrapper_kwargs={"b": 0.5,"reward_type":"dense","distance_threshold":0.08},
#         env_kwargs={
#             "reward_type": "sparse",
#             "control_type": "joints",
#             "distance_threshold":0.0001,
#             "goal_range":0.001,
#             "max_episode_steps": 88
#         }
#     )
