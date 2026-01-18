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


import cv2  # 导入 OpenCV 用于图像处理和显示
import numpy as np
import time
from datetime import datetime

def visualize_and_save_on_timeout(model_path):
    # 1. 关键修改：使用 'rgb_array' 模式
    # 这样 env.render() 会返回 numpy 数组，而不是直接弹窗
    env = gym.make('my-reach-08', render_mode='rgb_array')

    print(f"正在加载模型: {model_path}...")
    model = SAC.load(model_path)

    observation, info = env.reset()
    
    print("开始运行... (按 'q' 键退出)")
    
    episode_count = 0

    try:
        while True:
            # 获取模型动作
            action, _states = model.predict(observation, deterministic=True)
            
            # 执行一步
            observation, reward, terminated, truncated, info = env.step(action)
            
            # 2. 获取当前帧图像
            frame = env.render()
            
            # 3. 颜色转换：Gym 输出是 RGB，OpenCV 需要 BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 4. 显示画面 (替代原来的 Human 窗口)
            cv2.imshow("Panda Replay (Press 'q' to quit)", frame_bgr)
            
            # 处理键盘输入，按 q 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # -------------------------------------------------------
            # 5. 核心逻辑：检测是否达到最大步数 (Truncated)
            # -------------------------------------------------------
            if truncated:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"timeout_screenshot_ep{episode_count}_{timestamp}.png"
                
                # 保存图片
                cv2.imwrite(filename, frame_bgr)
                print(f"⚠️ 达到最大步数！已截图保存为: {filename}")
                
                # 可选：稍微停顿一下让你看到“案发现场”
                time.sleep(1)

            # 正常的回合结束处理
            if terminated or truncated:
                observation, info = env.reset()
                episode_count += 1

    except KeyboardInterrupt:
        print("\n演示结束")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    
    visualize_and_save_on_timeout("/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/distance_threshold_0.01_to_0_005/sac_without_reset_relable_goal_range_05_to_08/b_0_5/sac_10hz_128_128_b_0_5_1e5steps_threshold_0_01_0_005_range_05_08_seed_1_singleRL/best_model")

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
