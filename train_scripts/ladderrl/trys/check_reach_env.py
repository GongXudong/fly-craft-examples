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





def check_subproc_env(vec_env):
    print("=" * 40)
    print("🔍 SubprocVecEnv 环境配置核查")
    print("=" * 40)

    # 1. 获取最大步长 (Max Episode Steps)
    # TimeLimit 包装器通常会将限制存储在 _max_episode_steps 中
    # get_attr 会返回一个列表，包含所有并行环境的该属性值，我们取第0个即可
    try:
        max_steps_list = vec_env.get_attr("_max_episode_steps")
        current_max_steps = max_steps_list[0]
        print(f"✅ [设置确认] 最大步长 (TimeLimit): {current_max_steps}")
    except AttributeError:
        # 如果获取失败，尝试从 spec 中获取
        try:
            specs = vec_env.get_attr("spec")
            print(f"⚠️ [备选确认] 最大步长 (via Spec): {specs[0].max_episode_steps}")
        except:
            print(f"❌ 无法通过属性直接读取最大步长")

    # 2. 获取 Panda-Gym 特有参数 (如 reward_type)
    # get_attr 会自动穿透 Wrapper 层寻找属性
    try:
        reward_types = vec_env.get_attr("reward_type")
        print(f"✅ [设置确认] 奖励类型 (Reward Type): {reward_types[0]}")
    except AttributeError:
        print(f"❌ 无法找到 'reward_type' 属性 (可能是属性名不对或环境不支持)")

    # 3. 动态运行测试 (这是最稳妥的方法)
    # 无论单进程还是多进程，step() 的行为是一致的
    print("\n🚀 [动态测试] 正在运行一步步测试 (Step Test)...")
    
    vec_env.reset()
    steps = 0
    done = False
    
    # 既然是多进程，我们只关注第0个环境的行为
    while True:
        # 发送随机动作
        actions = [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
        obs, rewards, dones, infos = vec_env.step(actions)
        steps += 1
        
        # 检查第0个环境是否结束
        if dones[0]:
            print(f"🛑 第 0 号环境在第 [ {steps} ] 步结束")
            
            # 检查结束原因
            info = infos[0]
            if "TimeLimit.truncated" in info and info["TimeLimit.truncated"]:
                print(f"   -> 原因: 超时截断 (Truncated) ✅")
                if steps == current_max_steps:
                    print("   -> 结论: 步数设置完全符合预期！")
                else:
                    print(f"   -> ⚠️ 警告: 实际截断步数 ({steps}) 与读取到的属性 ({current_max_steps}) 不一致！")
            else:
                print(f"   -> 原因: 任务完成或失败 (Terminated) - 无法验证最大步长")
            
            break
        
        if steps > (current_max_steps + 10):
            print(f"❌ 测试失败: 已经运行了 {steps} 步，超过了设定的 {current_max_steps}，环境仍未重置！")
            break

import numpy as np
import gymnasium as gym

def verify_goal_and_threshold(vec_env):
    print("=" * 50)
    print("🎯 PandaReach 目标范围与阈值深度验证")
    print("=" * 50)

    # ---------------------------------------------------
    # 1. 静态属性读取 (通过 get_attr 穿透多进程获取)
    # ---------------------------------------------------
    print("\n[1] 读取内部属性配置:")
    
    try:
        # 获取 distance_threshold
        # 这是 RobotTaskEnv 的标准属性
        d_thresholds = vec_env.get_attr("distance_threshold")
        d_thresh = d_thresholds[0]
        print(f"   - ✅ 距离阈值 (distance_threshold): {d_thresh} (预期: < {d_thresh} 则成功)")
    except AttributeError:
        print(f"   - ❌ 无法读取 distance_threshold")
        d_thresh = None

    try:
        # 获取 goal_range
        # PandaReach 通常用这个值定义 x, y 的采样范围 [-val, +val]
        g_ranges = vec_env.get_attr("goal_range")
        g_range = g_ranges[0]
        print(f"   - ✅ 目标范围参数 (goal_range): {g_range}")
    except AttributeError:
        print(f"   - ❌ 无法读取 goal_range")
        g_range = None

    # ---------------------------------------------------
    # 2. 动态采样验证 (Empirical Verification)
    # ---------------------------------------------------
    print("\n[2] 动态采样验证 (采样 500 次 Reset):")
    print("   正在统计生成的 'desired_goal' 实际分布...")

    sampled_goals = []
    
    # 进行多次 reset 以收集目标点数据
    # 注意：reset() 返回的 obs 是一个字典 (Dict observation)
    # 结构: {'observation': ..., 'achieved_goal': ..., 'desired_goal': ...}
    for _ in range(50): 
        obs = vec_env.reset()
        # obs['desired_goal'] 的形状是 (n_envs, 3)
        # 我们把它展平并收集起来
        sampled_goals.append(obs['desired_goal'])
    
    # 转换为大数组 (总采样数 x 3)
    all_goals = np.vstack(sampled_goals)
    
    # 计算统计特征
    min_vals = all_goals.min(axis=0)
    max_vals = all_goals.max(axis=0)
    
    # 打印 X, Y, Z 的范围
    print(f"   - 实际生成的 X 范围: [{min_vals[0]:.4f}, {max_vals[0]:.4f}]")
    print(f"   - 实际生成的 Y 范围: [{min_vals[1]:.4f}, {max_vals[1]:.4f}]")
    print(f"   - 实际生成的 Z 范围: [{min_vals[2]:.4f}, {max_vals[2]:.4f}]")

    # 验证 goal_range
    if g_range is not None:
        # PandaReach 通常在 XY 平面上使用 [-goal_range, goal_range]
        # Z 轴通常是固定的或有单独的逻辑
        is_x_ok = -g_range - 0.05 <= min_vals[0] and max_vals[0] <= g_range + 0.05
        is_y_ok = -g_range - 0.05 <= min_vals[1] and max_vals[1] <= g_range + 0.05
        
        if is_x_ok and is_y_ok:
            print(f"   -> ✅ 统计结果与 goal_range={g_range} 一致 (XY轴在范围内)")
        else:
            print(f"   -> ⚠️ 警告: 统计范围与 goal_range 略有出入，请检查是否有额外偏移量(Offset)")

    # ---------------------------------------------------
    # 3. 距离逻辑手动校验
    # ---------------------------------------------------
    print("\n[3] 距离计算逻辑校验 (Distance Logic):")
    
    # 获取当前观测
    obs = vec_env.reset()
    ag = obs['achieved_goal'][0] # 第0个环境的当前位置
    dg = obs['desired_goal'][0]  # 第0个环境的目标位置
    
    # 手动计算欧氏距离
    dist = np.linalg.norm(ag - dg)
    print(f"   - 当前 Achieved Goal: {ag}")
    print(f"   - 当前 Desired Goal : {dg}")
    print(f"   - 💡 手算欧氏距离: {dist:.6f}")
    
    if d_thresh is not None:
        is_success = dist < d_thresh
        print(f"   - 判定结果: {'成功 (Success)' if is_success else '未成功 (Not Success)'}")
        print(f"     (基于 distance_threshold={d_thresh})")

# 调用验证函数
# verify_goal_and_threshold(env)

if __name__ == '__main__':
    vec_env = make_vec_env(
        env_id="my-reach",
        n_envs=4,
        seed=102,
        vec_env_cls=SubprocVecEnv, 
        wrapper_class=PowerRewardWrapper, 
        wrapper_kwargs={"b": 0.5,"reward_type":"dense","distance_threshold":0.08},
        env_kwargs={
            "reward_type": "sparse",
            "control_type": "joints",
            "distance_threshold":0.0001,
            "goal_range":0.001,
            "max_episode_steps": 88
        }
    )


    # vec_env
    check_subproc_env(vec_env)
    verify_goal_and_threshold(vec_env)

            # "id": "my-reach",
            # "__help": "goal_range, distance_threshold, max_episode_steps只用于记录，环境的实际配置需在脚本中指定！！！",
            # "goal_range": 0.5,
            # "distance_threshold": 0.01,
            # "max_episode_steps": 50,
            # "reward_type": "dense", 
            # "control_type": "joints",
            # "normalize": false,
            # "b": 0.5