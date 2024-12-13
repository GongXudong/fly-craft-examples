import gymnasium as gym
import numpy as np
from pathlib import Path
import logging
import torch as th
import argparse
from copy import deepcopy
import os
import sys

from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.vec_env import VecCheckNan

import flycraft
from flycraft.utils.load_config import load_config
from flycraft.utils.dict_utils import update_nested_dict

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.disc.algorithms.smooth_goal_ppo import SmoothGoalPPO
from utils_my.sb3.vec_env_helper import get_vec_env
from utils_my.sb3.my_eval_callback import MyEvalCallback
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate
from utils_my.sb3.my_schedule import linear_schedule
from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper

np.seterr(all="raise")  # 检查nan

def get_ppo_algo(env, env_used_in_attacker):
    policy_kwargs = dict(
        net_arch=dict(
            pi=NET_ARCH,
            vf=deepcopy(NET_ARCH)
        )
    )

    return SmoothGoalPPO(
        policy=MultiInputPolicy, 
        env=env, 
        seed=SEED,
        batch_size=PPO_BATCH_SIZE,
        gamma=GAMMA,
        n_steps=2048,  # 采样时每个环境采样的step数
        n_epochs=5,  # 采样的数据在训练中重复使用的次数
        ent_coef=RL_ENT_COEF,
        policy_kwargs=policy_kwargs,
        use_sde=True,  # 使用state dependant exploration,
        normalize_advantage=True,
        learning_rate=linear_schedule(3e-4),
        device=DEVICE,
        goal_noise_epsilon=np.array(GOAL_NOISE_EPSILON),
        goal_regularization_strength=GOAL_REGULARIZATION_STRENGTH,
        env_used_in_attacker=env_used_in_attacker,
    )

def train():

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / "disc" / RL_EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    tmp_custom_config = {"debug_mode": True, "flag_str": "Train"}
    update_nested_dict(tmp_custom_config, ENV_CUSTOM_CONFIG)
    print(ENV_CUSTOM_CONFIG, tmp_custom_config)
    env_config_dict_in_training = {
        "num_process": ROLLOUT_PROCESS_NUM, 
        "seed": SEED_IN_TRAINING_ENV,
        "config_file": str(PROJECT_ROOT_DIR / "configs" / "env" / ENV_CONFIG_FILE),
        "custom_config": tmp_custom_config,
    }

    env_num_used_in_eval = EVALUATE_PROCESS_NUM
    env_config_dict_in_eval = deepcopy(env_config_dict_in_training)
    tmp_custom_config_in_eval = {"debug_mode": False, "flag_str": "Evaluate"}
    update_nested_dict(tmp_custom_config_in_eval, ENV_CUSTOM_CONFIG)
    print(ENV_CUSTOM_CONFIG, tmp_custom_config_in_eval)
    update_nested_dict(env_config_dict_in_eval, {
        "num_process": env_num_used_in_eval,
        "custom_config": tmp_custom_config_in_eval,
    })
    
    env_num_used_in_callback = CALLBACK_PROCESS_NUM
    env_config_dict_in_callback = deepcopy(env_config_dict_in_training)
    tmp_custom_config_in_callback = {"debug_mode": True, "flag_str": "Callback"}
    update_nested_dict(tmp_custom_config_in_callback, ENV_CUSTOM_CONFIG)
    print(ENV_CUSTOM_CONFIG, tmp_custom_config_in_callback)
    update_nested_dict(env_config_dict_in_callback, {
        "seed": SEED_IN_CALLBACK_ENV,
        "num_process": env_num_used_in_callback,
        "custom_config": tmp_custom_config_in_callback,
    })

    vec_env = VecCheckNan(get_vec_env(
        **env_config_dict_in_training
    ))
    # evaluate_policy使用的测试环境
    eval_env = VecCheckNan(get_vec_env(
        **env_config_dict_in_eval
    ))
    # 回调函数中使用的测试环境
    eval_env_in_callback = VecCheckNan(get_vec_env(
        **env_config_dict_in_callback
    ))

    env_used_in_attacker = ScaledActionWrapper(
        ScaledObservationWrapper(
            gym.make(
                "FlyCraft-v0", 
                config_file=str(PROJECT_ROOT_DIR / "configs" / "env" / ENV_CONFIG_FILE)
            )
        )
    )

    algo_ppo = get_ppo_algo(vec_env, env_used_in_attacker)
    sb3_logger.info(str(algo_ppo.policy))

    # set sb3 logger
    algo_ppo.set_logger(sb3_logger)

    eval_callback = MyEvalCallback(
        eval_env_in_callback, 
        best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / "disc" / RL_EXPERIMENT_NAME).absolute()),
        log_path=str((PROJECT_ROOT_DIR / "logs" / "disc" / RL_EXPERIMENT_NAME).absolute()), 
        eval_freq=EVALUATE_FREQUENCE,
        n_eval_episodes=EVALUATE_NUMS_IN_CALLBACK*env_num_used_in_callback,
        deterministic=True, 
        render=False,
    )

    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path=str((PROJECT_ROOT_DIR / "checkpoints" / "disc" / RL_EXPERIMENT_NAME).absolute()))
    event_callback = EveryNTimesteps(n_steps=SAVE_CHECKPOINT_EVERY_N_TIMESTEPS, callback=checkpoint_on_event)

    algo_ppo.learn(
        total_timesteps=RL_TRAIN_STEPS, 
        callback=[eval_callback, event_callback]
    )

    # evaluate
    reward, _, success_rate = evaluate_policy_with_success_rate(algo_ppo.policy, eval_env, EVALUATE_NUMS_IN_EVALUATION*env_num_used_in_eval)
    sb3_logger.info(f"Reward after RL: {reward}")
    sb3_logger.info(f"Success rate after RL: {success_rate}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="传入配置文件")
    parser.add_argument("--config-file-name", type=str, help="配置文件名", default="ppo_bc_config_10hz_128_128_1.json")
    args = parser.parse_args()

    train_config = load_config(Path(os.getcwd()) / args.config_file_name)

    ENV_CONFIG_FILE = train_config["env"]["config_file"]
    ENV_CUSTOM_CONFIG = train_config["env"].get("custom_config", {})

    SEED = train_config["rl"]["seed"]
    SEED_IN_TRAINING_ENV = train_config["rl"].get("seed_in_train_env")
    SEED_IN_CALLBACK_ENV = train_config["rl"].get("seed_in_callback_env")

    RL_EXPERIMENT_NAME = train_config["rl"]["experiment_name"]
    GAMMA = train_config["rl"]["gamma"]
    NET_ARCH = train_config["rl"]["net_arch"]
    PPO_BATCH_SIZE = train_config["rl"]["batch_size"]
    RL_TRAIN_STEPS = train_config["rl"]["train_steps"]
    RL_ENT_COEF = train_config["rl"].get("ent_coef", 0.0)
    ROLLOUT_PROCESS_NUM = train_config["rl"]["rollout_process_num"]
    EVALUATE_PROCESS_NUM = train_config["rl"].get("evaluate_process_num", 32)
    CALLBACK_PROCESS_NUM = train_config["rl"].get("callback_process_num", 32)
    EVALUATE_FREQUENCE = train_config["rl"].get("evaluate_frequence", 2048)
    EVALUATE_NUMS_IN_EVALUATION = train_config["rl"].get("evaluate_nums_in_evaluation", 30)
    EVALUATE_NUMS_IN_CALLBACK = train_config["rl"].get("evaluate_nums_in_callback", 3)
    SAVE_CHECKPOINT_EVERY_N_TIMESTEPS = train_config["rl"].get("save_checkpoint_every_n_timesteps", 4_000_000)

    GOAL_NOISE_EPSILON = train_config["rl"].get("goal_noise_epsilon", [10., 3., 3.])
    GOAL_REGULARIZATION_STRENGTH = train_config["rl"].get("goal_regularization_strength", 1e-3)

    DEVICE = train_config["rl"].get("device", "cpu")

    train()
