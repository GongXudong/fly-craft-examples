import gymnasium as gym
import numpy as np
from pathlib import Path
import logging
import torch as th
import argparse
from copy import deepcopy
import os
import sys

from stable_baselines3.ppo import PPO, MultiInputPolicy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, EvalCallback
from stable_baselines3.common.vec_env import VecCheckNan

import flycraft
from flycraft.utils.load_config import load_config
from flycraft.utils.dict_utils import update_nested_dict

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.goal_adversary.utils.vec_env_helper import get_anti_goal_adversary_efficient_vec_env
from utils_my.sb3.my_eval_callback import MyEvalCallback
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate
from utils_my.sb3.my_schedule import linear_schedule
from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper

np.seterr(all="raise")  # 检查nan

def get_ppo_algo(env):
    policy_kwargs = dict(
        net_arch=dict(
            pi=NET_ARCH,
            vf=deepcopy(NET_ARCH)
        )
    )

    return PPO(
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
        learning_rate=3e-4,
        device=DEVICE,
    )

def load_algo(env):
    assert RL_INITIAL_POLICY != "none", "RL_INITIAL_POLICY should not be 'none' when loading an existing policy."
    algo = PPO.load(
        str(PROJECT_ROOT_DIR / RL_INITIAL_POLICY),
        env=env,
        device=DEVICE,
    )
    return algo


def train():

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / RL_EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    if ANTI_GOAL_ADVERSARY_REFERENCE_ALGO_TYPE == "ppo":
        referenced_goal_adversary_algo_class = PPO
    else:
        raise ValueError("Unsupported GOAL_ADVERSARY_REFERENCE_ALGO_TYPE: {}".format(ANTI_GOAL_ADVERSARY_REFERENCE_ALGO_TYPE))

    tmp_custom_config = {"debug_mode": True, "flag_str": "Train"}
    update_nested_dict(tmp_custom_config, ENV_CUSTOM_CONFIG)
    print(ENV_CUSTOM_CONFIG, tmp_custom_config)
    print(f"goal noise min: {GOAL_ADVERSARY_NOISE_MIN}, max: {GOAL_ADVERSARY_NOISE_MAX}")
    print(f"type: {ANTI_GOAL_ADVERSARY_REFERENCE_ALGO_TYPE}, referenced algo path: {ANTI_GOAL_ADVERSARY_REFERENCE_ALGO_PATH}")

    env_config_dict_in_training = {
        "num_process": ROLLOUT_PROCESS_NUM,
        "seed": SEED_IN_TRAINING_ENV,
        "config_file": str(PROJECT_ROOT_DIR / "configs" / "env" / ENV_CONFIG_FILE),
        "custom_config": tmp_custom_config,
        "goal_adversary_algo_class": referenced_goal_adversary_algo_class,
        "goal_adversary_algo_path": str(PROJECT_ROOT_DIR / ANTI_GOAL_ADVERSARY_REFERENCE_ALGO_PATH),
        "goal_noise_min": np.array(GOAL_ADVERSARY_NOISE_MIN),
        "goal_noise_max": np.array(GOAL_ADVERSARY_NOISE_MAX),
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

    vec_env = VecCheckNan(get_anti_goal_adversary_efficient_vec_env(
        **env_config_dict_in_training
    ))
    # evaluate_policy使用的测试环境
    eval_env = VecCheckNan(get_anti_goal_adversary_efficient_vec_env(
        **env_config_dict_in_eval
    ))
    # 回调函数中使用的测试环境
    eval_env_in_callback = VecCheckNan(get_anti_goal_adversary_efficient_vec_env(
        **env_config_dict_in_callback
    ))

    if RL_INITIAL_POLICY == "none":
        algo_ppo = get_ppo_algo(vec_env)
    else: 
        algo_ppo = load_algo(vec_env)

    sb3_logger.info(str(algo_ppo.policy))

    # set sb3 logger
    algo_ppo.set_logger(sb3_logger)

    sb3_logger.log(f"Check goal adversary configs, algorithm path: {ANTI_GOAL_ADVERSARY_REFERENCE_ALGO_PATH}, goal noise min: {GOAL_ADVERSARY_NOISE_MIN}, goal noise max: {GOAL_ADVERSARY_NOISE_MAX}.")

    # evaluate before training
    reward_mean, reward_std, success_rate = evaluate_policy_with_success_rate(algo_ppo.policy, eval_env, EVALUATE_NUMS_IN_EVALUATION*env_num_used_in_eval)
    sb3_logger.info(f"Reward before RL: {reward_mean} +- {reward_std}")
    sb3_logger.info(f"Success rate before RL: {success_rate}")

    eval_callback = EvalCallback(
        eval_env_in_callback, 
        best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / RL_EXPERIMENT_NAME).absolute()),
        log_path=str((PROJECT_ROOT_DIR / "logs" / RL_EXPERIMENT_NAME).absolute()), 
        eval_freq=EVALUATE_FREQUENCE,
        n_eval_episodes=EVALUATE_NUMS_IN_CALLBACK*env_num_used_in_callback,
        deterministic=True, 
        render=False,
    )

    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path=str((PROJECT_ROOT_DIR / "checkpoints" / RL_EXPERIMENT_NAME).absolute()))
    event_callback = EveryNTimesteps(n_steps=SAVE_CHECKPOINT_EVERY_N_TIMESTEPS, callback=checkpoint_on_event)

    algo_ppo.learn(
        total_timesteps=RL_TRAIN_STEPS, 
        callback=[eval_callback, event_callback]
    )

    # evaluate after training
    reward_mean, reward_std, success_rate = evaluate_policy_with_success_rate(algo_ppo.policy, eval_env, EVALUATE_NUMS_IN_EVALUATION*env_num_used_in_eval)
    sb3_logger.info(f"Reward after RL: {reward_mean} +- {reward_std}")
    sb3_logger.info(f"Success rate after RL: {success_rate}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="传入配置文件")
    parser.add_argument("--config-file-name", type=str, help="配置文件名", default="ppo_bc_config_10hz_128_128_1.json")
    args = parser.parse_args()

    train_config = load_config(Path(os.getcwd()) / args.config_file_name)

    ENV_CONFIG_FILE = train_config["env"]["config_file"]
    ENV_CUSTOM_CONFIG = train_config["env"].get("custom_config", {})
    ANTI_GOAL_ADVERSARY_REFERENCE_ALGO_TYPE = train_config["env"].get("referenced_goal_adversary_type", "ppo")
    ANTI_GOAL_ADVERSARY_REFERENCE_ALGO_PATH = train_config["env"].get("referenced_goal_adversary_path", "")
    GOAL_ADVERSARY_NOISE_BASE = np.array(train_config["env"].get("goal_noise_base", [0.0100, 0.0167, 0.0083]))
    GOAL_ADVERSARY_NOISE_MULTIPLIER = train_config["env"].get("goal_noise_multiplier", 1.0)
    GOAL_ADVERSARY_NOISE_MIN = - GOAL_ADVERSARY_NOISE_BASE * GOAL_ADVERSARY_NOISE_MULTIPLIER
    GOAL_ADVERSARY_NOISE_MAX = GOAL_ADVERSARY_NOISE_BASE * GOAL_ADVERSARY_NOISE_MULTIPLIER

    SEED = train_config["rl"]["seed"]
    SEED_IN_TRAINING_ENV = train_config["rl"].get("seed_in_train_env")
    SEED_IN_CALLBACK_ENV = train_config["rl"].get("seed_in_callback_env")

    RL_EXPERIMENT_NAME = train_config["rl"]["experiment_name"]
    RL_INITIAL_POLICY = train_config["rl"]["policy_trained_last_iteration"]
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
    DEVICE = train_config["rl"].get("device", "cpu")

    train()
