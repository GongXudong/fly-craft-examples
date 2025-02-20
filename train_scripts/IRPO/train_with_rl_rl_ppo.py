import gymnasium as gym
import numpy as np
from pathlib import Path
import logging
from time import time
from copy import deepcopy
import argparse
import os
import sys
import torch as th

from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import EvalCallback

import flycraft
from flycraft.env import FlyCraftEnv
from flycraft.utils.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.models.ppo_with_bc_loss import PPOWithBCLoss
from utils_my.sb3.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
from utils_my.sb3.vec_env_helper import get_vec_env
from utils_my.sb3.my_schedule import linear_schedule
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate
from utils_my.sb3.my_eval_callback import MyEvalCallback


np.seterr(all="raise")  # 检查nan


def train():
    
    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / "IRPO" / "rl_rl" / RL_EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    env_config_dict_in_training = {
        "num_process": ROLLOUT_PROCESS_NUM,
        "seed": SEED,
        "config_file": str(PROJECT_ROOT_DIR / "configs" / "env" / train_config["env"].get("config_file", "env_config_for_sac.json")),
        "custom_config": {"debug_mode": False, "flag_str": "Train"}
    }
    
    env_num_used_in_eval = EVALUATE_PROCESS_NUM
    env_config_dict_in_eval = deepcopy(env_config_dict_in_training)
    env_config_dict_in_eval.update({
        "num_process": env_num_used_in_eval,
        "custom_config": {"debug_mode": False, "flag_str": "Evaluate"}
    })

    env_num_used_in_callback = CALLBACK_PROCESS_NUM
    env_config_dict_in_callback = deepcopy(env_config_dict_in_training)
    env_config_dict_in_callback.update({
        "num_process": env_num_used_in_callback,
        "custom_config": {"debug_mode": True, "flag_str": "Callback"}
    })

    # 训练使用的环境
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

    # load model
    # bc_policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "bc" / BC_EXPERIMENT_NAME
    rl_reference_model_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "IRPO" / RL_REFERENCE_MODEL_DIR / RL_REFERENCE_MODEL
    algo_ppo_for_kl_loss = PPOWithBCLoss.load(
        str((rl_reference_model_save_dir / RL_REFERENCE_MODEL_FILE_NAME).absolute()),
        custom_objects={
            "observation_space": vec_env.observation_space,
            "action_space": vec_env.action_space,
        }
    )
    algo_ppo_for_kl_loss.policy.set_training_mode(False)
    algo_ppo = PPOWithBCLoss.load(
        str((rl_reference_model_save_dir / RL_REFERENCE_MODEL_FILE_NAME).absolute()), 
        env=vec_env, 
        seed=SEED_FOR_LOAD_ALGO,
        custom_objects={
            "bc_trained_algo": algo_ppo_for_kl_loss,
            "learning_rate": linear_schedule(RL_LR_RATE),
            "observation_space": vec_env.observation_space,
            "action_space": vec_env.action_space,
            "kl_coef_with_bc": KL_WITH_PRETRAINED_MODEL_COEF,
        },
    )
    sb3_logger.info(str(algo_ppo.policy))

    # set sb3 logger
    algo_ppo.set_logger(sb3_logger)

    # evaluate
    reward, _, success_rate = evaluate_policy_with_success_rate(
        algo_ppo.policy, 
        eval_env, 
        EVALUATE_NUMS_IN_EVALUATION * env_num_used_in_eval
    )
    sb3_logger.info(f"Reward before RL: {reward}")
    sb3_logger.info(f"Success rate before RL: {success_rate}")

    # sb3自带的EvalCallback根据最高平均reward保存最优策略；改成MyEvalCallback，根据最高胜率保存最优策略
    eval_callback = EvalCallback(
        eval_env_in_callback, 
        best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / "IRPO" / "rl_rl" / RL_EXPERIMENT_NAME).absolute()),
        log_path=str((PROJECT_ROOT_DIR / "logs" / "IRPO" / "rl_rl" / RL_EXPERIMENT_NAME).absolute()), 
        eval_freq=EVALUATE_FREQUENCE,  # 多少次env.step()评估一次，此处设置为1000，因为VecEnv有72个并行环境，所以实际相当于72*1000次step，评估一次
        n_eval_episodes=EVALUATE_NUMS_IN_CALLBACK * env_num_used_in_callback,  # 每次评估使用多少条轨迹
        deterministic=True, 
        render=False,
    )

    algo_ppo.learn(total_timesteps=RL_TRAIN_STEPS, callback=eval_callback)

    # evaluate
    reward, _, success_rate = evaluate_policy_with_success_rate(
        algo_ppo.policy, 
        eval_env, 
        EVALUATE_NUMS_IN_EVALUATION * env_num_used_in_eval
    )

    sb3_logger.info(f"Reward after RL: {reward}")
    sb3_logger.info(f"Success rate after RL: {success_rate}")

    # save model
    # rl_policy_save_dir = Path(__file__).parent / "checkpoints_sb3" / "rl" / RL_EXPERIMENT_NAME
    # algo_ppo.save(str(rl_policy_save_dir / RL_POLICY_FILE_NAME))

if __name__ == "__main__":

    # python examples/train_with_rl_rl_ppo.py --config_file_name configs/train/ppo_fixed_target/ppo_rl_rl_config_10hz_128_128_target_100_-25_75.json

    parser = argparse.ArgumentParser(description="传入配置文件")
    parser.add_argument("--config-file-name", type=str, help="配置文件名", default="ppo_bc_config_10hz_128_128_1.json")
    args = parser.parse_args()

    train_config = load_config(Path(os.getcwd()) / args.config_file_name)

    BC_EXPERIMENT_NAME = train_config["bc"]["experiment_name"]
    BC_POLICY_FILE_NAME = train_config["bc"]["policy_file_save_name"]
    BC_POLICY_AFTER_VALUE_HEAD_TRAINED_FILE_NAME = train_config["bc"]["policy_after_value_head_trained_file_save_name"]

    RL_EXPERIMENT_NAME = train_config["rl_rl"]["experiment_name"]
    SEED = train_config["rl_rl"]["seed"]
    SEED_FOR_LOAD_ALGO = train_config["rl_rl"]["seed_for_load_algo"]
    NET_ARCH = train_config["rl_rl"]["net_arch"]
    RL_REFERENCE_MODEL = train_config["rl_rl"]["reference_model"]
    RL_REFERENCE_MODEL_DIR = train_config["rl_rl"].get("reference_model_dir", "rl_single")
    RL_REFERENCE_MODEL_FILE_NAME = "best_model"
    PPO_BATCH_SIZE = train_config["rl_rl"]["batch_size"]
    GAMMA = train_config["rl_rl"]["gamma"]
    ACTIVATE_VALUE_HEAD_TRAIN_STEPS = train_config["rl_rl"]["activate_value_head_train_steps"]
    RL_TRAIN_STEPS = train_config["rl_rl"]["train_steps"]
    RL_ENT_COEF = train_config["rl_rl"].get("ent_coef", 0.0)
    RL_LR_RATE = train_config["rl_rl"].get("lr", 3e-4)
    ROLLOUT_PROCESS_NUM = train_config["rl_rl"]["rollout_process_num"]
    EVALUATE_PROCESS_NUM = train_config["rl_rl"].get("evaluate_process_num", 32)
    CALLBACK_PROCESS_NUM = train_config["rl_rl"].get("callback_process_num", 32)
    EVALUATE_ON_ALL_TASKS = train_config["rl_rl"].get("evaluate_on_all_tasks", False)
    EVALUATE_FREQUENCE = train_config["rl_rl"].get("evaluate_frequence", 2048)
    EVALUATE_NUMS_IN_EVALUATION = train_config["rl_rl"].get("evaluate_nums_in_evaluation", 30)
    EVALUATE_NUMS_IN_CALLBACK = train_config["rl_rl"].get("evaluate_nums_in_callback", 3)
    KL_WITH_PRETRAINED_MODEL_COEF = train_config["rl_rl"]["kl_with_pretrained_model_coef"]

    train()
