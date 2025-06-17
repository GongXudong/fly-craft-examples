import gymnasium as gym
import numpy as np
from pathlib import Path
import logging
import torch as th
import argparse
from copy import deepcopy
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, EvalCallback
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan

import flycraft
from flycraft.utils_common.load_config import load_config
from flycraft.utils_common.dict_utils import update_nested_dict

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.reward_norm.algorithms.normalizers.vec_normalize_goal_conditioned_reward_scaling import VecNormalizeGoalConditionedRewardScaling
from train_scripts.reward_norm.algorithms.normalizers.vec_reward_minmax_normalizer import VecRewardMinMaxNormalize
from utils_my.sb3.vec_env_helper import get_vec_env
from utils_my.sb3.my_eval_callback import MyEvalCallback
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate
from utils_my.sb3.my_schedule import linear_schedule
from utils_my.sb3.my_vec_normalize_save_callback import MyVecNormalizeSaveCallback
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
        learning_rate=LEARNING_RATE,
    )

def train():

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / RL_EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    env_config_dict_in_training = {
        "num_process": ROLLOUT_PROCESS_NUM, 
        "seed": SEED,
        "config_file": str(PROJECT_ROOT_DIR / "configs" / "env" / ENV_CONFIG),
        "custom_config": {"debug_mode": True, "flag_str": "Train"}
    }

    env_num_used_in_eval = EVALUATE_PROCESS_NUM
    env_config_dict_in_eval = deepcopy(env_config_dict_in_training)
    update_nested_dict(env_config_dict_in_eval, {
        "num_process": env_num_used_in_eval,
        "custom_config": {"debug_mode": False, "flag_str": "Evaluate"}
    })

    env_num_used_in_callback = CALLBACK_PROCESS_NUM
    env_config_dict_in_callback = deepcopy(env_config_dict_in_training)
    update_nested_dict(env_config_dict_in_callback, {
        "num_process": env_num_used_in_callback,
        "custom_config": {"debug_mode": True, "flag_str": "Callback"}
    })

    vec_env = VecCheckNan(get_vec_env(
        **env_config_dict_in_training
    ))

    helper_env = ScaledActionWrapper(
        ScaledObservationWrapper(
            gym.make(
                "FlyCraft-v0", 
                config_file=str(PROJECT_ROOT_DIR / "configs" / "env" / ENV_CONFIG)
            )
        )
    )

    if USE_ENV_NORMARLIZER:
        normarlizer_type = ENV_NORMARLIZER.get("type", None)
        if normarlizer_type == "reward_scaling_cluster":
            vec_env = VecNormalizeGoalConditionedRewardScaling(
                venv=vec_env,
                training=True,
                norm_obs=False,
                norm_reward=True,
                gamma=GAMMA,
            )
            sb3_logger.log("Using VecNormalizeGoalConditionedRewardScaling.")
        elif normarlizer_type == "reward_scaling_minmax":
            vec_env = VecRewardMinMaxNormalize(
                venv=vec_env,
                helper_env=helper_env,
                radius=ENV_NORMARLIZER.get("dg_search_radius", 3.0),
                default_max_return=ENV_NORMARLIZER.get("default_max_return", -100.0),
            )
            sb3_logger.log("Using VecRewardMinMaxNormalize.")

            reference_file_list = ENV_NORMARLIZER.get("reference_file_list", [])
            
            if len(reference_file_list) > 0:
                for rf in reference_file_list:
                    tmp_file_dir = PROJECT_ROOT_DIR / rf
                    sb3_logger.log(f"load referenced evaluation results from {tmp_file_dir}")
                    vec_env.update_data_from_csv_file(tmp_file_dir)
                
                vec_env.fit_data()
        else:
            # TODO: other reward scaling methods
            pass
    else:
        sb3_logger.log("Not using RewardNormalizer.")

    # evaluate_policy使用的测试环境
    eval_env = VecCheckNan(get_vec_env(
        **env_config_dict_in_eval
    ))
    # 回调函数中使用的测试环境
    eval_env_in_callback = VecCheckNan(get_vec_env(
        **env_config_dict_in_callback
    ))

    algo_ppo = get_ppo_algo(vec_env)
    sb3_logger.info(str(algo_ppo.policy))

    # set sb3 logger
    algo_ppo.set_logger(sb3_logger)

    # sb3自带的EvalCallback根据最高平均reward保存最优策略；改成MyEvalCallback，根据最高胜率保存最优策略
    eval_callback = EvalCallback(
        eval_env_in_callback, 
        best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / RL_EXPERIMENT_NAME).absolute()),
        log_path=str((PROJECT_ROOT_DIR / "logs" / RL_EXPERIMENT_NAME).absolute()), 
        eval_freq=EVALUATE_FREQUENCE,
        n_eval_episodes=EVALUATE_NUMS_IN_CALLBACK*env_num_used_in_callback,
        deterministic=True, 
        render=False,
    )

    checkpoint_on_event = CheckpointCallback(
        save_freq=1, 
        save_path=str((PROJECT_ROOT_DIR / "checkpoints" / RL_EXPERIMENT_NAME).absolute()),
        save_vecnormalize=True,
    )
    event_callback = EveryNTimesteps(
        n_steps=SAVE_CKPT_EVERY_N_TIMESTEPS, 
        callback=checkpoint_on_event
    )

    save_vec_normalize_callback = EveryNTimesteps(
        n_steps=SAVE_CKPT_EVERY_N_TIMESTEPS,
        callback=MyVecNormalizeSaveCallback(
            save_freq=1,
            save_path=str((PROJECT_ROOT_DIR / "checkpoints" / RL_EXPERIMENT_NAME).absolute()),
            name_prefix="my",
            verbose=2,
        )
    )

    algo_ppo.learn(
        total_timesteps=RL_TRAIN_STEPS, 
        callback=[eval_callback, event_callback, save_vec_normalize_callback],
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

    ENV_CONFIG = train_config["env"].get("config_file", "env_config_for_sac.json")
    ENV_NORMARLIZER = train_config["env"].get("normarlizer", None)
    if ENV_NORMARLIZER is None:
        USE_ENV_NORMARLIZER = False
    else:
        USE_ENV_NORMARLIZER = ENV_NORMARLIZER.get("use", False)

    ENV_NORMALIZE_REWARD = train_config["env"].get("normarlize_reward", "none")

    RL_EXPERIMENT_NAME = train_config["rl"]["experiment_name"]
    SEED = train_config["rl"]["seed"]
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
    LEARNING_RATE = train_config["rl"].get("learning_rate", 3e-4)

    SAVE_CKPT_EVERY_N_TIMESTEPS = train_config["rl"].get("save_checkpoint_every_n_timesteps", 4000000)

    train()
