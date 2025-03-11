import gymnasium as gym
import numpy as np
from pathlib import Path
import os
import sys
import torch as th
import argparse

from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, EvalCallback

import flycraft
from flycraft.utils.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.reward_norm.algorithms.normalizers.vec_normalize_goal_conditioned_reward_scaling import VecNormalizeGoalConditionedRewardScaling
from utils_my.sb3.vec_env_helper import get_vec_env
from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate
from utils_my.sb3.my_vec_normalize_save_callback import MyVecNormalizeSaveCallback

import warnings
warnings.filterwarnings("ignore")  # 过滤Gymnasium的UserWarning

def train():

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / RL_EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    vec_env = get_vec_env(
        num_process=RL_TRAIN_PROCESS_NUM,
        seed=SEED_IN_TRAINING_ENV,
        config_file=str(PROJECT_ROOT_DIR / "configs" / "env" / ENV_CONFIG_FILE),
        custom_config={"debug_mode": True, "flag_str": "Train"}
    )

    if ENV_NORMALIZE_REWARD == "reward_scaling_cluster":
        vec_env = VecNormalizeGoalConditionedRewardScaling(
            vec_env,
            training=True,
            norm_obs=False,
            norm_reward=True,
            gamma=GAMMA,
        )
        # TODO: other reward scaling methods
    else:
        pass

    eval_env_in_callback = get_vec_env(
        num_process=RL_EVALUATE_PROCESS_NUM,
        seed=SEED_IN_CALLBACK_ENV,
        config_file=str(PROJECT_ROOT_DIR / "configs" / "env" / ENV_CONFIG_FILE),
        custom_config={"debug_mode": True, "flag_str": "Callback"}
    )

    print(f"action space: {vec_env.action_space}")

    # SAC hyperparams:
    sac_algo = SAC(
        "MultiInputPolicy",
        vec_env,
        seed=SEED,
        replay_buffer_class=HerReplayBuffer if USE_HER else DictReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ) if USE_HER else None,
        verbose=1,
        buffer_size=int(BUFFER_SIZE),
        learning_starts=int(LEARNING_STARTS),
        gradient_steps=int(GRADIENT_STEPS),
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        batch_size=int(BATCH_SIZE),
        policy_kwargs=dict(
            net_arch=NET_ARCH,
            activation_fn=th.nn.Tanh
        ),
    )

    sac_algo.set_logger(sb3_logger)

    # callback: evaluate, save best
    eval_callback = EvalCallback(
        eval_env_in_callback, 
        best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / RL_EXPERIMENT_NAME).absolute()),
        log_path=str((PROJECT_ROOT_DIR / "logs" / RL_EXPERIMENT_NAME).absolute()), 
        eval_freq=EVAL_FREQ,  # 多少次env.step()评估一次，此处设置为1000，因为VecEnv有72个并行环境，所以实际相当于72*1000次step，评估一次
        n_eval_episodes=N_EVAL_EPISODES,  # 每次评估使用多少条轨迹
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
        callback=checkpoint_on_event,
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

    sac_algo.learn(
        total_timesteps=int(RL_TRAIN_STEPS), 
        callback=[eval_callback, event_callback, save_vec_normalize_callback],
    )
    # sac_algo.save(str(PROJECT_ROOT_DIR / "checkpoints" / RL_EXPERIMENT_NAME))

def test_single_traj():
    # Load saved model
    # Because it needs access to `env.compute_reward()`
    # HER must be loaded with the env
    env = gym.make(
        id="FlyCraft-v0",
        config_file=str(PROJECT_ROOT_DIR / "configs" / "env" / ENV_CONFIG_FILE),
        custom_config=ENV_CUSTOM_CONFIG
    )

    env = ScaledActionWrapper(ScaledObservationWrapper(env))

    model = SAC.load(
        str(PROJECT_ROOT_DIR / "checkpoints" / RL_EXPERIMENT_NAME / "best_model"), 
        env=env
    )

    obs, info = env.reset()

    # Evaluate the agent
    episode_reward = 0
    for _ in range(400):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        if terminated or truncated or info.get("is_success", False):
            print("Reward:", episode_reward, "Success?", info.get("is_success", False))
            episode_reward = 0.0
            obs, info = env.reset()

def test_multi_traj():
    vec_env = get_vec_env(
        num_process=RL_EVALUATE_PROCESS_NUM,
        config_file=str(PROJECT_ROOT_DIR / "configs" / "env" / ENV_CONFIG_FILE),
        custom_config=ENV_CUSTOM_CONFIG
    )
    sac_algo = SAC.load(
        str(PROJECT_ROOT_DIR / "checkpoints" / RL_EXPERIMENT_NAME / "best_model"), 
        env=vec_env,
        custom_objects={
            "observation_space": vec_env.observation_space,
            "action_space": vec_env.action_space
        }
    )

    res = evaluate_policy_with_success_rate(sac_algo.policy, vec_env, 100)

    print(res)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="传入配置文件")
    parser.add_argument("--config-file-name", type=str, help="配置文件名", default="sac_config_10hz_128_128_1.json")
    args = parser.parse_args()

    train_config = load_config(Path(os.getcwd()) / args.config_file_name)

    ENV_CONFIG_FILE = train_config["env"]["config_file"]
    ENV_CUSTOM_CONFIG = train_config["env"].get("custom_config", {})
    ENV_NORMALIZE_REWARD = train_config["env"].get("normarlize_reward", "none")

    SEED = train_config["rl"].get("seed")
    SEED_IN_TRAINING_ENV = train_config["rl"].get("seed_in_train_env")
    SEED_IN_CALLBACK_ENV = train_config["rl"].get("seed_in_callback_env")

    RL_EXPERIMENT_NAME = train_config["rl"]["experiment_name"]
    NET_ARCH = train_config["rl"]["net_arch"]
    RL_TRAIN_STEPS = train_config["rl"]["train_steps"]
    GAMMA = train_config["rl"].get("gamma", 0.995)
    BUFFER_SIZE = train_config["rl"].get("buffer_size", 1e6)
    BATCH_SIZE = train_config["rl"].get("batch_size", 1024)
    LEARNING_STARTS = train_config["rl"].get("learning_starts", 10240)
    RL_TRAIN_PROCESS_NUM = train_config["rl"].get("rollout_process_num", 32)
    RL_EVALUATE_PROCESS_NUM = train_config["rl"].get("evaluate_process_num", 32)
    CALLBACK_PROCESS_NUM = train_config["rl"].get("callback_process_num", 32)
    GRADIENT_STEPS = train_config["rl"].get("gradient_steps", 2)
    LEARNING_RATE = train_config["rl"].get("learning_rate", 3e-4)

    USE_HER = train_config["rl"].get("use_her", True)

    EVAL_FREQ = train_config["rl"].get("eval_freq", 1000)
    N_EVAL_EPISODES = train_config["rl"].get("n_eval_episodes", CALLBACK_PROCESS_NUM*10)
    SAVE_CKPT_EVERY_N_TIMESTEPS = train_config["rl"].get("save_checkpoint_every_n_timesteps", 50000)
    

    train()
    # test_single_traj()
    # test_multi_traj()
