import gymnasium as gym
import numpy as np
from pathlib import Path
import sys
import torch as th

from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger

import flycraft

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.vec_env_helper import get_vec_env
from utils_my.sb3.my_eval_callback import MyEvalCallback
from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate

def train():

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / RL_EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    vec_env = get_vec_env(
        num_process=RL_TRAIN_PROCESS_NUM,
        seed=1,
        config_file=str(PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_sac.json"),
        custom_config={"debug_mode": False}
    )

    eval_env_in_callback = get_vec_env(
        num_process=RL_EVALUATE_PROCESS_NUM,
        seed=5,
        config_file=str(PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_sac.json"),
        custom_config={"debug_mode": False}
    )

    # SAC hyperparams:
    sac_algo = SAC(
        "MultiInputPolicy",
        vec_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=1,
        buffer_size=int(1e6),
        learning_starts=RL_TRAIN_PROCESS_NUM*400,
        gradient_steps=2,
        learning_rate=3e-4,
        gamma=0.995,
        batch_size=256,
        policy_kwargs=dict(
            net_arch=[128, 128],
            activation_fn=th.nn.Tanh
        ),
    )

    sac_algo.set_logger(sb3_logger)

    # TODO: callback: evaluate, save best
    eval_callback = MyEvalCallback(
        eval_env_in_callback, 
        best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / RL_EXPERIMENT_NAME).absolute()),
        log_path=str((PROJECT_ROOT_DIR / "logs" / RL_EXPERIMENT_NAME).absolute()), 
        eval_freq=100,  # 多少次env.step()评估一次，此处设置为1000，因为VecEnv有72个并行环境，所以实际相当于72*1000次step，评估一次
        n_eval_episodes=RL_EVALUATE_PROCESS_NUM * 10,  # 每次评估使用多少条轨迹
        deterministic=True, 
        render=False,
    )

    sac_algo.learn(
        total_timesteps=int(1e6), 
        callback=eval_callback
    )
    sac_algo.save(str(PROJECT_ROOT_DIR / "checkpoints" / RL_EXPERIMENT_NAME))

def test_single_traj():
    # Load saved model
    # Because it needs access to `env.compute_reward()`
    # HER must be loaded with the env
    env = gym.make(
        id="FlyCraft-v0",
        config_file=str(PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_sac.json")
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
        num_process=4,
        config_file=str(PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_sac.json")
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

    RL_EXPERIMENT_NAME = "sac_her"
    RL_TRAIN_PROCESS_NUM = 4
    RL_EVALUATE_PROCESS_NUM = 4

    # train()
    # test_single_traj()
    test_multi_traj()
