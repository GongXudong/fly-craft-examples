import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import SACPolicy

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.msr.utils.reset_env_utils import reset_env_with_desired_goal


def my_evaluate_with_original_dg(policy: SACPolicy, env: gym.Env):
    policy.set_training_mode(False)
    obs, info = env.reset()
    desired_goal = obs["desired_goal"]
    cumulative_reward = 0.0

    while True:
        tmp_action, _ = policy.predict(observation=obs, deterministic=True)
        # tmp_action_dist = policy.actor.action_dist
        new_obs, reward, terminated, truncated, info = env.step(tmp_action)

        cumulative_reward += reward
        is_success = info["is_success"] if "is_success" in info else False

        if terminated or truncated:
            return is_success, desired_goal, cumulative_reward

        obs = new_obs

def my_evaluate_with_customized_dg(policy: SACPolicy, env: gym.Env, desired_goal: np.ndarray):
    policy.set_training_mode(False)

    # 注意：调用此函数后，需要手动将env.unwrapped.task.goal_sampler.use_fixed_goal恢复成原来的值！！！
    tmp_env_sample_fixed_goal = env.unwrapped.task.goal_sampler.use_fixed_goal
    obs, info = reset_env_with_desired_goal(env, desired_goal=desired_goal, validate_desired_goal_bound=True)
    env.unwrapped.task.goal_sampler.use_fixed_goal = tmp_env_sample_fixed_goal
    
    desired_goal = obs["desired_goal"]
    cumulative_reward = 0.0

    while True:
        tmp_action, _ = policy.predict(observation=obs, deterministic=True)
        # tmp_action_dist = policy.actor.action_dist
        new_obs, reward, terminated, truncated, info = env.step(tmp_action)

        cumulative_reward += reward
        is_success = info["is_success"] if "is_success" in info else False

        if terminated or truncated:
            return is_success, desired_goal, cumulative_reward

        obs = new_obs