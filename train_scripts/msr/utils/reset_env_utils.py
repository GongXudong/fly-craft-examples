import numpy as np
import gymnasium as gym
from typing import Union

import flycraft


def get_lower_bound_of_desired_goal(env: gym.Env):
    return np.array([env.unwrapped.env_config["goal"]["v_min"], env.unwrapped.env_config["goal"]["mu_min"], env.unwrapped.env_config["goal"]["chi_min"]])

def get_upper_bound_of_desired_goal(env: gym.Env):
    return np.array([env.unwrapped.env_config["goal"]["v_max"], env.unwrapped.env_config["goal"]["mu_max"], env.unwrapped.env_config["goal"]["chi_max"]])

def get_validate_part_of_noised_desired_goal(env: gym.Env, desired_goal: np.ndarray, noise: Union[np.ndarray, None]=None):
    """保证noised_desired_goal在合法范围内
    """
    tmp_noise = np.zeros_like(desired_goal) if noise is None else noise
    return np.clip(
        a=desired_goal + tmp_noise, 
        a_min=get_lower_bound_of_desired_goal(env), 
        a_max=get_upper_bound_of_desired_goal(env)
    )

def sample_a_noised_desired_goal_by_random(env: gym.Env, desired_goal: np.ndarray, epsilon: np.ndarray):
    """在epsilon范围内采样噪声，加到desired_goal中，并返回环境定义的合法范围内的加噪的desired_goal

    Args:
        env (gym.Env): _description_
        desired_goal (np.ndarray): _description_
        epsilon (np.ndarray): 噪声最大值，在[-epsilon, epsilon]之间采样噪声

    Returns:
        _type_: _description_
    """
    tmp_noise = 2 * (np.random.random(env.observation_space["desired_goal"].shape[0]) - 0.5) * epsilon
    return get_validate_part_of_noised_desired_goal(env, desired_goal + tmp_noise)

def reset_env_with_desired_goal(env: gym.Env, desired_goal: np.ndarray, validate_desired_goal_bound: bool=True):
    """重置环境时，将环境的desired_goal设置为预定值。
    注意：调用此函数后，需要手动将env.unwrapped.task.goal_sampler.use_fixed_goal恢复成原来的值！！！
    """
    assert env.observation_space["desired_goal"].shape[0] == len(desired_goal), "The shape of desired_goal does not satisfy the observation_space!"
    
    if validate_desired_goal_bound:
        tmp_desired_goal = get_validate_part_of_noised_desired_goal(env, desired_goal)
    else:
        tmp_desired_goal = desired_goal
    
    env.unwrapped.task.goal_sampler.use_fixed_goal = True
    env.unwrapped.task.goal_sampler.goal_v = tmp_desired_goal[0]
    env.unwrapped.task.goal_sampler.goal_mu = tmp_desired_goal[1]
    env.unwrapped.task.goal_sampler.goal_chi = tmp_desired_goal[2]

    obs, info = env.reset()

    return obs, info
