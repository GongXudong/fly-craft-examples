import logging
import sys
from pathlib import Path
import numpy as np
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from gymnasium import Wrapper, ObservationWrapper, ActionWrapper, Env, spaces
from flycraft.env import FlyCraftEnv
from typing import TypeVar, Dict, Union, List, SupportsFloat, Any
from copy import deepcopy

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.goal_adversary.utils.wrappers import FrameSkipWrapper
from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from train_scripts.goal_adversary.adversary_wrappers.goal_adversary_wrapper import GoalAdversaryWrapper
from train_scripts.goal_adversary.adversary_wrappers.goal_adversary_vec_wrapper import VecGoalAdversaryWrapper
from train_scripts.goal_adversary.adversary_wrappers.anti_goal_adversary_vec_wrapper import VecAntiGoalAdversaryWrapper

# --------------------------------begin: get_vec_env for FlyCraftEnv------------------------------

def make_env(rank: int, seed: int = 0, **kwargs):
    """
    Utility function for multiprocessed env.

    注意套wrapper的顺序，先套FrameSkipWrapper，再套ScaledActionWrapper和ScaledObservationWrapper

    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = FlyCraftEnv(
            config_file=kwargs["config_file"],
            custom_config=kwargs.get("custom_config", {})
        )
        
        frame_skip = kwargs.get("frame_skip", 1)
        if frame_skip > 1:
            print(f"frame_skip: {frame_skip}")
            env = FrameSkipWrapper(env, skip=frame_skip)

        env = ScaledActionWrapper(ScaledObservationWrapper(env))

        env.reset(seed=seed + rank)
        print(seed+rank, env.unwrapped.task.np_random, env.unwrapped.task.goal_sampler.np_random)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env(num_process: int=10, seed: int=0, **kwargs):
    return SubprocVecEnv([make_env(rank=i, seed=seed, **kwargs) for i in range(num_process)])

# --------------------------------env: get_vec_env for FlyCraftEnv--------------------------------


# --------------------------------begin: get_vec_env for GoalAdversaryWrapper---------------------

# 注意：每个被wrappered的env都包含一个policy，该版本计算量较大
def make_goal_adversary_env(
        algo_class: BaseAlgorithm,
        algo_path: Union[str, Path],
        goal_noise_min: np.ndarray,
        goal_noise_max: np.ndarray,
        rank: int,
        seed: int = 0,
        **kwargs
    ):
    def _init():
        env = FlyCraftEnv(
            config_file=kwargs["config_file"],
            custom_config=kwargs.get("custom_config", {})
        )
        
        frame_skip = kwargs.get("frame_skip", 1)
        if frame_skip > 1:
            print(f"frame_skip: {frame_skip}")
            env = FrameSkipWrapper(env, skip=frame_skip)

        env = ScaledActionWrapper(ScaledObservationWrapper(env))

        algo = algo_class.load(algo_path)

        env = GoalAdversaryWrapper(
            env=env,
            policy=algo.policy,
            noise_min=goal_noise_min,
            noise_max=goal_noise_max,
        )

        env.reset(seed=seed + rank)
        print(seed+rank, env.unwrapped.task.np_random, env.unwrapped.task.goal_sampler.np_random)
        return env
    set_random_seed(seed)
    return _init

def get_goal_adversary_vec_env(
        algo_class: BaseAlgorithm,
        algo_path: Union[str, Path],
        goal_noise_min: np.ndarray,
        goal_noise_max: np.ndarray,
        num_process: int=10, 
        seed: int=0, 
        **kwargs
    ):
    return SubprocVecEnv([make_goal_adversary_env(
        algo_class=algo_class,
        algo_path=algo_path,
        goal_noise_min=goal_noise_min,
        goal_noise_max=goal_noise_max,
        rank=i,
        seed=seed,
        **kwargs
        ) for i in range(num_process)])

# --------------------------------end: get_vec_env for GoalAdversaryWrapper-----------------------


# --------------------------------begin: efficient version of get_vec_env for GoalAdversaryWrapper---------------------
# 所有被wrappered的env共用一个policy，这个版本计算量较小
def get_goal_adversary_efficient_vec_env(
        algo_class: BaseAlgorithm,
        algo_path: Union[str, Path],
        goal_noise_min: np.ndarray,
        goal_noise_max: np.ndarray,
        num_process: int=10, 
        seed: int=0, 
        **kwargs
    ):
    """
    Get a vectorized environment for GoalAdversaryWrapper with efficient memory usage.
    """
    vec_env = SubprocVecEnv([make_env(rank=i, seed=seed, **kwargs) for i in range(num_process)])
    
    vec_env = VecGoalAdversaryWrapper(
        venv=vec_env,
        policy=algo_class.load(algo_path).policy,
        noise_min=goal_noise_min,
        noise_max=goal_noise_max,
        env_config=kwargs["config_file"],
    )
    
    return vec_env
# --------------------------------end: efficient version of get_vec_env for GoalAdversaryWrapper-----------------------

# --------------------------------begin: efficient version of get_vec_env for AntiGoalAdversaryWrapper---------------------
def get_anti_goal_adversary_efficient_vec_env(
        goal_adversary_algo_class: BaseAlgorithm,
        goal_adversary_algo_path: Union[str, Path],
        goal_noise_min: np.ndarray,
        goal_noise_max: np.ndarray,
        num_process: int=10, 
        seed: int=0, 
        **kwargs
    ):
    """
    Get a vectorized environment for AntiGoalAdversaryWrapper with efficient memory usage.
    """
    vec_env = SubprocVecEnv([make_env(rank=i, seed=seed, **kwargs) for i in range(num_process)])
    
    vec_env = VecAntiGoalAdversaryWrapper(
        venv=vec_env,
        goal_adversary_policy=goal_adversary_algo_class.load(goal_adversary_algo_path).policy,
        noise_min=goal_noise_min,
        noise_max=goal_noise_max,
        env_config=kwargs["config_file"],
    )
    
    return vec_env