import logging
import sys
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from flycraft.env import FlyCraftEnv

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper, FrameSkipWrapper

def make_env(rank: int, seed: int = 0, **kwargs):
    """
    Utility function for multiprocessed env.

    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = FlyCraftEnv(
            config_file=kwargs["config_file"],
            custom_config=kwargs.get("custom_config", {})
        )

        if "frame_skip" in kwargs and kwargs["frame_skip"]:
            print("Check wrapper: use frame_skip!!!!!")
            env = FrameSkipWrapper(
                env=env,
                skip=kwargs.get("skip", 1)
            )
        env = ScaledActionWrapper(ScaledObservationWrapper(env))

        env.reset(seed=seed + rank)
        print(seed+rank, env.unwrapped.task.np_random, env.unwrapped.task.goal_sampler.np_random)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env(num_process: int=10, seed: int=0, **kwargs):
    """_summary_

    Args:
        num_process (int, optional): _description_. Defaults to 10.
        seed (int, optional): _description_. Defaults to 0.
        kwargs: 
            1. env config: pass two params: config_file=xxx/xxx, custom_config={...};
            2. Wrappers: 
                2.1 If planning to use FrameSkip wrapper, pass these two params: frame_skip=True, skip=4;
    Returns:
        _type_: _description_
    """
    return SubprocVecEnv([make_env(rank=i, seed=seed, **kwargs) for i in range(num_process)])
