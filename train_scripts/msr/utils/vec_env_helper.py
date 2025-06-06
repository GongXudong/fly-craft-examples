import logging
import sys
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from gymnasium import Wrapper, ObservationWrapper, ActionWrapper, Env, spaces
from flycraft.env import FlyCraftEnv

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper

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
        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        if kwargs.get("frame_skip", 1) > 1:
            env = FrameSkipWrapper(env,skip=kwargs.get("frame_skip", 1))
        env.reset(seed=seed + rank)
        print(seed+rank, env.unwrapped.task.np_random, env.unwrapped.task.goal_sampler.np_random)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env(num_process: int=10, seed: int=0, **kwargs):
    return SubprocVecEnv([make_env(rank=i, seed=seed, **kwargs) for i in range(num_process)])


class FrameSkipWrapper(Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping).
    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
    The same action will be taken ``skip`` times.
    """

    def __init__(self, env: Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip
    
    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.
        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        terminated = truncated = False
        info_for_skip = []
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            info_for_skip.append({
            "obs": deepcopy(obs),
            "reward": reward
            })
            done = terminated or truncated
            total_reward += float(reward)
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        #info.update("frame_skip_info", info_for_skip)
        info.update({"frame_skip_info": info_for_skip})
        return obs, total_reward, terminated, truncated, info
