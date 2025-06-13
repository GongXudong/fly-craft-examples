from copy import deepcopy
from gymnasium import Env, Wrapper
from typing import TypeVar, SupportsFloat, Any

# from gymnasium core.py
ObsType = TypeVar("ObsType")
WrapperObsType = TypeVar("WrapperObsType")
ActType = TypeVar("ActType")
WrapperActType = TypeVar("WrapperActType")


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
