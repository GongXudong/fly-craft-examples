from gymnasium import Wrapper, ObservationWrapper, ActionWrapper, Env, spaces
from sklearn.preprocessing import MinMaxScaler
from typing import TypeVar, Dict, Union, List, SupportsFloat, Any
import numpy as np
from pathlib import Path
import sys

from flycraft.env import FlyCraftEnv
from flycraft.tasks.velocity_vector_control_task import VelocityVectorControlTask
from flycraft.planes.f16_plane import F16Plane

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.scalar import get_min_max_scalar
from copy  import deepcopy

# from gymnasium core.py
ObsType = TypeVar("ObsType")
WrapperObsType = TypeVar("WrapperObsType")
ActType = TypeVar("ActType")
WrapperActType = TypeVar("WrapperActType")


class ScaledObservationWrapper(ObservationWrapper):
    
    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

        # 缩放与仿真器无关，只在学习器中使用
        # 送进策略网络的观测，各分量的取值都在[0, 1]之间
        
        plane_state_mins = VelocityVectorControlTask.get_state_lower_bounds()
        plane_state_maxs = VelocityVectorControlTask.get_state_higher_bounds()
        plane_goal_mins = VelocityVectorControlTask.get_goal_lower_bounds()
        plane_goal_maxs = VelocityVectorControlTask.get_goal_higher_bounds()
        
        self.observation_space = spaces.Dict(
            dict(
                observation = spaces.Box(low=0., high=1., shape=(len(plane_state_mins),)),  # phi, theta, psi, v, mu, chi, p, h
                desired_goal = spaces.Box(low=0., high=1., shape=(len(plane_goal_mins),)),
                achieved_goal = spaces.Box(low=0., high=1., shape=(len(plane_goal_mins),)),
            )
        )

        self.state_scalar: MinMaxScaler = get_min_max_scalar(
            mins=np.array(plane_state_mins),
            maxs=np.array(plane_state_maxs),
            feature_range=(0., 1.),
        )
        self.goal_scalar: MinMaxScaler = get_min_max_scalar(
            mins=np.array(plane_goal_mins),
            maxs=np.array(plane_goal_maxs),
            feature_range=(0., 1.)
        )
    
    def scale_state(self, state_var: Union[Dict, np.ndarray]) -> Union[Dict, np.ndarray]:
        """将仿真器返回的state缩放到[0, 1]之间。
        每一步的状态是字典类型，
        包括三个键：observation，desired_goal，achieved_goal，对应的值的类型都是np.ndarray。
        """
        if isinstance(state_var, dict):
            tmp_state_var = [state_var]
            # return self.state_scalar.transform(tmp_state_var).reshape((-1))
        elif len(state_var.shape) == 2:
            tmp_state_var = state_var
            # return self.state_scalar.transform(state_var)
        else:
            raise TypeError("state_var只能是1维或者2维！")
        
        res = [
            dict(
                observation = self.state_scalar.transform(tmp_state["observation"].reshape((1,-1))).reshape((-1)),
                desired_goal = self.goal_scalar.transform(tmp_state["desired_goal"].reshape((1,-1))).reshape((-1)),
                achieved_goal = self.goal_scalar.transform(tmp_state["achieved_goal"].reshape((1,-1))).reshape((-1)),
            )
            for tmp_state in tmp_state_var
        ]

        if isinstance(state_var, dict):
            return res[0]
        else:
            return np.array(res)

    def observation(self, observation: ObsType) -> WrapperObsType:
        return self.scale_state(observation)
    
    def inverse_scale_state(self, state_var: Union[Dict, np.ndarray]) -> Union[Dict, np.ndarray]:
        """将[0, 1]之间state变回仿真器定义的原始state。用于测试！！！
        """
        if isinstance(state_var, dict):
            tmp_state_var = [state_var]
            # return self.state_scalar.inverse_transform(tmp_state_var).reshape((-1))
        elif len(state_var.shape) == 2:
            tmp_state_var = state_var
            # return self.state_scalar.inverse_transform(state_var)
        else:
            raise TypeError("state_var只能是1维或者2维！")
        
        res = [
            dict(
                observation = self.state_scalar.inverse_transform(tmp_state["observation"].reshape((1,-1))).reshape((-1)),
                desired_goal = self.goal_scalar.inverse_transform(tmp_state["desired_goal"].reshape((1,-1))).reshape((-1)),
                achieved_goal = self.goal_scalar.inverse_transform(tmp_state["achieved_goal"].reshape((1,-1))).reshape((-1)),
            )
            for tmp_state in tmp_state_var
        ]

        if isinstance(state_var, dict):
            return res[0]
        else:
            return np.array(res)

class ScaledActionWrapper(ActionWrapper):

    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

        action_mins = F16Plane.get_action_lower_bounds(env.unwrapped.plane.control_mode)
        action_maxs = F16Plane.get_action_higher_bounds(env.unwrapped.plane.control_mode)

        self.action_space = spaces.Box(low=0., high=1., shape=(len(action_mins),))  # p, nz, pla

        # 策略网络输出的动作，各分量的取值都在[0, 1]之间
        self.action_scalar: MinMaxScaler = get_min_max_scalar(
            mins=np.array(action_mins),
            maxs=np.array(action_maxs),
            feature_range=(0., 1.)
        )
    
    def inverse_scale_action(self, action_var: np.ndarray) -> np.ndarray:
        """将学习器推理出的动作放大到仿真器接收的动作范围
        """
        if len(action_var.shape) == 1:
            tmp_action_var = action_var.reshape((1, -1))
            return self.action_scalar.inverse_transform(tmp_action_var).reshape((-1))
        elif len(action_var.shape) == 2:
            return self.action_scalar.inverse_transform(action_var)
        else:
            raise TypeError("action_var只能是1维或者2维！") 
    
    def action(self, action: WrapperActType) -> ActType:
        # 检查action类型
        if type(action) == np.ndarray:
            return self.inverse_scale_action(action)
        else:
            return self.inverse_scale_action(np.array(action))
    
    def scale_action(self, action_var: np.ndarray) -> np.ndarray:
        """将仿真器接收范围的action缩放到[0, 1]之间。用于测试！！！
        """
        if len(action_var.shape) == 1:
            tmp_action_var = action_var.reshape((1, -1))
            return self.action_scalar.transform(tmp_action_var).reshape((-1))
        elif len(action_var.shape) == 2:
            return self.action_scalar.transform(action_var)
        else:
            raise TypeError("action_var只能是1维或者2维！")
        
# class FrameSkipWrapper(Wrapper):
#     """
#     Return only every ``skip``-th frame (frameskipping).

#     :param env: Environment to wrap
#     :param skip: Number of ``skip``-th frame
#         The same action will be taken ``skip`` times.
#     """

#     def __init__(self, env: Env, skip: int = 4) -> None:
#         super().__init__(env)
        
#         self._skip = skip

#     def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
#         """
#         Step the environment with the given action
#         Repeat action, sum reward, and max over last observations.

#         :param action: the action
#         :return: observation, reward, terminated, truncated, information
#         """
#         total_reward = 0.0
#         terminated = truncated = False
#         infos ={"next_observations":[],"rewards":[]}
#         for i in range(self._skip):
#             obs, reward, terminated, truncated, info = self.env.step(action)
#             infos["next_observations"].append(obs)
#             infos["rewards"].append(reward)
#             done = terminated or truncated
#             total_reward += float(reward)
#             if done:
#                 break
#         # Note that the observation on the done=True frame
#         # doesn't matter
#         return obs, total_reward, terminated, truncated, infos

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