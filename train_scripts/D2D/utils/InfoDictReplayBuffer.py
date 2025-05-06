from typing import NamedTuple
import numpy as np
import torch as th
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, TensorDict
from typing import Any, Dict, List, Optional, Union
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecNormalize


class InfoDictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    next_observations: TensorDict
    dones: th.Tensor
    rewards: th.Tensor
    infos: np.ndarray


# class InfoDictReplayBufferSamples(DictReplayBufferSamples):
#     """
#     包含info数据的样本类，继承自DictReplayBufferSamples
#     """
#     def __init__(
#         self,
#         observations: Dict[str, th.Tensor],
#         actions: th.Tensor,
#         next_observations: Dict[str, th.Tensor],
#         dones: th.Tensor,
#         rewards: th.Tensor,
#         infos: np.ndarray,
#     ):
#         super().__init__(observations, actions, next_observations, dones, rewards)
#         self.infos = infos



class InfoDictReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        # 调用父类初始化
        super().__init__(buffer_size, observation_space, action_space, device, n_envs,)
        self.infos = np.array([[{} for _ in range(self.n_envs)] for _ in range(self.buffer_size)])
        # 初始化为空字典避免None
        # self.infos[:] = {}


    def add(  # type: ignore[override]
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference

        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = np.array(next_obs[key])

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        self.infos[self.pos] = np.array(infos)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0


    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env)
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)
        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        infos_ =self.infos[batch_inds, env_indices]

        
        # for index,item in enumerate(infos_):
        #     item.update({"observation": {key: value[index] for key,value in observations.items()}})
        #     item.update({"next_observations": {key: value[index] for key,value in next_observations.items()}})


        return InfoDictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
            infos = infos_

        )




    

    # def _get_samples(
    #     self, 
    #     batch_inds: np.ndarray, 
    #     env: Optional[VecNormalize] = None,
    # ) -> InfoDictReplayBufferSamples:
    #     # Sample randomly the env idx
    #     # Sample randomly the env idx
    #     env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

    #     # Normalize if needed and remove extra dimension (we are using only one env for now)
    #     obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env)
    #     next_obs_ = self._normalize_obs(
    #         {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
    #     )

    #     assert isinstance(obs_, dict)
    #     assert isinstance(next_obs_, dict)
    #     # Convert to torch tensor
    #     observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
    #     next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}
    #     # 提取对应的info数据
    #     selected_infos =self.infos[batch_inds, env_indices]

    #     # 返回包含info的样本对象
    #     return InfoDictReplayBufferSamples(
    #         observations=observations,
    #         actions=self.to_torch(self.actions[batch_inds, env_indices]),
    #         next_observations=next_observations,
    #         # Only use dones that are not due to timeouts
    #         # deactivated by default (timeouts is initialized as an array of False)
    #         dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
    #             -1, 1
    #         ),
    #         rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
    #         infos=selected_infos
    #     )


 
# class InfoDictReplayBuffer(DictReplayBuffer):
#     """
#     自定义的DictReplayBuffer，能够存储环境返回的info信息
#     """
#     def __init__(
#         self,
#         buffer_size: int,
#         observation_space: spaces.Dict,
#         action_space: spaces.Space,
#         device: Union[th.device, str] = "auto",
#         n_envs: int = 1,
#         optimize_memory_usage: bool = False,
#         handle_timeout_termination: bool = True,
#     ):
#         super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage)
#         self.infos = np.empty((buffer_size, n_envs), dtype=object)  # 存储 info 的数组
#         self.handle_timeout_termination = handle_timeout_termination

#     def add(
#         self,
#         obs: Dict[str, np.ndarray],
#         next_obs: Dict[str, np.ndarray],
#         action: np.ndarray,
#         reward: np.ndarray,
#         done: np.ndarray,
#         infos: List[Dict[str, Any]],
#     ) -> None:
#         """
#         添加数据到回放池
#         """
#         for key in self.observations.keys():
#             self.observations[key][self.pos] = np.array(obs[key])
#         for key in self.next_observations.keys():
#             self.next_observations[key][self.pos] = np.array(next_obs[key])
#         self.actions[self.pos] = np.array(action)
#         self.rewards[self.pos] = np.array(reward)
#         self.dones[self.pos] = np.array(done)
#         self.infos[self.pos] = infos  # 存储 info 数据
#         if self.handle_timeout_termination:
#             self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
#         self.pos += 1
#         if self.pos == self.buffer_size:
#             self.full = True
#             self.pos = 0

#     def _get_samples(
#         self, 
#         batch_inds: np.ndarray, 
#         env: Optional[VecNormalize] = None,
#     ) -> InfoDictReplayBufferSamples:
#         """
#         从回放池中采样数据
#         """
#         # Sample randomly the env idx
#         env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

#         # Normalize if needed and remove extra dimension (we are using only one env for now)
#         obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env)
#         next_obs_ = self._normalize_obs(
#             {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
#         )

#         # Convert to torch tensor
#         observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
#         next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

#         # Extract the corresponding info data
#         selected_infos = self.infos[batch_inds, env_indices]

#         # Return the sample object containing info
#         return InfoDictReplayBufferSamples(
#             observations=observations,
#             actions=self.to_torch(self.actions[batch_inds, env_indices]),
#             next_observations=next_observations,
#             dones=self.to_torch(self.dones[batch_inds, env_indices]).reshape(-1, 1),
#             rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
#             infos=selected_infos,
#         )





