import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


def fill_replay_buffer(replay_buffer: ReplayBuffer, observations: np.ndarray, actions: np.ndarray, next_observations: np.ndarray, rewards: np.ndarray, dones: np.ndarray, infos: np.ndarray, n_envs: int) -> None:
    """_summary_

    Args:
        replay_buffer (ReplayBuffer): replay buffer
        observations (np.ndarray): (num_transitions,obs_shape)
        actions (np.ndarray): (num_transitions,action_shape)
        next_observations (np.ndarray): (num_transitions,next_obs_shape)
        rewards (np.ndarray): (num_transitions,re)
        dones (np.ndarray): _description_
        infos (np.ndarray): _description_
        n_envs (int): _description_
    """
    for tmp_obs, tmp_next_obs, tmp_action, tmp_reward, tmp_done, tmp_info in zip(observations, next_observations, actions, rewards, dones, infos):
        for key in tmp_obs:
            tmp_obs[key] = tmp_obs[key].reshape(n_envs, tmp_obs[key].shape[-1])
        for key in tmp_next_obs:
            tmp_next_obs[key] = tmp_next_obs[key].reshape(n_envs ,tmp_next_obs[key].shape[-1])
        replay_buffer.add(obs=tmp_obs,next_obs=tmp_next_obs,action=tmp_action,reward=tmp_reward,done=tmp_done,infos=tmp_info)