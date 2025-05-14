import pickle
import numpy as np
from stable_baselines3 import HerReplayBuffer
import math

def cut_buffer(buffer_path, cut_number):
    # 加载原始缓冲区
    with open(buffer_path, "rb") as f:
        original_buffer = pickle.load(f)

    N = cut_number  # 每N条保留1条

    indices = np.arange(original_buffer.buffer_size)
    # 每隔N个取一个索引
    filtered_indices = indices[::N]

    # 提取过滤后的数据
    actions = original_buffer.actions[filtered_indices]
    dones = original_buffer.dones[filtered_indices]
    ep_length = original_buffer.ep_length[filtered_indices]
    
    ep_length = (ep_length+ N -1) // N
    ep_start = original_buffer.ep_start[filtered_indices]
    infos = original_buffer.infos[filtered_indices]


    obs = {key: np.zeros_like(original_buffer.observations[key]) for key in original_buffer.observations.keys()}
    next_obs = {key: np.zeros_like(original_buffer.next_observations[key]) for key in original_buffer.next_observations.keys()}
    #next_obs = {}
    for key in original_buffer.next_observations.keys():
        next_obs[key][:len(filtered_indices)] = original_buffer.next_observations[key][filtered_indices]
    
        
    #obs = {}
    for key in original_buffer.observations.keys():
        obs[key][:len(filtered_indices)] = original_buffer.observations[key][filtered_indices]

    rewards = original_buffer.rewards[filtered_indices]
    timeouts = original_buffer.timeouts[filtered_indices]

    # 创建新缓冲区，调整buffer_size为过滤后的数据量或保持原大小
    new_buffer = HerReplayBuffer(
        env = None,
        buffer_size=original_buffer.buffer_size,
        observation_space=original_buffer.observation_space,
        action_space=original_buffer.action_space,
        device=original_buffer.device,
        goal_selection_strategy="future",
        copy_info_dict=original_buffer.copy_info_dict,
        # 传递其他必要参数，例如n_sampled_goal等
    )
    new_buffer.buffer_size = original_buffer.buffer_size
    new_buffer.observation_space = original_buffer.observation_space
    new_buffer.action_space = original_buffer.action_space
    new_buffer.device = original_buffer.device
    new_buffer.goal_selection_strategy = original_buffer.goal_selection_strategy
    new_buffer.copy_info_dict = original_buffer.copy_info_dict
    new_buffer.her_ratio = original_buffer.her_ratio
    new_buffer.n_sampled_goal = original_buffer.n_sampled_goal
    new_buffer.n_envs = original_buffer.n_envs

    # new_buffer.observations = {key: np.zeros_like(original_buffer.observations[key]) for key in original_buffer.observations.keys()}
    # new_buffer.next_observations = {key: np.zeros_like(original_buffer.next_observations[key]) for key in original_buffer.next_observations.keys()}

    # # 填充数据到新缓冲区
    # for key in new_buffer.observations:
    #     new_buffer.observations[key][:len(filtered_indices)] =obs[key][:len(filtered_indices)]

    # for key in new_buffer.next_observations:
    #     new_buffer.next_observations[key][:len(filtered_indices)] = next_obs[key][:len(filtered_indices)]
    new_buffer.observations = obs
    new_buffer.actions[:len(filtered_indices)] = actions
    new_buffer.rewards[:len(filtered_indices)] = rewards
    new_buffer.next_observations = next_obs
    new_buffer.dones[:len(filtered_indices)] = dones
    new_buffer.ep_length[:len(filtered_indices)] = ep_length
    new_buffer.ep_start[:len(filtered_indices)] = ep_start
    new_buffer.infos[:len(filtered_indices)]= infos
    new_buffer.timeouts[:len(filtered_indices)] = timeouts
    # new_buffer.size = len(filtered_indices)
    new_buffer.pos = len(filtered_indices)
    new_buffer.full = False
    # new_buffer.pos = len(filtered_indices) % new_buffer.buffer_size  # 循环缓冲区处理

    # # 保存处理后的缓冲区
    save_path  = "/".join(buffer_path.split("/")[:-1])
    save_path = save_path + "/"+ f"cut_{N}_"+"cut_buffer.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(new_buffer, f)


if __name__=="__main__":
    path_list = [ 
        "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/VVC/relative_hard/b_05/sac_her_10hz_128_128_b_05_1e6steps_seed_1_singleRL/replay_buffer.pkl",
        "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/VVC/relative_hard/b_05/sac_her_10hz_128_128_b_05_1e6steps_seed_2_singleRL/replay_buffer.pkl",
        "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/VVC/relative_hard/b_05/sac_her_medium2_10hz_128_128_b_05_1e6steps_seed_3_singleRL/replay_buffer.pkl",
        "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/VVC/relative_hard/b_05/sac_her_medium2_10hz_128_128_b_05_1e6steps_seed_4_singleRL/replay_buffer.pkl",
        "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/VVC/relative_hard/b_05/sac_her_medium2_10hz_128_128_b_05_1e6steps_seed_5_singleRL/replay_buffer.pkl"
    ]
    for buffer_path in path_list:
        N = 3
        cut_buffer(buffer_path,N)
    print("complete!")