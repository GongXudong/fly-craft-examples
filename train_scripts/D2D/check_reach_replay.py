import numpy as np
data = np.load('/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/VVC/relative_hard/b_05/sac_her_10hz_128_128_b_05_1e6steps_seed_1_singleRL/replay_buffer.pkl', allow_pickle=True)
count = 0
for reward  in data.rewards:
    if reward !=0.0:
        count +=1
print(count)
# from stable_baselines3 import HerReplayBuffer, SAC

# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.buffers import DictReplayBuffer
# import sys
# from pathlib import Path
# PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
# if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
#     sys.path.append(str(PROJECT_ROOT_DIR.absolute()))
# from utils_my.sb3.my_reach_reward_wrapper import PowerRewardWrapper

# if __name__ == '__main__':

#     vec_env = make_vec_env(
#         env_id="my-reach",
#         n_envs=8,
#         seed=1024,
#         vec_env_cls=SubprocVecEnv, 
#         wrapper_class=PowerRewardWrapper, 
#         wrapper_kwargs={"b": 0.5},
#         env_kwargs={
#             "reward_type": "dense",
#             "control_type": "joints",

#         }

#     )

#     USE_HER =True
#     sac_algo = SAC(
#         "MultiInputPolicy",
#         vec_env,
#         seed=2258,
#         replay_buffer_class=HerReplayBuffer if USE_HER else DictReplayBuffer,
#         replay_buffer_kwargs=dict(
#             n_sampled_goal=4,
#             goal_selection_strategy="future",
#         ) if USE_HER else None,
#         verbose=1,
#     )

#     sac_algo.load_replay_buffer('/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/distance_threshold_0_01/her/two_stage_relative_hard_b_2_b_05/sac_her_10hz_128_128_b_2_5e4steps_seed_1_singleRL/replay_buffer.pkl')

#     # sac_algo.replay_buffer.observations
#     loaded_replay_buffer_size = sac_algo.replay_buffer.size()
#     new_rewards = vec_env.env_method(
#         method_name="compute_reward",
#         indices=[0],
#         achieved_goal=sac_algo.replay_buffer.next_observations["achieved_goal"].squeeze()[:loaded_replay_buffer_size], 
#         desired_goal=sac_algo.replay_buffer.observations["desired_goal"].squeeze()[:loaded_replay_buffer_size],
#         info=sac_algo.replay_buffer.infos.squeeze()[:loaded_replay_buffer_size]
#     )[0]
#     new_rewards = new_rewards.reshape(-1, 1)  
#     sac_algo.replay_buffer.rewards[:loaded_replay_buffer_size] = new_rewards.reshape(-1, 1)
