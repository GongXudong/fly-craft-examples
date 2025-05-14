from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
import sys
from pathlib import Path
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))
from train_scripts.ladderrl.utils.wrappers import PowerRewardWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
# from utils_my.env_utils.register_env import register_all_with_default_dense_params,register_nsubsteps_all_with_sparse_params,register_nsubsteps_all_with_dense_params
# register_all_with_default_dense_params() 
# register_nsubsteps_all_with_sparse_params()
# register_nsubsteps_all_with_dense_params()
if __name__ == '__main__':
    eval_env = make_vec_env(env_id= "my-reach-dense-5",
                            n_envs= 1,
                            wrapper_class=PowerRewardWrapper, 
                            wrapper_kwargs={"b": 1.0,"reward_type":"dense","distance_threshold":0.01},
                            vec_env_cls=SubprocVecEnv, 
                            env_kwargs={
                                "reward_type": "dense",
                                "control_type": "joints",
                                "distance_threshold":0.01})

    policy_list = [
        # "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_3/b_1/sac_her_10hz_128_128_b_1_1e6steps_seed_1_singleRL/best_model.zip",
        # "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_3/b_1/sac_her_10hz_128_128_b_1_1e6steps_seed_2_singleRL/best_model.zip",
        # "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_3/b_1/sac_her_10hz_128_128_b_1_1e6steps_seed_3_singleRL/best_model.zip",
        # "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_3/b_1/sac_her_10hz_128_128_b_1_1e6steps_seed_4_singleRL/best_model.zip",
        # "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_3/b_1/sac_her_10hz_128_128_b_1_1e6steps_seed_5_singleRL/best_model.zip"


        "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_5/b_1/sac_her_10hz_128_128_b_1_1e6steps_seed_1_singleRL/best_model.zip",
        "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_5/b_1/sac_her_10hz_128_128_b_1_1e6steps_seed_2_singleRL/best_model.zip",
        "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_5/b_1/sac_her_10hz_128_128_b_1_1e6steps_seed_3_singleRL/best_model.zip",
        "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_5/b_1/sac_her_10hz_128_128_b_1_1e6steps_seed_4_singleRL/best_model.zip",
        "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_5/b_1/sac_her_10hz_128_128_b_1_1e6steps_seed_5_singleRL/best_model.zip",
    ]
    average_reward = 0.0
    average_std = 0.0
    for index, policy in enumerate(policy_list):
        sac = SAC.load(policy)
        mean_reward, std_reward = evaluate_policy(sac.policy, eval_env, n_eval_episodes=1000, deterministic=True)
        print(f"seed{index+1}  ：",f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        average_reward += mean_reward
        average_std += std_reward
    print(f"mean_reward={average_reward/len(policy_list):.2f} +/- {average_std/len(policy_list)}")


