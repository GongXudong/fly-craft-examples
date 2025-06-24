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
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate
import gymnasium as gym
import flycraft
import warnings
warnings.filterwarnings("ignore")  # 过滤Gymnasium的UserWarning
gym.register_envs(flycraft)
from train_scripts.ladderrl.utils.get_vec_env import get_vec_env

if __name__ == '__main__':


    env_config_in_evaluation = {
    "num_process": 32,
    "seed": 183,
    "config_file": str(PROJECT_ROOT_DIR / "configs" / "env" / "D2D/env_config_for_sac_medium_b_1.json"),
    "custom_config": {"debug_mode": True, "flag_str": "Callback"}
}

    eval_env =get_vec_env(
            **env_config_in_evaluation
        )

    policy_list = [
        "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_5/b_1/sac_her_10hz_128_128_b_1_1e6steps_seed_1_singleRL/best_model.zip",
        "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_5/b_1/sac_her_10hz_128_128_b_1_1e6steps_seed_2_singleRL/best_model.zip",
        "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_5/b_1/sac_her_10hz_128_128_b_1_1e6steps_seed_3_singleRL/best_model.zip",
        "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_5/b_1/sac_her_10hz_128_128_b_1_1e6steps_seed_4_singleRL/best_model.zip",
        "/home/sen/pythonprojects/fly-craft-examples/checkpoints/D2D/panda_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_5/b_1/sac_her_10hz_128_128_b_1_1e6steps_seed_5_singleRL/best_model.zip",
    ]
    average_reward = 0.0
    average_std = 0.0
    average_successrate = 0.0 
    for index, policy in enumerate(policy_list):
        sac = SAC.load(policy)
        eval_reward, std_reward, eval_success_rate = evaluate_policy_with_success_rate(sac.policy, eval_env, 1000)


        print(f"seed{index+1}  ：",f"mean_reward={eval_reward:.2f} +/- {std_reward}")
        average_reward += eval_reward
        average_std += std_reward
        average_successrate +=eval_success_rate
    print(f"mean_reward={average_reward/len(policy_list):.2f} +/- {average_std/len(policy_list)}")


