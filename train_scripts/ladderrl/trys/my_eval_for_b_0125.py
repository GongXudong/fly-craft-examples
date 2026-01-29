from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
import sys
from pathlib import Path
import os
current_dir = os.getcwd()
print(current_dir)
PROJECT_ROOT_DIR = Path(current_dir)#.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))
print(PROJECT_ROOT_DIR)
from train_scripts.ladderrl.utils.wrappers import PowerRewardWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate
import gymnasium as gym
import flycraft
import warnings
warnings.filterwarnings("ignore")  # 过滤Gymnasium的UserWarning
gym.register_envs(flycraft)
from train_scripts.ladderrl.utils.get_vec_env import get_vec_env
import numpy as np

def eval(env_config_in_evaluation,policy_list):
    eval_env =get_vec_env(
            **env_config_in_evaluation
        )

    average_reward = 0.0
    average_std = 0.0
    average_successrate = 0.0 
    success_rates = [] 
    for index, policy in enumerate(policy_list):
        sac = SAC.load(PROJECT_ROOT_DIR/policy)
        eval_reward, std_reward, eval_success_rate = evaluate_policy_with_success_rate(sac.policy, eval_env, 1000)
        print(f"seed{index+1}  ：",f"mean_reward={eval_reward:.2f} +/- {std_reward}")
        average_reward += eval_reward
        average_std += std_reward
        average_successrate +=eval_success_rate
        success_rates.append(eval_success_rate)
    std_success_rate = np.std(success_rates)
    print(f"eval policy from {policy_list[-1]}")
    return average_reward/len(policy_list),average_std/len(policy_list), average_successrate/len(policy_list),std_success_rate





if __name__ =='__main__':
    policy_list = [
        "checkpoints/D2D/VVC/medium/b_0125/1e6_weight_change_indicator/sac_10hz_128_128_b_0125_1e6steps_seed_1_singleRL/best_model.zip",
        "checkpoints/D2D/VVC/medium/b_0125/1e6_weight_change_indicator/sac_10hz_128_128_b_0125_1e6steps_seed_2_singleRL/best_model.zip",
        "checkpoints/D2D/VVC/medium/b_0125/1e6_weight_change_indicator/sac_10hz_128_128_b_0125_1e6steps_seed_3_singleRL/best_model.zip",
        "checkpoints/D2D/VVC/medium/b_0125/1e6_weight_change_indicator/sac_10hz_128_128_b_0125_1e6steps_seed_4_singleRL/best_model.zip",
        "checkpoints/D2D/VVC/medium/b_0125/1e6_weight_change_indicator/sac_10hz_128_128_b_0125_1e6steps_seed_5_singleRL/best_model.zip",
    
    ]
    env_config_in_evaluation = {
    "num_process": 8,
    "seed": 183,
    "config_file": str(PROJECT_ROOT_DIR / "configs" / "env" / "D2D/env_config_for_sac_medium_b_0125.json"),
    "custom_config": {"debug_mode": True, "flag_str": "Callback"}
    }
    reward,std_reward,success_rate,std_success_rate  = eval(env_config_in_evaluation,policy_list)
    print(f"mean_reward={reward:.2f} +/- {std_reward}", f"success_rate = {success_rate} +/- {std_success_rate}") 