import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC

PROJECT_ROOT_DIR = Path(__file__).absolute().parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_stat
from utils_my.sb3.vec_env_helper import get_vec_env

def work(algo: str, algo_ckpt: str, env_config: str, n_envs: int, eval_episode_num: int, seed: int, save_result: bool, save_result_file_name: str):
    # 1.prepare env
    env_config_dict_in_training = {
        "num_process": n_envs,
        "seed": seed,
        "config_file": str(PROJECT_ROOT_DIR / env_config),
        "custom_config": {"debug_mode": False, "flag_str": "Train"}
    }

    vec_env = get_vec_env(
        **env_config_dict_in_training
    )

    # 2.load algo
    if algo == "ppo":
        algo_class = PPO
    elif algo == "bc":
        algo_class = PPO
    elif algo == "sac":
        algo_class = SAC
    elif algo == "her":
        algo_class = SAC
    else:
        raise ValueError("algo must be one of: ppo, bc, sac, her!")
    
    loaded_algo = algo_class.load(PROJECT_ROOT_DIR / algo_ckpt)

    # 3.evaluate
    mean_reward, std_reward, success_rate, res_dict_arr = evaluate_policy_with_stat(loaded_algo, vec_env, n_eval_episodes=eval_episode_num, deterministic=True)

    # 4.save result
    res_log = {
        "dg_v": [],
        "dg_mu": [],
        "dg_chi": [],
        "is_success": [],
        "termination": [],
        "cumulative_reward": [],
        "episode_length": [],
    }

    for res_dict in res_dict_arr:
        res_log["dg_v"].append(res_dict["last_info"]["desired_goal"][0])
        res_log["dg_mu"].append(res_dict["last_info"]["desired_goal"][1])
        res_log["dg_chi"].append(res_dict["last_info"]["desired_goal"][2])
        res_log["is_success"].append(res_dict["last_info"]["is_success"])
        res_log["termination"].append(res_dict["last_info"]["termination"])
        res_log["cumulative_reward"].append(res_dict["cumulative_reward"])
        res_log["episode_length"].append(res_dict["episode_length"])

    res_df = pd.DataFrame(res_log)

    if save_result:
        res_df.to_csv(PROJECT_ROOT_DIR / save_result_file_name, index=False)

    return mean_reward, std_reward, success_rate, res_df

# python train_scripts/reward_norm/evaluations/evaluation.py --algo ppo --algo-ckpt checkpoints/IRPO/rl/10hz_128_128_2e8steps_easy_1/best_model.zip --env-config configs/env/D2D/env_config_for_sac_medium_b_05.json --n-envs 8 --eval-episode-num 100 --seed 42 --save-result --result-file-save-name train_scripts/reward_norm/evaluations/results/result.csv
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="传入配置文件")
    parser.add_argument("--algo", type=str, help="ppo, bc, sac, her", default="ppo")
    parser.add_argument("--algo-ckpt", type=str, help="algorithm checkpoint", default="checkpoints/IRPO/rl/10hz_128_128_2e8steps_easy_1/best_model.zip")
    parser.add_argument("--env-config", type=str, help="environment configuration", default="configs/env/D2D/env_config_for_sac_medium_b_05.json")
    parser.add_argument("--n-envs", type=int, help="number of environment used in evaluation", default=8)
    parser.add_argument("--eval-episode-num", type=int, help="number of episode to evaluate", default=100)
    parser.add_argument("--seed", type=int, help="seed used in evaluation", default=42)
    parser.add_argument("--save-result", action="store_true", help="whether to save result dict")
    parser.add_argument("--result-file-save-name", type=str, help="name of the file to save result dict", default="result.csv")
    args = parser.parse_args()

    mean_reward, std_reward, success_rate, res_df = work(
        algo=args.algo,
        algo_ckpt=args.algo_ckpt,
        env_config=args.env_config,
        n_envs=args.n_envs,
        eval_episode_num=args.eval_episode_num,
        seed=args.seed,
        save_result=args.save_result,
        save_result_file_name=args.result_file_save_name,
    )

    print(f"reward: {round(res_df['cumulative_reward'].mean(), 2)} +- {round(res_df['cumulative_reward'].std(), 2)}, length: {round(res_df['episode_length'].mean(), 2)} += {round(res_df['episode_length'].std(), 2)}")
