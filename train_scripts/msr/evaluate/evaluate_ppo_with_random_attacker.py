import sys
import os
from pathlib import Path
from copy import deepcopy
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import pandas as pd
import argparse

from stable_baselines3.ppo import PPO

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from train_scripts.msr.algorithms.smooth_goal_ppo import SmoothGoalPPO
from train_scripts.msr.attackers.ppo.random_attackers_ppo import RandomAttacker
from train_scripts.msr.utils.evaluation import my_evaluate_with_customized_dg


def evaluate(args):
    # prepare env
    env_id = "FlyCraft-v0"
    env = gym.make(
        env_id,
        config_file=PROJECT_ROOT_DIR / args.env_config,
    )
    env = ScaledActionWrapper(ScaledObservationWrapper(env))

    res_log = {
        # env
        "env": [],
        # algo
        "algo": [],
        "algo_epsilon": [],
        "algo_reg": [],
        "seed": [],
        # eval
        "random_noise_num": [],
        "noise_base": [],
        "noise_multiplier": [],
        # eval result
        "attack_method": [],
        "desired_goal": [],
        "noised_desired_goal": [],
        "goal_distance": [],
        "policy_discrepency": [],
        "cumulative_reward_dg": [],
        "is_noised_dg_success": [],
        "cumulative_reward_noised_dg": [],
    }

    # iterate over checkpoints
    for tmp_seed in tqdm(args.algo_seeds, total=len(args.algo_seeds)):
        tmp = args.algo_ckpt_dir
        tmp = tmp.format(tmp_seed)
        policy_dir = PROJECT_ROOT_DIR / tmp / args.algo_ckpt_model_name
        if args.algo_class == "PPO":
            algo_class = PPO
        elif args.algo_class == "SmoothGoalPPO":
            algo_class = SmoothGoalPPO
        else:
            raise ValueError("argo-class only support PPO or SmoothGoalPPO currently!")
        
        ppo_algo = algo_class.load(
            policy_dir,
            env=env
        )

        # prepare attacker
        randomAttacker = RandomAttacker(
            policy=ppo_algo.policy,
            env=env,
            epsilon=np.array(args.evaluate_noise_base) * args.evaluate_noise_multiplier
        )

        for _ in tqdm(range(args.evaluate_dg_num), total=args.evaluate_dg_num):
            flag, desired_goal, cumulative_reward, obs_list, action_dist_list = randomAttacker.get_an_achievable_desired_goal()
            
            if flag:  # 如果能找到一个能够到达的目标
                tmp_obs = deepcopy(obs_list[0])
                tmp_obs["desired_goal"] = desired_goal
                unscaled_desired_goal = randomAttacker.env.env.inverse_scale_state(tmp_obs)["desired_goal"]
                
                noised_goal, discrepency = randomAttacker.attack(
                    desired_goal=unscaled_desired_goal,
                    observation_history=obs_list,
                    action_distribution_list=action_dist_list,
                    random_noise_num=args.evaluate_random_noise_num,
                )

                # 测试policy在noised_dg上的表现
                is_noised_dg_success, _, cumulative_reward_noised_dg = my_evaluate_with_customized_dg(
                    policy=randomAttacker.policy,
                    env=env,
                    desired_goal=noised_goal
                )
                
                res_log["env"].append(args.env_flag_str)
                res_log["algo"].append(args.algo_flag_str)
                res_log["algo_epsilon"].append(args.algo_epsilon)
                res_log["algo_reg"].append(args.algo_reg)
                res_log["seed"].append(tmp_seed)
                res_log["random_noise_num"].append(args.evaluate_random_noise_num)
                res_log["noise_base"].append(np.array(args.evaluate_noise_base))
                res_log["noise_multiplier"].append(args.evaluate_noise_multiplier)
                res_log["attack_method"].append(args.attacker_flag_str)

                res_log["desired_goal"].append(unscaled_desired_goal)
                res_log["noised_desired_goal"].append(noised_goal)
                res_log["goal_distance"].append(np.linalg.norm(unscaled_desired_goal-noised_goal, ord=2))
                res_log["policy_discrepency"].append(discrepency)
                res_log["cumulative_reward_dg"].append(cumulative_reward)
                res_log["is_noised_dg_success"].append(is_noised_dg_success)
                res_log["cumulative_reward_noised_dg"].append(cumulative_reward_noised_dg)
    
    save_path: Path = PROJECT_ROOT_DIR / args.res_file_save_name
    if not save_path.parent.exists():
        os.makedirs(save_path.parent)
    
    res_log_df = pd.DataFrame(data=res_log)
    res_log_df.to_csv(save_path, index=False)


# python train_scripts/disc/evaluate_with_random_attacker.py --env-config configs/env/D2D/env_config_for_sac_hard_b_025.json --env-flag-str Hard-025 --algo-class PPO_BC --algo-ckpt-dir checkpoints/rl_single/D2D/hard_sac_her_b_025/sac_her_10hz_128_128_b_025_1e6steps_seed_{0}_singleRL --algo-ckpt-model-name best_model --algo-seeds 1 2 3 4 5 --algo-flag-str HER --evaluate-dg-num 20 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/res_log_hard_her_random_10_01.csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pass configurations")
    # environment
    parser.add_argument("--env-config", type=str, default="configs/env/env_config_for_sac.json", help="environment configuration file")
    parser.add_argument("--env-flag-str", type=str, default="Hard", help="log str for environment")
    # algorithm
    parser.add_argument("--algo-class", type=str, default="PPO", help="algorithm class: PPO, SmoothGoalPPO")
    parser.add_argument("--algo-ckpt-dir", type=str, default="checkpoints/rl_single/sac_her_easy_10hz_128_128_1e6steps_loss_{0}_singleRL", help="algorithm checkpoint file")
    parser.add_argument("--algo-ckpt-model-name", type=str, default="best_model", help="algorithm checkpoint model name")
    parser.add_argument("--algo-seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5], help="algorithm random seeds")
    parser.add_argument("--algo-flag-str", type=str, default="HER", help="log str for algorithm")
    parser.add_argument("--algo-epsilon", type=float, default=0.1, help="the noise epsilon used when training models")
    parser.add_argument("--algo-reg", type=float, default=0.001, help="the regularization used when training models")
    # evaluation
    parser.add_argument("--evaluate-dg-num", type=int, default=100, help="how many desired goals are planed to evaluate")
    parser.add_argument("--evaluate-random-noise-num", type=int, default=10, help="how many noise are sampled on a desired goal")
    parser.add_argument("--evaluate-noise-base", nargs="*", type=float, default=[10.0, 3.0, 3.0], help="base noise")
    parser.add_argument("--evaluate-noise-multiplier", type=float, default=1.0, help="noise multiplier")
    parser.add_argument("--attacker-flag-str", type=str, default="Random 10", help="log str for attacker")
    # save res file
    parser.add_argument("--res-file-save-name", type=str, default="train_scripts/disc/evaluate/results/res_log.csv", help="result file save name")


    args = parser.parse_args()

    evaluate(args)