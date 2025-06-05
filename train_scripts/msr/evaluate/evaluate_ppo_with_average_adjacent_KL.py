import sys
import os
from pathlib import Path
from copy import deepcopy
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import pandas as pd
import argparse
from typing import List

import torch as th
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MultiInputPolicy as PPOMultiInputPolicy
from stable_baselines3.common.distributions import Distribution, kl_divergence

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from train_scripts.msr.algorithms.smooth_goal_ppo import SmoothGoalPPO
from train_scripts.msr.attackers.ppo.gradient_ascent_attackers_ppo import GradientAscentAttacker
from train_scripts.msr.utils.evaluation import my_evaluate_with_customized_dg


def calc_KL(policy: PPOMultiInputPolicy, new_desired_goal: th.Tensor, obs_list: List[th.Tensor], action_dist_list: List[Distribution]):
    KL_list = []
    for obs, reference_action_distribution in zip(obs_list, action_dist_list):
        obs_tensor, _ = policy.obs_to_tensor(obs)
        # print(f"obs_tensor: {obs_tensor}")
        # print(f"obs_tensor dg: {obs_tensor['desired_goal']}")
        obs_tensor["desired_goal"] = new_desired_goal.reshape((1, -1))

        tmp_action_dist = policy.get_distribution(obs=obs_tensor)

        distance = th.distributions.kl_divergence(reference_action_distribution, tmp_action_dist.distribution).sum(axis=-1)
        KL_list.append(distance)
    
    return th.mean(th.concat(KL_list))


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
        "algo_reg_beta": [],
        "seed": [],
        # eval
        "evaluate_adjacent_num": [],
        # eval result
        "desired_goal": [],
        "noised_desired_goal": [],
        "noise_mu": [],
        "noise_chi": [],
        "goal_distance": [],
        "cumulative_reward": [],
        "noised_goal_cumulative_reward": [],
        "KL_value": [],
        "success": [],
    }

    # iterate over checkpoints
    for tmp_seed in tqdm(args.algo_seeds, total=len(args.algo_seeds)):
        tmp = args.algo_ckpt_dir
        tmp = tmp.format(tmp_seed)
        policy_dir = PROJECT_ROOT_DIR / tmp / args.algo_ckpt_model_name
        if args.algo_class == "PPO":
            policy_class = PPO
        elif args.algo_class == "SmoothGoalPPO":
            policy_class = SmoothGoalPPO
        else:
            raise ValueError("algo_class can only be PPO or SmoothGoalPPO!")
        
        algo = policy_class.load(
            policy_dir,
            env=env
        )

        attacker = GradientAscentAttacker(
            policy=algo.policy,
            env=env,
            epsilon=np.array(args.evaluate_noise_base) * args.evaluate_noise_multiplier,
        )

        for _ in tqdm(range(args.evaluate_dg_num), total=args.evaluate_dg_num):
            # 找到一个能完成的goal
            achieved, achievable_dg, cumulative_reward, obs_list, action_dist_list = attacker.get_an_achievable_desired_goal()
            
            if not achieved:
                continue
            
            original_achievable_dg = env.env.goal_scalar.inverse_transform(achievable_dg.reshape((1, -1))).reshape((-1))
            print(f"achievable dg: {achievable_dg}, original desired dg: {original_achievable_dg}")

            # 计算对这个goal能施加的最小和最大噪声
            attacker._calc_noise_min_max(desired_goal=original_achievable_dg)
            noise_min, noise_max = attacker.noise_min.cpu().numpy(), attacker.noise_max.cpu().numpy()
            print(f"min noise: {noise_min}, max noise: {noise_max}")

            tmp_mu_index = - args.evaluate_adjacent_num
            for tmp_mu in np.linspace(noise_min[1], noise_max[1], 2*args.evaluate_adjacent_num+1):
                
                tmp_chi_index = - args.evaluate_adjacent_num
                for tmp_chi in np.linspace(noise_min[2], noise_max[2], 2*args.evaluate_adjacent_num+1):
                    new_dg = th.tensor(
                        np.array([
                            achievable_dg[0], 
                            achievable_dg[1] + tmp_mu, 
                            achievable_dg[2] + tmp_chi
                        ]), 
                        requires_grad=False, 
                        device=attacker.device
                    )
                    # print(new_dg)

                    # 计算KL
                    tmp_KL = calc_KL(policy=algo.policy, new_desired_goal=new_dg, obs_list=obs_list, action_dist_list=action_dist_list)
                    # print(tmp_KL)
                    
                    tmp_success, _, noised_goal_cumulative_reward = my_evaluate_with_customized_dg(
                        policy=algo.policy, 
                        env=env, 
                        desired_goal=env.env.goal_scalar.inverse_transform(new_dg.cpu().numpy().reshape((1, -1))).reshape((-1))
                    )
                    
                    res_log["env"].append(args.env_flag_str)
                    res_log["algo"].append(args.algo_flag_str)
                    res_log["algo_epsilon"].append(args.algo_epsilon)
                    res_log["algo_reg"].append(args.algo_reg)
                    res_log["algo_reg_beta"].append(args.algo_reg_beta)
                    res_log["seed"].append(tmp_seed)
                    res_log["evaluate_adjacent_num"].append(args.evaluate_adjacent_num)
                    res_log["desired_goal"].append(achievable_dg)
                    res_log["noised_desired_goal"].append(new_dg.cpu().numpy())
                    res_log["noise_mu"].append(tmp_mu_index)
                    res_log["noise_chi"].append(tmp_chi_index)
                    res_log["goal_distance"].append(np.linalg.norm([tmp_mu, tmp_chi]))
                    res_log["cumulative_reward"].append(cumulative_reward)
                    res_log["noised_goal_cumulative_reward"].append(noised_goal_cumulative_reward)
                    res_log["KL_value"].append(tmp_KL.detach().item())
                    res_log["success"].append(tmp_success)

                    tmp_chi_index += 1

                tmp_mu_index += 1
    
    res_df = pd.DataFrame(res_log)
    
    save_path: Path = PROJECT_ROOT_DIR / args.res_file_save_name
    if not save_path.parent.exists():
        os.makedirs(save_path.parent)

    res_df.to_csv(save_path, index=False)

# evaluate PPO
# python train_scripts/disc/evaluate/evaluate_ppo_with_agerage_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class PPO --algo-ckpt-dir checkpoints/disc/medium/ppo/epsilon_0_1_reg_0/128_128_2e8steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str PPO --algo-epsilon 0.0 --algo-reg 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/ppo/results/ppo_epsilon_0_reg_0_N_16_noise_0_1.csv

# evaluate SmoothGoalPPO
# python train_scripts/disc/evaluate/evaluate_ppo_with_agerage_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/medium/ppo/epsilon_0_1_reg_0_001_N_16/128_128_2e8steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO --algo-epsilon 0.1 --algo-reg 0.001 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/ppo/results/ppo_epsilon_0_1_reg_0_001_N_16_noise_0_1.csv
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pass configurations")
    # environment
    parser.add_argument("--env-config", type=str, default="configs/env/env_config_for_sac.json", help="environment configuration file")
    parser.add_argument("--env-flag-str", type=str, default="Hard", help="log str for environment")
    # algorithm
    parser.add_argument("--algo-class", type=str, default="SAC", help="algorithm class, can be one of: SAC, SmoothGoalSAC")
    parser.add_argument("--algo-ckpt-dir", type=str, default="checkpoints/rl_single/sac_her_easy_10hz_128_128_1e6steps_loss_{0}_singleRL", help="algorithm checkpoint file")
    parser.add_argument("--algo-ckpt-model-name", type=str, default="best_model", help="algorithm checkpoint model name")
    parser.add_argument("--algo-seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5], help="algorithm random seeds")
    parser.add_argument("--algo-flag-str", type=str, default="HER", help="log str for algorithm")
    parser.add_argument("--algo-epsilon", type=float, default=0.1, help="the noise epsilon used when training models")
    parser.add_argument("--algo-reg", type=float, default=0.001, help="the regularization used when training models")
    parser.add_argument("--algo-reg-beta", type=float, default=0.0, help="the regularization beta used when calculating goal-regularization-loss")
    # evaluation
    parser.add_argument("--evaluate-dg-num", type=int, default=100, help="how many desired goals are planed to evaluate")
    parser.add_argument("--evaluate-noise-base", nargs="*", type=float, default=[10.0, 3.0, 3.0], help="base noise")
    parser.add_argument("--evaluate-noise-multiplier", type=float, default=1.0, help="noise multiplier")
    parser.add_argument("--evaluate-adjacent-num", type=int, default=10, help="")
    # save res file
    parser.add_argument("--res-file-save-name", type=str, default="train_scripts/disc/evaluate/results/res_log.csv", help="result file save name")


    args = parser.parse_args()

    evaluate(args)
