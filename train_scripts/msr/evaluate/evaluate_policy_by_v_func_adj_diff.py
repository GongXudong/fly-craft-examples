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
from torch.nn import functional as F
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MultiInputPolicy as PPOMultiInputPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution, kl_divergence
from stable_baselines3.common.type_aliases import PyTorchObs

import flycraft

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from utils_my.sb3.vec_env_helper import get_vec_env
from train_scripts.msr.algorithms.smooth_goal_ppo import SmoothGoalPPO
from train_scripts.msr.algorithms.smooth_goal_sac import SmoothGoalSAC
from train_scripts.msr.attackers.ppo.gradient_ascent_attackers_ppo import GradientAscentAttacker
from train_scripts.msr.utils.evaluation import my_evaluate_with_customized_dg

gym.register_envs(flycraft)


def get_v(policy: ActorCriticPolicy, obs: PyTorchObs) -> np.ndarray:
    return policy.predict_values(obs)

def evaluate(args):
    # prepare env
    env_id = "FlyCraft-v0"
    helper_env = gym.make(
        env_id,
        config_file=PROJECT_ROOT_DIR / args.env_config_file,
    )
    helper_env = ScaledActionWrapper(ScaledObservationWrapper(helper_env))

    vec_env = get_vec_env(
        num_process=args.n_envs,
        seed=args.eval_seed,
        config_file=PROJECT_ROOT_DIR /  args.env_config_file,
    )

    if args.algo_class == "ppo":
        algo_class = PPO
    
    tmp_res_dfs = []
    tmp_res_val = []

    # iterate over checkpoints
    for tmp_seed in tqdm(args.algo_seeds, total=len(args.algo_seeds)):

        algo_ckpt_dir = args.algo_ckpt_dir
        algo_ckpt_dir = algo_ckpt_dir.format(tmp_seed)

        trained_algo = algo_class.load(
            path=PROJECT_ROOT_DIR / algo_ckpt_dir / args.algo_ckpt_model_name
        )

        helper_algo = SmoothGoalPPO(
            policy=PPOMultiInputPolicy,
            env=vec_env,
            goal_noise_epsilon=np.array(args.evaluate_noise_base) * args.evaluate_noise_multiplier
        )

        helper_algo.init_desired_goal_params(helper_env)

        obss_list, noised_obss_list, value_list, noised_value_list = [], [], [], []

        for i in tqdm(range(int(args.n_eval_episodes / args.n_envs))):
            obss = vec_env.reset()
            obss_th, _ = trained_algo.policy.obs_to_tensor(obss)
            values = get_v(trained_algo.policy, obss_th)

            obss_list.extend(obss["desired_goal"])
            value_list.extend(values.squeeze().cpu().detach().numpy())
            # print(f"original obs: {obss['desired_goal']} \n original values: {values}")

            noised_obss = deepcopy(obss)
            noised_obss_th, _ = trained_algo.policy.obs_to_tensor(noised_obss)
            helper_algo.add_noise_to_desired_goals(noised_obss_th)
            noised_obss_list.extend(noised_obss_th["desired_goal"].cpu().detach().numpy())
            # print(f"noised obs: {noised_obss_th['desired_goal']}")

            noised_obs_values = get_v(trained_algo.policy, noised_obss_th)

            noised_value_list.extend(noised_obs_values.squeeze().cpu().detach().numpy())
            # print(f"noised obs values: {noised_obs_values}")

        res_df = pd.DataFrame(data={
            "env": [args.env_flag_str] * len(obss_list),
            "algo": [args.algo_flag_str] * len(obss_list),
            "seed": [tmp_seed] * len(obss_list),
            "original_obs": obss_list,
            "noised_obs": noised_obss_list,
            "original_obs_value": value_list,
            "noised_obs_value": noised_value_list,
        })

        tmp_res_val.append(np.average(np.power(res_df["original_obs_value"] - res_df["noised_obs_value"], 2)))
        tmp_res_dfs.append(res_df)
        print(f"seed: {tmp_seed}: {tmp_res_val[-1]}")

    total_res_df = pd.concat(tmp_res_dfs)

    save_path: Path = PROJECT_ROOT_DIR / args.res_file_save_name
    if not save_path.parent.exists():
        os.makedirs(save_path.parent)

    total_res_df.to_csv(save_path, index=False)

    tmp_res_val = np.array(tmp_res_val)
    print(f"{tmp_res_val.mean()} += {tmp_res_val.std()}")


# python train_scripts/msr/evaluate/evaluate_policy_by_v_func_adj_diff.py --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --env-flag-str Hard-05 --algo-class ppo --algo-ckpt-dir checkpoints/msr/medium/smooth_goal_ppo_v/epsilon_0_1_reg_0_001_beta_0_N_16_v_reg_10_v_beta_0/128_128_2e8steps_seed_{0} --algo-ckpt-model-name best_model --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO_V --n-envs 8 --n-eval-episodes 800 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1 --eval-seed 1234 --res-file-save-name train_scripts/msr/plots/smooth_goal_ppo/results/smooth_goal_ppo_v__v_reg_10_v_beta_0.csv
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pass configurations")
    
    # environment
    parser.add_argument("--env-config-file", type=str, help="the configuration of environment used in evaluation", default="configs/env/env_hard_config_for_sac.json")
    parser.add_argument("--env-flag-str", type=str, default="Hard", help="log str for environment")
    # algorithm
    parser.add_argument("--algo-class", type=str, help="the algorithm model to be evaluated, can be sac, or ppo", default="sac")
    parser.add_argument("--algo-ckpt-dir", type=str, help="train configuration file", default="checkpoints/IRPO/bc/guidance_law_mode/iter_1/128_128_300epochs_{0}")
    parser.add_argument("--algo-ckpt-model-name", type=str, default="best_model", help="algorithm checkpoint model name")
    parser.add_argument("--algo-seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5], help="algorithm random seeds")
    parser.add_argument("--algo-flag-str", type=str, default="HER", help="log str for algorithm")
    # evaluation
    parser.add_argument("--n-envs", type=int, default=8, help="the number of environments used in this evaluation")
    parser.add_argument("--n-eval-episodes", type=int, default=100, help="the number of episodes used in this evaluation")
    parser.add_argument("--evaluate-noise-base", nargs="*", type=float, default=[10.0, 3.0, 3.0], help="base noise")
    parser.add_argument("--evaluate-noise-multiplier", type=float, default=1.0, help="noise multiplier")
    parser.add_argument("--eval-seed", type=int, help="seed used in evaluation", default=1)
    # save res file
    parser.add_argument("--res-file-save-name", type=str, default="train_scripts/disc/evaluate/results/res_log.csv", help="result file save name")
    
    args = parser.parse_args()

    evaluate(args)
