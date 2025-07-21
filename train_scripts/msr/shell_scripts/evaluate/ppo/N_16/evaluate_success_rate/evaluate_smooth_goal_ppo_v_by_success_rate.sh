#!/bin/bash

# bash train_scripts/msr/shell_scripts/evaluate/ppo/N_16/evaluate_success_rate/evaluate_smooth_goal_ppo_v_by_success_rate.sh &> tmp_smooth_goal_ppo_v_res.txt

# v_reg = 10.0,  v_beta = 0.0
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_v/medium/v_reg_10_v_beta_0/128_128_seed_1.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 6327 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_v/medium/v_reg_10_v_beta_0/128_128_seed_2.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 6321237 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_v/medium/v_reg_10_v_beta_0/128_128_seed_3.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 24536 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_v/medium/v_reg_10_v_beta_0/128_128_seed_4.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 21346 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_v/medium/v_reg_10_v_beta_0/128_128_seed_5.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 3127628 --n-envs 32 --n-eval-episode 1000

# v_reg = 10.0,  v_beta = 0.0
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_v/medium/v_reg_100_v_beta_0/128_128_seed_1.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 43259 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_v/medium/v_reg_100_v_beta_0/128_128_seed_2.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 6580 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_v/medium/v_reg_100_v_beta_0/128_128_seed_3.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 34509 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_v/medium/v_reg_100_v_beta_0/128_128_seed_4.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 227622 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_v/medium/v_reg_100_v_beta_0/128_128_seed_5.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 467598211 --n-envs 32 --n-eval-episode 1000
