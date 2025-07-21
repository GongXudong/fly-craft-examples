#!/bin/bash

# bash train_scripts/msr/shell_scripts/evaluate/ppo/N_16/evaluate_success_rate/evaluate_smooth_goal_ppo_pi_and_v_by_success_rate.sh &> tmp_smooth_goal_ppo_pi_and_v_res.txt

# epsilon = 0.1,  pi_reg = 0.001,  pi_beta = 0.0,  v_reg = 0.0,  v_beta = 0.0
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_v_beta_0/128_128_seed_1.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 543 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_v_beta_0/128_128_seed_2.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 3568 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_v_beta_0/128_128_seed_3.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 3445 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_v_beta_0/128_128_seed_4.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 4527 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_v_beta_0/128_128_seed_5.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 5211145 --n-envs 32 --n-eval-episode 1000


# epsilon = 0.1,  pi_reg = 0.001,  pi_beta = 0.0,  v_reg = 0.001,  v_beta = 0.0
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_001_v_beta_0/128_128_seed_1.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 442321 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_001_v_beta_0/128_128_seed_2.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 13457 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_001_v_beta_0/128_128_seed_3.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 54321253 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_001_v_beta_0/128_128_seed_4.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 54279 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_001_v_beta_0/128_128_seed_5.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 5689034 --n-envs 32 --n-eval-episode 1000


# epsilon = 0.1,  pi_reg = 0.001,  pi_beta = 0.0,  v_reg = 0.0001,  v_beta = 0.0
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_0001_v_beta_0/128_128_seed_1.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 3166 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_0001_v_beta_0/128_128_seed_2.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 34608 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_0001_v_beta_0/128_128_seed_3.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 34177 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_0001_v_beta_0/128_128_seed_4.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 11547 --n-envs 32 --n-eval-episode 1000
python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_0001_v_beta_0/128_128_seed_5.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 658543 --n-envs 32 --n-eval-episode 1000


# # epsilon = 0.1,  pi_reg = 0.001,  pi_beta = 0.0,  v_reg = 1.0,  v_beta = 0.0
# python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_1_v_beta_0/128_128_seed_1.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 34783 --n-envs 32 --n-eval-episode 1000
# python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_1_v_beta_0/128_128_seed_2.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 3431787 --n-envs 32 --n-eval-episode 1000
# python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_1_v_beta_0/128_128_seed_3.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 546248 --n-envs 32 --n-eval-episode 1000
# python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_1_v_beta_0/128_128_seed_4.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 3321517 --n-envs 32 --n-eval-episode 1000
# python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_1_v_beta_0/128_128_seed_5.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 1432320 --n-envs 32 --n-eval-episode 1000


# # epsilon = 0.1,  pi_reg = 0.001,  pi_beta = 0.0,  v_reg = 10.0,  v_beta = 0.0
# python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_10_v_beta_0/128_128_seed_1.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 11687 --n-envs 32 --n-eval-episode 1000
# python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_10_v_beta_0/128_128_seed_2.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 4258799 --n-envs 32 --n-eval-episode 1000
# python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_10_v_beta_0/128_128_seed_3.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 345634 --n-envs 32 --n-eval-episode 1000
# python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_10_v_beta_0/128_128_seed_4.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 1092 --n-envs 32 --n-eval-episode 1000
# python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_10_v_beta_0/128_128_seed_5.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 223400031 --n-envs 32 --n-eval-episode 1000


# # epsilon = 0.1,  pi_reg = 0.001,  pi_beta = 0.0,  v_reg = 100.0,  v_beta = 0.0
# python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_100_v_beta_0/128_128_seed_1.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 5630921 --n-envs 32 --n-eval-episode 1000
# python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_100_v_beta_0/128_128_seed_2.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 679411 --n-envs 32 --n-eval-episode 1000
# python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_100_v_beta_0/128_128_seed_3.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 80654 --n-envs 32 --n-eval-episode 1000
# python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_100_v_beta_0/128_128_seed_4.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 9059 --n-envs 32 --n-eval-episode 1000
# python train_scripts/msr/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_100_v_beta_0/128_128_seed_5.json --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --algo ppo --seed 42476 --n-envs 32 --n-eval-episode 1000






