#!/bin/bash

# ----------------------------------------------------------- smooth_goal_ppo_v -------------------------------------------------------------
# v_reg = 10, v_beta = 0
# eval noise = 1
python train_scripts/msr/evaluate/evaluate_policy_by_v_func_adj_diff.py --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --env-flag-str Hard-05 --algo-class ppo --algo-ckpt-dir checkpoints/msr/medium/smooth_goal_ppo_v/epsilon_0_1_reg_0_001_beta_0_N_16_v_reg_10_v_beta_0/128_128_2e8steps_seed_{0} --algo-ckpt-model-name best_model --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO_V --n-envs 8 --n-eval-episodes 10000 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1 --eval-seed 1234 --res-file-save-name train_scripts/msr/plots/smooth_goal_ppo/results/smooth_goal_ppo_v__v_reg_10_v_beta_0_eval_noise_1.csv


# ----------------------------------------------------------- smooth_goal_ppo_pi -------------------------------------------------------------
# epsilon = 0.1, reg = 0.001, beta = 0
# eval noise = 0.1
python train_scripts/msr/evaluate/evaluate_policy_by_v_func_adj_diff.py --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --env-flag-str Hard-05 --algo-class ppo --algo-ckpt-dir checkpoints/msr/medium/smooth_goal_ppo_pi/epsilon_0_1_reg_0_001_N_16/128_128_2e8steps_seed_{0} --algo-ckpt-model-name best_model --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO_V --n-envs 8 --n-eval-episodes 10000 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --eval-seed 4312 --res-file-save-name train_scripts/msr/plots/smooth_goal_ppo/results/smooth_goal_ppo_pi__epsilon_0_1_reg_0_001_beta_0_eval_noise_0_1.csv
# eval noise = 1
python train_scripts/msr/evaluate/evaluate_policy_by_v_func_adj_diff.py --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --env-flag-str Hard-05 --algo-class ppo --algo-ckpt-dir checkpoints/msr/medium/smooth_goal_ppo_pi/epsilon_0_1_reg_0_001_N_16/128_128_2e8steps_seed_{0} --algo-ckpt-model-name best_model --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO_V --n-envs 8 --n-eval-episodes 10000 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1 --eval-seed 4312 --res-file-save-name train_scripts/msr/plots/smooth_goal_ppo/results/smooth_goal_ppo_pi__epsilon_0_1_reg_0_001_beta_0_eval_noise_1.csv



# ----------------------------------------------------------- smooth_goal_ppo_pi_and_v -------------------------------------------------------------
# epsilon = 0.1, reg = 0.001, beta = 0, v_reg = 100, v_beta = 0
# eval noise = 0.1
python train_scripts/msr/evaluate/evaluate_policy_by_v_func_adj_diff.py --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --env-flag-str Hard-05 --algo-class ppo --algo-ckpt-dir checkpoints/msr/medium/smooth_goal_ppo_pi_and_v/epsilon_0_1_reg_0_001_beta_0_N_16_v_reg_100_v_beta_0/128_128_2e8steps_seed_{0} --algo-ckpt-model-name best_model --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO_V --n-envs 8 --n-eval-episodes 10000 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --eval-seed 346 --res-file-save-name train_scripts/msr/plots/smooth_goal_ppo/results/smooth_goal_ppo_pi_and_v__epsilon_0_1_reg_0_001_beta_0_v_reg_100_v_beta_0_eval_noise_0_1.csv
# eval noise = 1
python train_scripts/msr/evaluate/evaluate_policy_by_v_func_adj_diff.py --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --env-flag-str Hard-05 --algo-class ppo --algo-ckpt-dir checkpoints/msr/medium/smooth_goal_ppo_pi_and_v/epsilon_0_1_reg_0_001_beta_0_N_16_v_reg_100_v_beta_0/128_128_2e8steps_seed_{0} --algo-ckpt-model-name best_model --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO_V --n-envs 8 --n-eval-episodes 10000 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1 --eval-seed 469 --res-file-save-name train_scripts/msr/plots/smooth_goal_ppo/results/smooth_goal_ppo_pi_and_v__epsilon_0_1_reg_0_001_beta_0_v_reg_100_v_beta_0_eval_noise_1.csv

# epsilon = 0.1, reg = 0.001, beta = 0, v_reg = 10, v_beta = 0
# eval noise = 0.1
python train_scripts/msr/evaluate/evaluate_policy_by_v_func_adj_diff.py --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --env-flag-str Hard-05 --algo-class ppo --algo-ckpt-dir checkpoints/msr/medium/smooth_goal_ppo_pi_and_v/epsilon_0_1_reg_0_001_beta_0_N_16_v_reg_10_v_beta_0/128_128_2e8steps_seed_{0} --algo-ckpt-model-name best_model --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO_V --n-envs 8 --n-eval-episodes 10000 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --eval-seed 346 --res-file-save-name train_scripts/msr/plots/smooth_goal_ppo/results/smooth_goal_ppo_pi_and_v__epsilon_0_1_reg_0_001_beta_0_v_reg_10_v_beta_0_eval_noise_0_1.csv
# eval noise = 1
python train_scripts/msr/evaluate/evaluate_policy_by_v_func_adj_diff.py --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --env-flag-str Hard-05 --algo-class ppo --algo-ckpt-dir checkpoints/msr/medium/smooth_goal_ppo_pi_and_v/epsilon_0_1_reg_0_001_beta_0_N_16_v_reg_10_v_beta_0/128_128_2e8steps_seed_{0} --algo-ckpt-model-name best_model --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO_V --n-envs 8 --n-eval-episodes 10000 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1 --eval-seed 469 --res-file-save-name train_scripts/msr/plots/smooth_goal_ppo/results/smooth_goal_ppo_pi_and_v__epsilon_0_1_reg_0_001_beta_0_v_reg_10_v_beta_0_eval_noise_1.csv

# epsilon = 0.1, reg = 0.001, beta = 0, v_reg = 1, v_beta = 0
# eval noise = 0.1
python train_scripts/msr/evaluate/evaluate_policy_by_v_func_adj_diff.py --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --env-flag-str Hard-05 --algo-class ppo --algo-ckpt-dir checkpoints/msr/medium/smooth_goal_ppo_pi_and_v/epsilon_0_1_reg_0_001_beta_0_N_16_v_reg_1_v_beta_0/128_128_2e8steps_seed_{0} --algo-ckpt-model-name best_model --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO_V --n-envs 8 --n-eval-episodes 10000 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --eval-seed 346 --res-file-save-name train_scripts/msr/plots/smooth_goal_ppo/results/smooth_goal_ppo_pi_and_v__epsilon_0_1_reg_0_001_beta_0_v_reg_1_v_beta_0_eval_noise_0_1.csv
# eval noise = 1
python train_scripts/msr/evaluate/evaluate_policy_by_v_func_adj_diff.py --env-config-file configs/env/MSR/env_config_for_ppo_10hz_medium_b_05.json --env-flag-str Hard-05 --algo-class ppo --algo-ckpt-dir checkpoints/msr/medium/smooth_goal_ppo_pi_and_v/epsilon_0_1_reg_0_001_beta_0_N_16_v_reg_1_v_beta_0/128_128_2e8steps_seed_{0} --algo-ckpt-model-name best_model --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO_V --n-envs 8 --n-eval-episodes 10000 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1 --eval-seed 469 --res-file-save-name train_scripts/msr/plots/smooth_goal_ppo/results/smooth_goal_ppo_pi_and_v__epsilon_0_1_reg_0_001_beta_0_v_reg_1_v_beta_0_eval_noise_1.csv
