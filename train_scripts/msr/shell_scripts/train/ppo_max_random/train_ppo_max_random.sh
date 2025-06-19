#!/bin/bash

# #---------------------------------------- MSR-GC-PPO-MAX epsilon = [1.0, 0.3, 0.3],-------------------------------------------------------------

# # Env  Params: Medium, step_frequence = 10hz, no skip
# # Algo Params: regularization_strength = 0.001, beta = 0, N=16, gamma = 0.995
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/step_freq_10hz/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/step_freq_10hz/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/step_freq_10hz/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/step_freq_10hz/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/step_freq_10hz/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json


# # Env  Params: Medium, step_frequence = 10hz, no skip
# # Algo Params: regularization_strength = 0.001, beta = 0.1, N=16, gamma = 0.995
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/step_freq_10hz/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/step_freq_10hz/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/step_freq_10hz/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/step_freq_10hz/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/step_freq_10hz/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json

# # Env  Params: Medium, step_frequence = 10hz, no skip
# # Algo Params: regularization_strength = 0.001, beta = 0, N=64, gamma = 0.98 
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/step_freq_10hz/beta_0/epsilon_0_1_reg_0_001_N_64_gamma_0_98/128_128_seed_1.json
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/step_freq_10hz/beta_0/epsilon_0_1_reg_0_001_N_64_gamma_0_98/128_128_seed_2.json
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/step_freq_10hz/beta_0/epsilon_0_1_reg_0_001_N_64_gamma_0_98/128_128_seed_3.json
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/step_freq_10hz/beta_0/epsilon_0_1_reg_0_001_N_64_gamma_0_98/128_128_seed_4.json
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/step_freq_10hz/beta_0/epsilon_0_1_reg_0_001_N_64_gamma_0_98/128_128_seed_5.json

