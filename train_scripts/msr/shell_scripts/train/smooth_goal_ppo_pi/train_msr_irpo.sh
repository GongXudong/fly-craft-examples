#!/bin/bash


# 1. w/o IRPO, w/o MSR     IRPO: lambda = 0   MSR: epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0  beta = 0  N = 16
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0__msr_epsilon_0_1_reg_0_beta_0_N_16/128_128_seed_1.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0__msr_epsilon_0_1_reg_0_beta_0_N_16/128_128_seed_2.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0__msr_epsilon_0_1_reg_0_beta_0_N_16/128_128_seed_3.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0__msr_epsilon_0_1_reg_0_beta_0_N_16/128_128_seed_4.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0__msr_epsilon_0_1_reg_0_beta_0_N_16/128_128_seed_5.json


# 2. w/o IRPO, w/ MSR     IRPO: lambda = 0   MSR: epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001  beta = 0  N = 16
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0__msr_epsilon_0_1_reg_0_001_beta_0_N_16/128_128_seed_1.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0__msr_epsilon_0_1_reg_0_001_beta_0_N_16/128_128_seed_2.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0__msr_epsilon_0_1_reg_0_001_beta_0_N_16/128_128_seed_3.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0__msr_epsilon_0_1_reg_0_001_beta_0_N_16/128_128_seed_4.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0__msr_epsilon_0_1_reg_0_001_beta_0_N_16/128_128_seed_5.json


# 3. w/ IRPO, w/o MSR     IRPO: lambda = 0.001   MSR: epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0  beta = 0  N = 16
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0_001__msr_epsilon_0_1_reg_0_beta_0_N_16/128_128_seed_1.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0_001__msr_epsilon_0_1_reg_0_beta_0_N_16/128_128_seed_2.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0_001__msr_epsilon_0_1_reg_0_beta_0_N_16/128_128_seed_3.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0_001__msr_epsilon_0_1_reg_0_beta_0_N_16/128_128_seed_4.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0_001__msr_epsilon_0_1_reg_0_beta_0_N_16/128_128_seed_5.json


# 4. w/ IRPO, w/ MSR     IRPO: lambda = 0.001   MSR: epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001  beta = 0  N = 16
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0_001__msr_epsilon_0_1_reg_0_001_beta_0_N_16/128_128_seed_1.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0_001__msr_epsilon_0_1_reg_0_001_beta_0_N_16/128_128_seed_2.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0_001__msr_epsilon_0_1_reg_0_001_beta_0_N_16/128_128_seed_3.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0_001__msr_epsilon_0_1_reg_0_001_beta_0_N_16/128_128_seed_4.json
python train_scripts/msr/train/from_bc/train_ppo_msr_irpo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc_with_irpo/irpo_lambda_0_001__msr_epsilon_0_1_reg_0_001_beta_0_N_16/128_128_seed_5.json

