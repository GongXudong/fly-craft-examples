#!/bin/bash


# 1.epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001  beta = 0  N = 16

## 1.1 v_regularization_strength = 0  v_beta = 0
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_v_beta_0/128_128_seed_1.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_v_beta_0/128_128_seed_2.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_v_beta_0/128_128_seed_3.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_v_beta_0/128_128_seed_4.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_v_beta_0/128_128_seed_5.json

## 1.2 v_regularization_strength = 0.0001  v_beta = 0
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_0001_v_beta_0/128_128_seed_1.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_0001_v_beta_0/128_128_seed_2.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_0001_v_beta_0/128_128_seed_3.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_0001_v_beta_0/128_128_seed_4.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_0001_v_beta_0/128_128_seed_5.json

## 1.3 v_regularization_strength = 0.001  v_beta = 0
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_001_v_beta_0/128_128_seed_1.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_001_v_beta_0/128_128_seed_2.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_001_v_beta_0/128_128_seed_3.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_001_v_beta_0/128_128_seed_4.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_pi_and_v/medium/epsilon_0_1_reg_0_001_beta_0_N_16/v_reg_0_001_v_beta_0/128_128_seed_5.json

