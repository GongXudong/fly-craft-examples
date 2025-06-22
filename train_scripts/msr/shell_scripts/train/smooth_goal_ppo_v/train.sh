#!/bin/bash

# 1. v_regularization_strength = 0.1  v_beta = 0
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_v/medium/v_reg_0_1_v_beta_0/128_128_seed_1.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_v/medium/v_reg_0_1_v_beta_0/128_128_seed_2.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_v/medium/v_reg_0_1_v_beta_0/128_128_seed_3.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_v/medium/v_reg_0_1_v_beta_0/128_128_seed_4.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_v/medium/v_reg_0_1_v_beta_0/128_128_seed_5.json

# 2. v_regularization_strength = 10.0  v_beta = 0
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_v/medium/v_reg_10_v_beta_0/128_128_seed_1.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_v/medium/v_reg_10_v_beta_0/128_128_seed_2.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_v/medium/v_reg_10_v_beta_0/128_128_seed_3.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_v/medium/v_reg_10_v_beta_0/128_128_seed_4.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/smooth_goal_ppo_v/medium/v_reg_10_v_beta_0/128_128_seed_5.json
