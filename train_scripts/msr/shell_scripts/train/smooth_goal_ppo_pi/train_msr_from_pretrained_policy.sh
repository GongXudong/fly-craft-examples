#!/bin/bash


# 1.MSR: epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001  beta = 0  N = 16
python train_scripts/msr/train/from_bc/train_ppo_msr.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json
python train_scripts/msr/train/from_bc/train_ppo_msr.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json
python train_scripts/msr/train/from_bc/train_ppo_msr.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json
python train_scripts/msr/train/from_bc/train_ppo_msr.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json
python train_scripts/msr/train/from_bc/train_ppo_msr.py --config-file-name configs/train/msr/smooth_goal_ppo_pi/hard/from_bc/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json

