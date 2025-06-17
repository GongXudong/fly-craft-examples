#!/bin/bash

# #---------------------------------------- MSR-GC-PPO-MAX epsilon = [1.0, 0.3, 0.3],-------------------------------------------------------------
# beta_0/epsilon_0_1_reg_0_001
# # epsilon = [1.0, 0.3, 0.3], regularization_strength = 0.001  N=16
# 242
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json
# 20
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json
# 10
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json
# 100
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json

python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json


# beta_0_1/epsilon_0_1_reg_0_001
# 120
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json
# 140
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json
# 166
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json
# 110
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json

python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json



# beta_0/epsilon_0_1_reg_0_001 N = 64  frame skip 5
# 20
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/beta_0/epsilon_0_1_reg_0_001_N_64/128_128_seed_1.json
# 166
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/beta_0/epsilon_0_1_reg_0_001_N_64/128_128_seed_2.json
# 110
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/beta_0/epsilon_0_1_reg_0_001_N_64/128_128_seed_3.json
# 100
python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/beta_0/epsilon_0_1_reg_0_001_N_64/128_128_seed_4.json

python train_scripts/msr/train/train_ppo_max_random.py --config-file-name configs/train/msr/smooth_goal_ppo_max_random/medium/beta_0/epsilon_0_1_reg_0_001_N_64/128_128_seed_5.json