#!/bin/bash

# 200 20 30
python train_scripts/IRPO/train_with_rl_ppo.py --config-file-name configs/train/ppo/single_goal/200_20_30/128_128_1.json
python train_scripts/IRPO/train_with_rl_ppo.py --config-file-name configs/train/ppo/single_goal/200_20_30/128_128_2.json
python train_scripts/IRPO/train_with_rl_ppo.py --config-file-name configs/train/ppo/single_goal/200_20_30/128_128_3.json
python train_scripts/IRPO/train_with_rl_ppo.py --config-file-name configs/train/ppo/single_goal/200_20_30/128_128_4.json
python train_scripts/IRPO/train_with_rl_ppo.py --config-file-name configs/train/ppo/single_goal/200_20_30/128_128_5.json


# #---------------------------------------- epsilon = [1.0, 0.3, 0.3] -------------------------------------------------------------
# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001
python train_scripts/disc/train/train_ppo.py --config-file-name configs/train/disc/ppo/medium/epsilon_0_1_reg_0_001/128_128_seed_1.json
python train_scripts/disc/train/train_ppo.py --config-file-name configs/train/disc/ppo/medium/epsilon_0_1_reg_0_001/128_128_seed_2.json
python train_scripts/disc/train/train_ppo.py --config-file-name configs/train/disc/ppo/medium/epsilon_0_1_reg_0_001/128_128_seed_3.json
python train_scripts/disc/train/train_ppo.py --config-file-name configs/train/disc/ppo/medium/epsilon_0_1_reg_0_001/128_128_seed_4.json
python train_scripts/disc/train/train_ppo.py --config-file-name configs/train/disc/ppo/medium/epsilon_0_1_reg_0_001/128_128_seed_5.json
