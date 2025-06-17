#!/bin/bash

python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_test_time.json
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/medium_without_her/sac_config_10hz_128_128_test_time.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_test_time.json
python train_scripts/IRPO/train_with_rl_ppo.py --config-file-name configs/train/ppo/medium/ppo_config_10hz_128_128_test_time.json
python train_scripts/disc/train/train_ppo.py --config-file-name configs/train/disc/ppo/medium/epsilon_0_1_reg_0_001_N_16/128_128_test_time.json
