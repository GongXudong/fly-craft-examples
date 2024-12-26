#!/bin/bash


#---------------------------------------- epsilon = [1.0, 0.3, 0.3] -------------------------------------------------------------
# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.0  noise_num = 16
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_N_16/128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_N_16/128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_N_16/128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_N_16/128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_N_16/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.0001  noise_num = 16
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_0001_N_16/128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_0001_N_16/128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_0001_N_16/128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_0001_N_16/128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_0001_N_16/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001  noise_num = 16
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json