#!/bin/bash

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0/sac_config_10hz_128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0/sac_config_10hz_128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0/sac_config_10hz_128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0/sac_config_10hz_128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0/sac_config_10hz_128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_001/sac_config_10hz_128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_001/sac_config_10hz_128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_001/sac_config_10hz_128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_001/sac_config_10hz_128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_001/sac_config_10hz_128_128_seed_5.json

# epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0.001
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_5_reg_0_001/sac_config_10hz_128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_5_reg_0_001/sac_config_10hz_128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_5_reg_0_001/sac_config_10hz_128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_5_reg_0_001/sac_config_10hz_128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_5_reg_0_001/sac_config_10hz_128_128_seed_5.json

# epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.001
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_1_reg_0_001/sac_config_10hz_128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_1_reg_0_001/sac_config_10hz_128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_1_reg_0_001/sac_config_10hz_128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_1_reg_0_001/sac_config_10hz_128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_1_reg_0_001/sac_config_10hz_128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.1
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_1/sac_config_10hz_128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_1/sac_config_10hz_128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_1/sac_config_10hz_128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_1/sac_config_10hz_128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_1/sac_config_10hz_128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 1.0
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_1/sac_config_10hz_128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_1/sac_config_10hz_128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_1/sac_config_10hz_128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_1/sac_config_10hz_128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_1/sac_config_10hz_128_128_seed_5.json
