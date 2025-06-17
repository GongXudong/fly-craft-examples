#!/bin/bash


python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/medium_her/sac_config_10hz_128_128_1.json
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/medium_her/sac_config_10hz_128_128_2.json
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/medium_her/sac_config_10hz_128_128_3.json
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/medium_her/sac_config_10hz_128_128_4.json
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/medium_her/sac_config_10hz_128_128_5.json
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/medium_her/sac_config_10hz_128_128_6.json
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/medium_her/sac_config_10hz_128_128_7.json
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/medium_her/sac_config_10hz_128_128_8.json
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/medium_her/sac_config_10hz_128_128_9.json
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/medium_her/sac_config_10hz_128_128_10.json


#---------------------------------------- epsilon = [1.0, 0.3, 0.3] -------------------------------------------------------------

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.0001  noise_num = 16
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_0001_N_16/128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_0001_N_16/128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_0001_N_16/128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_0001_N_16/128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_0001_N_16/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001  noise_num = 16
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_6.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_7.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_8.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_9.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_10.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.01  noise_num = 16
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_01_N_16/128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_01_N_16/128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_01_N_16/128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_01_N_16/128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_01_N_16/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.1  noise_num = 16
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_1_N_16/128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_1_N_16/128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_1_N_16/128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_1_N_16/128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_1_N_16/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 1.0  noise_num = 16
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_1_N_16/128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_1_N_16/128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_1_N_16/128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_1_N_16/128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_1_N_16/128_128_seed_5.json



# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.0001  beta = 0.1  noise_num = 16
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0_1/epsilon_0_1_reg_0_0001_N_16/128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0_1/epsilon_0_1_reg_0_0001_N_16/128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0_1/epsilon_0_1_reg_0_0001_N_16/128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0_1/epsilon_0_1_reg_0_0001_N_16/128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0_1/epsilon_0_1_reg_0_0001_N_16/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001  beta = 0.1  noise_num = 16
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json
python train_scripts/disc/train/train_sac.py --config-file-name configs/train/disc/her/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json



