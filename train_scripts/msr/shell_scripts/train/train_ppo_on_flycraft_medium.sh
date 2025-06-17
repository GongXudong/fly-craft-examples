#!/bin/bash


#---------------------------------------------------- Baseline GC-PPO -------------------------------------------------------------
python train_scripts/IRPO/train_with_rl_ppo.py --config-file-name configs/train/ppo/medium/ppo_config_10hz_128_128_1.json
python train_scripts/IRPO/train_with_rl_ppo.py --config-file-name configs/train/ppo/medium/ppo_config_10hz_128_128_2.json
python train_scripts/IRPO/train_with_rl_ppo.py --config-file-name configs/train/ppo/medium/ppo_config_10hz_128_128_3.json
python train_scripts/IRPO/train_with_rl_ppo.py --config-file-name configs/train/ppo/medium/ppo_config_10hz_128_128_4.json
python train_scripts/IRPO/train_with_rl_ppo.py --config-file-name configs/train/ppo/medium/ppo_config_10hz_128_128_5.json


# #---------------------------------------- MSR-GC-PPO epsilon = [0.1, 0.03, 0.03] -------------------------------------------------------------
# # epsilon = [0.1, 0.03, 0.03]  regularization_strength = 0.0001  N=16
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_0001_N_16/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_0001_N_16/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_0001_N_16/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_0001_N_16/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_0001_N_16/128_128_seed_5.json

# # epsilon = [0.1, 0.03, 0.03]  regularization_strength = 0.001  N=16
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_001_N_16/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_001_N_16/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_001_N_16/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_001_N_16/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_001_N_16/128_128_seed_5.json

# # epsilon = [0.1, 0.03, 0.03]  regularization_strength = 0.01  N=16
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_01_N_16/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_01_N_16/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_01_N_16/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_01_N_16/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_01_N_16/128_128_seed_5.json

# # epsilon = [0.1, 0.03, 0.03]  regularization_strength = 0.1  N=16
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_1_N_16/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_1_N_16/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_1_N_16/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_1_N_16/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_0_1_N_16/128_128_seed_5.json

# # epsilon = [0.1, 0.03, 0.03]  regularization_strength = 1.0  N=16
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_1_N_16/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_1_N_16/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_1_N_16/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_1_N_16/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_01_reg_1_N_16/128_128_seed_5.json


# #---------------------------------------- epsilon = [1.0, 0.3, 0.3] -------------------------------------------------------------
# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0/128_128_seed_5.json

# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_001/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_001/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_001/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_001/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_001/128_128_seed_5.json

# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.01
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_01/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_01/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_01/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_01/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_01/128_128_seed_5.json

# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.1
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_1/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_1/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_1/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_1/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_1/128_128_seed_5.json

# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 1.0
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_1/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_1/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_1/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_1/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_1/128_128_seed_5.json


# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.0001  N=16
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_0001_N_16/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_0001_N_16/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_0001_N_16/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_0001_N_16/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_0001_N_16/128_128_seed_5.json

# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001  N=16
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json

# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.01  N=16
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_01_N_16/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_01_N_16/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_01_N_16/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_01_N_16/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_01_N_16/128_128_seed_5.json

# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.1  N=16
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_1_N_16/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_1_N_16/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_1_N_16/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_1_N_16/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_0_1_N_16/128_128_seed_5.json

# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 1.0  N=16
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_1_N_16/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_1_N_16/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_1_N_16/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_1_N_16/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_1_reg_1_N_16/128_128_seed_5.json


# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001  beta = 0.0001  N=16
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_0001/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_0001/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_0001/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_0001/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_0001/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001  beta = 0.001  N=16
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_001/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_001/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_001/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_001/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_001/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001  beta = 0.01  N=16
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_01/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_01/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_01/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_01/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_01/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001  beta = 0.1  N=16
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json

# #---------------------------------------- epsilon = [5.0, 1.5, 1.5] -------------------------------------------------------------
# # epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0/128_128_seed_5.json

# # epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0.001
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0_001/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0_001/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0_001/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0_001/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0_001/128_128_seed_5.json

# # epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0.01
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0_01/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0_01/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0_01/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0_01/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0_01/128_128_seed_5.json

# # epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0.1
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0_1/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0_1/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0_1/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0_1/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_0_1/128_128_seed_5.json

# # epsilon = [5.0, 1.5, 1.5]  regularization_strength = 1.0
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_1/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_1/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_1/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_1/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_0_5_reg_1/128_128_seed_5.json

# #---------------------------------------- epsilon = [10.0, 3.0, 3.0] -------------------------------------------------------------
# # epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0/128_128_seed_5.json

# # epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.001
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_001/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_001/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_001/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_001/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_001/128_128_seed_5.json

# # epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.01
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_01/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_01/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_01/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_01/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_01/128_128_seed_5.json

# # epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.1
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_1/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_1/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_1/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_1/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_1/128_128_seed_5.json

# # epsilon = [10.0, 3.0, 3.0]  regularization_strength = 1.0
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_1/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_1/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_1/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_1/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_1/128_128_seed_5.json



# # epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0  N = 16
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_N_16/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_N_16/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_N_16/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_N_16/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_N_16/128_128_seed_5.json

# # epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.001  N = 16
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_001_N_16/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_001_N_16/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_001_N_16/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_001_N_16/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_001_N_16/128_128_seed_5.json

# # epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.01  N = 16
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_01_N_16/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_01_N_16/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_01_N_16/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_01_N_16/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_01_N_16/128_128_seed_5.json

# # epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.1  N = 16
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_1_N_16/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_1_N_16/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_1_N_16/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_1_N_16/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_0_1_N_16/128_128_seed_5.json

# # epsilon = [10.0, 3.0, 3.0]  regularization_strength = 1.0  N = 16
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_1_N_16/128_128_seed_1.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_1_N_16/128_128_seed_2.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_1_N_16/128_128_seed_3.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_1_N_16/128_128_seed_4.json
# python train_scripts/msr/train/train_ppo.py --config-file-name configs/train/msr/ppo/medium/epsilon_1_reg_1_N_16/128_128_seed_5.json
