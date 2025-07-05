#!/bin/bash

python train_scripts/IRPO/train_with_bc_ppo.py --config-file-name configs/train/IRPO/guidance_law_mode/iter_3/MR/lambda_1e-1/ppo_bc_config_10hz_256_256_128_128_64_hard_lambda_1e-1_seed_1.json
python train_scripts/IRPO/train_with_bc_ppo.py --config-file-name configs/train/IRPO/guidance_law_mode/iter_3/MR/lambda_1e-1/ppo_bc_config_10hz_256_256_128_128_64_hard_lambda_1e-1_seed_2.json
python train_scripts/IRPO/train_with_bc_ppo.py --config-file-name configs/train/IRPO/guidance_law_mode/iter_3/MR/lambda_1e-1/ppo_bc_config_10hz_256_256_128_128_64_hard_lambda_1e-1_seed_3.json
python train_scripts/IRPO/train_with_bc_ppo.py --config-file-name configs/train/IRPO/guidance_law_mode/iter_3/MR/lambda_1e-1/ppo_bc_config_10hz_256_256_128_128_64_hard_lambda_1e-1_seed_4.json
python train_scripts/IRPO/train_with_bc_ppo.py --config-file-name configs/train/IRPO/guidance_law_mode/iter_3/MR/lambda_1e-1/ppo_bc_config_10hz_256_256_128_128_64_hard_lambda_1e-1_seed_5.json

#---------------------------------------- epsilon = [1.0, 0.3, 0.3] -------------------------------------------------------------
# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0  N=16
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_3_aug/epsilon_0_1_reg_0_N_16/256_256_128_128_64_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_3_aug/epsilon_0_1_reg_0_N_16/256_256_128_128_64_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_3_aug/epsilon_0_1_reg_0_N_16/256_256_128_128_64_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_3_aug/epsilon_0_1_reg_0_N_16/256_256_128_128_64_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_3_aug/epsilon_0_1_reg_0_N_16/256_256_128_128_64_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.0001  N=16
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_3_aug/epsilon_0_1_reg_0_0001_N_16/256_256_128_128_64_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_3_aug/epsilon_0_1_reg_0_0001_N_16/256_256_128_128_64_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_3_aug/epsilon_0_1_reg_0_0001_N_16/256_256_128_128_64_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_3_aug/epsilon_0_1_reg_0_0001_N_16/256_256_128_128_64_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_3_aug/epsilon_0_1_reg_0_0001_N_16/256_256_128_128_64_seed_5.json