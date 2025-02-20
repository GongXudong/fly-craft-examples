#!/bin/bash

# NMR  lambda_1e-3     242
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/NMR/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_1.json
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/NMR/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_2.json
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/NMR/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_3.json

#  120
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/NMR/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_4.json
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/NMR/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_5.json

python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-2/ppo_bc_config_10hz_128_128_hard_lambda_1e-2_seed_4.json


# MR 
# lambda_1e-3    166
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-3/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_1.json
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-3/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_2.json
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-3/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_3.json

# 140
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-3/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_4.json
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-3/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_5.json

python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-2/ppo_bc_config_10hz_128_128_hard_lambda_1e-2_seed_5.json

# lambda_1e0   20
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e0/ppo_bc_config_10hz_128_128_hard_lambda_1e0_seed_1.json
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e0/ppo_bc_config_10hz_128_128_hard_lambda_1e0_seed_2.json
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e0/ppo_bc_config_10hz_128_128_hard_lambda_1e0_seed_3.json


# 251 
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e0/ppo_bc_config_10hz_128_128_hard_lambda_1e0_seed_4.json
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e0/ppo_bc_config_10hz_128_128_hard_lambda_1e0_seed_5.json

python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-4/ppo_bc_config_10hz_128_128_hard_lambda_1e-4_seed_1.json


# lambda_1e-1 100
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-1/ppo_bc_config_10hz_128_128_hard_lambda_1e-1_seed_1.json
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-1/ppo_bc_config_10hz_128_128_hard_lambda_1e-1_seed_2.json
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-1/ppo_bc_config_10hz_128_128_hard_lambda_1e-1_seed_3.json

#110
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-1/ppo_bc_config_10hz_128_128_hard_lambda_1e-1_seed_4.json
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-1/ppo_bc_config_10hz_128_128_hard_lambda_1e-1_seed_5.json

python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-4/ppo_bc_config_10hz_128_128_hard_lambda_1e-4_seed_2.json

# lambda_1e-2  10
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-2/ppo_bc_config_10hz_128_128_hard_lambda_1e-2_seed_1.json
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-2/ppo_bc_config_10hz_128_128_hard_lambda_1e-2_seed_2.json
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-2/ppo_bc_config_10hz_128_128_hard_lambda_1e-2_seed_3.json



# 已分散
# python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-2/ppo_bc_config_10hz_128_128_hard_lambda_1e-2_seed_4.json
# python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-2/ppo_bc_config_10hz_128_128_hard_lambda_1e-2_seed_5.json

# lambda_1e-4 
# python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-4/ppo_bc_config_10hz_128_128_hard_lambda_1e-4_seed_1.json
# python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-4/ppo_bc_config_10hz_128_128_hard_lambda_1e-4_seed_2.json

# 待分散
# python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-4/ppo_bc_config_10hz_128_128_hard_lambda_1e-4_seed_3.json
# python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-4/ppo_bc_config_10hz_128_128_hard_lambda_1e-4_seed_4.json
# python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-4/ppo_bc_config_10hz_128_128_hard_lambda_1e-4_seed_5.json