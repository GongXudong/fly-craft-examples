#!/bin/bash

# This script is used to run the PPO algorithm on the VVCGym environment.

# Figure 3(b) of VVCGym paper, w/ or w/o environment designs

## [Exp 1] w/ environment designs, train PPO on the easy version of the VVCGym environment (5 random seeds)
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy/ppo_bc_config_10hz_128_128_easy_2.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy/ppo_bc_config_10hz_128_128_easy_3.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy/ppo_bc_config_10hz_128_128_easy_4.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy/ppo_bc_config_10hz_128_128_easy_5.json

## [Exp 2] w/o environment designs (5 random seeds)
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_no_terminations_b_1/ppo_bc_config_10hz_128_128_1.jso
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_no_terminations_b_1/ppo_bc_config_10hz_128_128_2.jso
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_no_terminations_b_1/ppo_bc_config_10hz_128_128_3.jso
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_no_terminations_b_1/ppo_bc_config_10hz_128_128_4.jso
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_no_terminations_b_1/ppo_bc_config_10hz_128_128_5.jso


# Figure 6 of VVCGym paper， w/ or w/o termination conditions

## w/ terminations, reuse [Exp 1]

## [Exp 3] w/o terminations (5 random seeds), use the 'script -a' command to log the terminations triggered during training to analysis Figure 6(c) and 6(d) of VVCGym paper
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_no_terminations/ppo_bc_config_10hz_128_128_easy_no_terminations_1.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_no_terminations/ppo_bc_config_10hz_128_128_easy_no_terminations_2.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_no_terminations/ppo_bc_config_10hz_128_128_easy_no_terminations_3.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_no_terminations/ppo_bc_config_10hz_128_128_easy_no_terminations_4.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_no_terminations/ppo_bc_config_10hz_128_128_easy_no_terminations_5.json


# Figure 7 and 8 of VVCGym paper， ablation on reward function

## ablation on $r_{penalty}$, Figure 7 of VVCGym paper

### $r_{penalty} = - \frac{1 - \gamma^{T_{max} - T_\tau}}{1 - \gamma}$, reuse [Exp 1]

### [Exp 4] $r_{penalty} = -1000$ (5 random seeds), use the 'script -a' command to log the terminations triggered during training
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_big_punishment/ppo_bc_config_10hz_128_128_easy_1.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_big_punishment/ppo_bc_config_10hz_128_128_easy_2.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_big_punishment/ppo_bc_config_10hz_128_128_easy_3.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_big_punishment/ppo_bc_config_10hz_128_128_easy_4.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_big_punishment/ppo_bc_config_10hz_128_128_easy_5.json

## ablation on $b$, Figure 8 of VVCGym paper

### $b = 0.5$, reuse [Exp 1]

### [Exp 5] $b = 0.25$ (5 random seeds), use the 'script -a' command to log the terminations triggered during training
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_025/ppo_bc_config_10hz_128_128_easy_b_025_1.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_025/ppo_bc_config_10hz_128_128_easy_b_025_2.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_025/ppo_bc_config_10hz_128_128_easy_b_025_3.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_025/ppo_bc_config_10hz_128_128_easy_b_025_4.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_025/ppo_bc_config_10hz_128_128_easy_b_025_5.json

# [Exp 6] $b = 1$ (5 random seeds), use the 'script -a' command to log the terminations triggered during training
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_1/ppo_bc_config_10hz_128_128_easy_b_1_1.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_1/ppo_bc_config_10hz_128_128_easy_b_1_2.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_1/ppo_bc_config_10hz_128_128_easy_b_1_3.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_1/ppo_bc_config_10hz_128_128_easy_b_1_4.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_1/ppo_bc_config_10hz_128_128_easy_b_1_5.json

# [Exp 7] $b = 2$ (5 random seeds), use the 'script -a' command to log the terminations triggered during training
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_2/ppo_bc_config_10hz_128_128_easy_b_2_1.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_2/ppo_bc_config_10hz_128_128_easy_b_2_2.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_2/ppo_bc_config_10hz_128_128_easy_b_2_3.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_2/ppo_bc_config_10hz_128_128_easy_b_2_4.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_2/ppo_bc_config_10hz_128_128_easy_b_2_5.json

# [Exp 8] $b = 4$ (5 random seeds), use the 'script -a' command to log the terminations triggered during training
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_4/ppo_bc_config_10hz_128_128_easy_b_4_1.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_4/ppo_bc_config_10hz_128_128_easy_b_4_2.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_4/ppo_bc_config_10hz_128_128_easy_b_4_3.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_4/ppo_bc_config_10hz_128_128_easy_b_4_4.json
python train_scripts/VVCGym/train/train_with_rl_ppo.py --config-file-name configs/train/VVCGym/ppo/easy_b_4/ppo_bc_config_10hz_128_128_easy_b_4_5.json


# Figure 4 of VVCGym paper, w/ or w/o pre-train on the hard version of the VVCGym environment. Refer to train_scripts/IRPO/README.md.
