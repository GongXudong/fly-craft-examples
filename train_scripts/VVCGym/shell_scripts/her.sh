#!/bin/bash

# This script is used to run the HER algorithm on the VVCGym environment.

# Figure 3(a) of VVCGym paper, w/ or w/o environment designs

## [Exp 1] w/ environment designs, train HER on the easy version of the VVCGym environment (5 random seeds)
python train_scripts/VVCGym/train/train_with_rl_sac_her.py --config-file-name configs/train/VVCGym/sac/easy_her/sac_config_10hz_128_128_1.json
python train_scripts/VVCGym/train/train_with_rl_sac_her.py --config-file-name configs/train/VVCGym/sac/easy_her/sac_config_10hz_128_128_2.json
python train_scripts/VVCGym/train/train_with_rl_sac_her.py --config-file-name configs/train/VVCGym/sac/easy_her/sac_config_10hz_128_128_3.json
python train_scripts/VVCGym/train/train_with_rl_sac_her.py --config-file-name configs/train/VVCGym/sac/easy_her/sac_config_10hz_128_128_4.json
python train_scripts/VVCGym/train/train_with_rl_sac_her.py --config-file-name configs/train/VVCGym/sac/easy_her/sac_config_10hz_128_128_5.json

## [Exp 2] w/o environment designs (5 random seeds)
python train_scripts/VVCGym/train/train_with_rl_sac_her.py --config-file-name configs/train/VVCGym/sac/easy_her_no_terminations_b_1/sac_config_10hz_128_128_1.json
python train_scripts/VVCGym/train/train_with_rl_sac_her.py --config-file-name configs/train/VVCGym/sac/easy_her_no_terminations_b_1/sac_config_10hz_128_128_2.json
python train_scripts/VVCGym/train/train_with_rl_sac_her.py --config-file-name configs/train/VVCGym/sac/easy_her_no_terminations_b_1/sac_config_10hz_128_128_3.json
python train_scripts/VVCGym/train/train_with_rl_sac_her.py --config-file-name configs/train/VVCGym/sac/easy_her_no_terminations_b_1/sac_config_10hz_128_128_4.json
python train_scripts/VVCGym/train/train_with_rl_sac_her.py --config-file-name configs/train/VVCGym/sac/easy_her_no_terminations_b_1/sac_config_10hz_128_128_5.json
