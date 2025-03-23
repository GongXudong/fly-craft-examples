#!/bin/bash


# ------------------------------------------------------ end-to-end mode ----------------------------------------------------------------------

## baseline
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_end_to_end/baseline/128_128_seed_1.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_end_to_end/baseline/128_128_seed_2.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_end_to_end/baseline/128_128_seed_3.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_end_to_end/baseline/128_128_seed_4.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_end_to_end/baseline/128_128_seed_5.json


# ----------------------------------------------------- guidance-law mode ---------------------------------------------------------------------

## baseline
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_guidance/baseline/128_128_seed_1.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_guidance/baseline/128_128_seed_2.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_guidance/baseline/128_128_seed_3.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_guidance/baseline/128_128_seed_4.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_guidance/baseline/128_128_seed_5.json

## reward_scaling_cluster
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_guidance/reward_scaling_cluster/128_128_seed_1.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_guidance/reward_scaling_cluster/128_128_seed_2.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_guidance/reward_scaling_cluster/128_128_seed_3.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_guidance/reward_scaling_cluster/128_128_seed_4.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_guidance/reward_scaling_cluster/128_128_seed_5.json


# -------------------- guidance-law mode   env_config_guidance_MR_medium_mu_chi_b_1   v 200, mu [-30, 30], chi [-60, 60] ----------------------

## baseline
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_mu_chi_b_1_guidance/baseline/128_128_seed_1.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_mu_chi_b_1_guidance/baseline/128_128_seed_2.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_mu_chi_b_1_guidance/baseline/128_128_seed_3.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_mu_chi_b_1_guidance/baseline/128_128_seed_4.json
python train_scripts/reward_norm/train/train_with_rl_sac_her.py --config-file-name configs/train/reward_norm/her/medium_mu_chi_b_1_guidance/baseline/128_128_seed_5.json


