#!/bin/bash

# bash train_scripts/disc/shell_scripts/evaluate/her/medium/evaluate_her_by_success_rate.sh &> tmp_her_medium_res.txt

python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/sac/medium_her/sac_config_10hz_128_128_1.json --env-config-file configs/env/D2D/env_config_for_ppo_medium_b_05.json --algo sac_only --seed 8234 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/sac/medium_her/sac_config_10hz_128_128_2.json --env-config-file configs/env/D2D/env_config_for_ppo_medium_b_05.json --algo sac_only --seed 341 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/sac/medium_her/sac_config_10hz_128_128_3.json --env-config-file configs/env/D2D/env_config_for_ppo_medium_b_05.json --algo sac_only --seed 2346 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/sac/medium_her/sac_config_10hz_128_128_4.json --env-config-file configs/env/D2D/env_config_for_ppo_medium_b_05.json --algo sac_only --seed 94 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/sac/medium_her/sac_config_10hz_128_128_5.json --env-config-file configs/env/D2D/env_config_for_ppo_medium_b_05.json --algo sac_only --seed 164 --n-envs 32 --n-eval-episode 1000


#---------------------------------------- epsilon = [1.0, 0.3, 0.3] -------------------------------------------------------------
# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001  N = 16
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json --env-config-file configs/env/D2D/env_config_for_ppo_medium_b_05.json --algo sac --seed 35210 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json --env-config-file configs/env/D2D/env_config_for_ppo_medium_b_05.json --algo sac --seed 723 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json --env-config-file configs/env/D2D/env_config_for_ppo_medium_b_05.json --algo sac --seed 93403 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json --env-config-file configs/env/D2D/env_config_for_ppo_medium_b_05.json --algo sac --seed 42006 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium/beta_0/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json --env-config-file configs/env/D2D/env_config_for_ppo_medium_b_05.json --algo sac --seed 42451 --n-envs 32 --n-eval-episode 1000