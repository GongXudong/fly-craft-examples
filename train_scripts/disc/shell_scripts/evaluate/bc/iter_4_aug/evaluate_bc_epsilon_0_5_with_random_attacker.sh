#!/bin/bash

# algo: epsilon 0.5 reg 0  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_ppo_with_random_attacker.py --env-config configs/env/env_hard_config_for_sac.json --env-flag-str Hard --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/hard/bc/iter_4_aug/epsilon_0_5_reg_0_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.5 --algo-reg 0.0 --evaluate-dg-num 300 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/bc/iter_4_aug/res_log_hard_bc_epsilon_0_5_reg_0_random_10_noise_0_1.csv

# algo: epsilon 0.5 reg 0  eval: epsilon 0.5
python train_scripts/disc/evaluate/evaluate_ppo_with_random_attacker.py --env-config configs/env/env_hard_config_for_sac.json --env-flag-str Hard --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/hard/bc/iter_4_aug/epsilon_0_5_reg_0_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.5 --algo-reg 0.0 --evaluate-dg-num 300 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.5 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/bc/iter_4_aug/res_log_hard_bc_epsilon_0_5_reg_0_random_10_noise_0_5.csv

# algo: epsilon 0.5 reg 0  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_ppo_with_random_attacker.py --env-config configs/env/env_hard_config_for_sac.json --env-flag-str Hard --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/hard/bc/iter_4_aug/epsilon_0_5_reg_0_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.5 --algo-reg 0.0 --evaluate-dg-num 300 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/bc/iter_4_aug/res_log_hard_bc_epsilon_0_5_reg_0_random_10_noise_1.csv

# algo: epsilon 0.5 reg 0.001  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_ppo_with_random_attacker.py --env-config configs/env/env_hard_config_for_sac.json --env-flag-str Hard --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/hard/bc/iter_4_aug/epsilon_0_5_reg_0_001_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.5 --algo-reg 0.001 --evaluate-dg-num 300 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/bc/iter_4_aug/res_log_hard_bc_epsilon_0_5_reg_0_001_random_10_noise_0_1.csv

# algo: epsilon 0.5 reg 0.001  eval: epsilon 0.5
python train_scripts/disc/evaluate/evaluate_ppo_with_random_attacker.py --env-config configs/env/env_hard_config_for_sac.json --env-flag-str Hard --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/hard/bc/iter_4_aug/epsilon_0_5_reg_0_001_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.5 --algo-reg 0.001 --evaluate-dg-num 300 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.5 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/bc/iter_4_aug/res_log_hard_bc_epsilon_0_5_reg_0_001_random_10_noise_0_5.csv

# algo: epsilon 0.5 reg 0.001  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_ppo_with_random_attacker.py --env-config configs/env/env_hard_config_for_sac.json --env-flag-str Hard --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/hard/bc/iter_4_aug/epsilon_0_5_reg_0_001_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.5 --algo-reg 0.001 --evaluate-dg-num 300 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/bc/iter_4_aug/res_log_hard_bc_epsilon_0_5_reg_0_001_random_10_noise_1.csv

# algo: epsilon 0.5 reg 0.01  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_ppo_with_random_attacker.py --env-config configs/env/env_hard_config_for_sac.json --env-flag-str Hard --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/hard/bc/iter_4_aug/epsilon_0_5_reg_0_01_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.5 --algo-reg 0.01 --evaluate-dg-num 300 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/bc/iter_4_aug/res_log_hard_bc_epsilon_0_5_reg_0_01_random_10_noise_0_1.csv

# algo: epsilon 0.5 reg 0.01  eval: epsilon 0.5
python train_scripts/disc/evaluate/evaluate_ppo_with_random_attacker.py --env-config configs/env/env_hard_config_for_sac.json --env-flag-str Hard --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/hard/bc/iter_4_aug/epsilon_0_5_reg_0_01_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.5 --algo-reg 0.01 --evaluate-dg-num 300 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.5 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/bc/iter_4_aug/res_log_hard_bc_epsilon_0_5_reg_0_01_random_10_noise_0_5.csv

# algo: epsilon 0.5 reg 0.01  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_ppo_with_random_attacker.py --env-config configs/env/env_hard_config_for_sac.json --env-flag-str Hard --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/hard/bc/iter_4_aug/epsilon_0_5_reg_0_01_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.5 --algo-reg 0.01 --evaluate-dg-num 300 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/bc/iter_4_aug/res_log_hard_bc_epsilon_0_5_reg_0_01_random_10_noise_1.csv

# algo: epsilon 0.5 reg 0.1  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_ppo_with_random_attacker.py --env-config configs/env/env_hard_config_for_sac.json --env-flag-str Hard --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/hard/bc/iter_4_aug/epsilon_0_5_reg_0_1_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.5 --algo-reg 0.1 --evaluate-dg-num 300 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/bc/iter_4_aug/res_log_hard_bc_epsilon_0_5_reg_0_1_random_10_noise_0_1.csv

# algo: epsilon 0.5 reg 0.1  eval: epsilon 0.5
python train_scripts/disc/evaluate/evaluate_ppo_with_random_attacker.py --env-config configs/env/env_hard_config_for_sac.json --env-flag-str Hard --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/hard/bc/iter_4_aug/epsilon_0_5_reg_0_1_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.5 --algo-reg 0.1 --evaluate-dg-num 300 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.5 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/bc/iter_4_aug/res_log_hard_bc_epsilon_0_5_reg_0_1_random_10_noise_0_5.csv

# algo: epsilon 0.5 reg 0.1  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_ppo_with_random_attacker.py --env-config configs/env/env_hard_config_for_sac.json --env-flag-str Hard --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/hard/bc/iter_4_aug/epsilon_0_5_reg_0_1_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.5 --algo-reg 0.1 --evaluate-dg-num 300 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/bc/iter_4_aug/res_log_hard_bc_epsilon_0_5_reg_0_1_random_10_noise_1.csv

# algo: epsilon 0.5 reg 1  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_ppo_with_random_attacker.py --env-config configs/env/env_hard_config_for_sac.json --env-flag-str Hard --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/hard/bc/iter_4_aug/epsilon_0_5_reg_1_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.5 --algo-reg 1.0 --evaluate-dg-num 300 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/bc/iter_4_aug/res_log_hard_bc_epsilon_0_5_reg_1_random_10_noise_0_1.csv

# algo: epsilon 0.5 reg 1  eval: epsilon 0.5
python train_scripts/disc/evaluate/evaluate_ppo_with_random_attacker.py --env-config configs/env/env_hard_config_for_sac.json --env-flag-str Hard --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/hard/bc/iter_4_aug/epsilon_0_5_reg_1_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.5 --algo-reg 1.0 --evaluate-dg-num 300 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.5 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/bc/iter_4_aug/res_log_hard_bc_epsilon_0_5_reg_1_random_10_noise_0_5.csv

# algo: epsilon 0.5 reg 1  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_ppo_with_random_attacker.py --env-config configs/env/env_hard_config_for_sac.json --env-flag-str Hard --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/disc/hard/bc/iter_4_aug/epsilon_0_5_reg_1_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.5 --algo-reg 1.0 --evaluate-dg-num 300 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/bc/iter_4_aug/res_log_hard_bc_epsilon_0_5_reg_1_random_10_noise_1.csv