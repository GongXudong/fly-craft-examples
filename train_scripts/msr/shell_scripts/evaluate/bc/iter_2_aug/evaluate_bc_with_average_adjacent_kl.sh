#!/bin/bash


# Iter 2
# algo: epsilon 0.0 reg 0.0  eval: epsilon 0.01
python train_scripts/msr/evaluate/evaluate_ppo_with_average_adjacent_KL.py --env-config configs/env/VVCGym/env_hard_config_for_sac.json --env-flag-str Hard-05 --algo-class PPO --algo-ckpt-dir checkpoints/IRPO/bc/guidance_law_mode/iter_2/128_128_300epochs_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str PPO --algo-epsilon 0.0 --algo-reg 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/msr/plots/bc/results/bc_iter_2_epsilon_0_reg_0_noise_0_01.csv

# algo: epsilon 0.0 reg 0.0  eval: epsilon 0.1
python train_scripts/msr/evaluate/evaluate_ppo_with_average_adjacent_KL.py --env-config configs/env/VVCGym/env_hard_config_for_sac.json --env-flag-str Hard-05 --algo-class PPO --algo-ckpt-dir checkpoints/IRPO/bc/guidance_law_mode/iter_2/128_128_300epochs_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str PPO --algo-epsilon 0.0 --algo-reg 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/msr/plots/bc/results/bc_iter_2_epsilon_0_reg_0_noise_0_1.csv

# algo: epsilon 0.0 reg 0.0  eval: epsilon 1.0
python train_scripts/msr/evaluate/evaluate_ppo_with_average_adjacent_KL.py --env-config configs/env/VVCGym/env_hard_config_for_sac.json --env-flag-str Hard-05 --algo-class PPO --algo-ckpt-dir checkpoints/IRPO/bc/guidance_law_mode/iter_2/128_128_300epochs_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str PPO --algo-epsilon 0.0 --algo-reg 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/msr/plots/bc/results/bc_iter_2_epsilon_0_reg_0_noise_1.csv


#--------------------------------------------------------------------SmoothGoalBC  Iter 2  epsilon=0.1--------------------------------------------------------------

# algo: epsilon 0.1 reg 0  eval: epsilon 0.01  noise_num: 16
python train_scripts/msr/evaluate/evaluate_ppo_with_average_adjacent_KL.py --env-config configs/env/VVCGym/env_hard_config_for_sac.json --env-flag-str Hard-05 --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/msr/hard/bc/iter_2_aug/epsilon_0_1_reg_0_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO --algo-epsilon 0.1 --algo-reg 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/msr/plots/bc/results/bc_iter_2_epsilon_0_1_reg_0_N_16_noise_0_01.csv

# algo: epsilon 0.1 reg 0  eval: epsilon 0.1  noise_num: 16
python train_scripts/msr/evaluate/evaluate_ppo_with_average_adjacent_KL.py --env-config configs/env/VVCGym/env_hard_config_for_sac.json --env-flag-str Hard-05 --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/msr/hard/bc/iter_2_aug/epsilon_0_1_reg_0_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO --algo-epsilon 0.1 --algo-reg 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/msr/plots/bc/results/bc_iter_2_epsilon_0_1_reg_0_N_16_noise_0_1.csv

# algo: epsilon 0.1 reg 0  eval: epsilon 1.0  noise_num: 16
python train_scripts/msr/evaluate/evaluate_ppo_with_average_adjacent_KL.py --env-config configs/env/VVCGym/env_hard_config_for_sac.json --env-flag-str Hard-05 --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/msr/hard/bc/iter_2_aug/epsilon_0_1_reg_0_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO --algo-epsilon 0.1 --algo-reg 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/msr/plots/bc/results/bc_iter_2_epsilon_0_1_reg_0_N_16_noise_1.csv




# algo: epsilon 0.1 reg 0.0001  eval: epsilon 0.01  noise_num: 16
python train_scripts/msr/evaluate/evaluate_ppo_with_average_adjacent_KL.py --env-config configs/env/VVCGym/env_hard_config_for_sac.json --env-flag-str Hard-05 --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/msr/hard/bc/iter_2_aug/epsilon_0_1_reg_0_0001_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO --algo-epsilon 0.1 --algo-reg 0.0001 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/msr/plots/bc/results/bc_iter_2_epsilon_0_1_reg_0_0001_N_16_noise_0_01.csv

# algo: epsilon 0.1 reg 0.0001  eval: epsilon 0.1  noise_num: 16
python train_scripts/msr/evaluate/evaluate_ppo_with_average_adjacent_KL.py --env-config configs/env/VVCGym/env_hard_config_for_sac.json --env-flag-str Hard-05 --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/msr/hard/bc/iter_2_aug/epsilon_0_1_reg_0_0001_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO --algo-epsilon 0.1 --algo-reg 0.0001 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/msr/plots/bc/results/bc_iter_2_epsilon_0_1_reg_0_0001_N_16_noise_0_1.csv

# algo: epsilon 0.1 reg 0.0001  eval: epsilon 1.0  noise_num: 16
python train_scripts/msr/evaluate/evaluate_ppo_with_average_adjacent_KL.py --env-config configs/env/VVCGym/env_hard_config_for_sac.json --env-flag-str Hard-05 --algo-class SmoothGoalPPO --algo-ckpt-dir checkpoints/msr/hard/bc/iter_2_aug/epsilon_0_1_reg_0_0001_N_16/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalPPO --algo-epsilon 0.1 --algo-reg 0.0001 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/msr/plots/bc/results/bc_iter_2_epsilon_0_1_reg_0_0001_N_16_noise_1.csv


