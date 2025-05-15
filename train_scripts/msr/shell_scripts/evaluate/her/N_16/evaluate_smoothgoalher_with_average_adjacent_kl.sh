#!/bin/bash


# algo: epsilon 0.01 reg 0.0  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/rl_single/sac_medium_128_128_1e6steps_loss_{0}_singleRL --algo-seeds 1 2 3 4 5 --algo-flag-str SAC --algo-epsilon 0.0 --algo-reg 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac/results/sac_epsilon_0_reg_0_noise_0_01.csv

# algo: epsilon 0.01 reg 0.0  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/rl_single/sac_medium_128_128_1e6steps_loss_{0}_singleRL --algo-seeds 1 2 3 4 5 --algo-flag-str SAC --algo-epsilon 0.0 --algo-reg 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac/results/sac_epsilon_0_reg_0_noise_0_1.csv

# algo: epsilon 0.01 reg 0.0  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/rl_single/sac_medium_128_128_1e6steps_loss_{0}_singleRL --algo-seeds 1 2 3 4 5 --algo-flag-str SAC --algo-epsilon 0.0 --algo-reg 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac/results/sac_epsilon_0_reg_0_noise_1.csv


#--------------------------------------------------------------------SmoothGoalSAC epsilon=0.01--------------------------------------------------------------
