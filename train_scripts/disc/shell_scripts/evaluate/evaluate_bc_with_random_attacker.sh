#!/bin/bash

# algo: epsilon 0.1 reg 0  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_random_attacker.py --env-config configs/env/D2D/env_config_for_sac_medium2_b_05.json --env-flag-str Medium2-05 --algo-ckpt-dir checkpoints/disc/medium2/bc/epsilon_0_1_reg_0/128_128_300epochs_seed_{0} --algo-ckpt-model-name bc_checkpoint --algo-seeds 1 2 3 4 5 --algo-flag-str BC --algo-epsilon 0.1 --algo-reg 0.0 --evaluate-dg-num 1000 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/res_log_medium2_bc_epsilon_0_1_reg_0_random_10_noise_0_1.csv


