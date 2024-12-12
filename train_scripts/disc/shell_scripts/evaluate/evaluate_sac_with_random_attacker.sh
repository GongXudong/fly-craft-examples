#!/bin/bash

python train_scripts/disc/evaluate_sac_with_random_attacker.py --env-config configs/env/D2D/env_config_for_sac_hard_b_025.json --env-flag-str Hard-025 --algo-ckpt-dir checkpoints/rl_single/D2D/hard_sac_her_b_025/sac_her_10hz_128_128_b_025_1e6steps_seed_{0}_singleRL --algo-ckpt-model-name best_model --algo-seeds 1 2 3 4 5 --algo-flag-str HER --evaluate-dg-num 20 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/res_log_hard_her_random_10_noise_0_1.csv
