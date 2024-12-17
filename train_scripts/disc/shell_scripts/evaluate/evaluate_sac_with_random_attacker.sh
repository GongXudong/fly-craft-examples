#!/bin/bash

# noise multiplier 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_random_attacker.py --env-config configs/env/D2D/env_config_for_sac_medium2_b_05.json --env-flag-str Medium2-05 --algo-ckpt-dir checkpoints/rl_single/D2D/single_b/sac_her_medium2_10hz_128_128_b_05_1e6steps_seed_{0}_singleRL --algo-ckpt-model-name best_model --algo-seeds 3 4 5 --algo-flag-str HER --evaluate-dg-num 100 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/res_log_medium2_b_05_her_random_10_noise_0_1.csv

# noise multiplier 0.3
python train_scripts/disc/evaluate/evaluate_sac_with_random_attacker.py --env-config configs/env/D2D/env_config_for_sac_medium2_b_05.json --env-flag-str Medium2-05 --algo-ckpt-dir checkpoints/rl_single/D2D/single_b/sac_her_medium2_10hz_128_128_b_05_1e6steps_seed_{0}_singleRL --algo-ckpt-model-name best_model --algo-seeds 3 4 5 --algo-flag-str HER --evaluate-dg-num 100 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.3 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/res_log_medium2_b_05_her_random_10_noise_0_3.csv

# noise multiplier 0.5
python train_scripts/disc/evaluate/evaluate_sac_with_random_attacker.py --env-config configs/env/D2D/env_config_for_sac_medium2_b_05.json --env-flag-str Medium2-05 --algo-ckpt-dir checkpoints/rl_single/D2D/single_b/sac_her_medium2_10hz_128_128_b_05_1e6steps_seed_{0}_singleRL --algo-ckpt-model-name best_model --algo-seeds 3 4 5 --algo-flag-str HER --evaluate-dg-num 100 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.5 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/res_log_medium2_b_05_her_random_10_noise_0_5.csv

# noise multiplier 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_random_attacker.py --env-config configs/env/D2D/env_config_for_sac_medium2_b_05.json --env-flag-str Medium2-05 --algo-ckpt-dir checkpoints/rl_single/D2D/single_b/sac_her_medium2_10hz_128_128_b_05_1e6steps_seed_{0}_singleRL --algo-ckpt-model-name best_model --algo-seeds 3 4 5 --algo-flag-str HER --evaluate-dg-num 100 --evaluate-random-noise-num 10 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --attacker-flag-str Random --res-file-save-name train_scripts/disc/evaluate/results/res_log_medium2_b_05_her_random_10_noise_1_0.csv
