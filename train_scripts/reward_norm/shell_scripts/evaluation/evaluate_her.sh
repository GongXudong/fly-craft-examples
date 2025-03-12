#!/bin/bash


# baseline
python train_scripts/reward_norm/evaluations/evaluation.py --algo her --algo-ckpt checkpoints/reward_norm/medium_guidance/her/baseline/128_128_1e6steps_seed_1/best_model.zip --env-config configs/env/reward_norm/env_config_guidance_MR_medium_b_05.json --n-envs 8 --eval-episode-num 1000 --seed 34233 --save-result --result-file-save-name train_scripts/reward_norm/evaluations/results/medium_guidance/her/baseline/seed_1.csv
python train_scripts/reward_norm/evaluations/evaluation.py --algo her --algo-ckpt checkpoints/reward_norm/medium_guidance/her/baseline/128_128_1e6steps_seed_2/best_model.zip --env-config configs/env/reward_norm/env_config_guidance_MR_medium_b_05.json --n-envs 8 --eval-episode-num 1000 --seed 24687 --save-result --result-file-save-name train_scripts/reward_norm/evaluations/results/medium_guidance/her/baseline/seed_2.csv
python train_scripts/reward_norm/evaluations/evaluation.py --algo her --algo-ckpt checkpoints/reward_norm/medium_guidance/her/baseline/128_128_1e6steps_seed_3/best_model.zip --env-config configs/env/reward_norm/env_config_guidance_MR_medium_b_05.json --n-envs 8 --eval-episode-num 1000 --seed 52889 --save-result --result-file-save-name train_scripts/reward_norm/evaluations/results/medium_guidance/her/baseline/seed_3.csv
python train_scripts/reward_norm/evaluations/evaluation.py --algo her --algo-ckpt checkpoints/reward_norm/medium_guidance/her/baseline/128_128_1e6steps_seed_4/best_model.zip --env-config configs/env/reward_norm/env_config_guidance_MR_medium_b_05.json --n-envs 8 --eval-episode-num 1000 --seed 23457 --save-result --result-file-save-name train_scripts/reward_norm/evaluations/results/medium_guidance/her/baseline/seed_4.csv
python train_scripts/reward_norm/evaluations/evaluation.py --algo her --algo-ckpt checkpoints/reward_norm/medium_guidance/her/baseline/128_128_1e6steps_seed_5/best_model.zip --env-config configs/env/reward_norm/env_config_guidance_MR_medium_b_05.json --n-envs 8 --eval-episode-num 1000 --seed 18662 --save-result --result-file-save-name train_scripts/reward_norm/evaluations/results/medium_guidance/her/baseline/seed_5.csv

# reward_scaling_cluster
