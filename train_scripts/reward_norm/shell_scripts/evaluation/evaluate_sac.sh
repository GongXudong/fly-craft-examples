#!/bin/bash


# baseline
python train_scripts/reward_norm/evaluations/evaluation.py --algo sac --algo-ckpt checkpoints/reward_norm/medium_mu_chi_b_1_guidance/sac/baseline/128_128_1e6steps_seed_1/best_model.zip --env-config configs/env/reward_norm/env_config_guidance_MR_medium_mu_chi_b_1.json --n-envs 8 --eval-episode-num 1000 --seed 2346 --save-result --result-file-save-name train_scripts/reward_norm/evaluations/results/medium_mu_chi_b_1_guidance/sac/baseline/seed_1.csv
python train_scripts/reward_norm/evaluations/evaluation.py --algo sac --algo-ckpt checkpoints/reward_norm/medium_mu_chi_b_1_guidance/sac/baseline/128_128_1e6steps_seed_2/best_model.zip --env-config configs/env/reward_norm/env_config_guidance_MR_medium_mu_chi_b_1.json --n-envs 8 --eval-episode-num 1000 --seed 4822 --save-result --result-file-save-name train_scripts/reward_norm/evaluations/results/medium_mu_chi_b_1_guidance/sac/baseline/seed_2.csv
python train_scripts/reward_norm/evaluations/evaluation.py --algo sac --algo-ckpt checkpoints/reward_norm/medium_mu_chi_b_1_guidance/sac/baseline/128_128_1e6steps_seed_3/best_model.zip --env-config configs/env/reward_norm/env_config_guidance_MR_medium_mu_chi_b_1.json --n-envs 8 --eval-episode-num 1000 --seed 3590 --save-result --result-file-save-name train_scripts/reward_norm/evaluations/results/medium_mu_chi_b_1_guidance/sac/baseline/seed_3.csv
python train_scripts/reward_norm/evaluations/evaluation.py --algo sac --algo-ckpt checkpoints/reward_norm/medium_mu_chi_b_1_guidance/sac/baseline/128_128_1e6steps_seed_4/best_model.zip --env-config configs/env/reward_norm/env_config_guidance_MR_medium_mu_chi_b_1.json --n-envs 8 --eval-episode-num 1000 --seed 6704 --save-result --result-file-save-name train_scripts/reward_norm/evaluations/results/medium_mu_chi_b_1_guidance/sac/baseline/seed_4.csv
python train_scripts/reward_norm/evaluations/evaluation.py --algo sac --algo-ckpt checkpoints/reward_norm/medium_mu_chi_b_1_guidance/sac/baseline/128_128_1e6steps_seed_5/best_model.zip --env-config configs/env/reward_norm/env_config_guidance_MR_medium_mu_chi_b_1.json --n-envs 8 --eval-episode-num 1000 --seed 5008 --save-result --result-file-save-name train_scripts/reward_norm/evaluations/results/medium_mu_chi_b_1_guidance/sac/baseline/seed_5.csv

# reward_scaling_cluster

