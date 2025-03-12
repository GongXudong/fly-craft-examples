#!/bin/bash


# baseline


# rl_bc iter_1
python train_scripts/reward_norm/evaluations/evaluation.py --algo ppo --algo-ckpt checkpoints/IRPO/rl/guidance_law_mode/iter_1/128_128_2e8steps_lambda_1e-3_1/best_model.zip --env-config configs/env/reward_norm/env_config_guidance_MR_medium_mu_chi_b_1.json --n-envs 8 --eval-episode-num 1000 --seed 48793 --save-result --result-file-save-name train_scripts/reward_norm/evaluations/results/medium_guidance/ppo/rl_bc/iter_1_seed_1.csv