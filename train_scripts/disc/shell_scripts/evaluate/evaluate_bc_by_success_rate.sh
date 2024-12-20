#!/bin/bash

# bash train_scripts/disc/shell_scripts/evaluate/evaluate_bc_by_success_rate.sh &> tmp_bc_res.txt

#---------------------------------------- epsilon = [1.0, 0.3, 0.3] -------------------------------------------------------------
# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0/128_128_seed_1.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 11 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0/128_128_seed_2.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 2 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0/128_128_seed_3.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 123 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0/128_128_seed_4.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 875 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0/128_128_seed_5.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 97738 --n-envs 32 --n-eval-episode 1000

# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0_001/128_128_seed_1.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 23 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0_001/128_128_seed_2.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 298 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0_001/128_128_seed_3.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 23 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0_001/128_128_seed_4.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 385 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0_001/128_128_seed_5.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 6922 --n-envs 32 --n-eval-episode 1000

# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.01
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0_01/128_128_seed_1.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 234 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0_01/128_128_seed_2.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 1261 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0_01/128_128_seed_3.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 672 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0_01/128_128_seed_4.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 2346 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0_01/128_128_seed_5.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 322458 --n-envs 32 --n-eval-episode 1000

# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.1
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0_1/128_128_seed_1.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 76 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0_1/128_128_seed_2.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 333 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0_1/128_128_seed_3.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 231 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0_1/128_128_seed_4.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 9 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_0_1/128_128_seed_5.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 43 --n-envs 32 --n-eval-episode 1000

# # epsilon = [1.0, 0.3, 0.3]  regularization_strength = 1.0
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_1/128_128_seed_1.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 8 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_1/128_128_seed_2.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 34 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_1/128_128_seed_3.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 122 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_1/128_128_seed_4.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 231 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_1_reg_1/128_128_seed_5.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 87 --n-envs 32 --n-eval-episode 1000


# #---------------------------------------- epsilon = [5.0, 1.5, 1.5] -------------------------------------------------------------
# # epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_5_reg_0/128_128_seed_1.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 6835 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_5_reg_0/128_128_seed_2.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 1234 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_5_reg_0/128_128_seed_3.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 3642 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_5_reg_0/128_128_seed_4.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 427 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_5_reg_0/128_128_seed_5.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 32528 --n-envs 32 --n-eval-episode 1000

# # epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0.001
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_5_reg_0_001/128_128_seed_1.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 243 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_5_reg_0_001/128_128_seed_2.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 613 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_5_reg_0_001/128_128_seed_3.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 24358 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_5_reg_0_001/128_128_seed_4.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 125754 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_0_5_reg_0_001/128_128_seed_5.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 1264134 --n-envs 32 --n-eval-episode 1000


# #---------------------------------------- epsilon = [10.0, 3.0, 3.0] -------------------------------------------------------------
# # epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_1_reg_0/128_128_seed_1.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 235 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_1_reg_0/128_128_seed_2.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 1002 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_1_reg_0/128_128_seed_3.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 274 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_1_reg_0/128_128_seed_4.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 20365 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_1_reg_0/128_128_seed_5.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 34098 --n-envs 32 --n-eval-episode 1000

# # epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.001
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_1_reg_0_001/128_128_seed_1.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 1200 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_1_reg_0_001/128_128_seed_2.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 9112 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_1_reg_0_001/128_128_seed_3.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 20974 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_1_reg_0_001/128_128_seed_4.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 215 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/bc/medium2/epsilon_1_reg_0_001/128_128_seed_5.json --env-config-file configs/env/env_hard_config_for_sac.json --algo bc --seed 98006 --n-envs 32 --n-eval-episode 1000
