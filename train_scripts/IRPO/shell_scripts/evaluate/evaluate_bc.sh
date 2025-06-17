#!/bin/bash 
# bash  train_scripts/IRPO/shell_scripts/evaluate/evaluate_bc.sh &> tmp_bc_successrate.txt

# bc MR 
#seed_3
python train_scripts/IRPO/evaluate/evaluate_policy_by_success_rate_on_specific_env.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-3/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_3.json --env-config configs/env/IRPO/env_hard_end2end_MR_config_for_ppo.json --algo bc --seed 981 --n-envs 64 --n-eval-episode 1000

#seed_4
python train_scripts/IRPO/evaluate/evaluate_policy_by_success_rate_on_specific_env.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-3/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_4.json --env-config configs/env/IRPO/env_hard_end2end_MR_config_for_ppo.json --algo bc --seed 891 --n-envs 64 --n-eval-episode 1000

#seed_5
python train_scripts/IRPO/evaluate/evaluate_policy_by_success_rate_on_specific_env.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-3/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_5.json --env-config configs/env/IRPO/env_hard_end2end_MR_config_for_ppo.json --algo bc --seed 97 --n-envs 64 --n-eval-episode 1000


# bc NMR   

#seed_1
python train_scripts/IRPO/evaluate/evaluate_policy_by_success_rate_on_specific_env.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-3/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_1.json --env-config configs/env/IRPO/env_hard_end2end_NMR_config_for_ppo.json --algo bc --seed 17 --n-envs 64 --n-eval-episode 1000

#seed_2
python train_scripts/IRPO/evaluate/evaluate_policy_by_success_rate_on_specific_env.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-3/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_2.json --env-config configs/env/IRPO/env_hard_end2end_NMR_config_for_ppo.json --algo bc --seed 916 --n-envs 64 --n-eval-episode 1000


#seed_3
python train_scripts/IRPO/evaluate/evaluate_policy_by_success_rate_on_specific_env.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-3/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_3.json --env-config configs/env/IRPO/env_hard_end2end_NMR_config_for_ppo.json --algo bc --seed 20 --n-envs 64 --n-eval-episode 1000

#seed_4
python train_scripts/IRPO/evaluate/evaluate_policy_by_success_rate_on_specific_env.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-3/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_4.json --env-config configs/env/IRPO/env_hard_end2end_NMR_config_for_ppo.json --algo bc --seed 64 --n-envs 64 --n-eval-episode 1000

#seed_5
python train_scripts/IRPO/evaluate/evaluate_policy_by_success_rate_on_specific_env.py --config-file-name configs/train/IRPO/end_to_end_mode/iter_1/MR/lambda_1e-3/ppo_bc_config_10hz_128_128_hard_lambda_1e-3_seed_5.json --env-config configs/env/IRPO/env_hard_end2end_NMR_config_for_ppo.json --algo bc --seed 917 --n-envs 64 --n-eval-episode 1000

