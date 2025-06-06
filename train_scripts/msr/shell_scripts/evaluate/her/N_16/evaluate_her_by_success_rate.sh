#!/bin/bash

# bash train_scripts/disc/shell_scripts/evaluate/her/N_16/evaluate_her_by_success_rate.sh &> tmp_her_res.txt





#---------------------------------------- epsilon = [1.0, 0.3, 0.3] -------------------------------------------------------------
# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001  N = 16
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 6745 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 1233 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 6706 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 4305 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 2341145 --n-envs 32 --n-eval-episode 1000

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.01  N = 16
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_0_01_N_16/128_128_seed_1.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 631 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_0_01_N_16/128_128_seed_2.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 281334 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_0_01_N_16/128_128_seed_3.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 6125 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_0_01_N_16/128_128_seed_4.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 2745 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_0_01_N_16/128_128_seed_5.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 71151 --n-envs 32 --n-eval-episode 1000

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.1  N = 16
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_0_1_N_16/128_128_seed_1.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 214 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_0_1_N_16/128_128_seed_2.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 416 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_0_1_N_16/128_128_seed_3.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 16437 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_0_1_N_16/128_128_seed_4.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 17134 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_0_1_N_16/128_128_seed_5.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 137814 --n-envs 32 --n-eval-episode 1000

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 1.0  N = 16
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_1_N_16/128_128_seed_1.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 1768453 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_1_N_16/128_128_seed_2.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 13246 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_1_N_16/128_128_seed_3.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 7245 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_1_N_16/128_128_seed_4.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 137753 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_1_reg_1_N_16/128_128_seed_5.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 5648 --n-envs 32 --n-eval-episode 1000


#---------------------------------------- epsilon = [5.0, 1.5, 1.5] -------------------------------------------------------------
# epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0.001  N = 16
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_0_001_N_16/128_128_seed_1.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 6702 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_0_001_N_16/128_128_seed_2.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 2456 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_0_001_N_16/128_128_seed_3.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 8476 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_0_001_N_16/128_128_seed_4.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 38786 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_0_001_N_16/128_128_seed_5.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 555 --n-envs 32 --n-eval-episode 1000

# epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0.01  N = 16
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_0_01_N_16/128_128_seed_1.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 5585 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_0_01_N_16/128_128_seed_2.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 5892 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_0_01_N_16/128_128_seed_3.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 98254 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_0_01_N_16/128_128_seed_4.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 154731 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_0_01_N_16/128_128_seed_5.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 16534875 --n-envs 32 --n-eval-episode 1000

# # epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0.1  N = 16
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_0_1_N_16/128_128_seed_1.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 245895 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_0_1_N_16/128_128_seed_2.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 8512 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_0_1_N_16/128_128_seed_3.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 4136 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_0_1_N_16/128_128_seed_4.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 18741 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_0_1_N_16/128_128_seed_5.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 55500436 --n-envs 32 --n-eval-episode 1000

# # epsilon = [5.0, 1.5, 1.5]  regularization_strength = 1  N = 16
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_1_N_16/128_128_seed_1.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 25436 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_1_N_16/128_128_seed_2.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 2489 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_1_N_16/128_128_seed_3.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 245898 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_1_N_16/128_128_seed_4.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 13475648 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_0_5_reg_1_N_16/128_128_seed_5.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 3457621 --n-envs 32 --n-eval-episode 1000

#---------------------------------------- epsilon = [10.0, 3.0, 3.0] -------------------------------------------------------------
# epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.001  N = 16
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_0_001_N_16/128_128_seed_1.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 5004 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_0_001_N_16/128_128_seed_2.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 45655 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_0_001_N_16/128_128_seed_3.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 1454321 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_0_001_N_16/128_128_seed_4.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 76548 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_0_001_N_16/128_128_seed_5.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 324521 --n-envs 32 --n-eval-episode 1000

# epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.01  N = 16
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_0_01_N_16/128_128_seed_1.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 476 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_0_01_N_16/128_128_seed_2.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 652 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_0_01_N_16/128_128_seed_3.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 73125 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_0_01_N_16/128_128_seed_4.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 143657 --n-envs 32 --n-eval-episode 1000
python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_0_01_N_16/128_128_seed_5.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 829 --n-envs 32 --n-eval-episode 1000

# # epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.1  N = 16
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_0_1_N_16/128_128_seed_1.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 92546 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_0_1_N_16/128_128_seed_2.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 3145 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_0_1_N_16/128_128_seed_3.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 852 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_0_1_N_16/128_128_seed_4.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 31476 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_0_1_N_16/128_128_seed_5.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 1286004 --n-envs 32 --n-eval-episode 1000

# # epsilon = [10.0, 3.0, 3.0]  regularization_strength = 1  N = 16
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_1_N_16/128_128_seed_1.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 90167 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_1_N_16/128_128_seed_2.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 325 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_1_N_16/128_128_seed_3.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 7245 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_1_N_16/128_128_seed_4.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 28456 --n-envs 32 --n-eval-episode 1000
# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --algo-config-file configs/train/disc/her/medium2/epsilon_1_reg_1_N_16/128_128_seed_5.json --env-config-file configs/env/D2D/env_config_for_sac_medium2_b_05.json --algo sac --seed 1385007 --n-envs 32 --n-eval-episode 1000
