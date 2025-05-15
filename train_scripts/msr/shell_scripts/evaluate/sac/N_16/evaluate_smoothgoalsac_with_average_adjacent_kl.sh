#!/bin/bash



# algo: epsilon 0.0 reg 0.0  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/rl_single/sac_medium_128_128_1e6steps_loss_{0}_singleRL --algo-seeds 1 2 3 4 5 --algo-flag-str SAC --algo-epsilon 0.0 --algo-reg 0.0 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_reg_0_noise_0_01.csv

# algo: epsilon 0.0 reg 0.0  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/rl_single/sac_medium_128_128_1e6steps_loss_{0}_singleRL --algo-seeds 1 2 3 4 5 --algo-flag-str SAC --algo-epsilon 0.0 --algo-reg 0.0 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_reg_0_noise_0_1.csv

# algo: epsilon 0.0 reg 0.0  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/rl_single/sac_medium_128_128_1e6steps_loss_{0}_singleRL --algo-seeds 1 2 3 4 5 --algo-flag-str SAC --algo-epsilon 0.0 --algo-reg 0.0 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_reg_0_noise_1.csv


#--------------------------------------------------------------------SmoothGoalSAC epsilon=0.01--------------------------------------------------------------

# algo: epsilon 0.01 reg 0.0001  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_01_reg_0_0001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.01 --algo-reg 0.0001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_01_reg_0_0001_N_16_noise_0_01.csv

# algo: epsilon 0.01 reg 0.0001  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_01_reg_0_0001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.01 --algo-reg 0.0001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_01_reg_0_0001_N_16_noise_0_1.csv

# algo: epsilon 0.01 reg 0.0001  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_01_reg_0_0001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.01 --algo-reg 0.0001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_01_reg_0_0001_N_16_noise_1.csv


# algo: epsilon 0.01 reg 0.001  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_01_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.01 --algo-reg 0.001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_01_reg_0_001_N_16_noise_0_01.csv

# algo: epsilon 0.01 reg 0.001  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_01_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.01 --algo-reg 0.001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_01_reg_0_001_N_16_noise_0_1.csv

# algo: epsilon 0.01 reg 0.001  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_01_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.01 --algo-reg 0.001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_01_reg_0_001_N_16_noise_1.csv


# algo: epsilon 0.01 reg 0.01  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_01_reg_0_01_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.01 --algo-reg 0.01 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_01_reg_0_01_N_16_noise_0_01.csv

# algo: epsilon 0.01 reg 0.01  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_01_reg_0_01_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.01 --algo-reg 0.01 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_01_reg_0_01_N_16_noise_0_1.csv

# algo: epsilon 0.01 reg 0.01  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_01_reg_0_01_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.01 --algo-reg 0.01 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_01_reg_0_01_N_16_noise_1.csv


# algo: epsilon 0.01 reg 0.1  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_01_reg_0_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.01 --algo-reg 0.1 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_01_reg_0_1_N_16_noise_0_01.csv

# algo: epsilon 0.01 reg 0.1  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_01_reg_0_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.01 --algo-reg 0.1 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_01_reg_0_1_N_16_noise_0_1.csv

# algo: epsilon 0.01 reg 0.1  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_01_reg_0_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.01 --algo-reg 0.1 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_01_reg_0_1_N_16_noise_1.csv


# algo: epsilon 0.01 reg 1.0  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_01_reg_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.01 --algo-reg 1.0 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_01_reg_1_N_16_noise_0_01.csv

# algo: epsilon 0.01 reg 1.0  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_01_reg_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.01 --algo-reg 1.0 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_01_reg_1_N_16_noise_0_1.csv

# algo: epsilon 0.01 reg 1.0  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_01_reg_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.01 --algo-reg 1.0 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_01_reg_1_N_16_noise_1.csv


#--------------------------------------------------------------------SmoothGoalSAC epsilon=0.1--------------------------------------------------------------

# algo: epsilon 0.1 reg 0.0001  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_1_reg_0_0001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.0001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_1_reg_0_0001_N_16_noise_0_01.csv

# algo: epsilon 0.1 reg 0.0001  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_1_reg_0_0001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.0001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_1_reg_0_0001_N_16_noise_0_1.csv

# algo: epsilon 0.1 reg 0.0001  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_1_reg_0_0001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.0001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_1_reg_0_0001_N_16_noise_1.csv


# algo: epsilon 0.1 reg 0.001  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_1_reg_0_001_N_16_noise_0_01.csv

# algo: epsilon 0.1 reg 0.001  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_1_reg_0_001_N_16_noise_0_1.csv

# algo: epsilon 0.1 reg 0.001  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_1_reg_0_001_N_16_noise_1.csv


# algo: epsilon 0.1 reg 0.01  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_1_reg_0_01_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.01 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_1_reg_0_01_N_16_noise_0_01.csv

# algo: epsilon 0.1 reg 0.01  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_1_reg_0_01_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.01 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_1_reg_0_01_N_16_noise_0_1.csv

# algo: epsilon 0.1 reg 0.01  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_1_reg_0_01_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.01 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_1_reg_0_01_N_16_noise_1.csv


# algo: epsilon 0.1 reg 0.1  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_1_reg_0_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.1 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_1_reg_0_1_N_16_noise_0_01.csv

# algo: epsilon 0.1 reg 0.1  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_1_reg_0_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.1 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_1_reg_0_1_N_16_noise_0_1.csv

# algo: epsilon 0.1 reg 0.1  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_1_reg_0_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.1 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_1_reg_0_1_N_16_noise_1.csv


# algo: epsilon 0.1 reg 1.0  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_1_reg_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 1.0 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_1_reg_1_N_16_noise_0_01.csv

# algo: epsilon 0.1 reg 1.0  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_1_reg_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 1.0 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_1_reg_1_N_16_noise_0_1.csv

# algo: epsilon 0.1 reg 1.0  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_0_1_reg_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 1.0 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_0_1_reg_1_N_16_noise_1.csv



# algo: epsilon 0.1 reg 0.0001  beta = 0.0001  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/beta_0_0001/epsilon_0_1_reg_0_0001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.0001 --algo-reg-beta 0.0001 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0_0001/sac_epsilon_0_1_reg_0_0001_beta_0_0001_N_16_noise_0_01.csv

# algo: epsilon 0.1 reg 0.0001  beta = 0.0001  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/beta_0_0001/epsilon_0_1_reg_0_0001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.0001 --algo-reg-beta 0.0001 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0_0001/sac_epsilon_0_1_reg_0_0001_beta_0_0001_N_16_noise_0_1.csv

# algo: epsilon 0.1 reg 0.0001  beta = 0.0001  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/beta_0_0001/epsilon_0_1_reg_0_0001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.0001 --algo-reg-beta 0.0001 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0_0001/sac_epsilon_0_1_reg_0_0001_beta_0_0001_N_16_noise_1.csv




# algo: epsilon 0.1 reg 0.001  beta = 0.0001  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/beta_0_0001/epsilon_0_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.001 --algo-reg-beta 0.0001 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0_0001/sac_epsilon_0_1_reg_0_001_beta_0_0001_N_16_noise_0_01.csv

# algo: epsilon 0.1 reg 0.001  beta = 0.0001  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/beta_0_0001/epsilon_0_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.001 --algo-reg-beta 0.0001 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0_0001/sac_epsilon_0_1_reg_0_001_beta_0_0001_N_16_noise_0_1.csv

# algo: epsilon 0.1 reg 0.001  beta = 0.0001  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/beta_0_0001/epsilon_0_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.001 --algo-reg-beta 0.0001 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0_0001/sac_epsilon_0_1_reg_0_001_beta_0_0001_N_16_noise_1.csv


# algo: epsilon 0.1 reg 0.001  beta = 0.001  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/beta_0_001/epsilon_0_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.001 --algo-reg-beta 0.001 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0_001/sac_epsilon_0_1_reg_0_001_beta_0_001_N_16_noise_0_01.csv

# algo: epsilon 0.1 reg 0.001  beta = 0.001  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/beta_0_001/epsilon_0_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.001 --algo-reg-beta 0.001 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0_001/sac_epsilon_0_1_reg_0_001_beta_0_001_N_16_noise_0_1.csv

# algo: epsilon 0.1 reg 0.001  beta = 0.001  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/beta_0_001/epsilon_0_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.001 --algo-reg-beta 0.001 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0_001/sac_epsilon_0_1_reg_0_001_beta_0_001_N_16_noise_1.csv


# algo: epsilon 0.1 reg 0.001  beta = 0.01  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/beta_0_01/epsilon_0_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.001 --algo-reg-beta 0.01 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0_01/sac_epsilon_0_1_reg_0_001_beta_0_01_N_16_noise_0_01.csv

# algo: epsilon 0.1 reg 0.001  beta = 0.01  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/beta_0_01/epsilon_0_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.001 --algo-reg-beta 0.01 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0_01/sac_epsilon_0_1_reg_0_001_beta_0_01_N_16_noise_0_1.csv

# algo: epsilon 0.1 reg 0.001  beta = 0.01  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/beta_0_01/epsilon_0_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.001 --algo-reg-beta 0.01 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0_01/sac_epsilon_0_1_reg_0_001_beta_0_01_N_16_noise_1.csv


# algo: epsilon 0.1 reg 0.001  beta = 0.1  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.001 --algo-reg-beta 0.1 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0_1/sac_epsilon_0_1_reg_0_001_beta_0_1_N_16_noise_0_01.csv

# algo: epsilon 0.1 reg 0.001  beta = 0.1  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.001 --algo-reg-beta 0.1 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0_1/sac_epsilon_0_1_reg_0_001_beta_0_1_N_16_noise_0_1.csv

# algo: epsilon 0.1 reg 0.001  beta = 0.1  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/beta_0_1/epsilon_0_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 0.1 --algo-reg 0.001 --algo-reg-beta 0.1 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0_1/sac_epsilon_0_1_reg_0_001_beta_0_1_N_16_noise_1.csv


#--------------------------------------------------------------------SmoothGoalSAC epsilon=1.0--------------------------------------------------------------

# algo: epsilon 1.0 reg 0.0001  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_1_reg_0_0001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 1.0 --algo-reg 0.0001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_1_reg_0_0001_N_16_noise_0_01.csv

# algo: epsilon 1.0 reg 0.0001  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_1_reg_0_0001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 1.0 --algo-reg 0.0001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_1_reg_0_0001_N_16_noise_0_1.csv

# algo: epsilon 1.0 reg 0.0001  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_1_reg_0_0001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 1.0 --algo-reg 0.0001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_1_reg_0_0001_N_16_noise_1.csv


# algo: epsilon 1.0 reg 0.001  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 1.0 --algo-reg 0.001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_1_reg_0_001_N_16_noise_0_01.csv

# algo: epsilon 1.0 reg 0.001  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 1.0 --algo-reg 0.001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_1_reg_0_001_N_16_noise_0_1.csv

# algo: epsilon 1.0 reg 0.001  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_1_reg_0_001_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 1.0 --algo-reg 0.001 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_1_reg_0_001_N_16_noise_1.csv


# algo: epsilon 1.0 reg 0.01  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_1_reg_0_01_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 1.0 --algo-reg 0.01 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_1_reg_0_01_N_16_noise_0_01.csv

# algo: epsilon 1.0 reg 0.01  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_1_reg_0_01_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 1.0 --algo-reg 0.01 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_1_reg_0_01_N_16_noise_0_1.csv

# algo: epsilon 1.0 reg 0.01  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_1_reg_0_01_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 1.0 --algo-reg 0.01 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_1_reg_0_01_N_16_noise_1.csv


# algo: epsilon 1.0 reg 0.1  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_1_reg_0_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 1.0 --algo-reg 0.1 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_1_reg_0_1_N_16_noise_0_01.csv

# algo: epsilon 1.0 reg 0.1  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_1_reg_0_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 1.0 --algo-reg 0.1 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_1_reg_0_1_N_16_noise_0_1.csv

# algo: epsilon 1.0 reg 0.1  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_1_reg_0_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 1.0 --algo-reg 0.1 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_1_reg_0_1_N_16_noise_1.csv


# algo: epsilon 1.0 reg 1.0  eval: epsilon 0.01
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_1_reg_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 1.0 --algo-reg 1.0 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.01 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_1_reg_1_N_16_noise_0_01.csv

# algo: epsilon 1.0 reg 1.0  eval: epsilon 0.1
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_1_reg_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 1.0 --algo-reg 1.0 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 0.1 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_1_reg_1_N_16_noise_0_1.csv

# algo: epsilon 1.0 reg 1.0  eval: epsilon 1.0
python train_scripts/disc/evaluate/evaluate_sac_with_average_adjacent_KL.py --env-config configs/env/D2D/env_config_for_ppo_medium_b_05.json --env-flag-str Medium-05 --algo-class SmoothGoalSAC --algo-ckpt-dir checkpoints/disc/medium/sac/epsilon_1_reg_1_N_16/128_128_1e6steps_seed_{0} --algo-seeds 1 2 3 4 5 --algo-flag-str SmoothGoalSAC --algo-epsilon 1.0 --algo-reg 1.0 --algo-reg-beta 0.0 --evaluate-dg-num 100 --evaluate-noise-base 10.0 3.0 3.0 --evaluate-noise-multiplier 1.0 --evaluate-adjacent-num 5 --res-file-save-name train_scripts/disc/plots/sac_medium/results/beta_0/sac_epsilon_1_reg_1_N_16_noise_1.csv
