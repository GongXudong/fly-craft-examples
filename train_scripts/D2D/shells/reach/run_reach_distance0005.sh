#!/bin/bash

# python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/distance_threshold_0_005/her/b_2/sac_seed_1.json
# python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/distance_threshold_0_005/her/b_1/sac_seed_1.json
# python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/distance_threshold_0_005/her/b_05/sac_seed_1.json

# python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/distance_threshold_0_005/her/two_stage_0.01_b_2_b_05/sac_seed_1.json


python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08_step_80/distance_threshold_0_01/her/b_1/sac_seed_1.json

python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08_step_80/distance_threshold_0_01/her/b_2/sac_seed_1.json

python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08_step_80/distance_threshold_0_01/her/b_05/sac_seed_1.json