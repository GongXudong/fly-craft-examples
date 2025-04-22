#!/bin/bash

# python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/distance_threshold_0_005/her/b_2/sac_seed_1.json
# python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/distance_threshold_0_005/her/b_1/sac_seed_1.json
# python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/distance_threshold_0_005/her/b_05/sac_seed_1.json

# python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/distance_threshold_0_005/her/two_stage_0.01_b_2_b_05/sac_seed_1.json


# python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08_step_80/distance_threshold_0_01/her/b_1/sac_seed_1.json

# python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08_step_80/distance_threshold_0_01/her/b_2/sac_seed_1.json

# python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08_step_80/distance_threshold_0_01/her/b_05/sac_seed_1.json

# her---her---her
# n_substeps = 1  166
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/n_substeps_1/b_05/sac_seed_1.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/n_substeps_1/b_05/sac_seed_2.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/n_substeps_1/b_05/sac_seed_3.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/n_substeps_1/b_05/sac_seed_4.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/n_substeps_1/b_05/sac_seed_5.json

# n_substeps = 3   251
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/n_substeps_3/b_05/sac_seed_1.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/n_substeps_3/b_05/sac_seed_2.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/n_substeps_3/b_05/sac_seed_3.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/n_substeps_3/b_05/sac_seed_4.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/n_substeps_3/b_05/sac_seed_5.json

# n_substeps = 5    251
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/n_substeps_5/b_05/sac_seed_1.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/n_substeps_5/b_05/sac_seed_2.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/n_substeps_5/b_05/sac_seed_3.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/n_substeps_5/b_05/sac_seed_4.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/n_substeps_5/b_05/sac_seed_5.json

# n_substeps = 3 to n_substeps = 1   100
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/two_stage_n_substeps_3_to_n_substeps_1/b_05/sac_seed_1.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/two_stage_n_substeps_3_to_n_substeps_1/b_05/sac_seed_2.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/two_stage_n_substeps_3_to_n_substeps_1/b_05/sac_seed_3.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/two_stage_n_substeps_3_to_n_substeps_1/b_05/sac_seed_4.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/two_stage_n_substeps_3_to_n_substeps_1/b_05/sac_seed_5.json



# n_substeps = 5 to n_substeps = 1   110
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/two_stage_n_substeps_5_to_n_substeps_1/b_05/sac_seed_1.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/two_stage_n_substeps_5_to_n_substeps_1/b_05/sac_seed_2.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/two_stage_n_substeps_5_to_n_substeps_1/b_05/sac_seed_3.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/two_stage_n_substeps_5_to_n_substeps_1/b_05/sac_seed_4.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/two_stage_n_substeps_5_to_n_substeps_1/b_05/sac_seed_5.json




# sac---sac---sac
#sparse 
#n_substeps = 1  100
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/n_substeps_1/b_1/sac_seed_1.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/n_substeps_1/b_1/sac_seed_2.json
CUDA_VISIBLE_DEVICES=1 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/n_substeps_1/b_1/sac_seed_3.json
CUDA_VISIBLE_DEVICES=2 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/n_substeps_1/b_1/sac_seed_4.json
CUDA_VISIBLE_DEVICES=3 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/n_substeps_1/b_1/sac_seed_5.json


#n_substeps = 3  110
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/n_substeps_3/b_1/sac_seed_1.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/n_substeps_3/b_1/sac_seed_2.json
CUDA_VISIBLE_DEVICES=1 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/n_substeps_3/b_1/sac_seed_3.json
CUDA_VISIBLE_DEVICES=2 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/n_substeps_3/b_1/sac_seed_4.json
CUDA_VISIBLE_DEVICES=3 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/n_substeps_3/b_1/sac_seed_5.json

#n_substeps = 5 251
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/n_substeps_5/b_1/sac_seed_1.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/n_substeps_5/b_1/sac_seed_2.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/n_substeps_5/b_1/sac_seed_3.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/n_substeps_5/b_1/sac_seed_4.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/n_substeps_5/b_1/sac_seed_5.json

# n_substeps = 3 to n_substeps = 1 251
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/two_stage_n_substeps_3_to_n_substeps_1/b_1/sac_seed_1.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/two_stage_n_substeps_3_to_n_substeps_1/b_1/sac_seed_2.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/two_stage_n_substeps_3_to_n_substeps_1/b_1/sac_seed_3.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/two_stage_n_substeps_3_to_n_substeps_1/b_1/sac_seed_4.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/two_stage_n_substeps_3_to_n_substeps_1/b_1/sac_seed_5.json


# n_substeps = 5 to n_substeps = 1  166
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/two_stage_n_substeps_5_to_n_substeps_1/b_1/sac_seed_1.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/two_stage_n_substeps_5_to_n_substeps_1/b_1/sac_seed_2.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/two_stage_n_substeps_5_to_n_substeps_1/b_1/sac_seed_3.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/two_stage_n_substeps_5_to_n_substeps_1/b_1/sac_seed_4.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/sparse/sac/two_stage_n_substeps_5_to_n_substeps_1/b_1/sac_seed_5.json


# F2F sac  dense !!!!
#n_substeps = 1  100
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_1/b_1/sac_seed_1.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_1/b_1/sac_seed_2.json
CUDA_VISIBLE_DEVICES=1 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_1/b_1/sac_seed_3.json
CUDA_VISIBLE_DEVICES=2 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_1/b_1/sac_seed_4.json
CUDA_VISIBLE_DEVICES=3 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_1/b_1/sac_seed_5.json


#n_substeps = 3  110
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_3/b_1/sac_seed_1.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_3/b_1/sac_seed_2.json
CUDA_VISIBLE_DEVICES=1 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_3/b_1/sac_seed_3.json
CUDA_VISIBLE_DEVICES=2 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_3/b_1/sac_seed_4.json
CUDA_VISIBLE_DEVICES=3 python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_3/b_1/sac_seed_5.json

#n_substeps = 5 251
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_5/b_1/sac_seed_1.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_5/b_1/sac_seed_2.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_5/b_1/sac_seed_3.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_5/b_1/sac_seed_4.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/n_substeps_5/b_1/sac_seed_5.json

# n_substeps = 3 to n_substeps = 1 251
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/two_stage_n_substeps_3_to_n_substeps_1/b_1/sac_seed_1.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/two_stage_n_substeps_3_to_n_substeps_1/b_1/sac_seed_2.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/two_stage_n_substeps_3_to_n_substeps_1/b_1/sac_seed_3.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/two_stage_n_substeps_3_to_n_substeps_1/b_1/sac_seed_4.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/two_stage_n_substeps_3_to_n_substeps_1/b_1/sac_seed_5.json


# n_substeps = 5 to n_substeps = 1  166
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/two_stage_n_substeps_5_to_n_substeps_1/b_1/sac_seed_1.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/two_stage_n_substeps_5_to_n_substeps_1/b_1/sac_seed_2.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/two_stage_n_substeps_5_to_n_substeps_1/b_1/sac_seed_3.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/two_stage_n_substeps_5_to_n_substeps_1/b_1/sac_seed_4.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_panda_reach.py --config-file-name=configs/train/D2D/pand_reach_dense/goal_range_08/distance_0_01_change_z_scope/dense/sac/two_stage_n_substeps_5_to_n_substeps_1/b_1/sac_seed_5.json
