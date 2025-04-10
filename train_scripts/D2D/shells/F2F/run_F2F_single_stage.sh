#!/bin/bash

# skip3 1e6
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/skip_3/sac_config_10hz_128_128_1.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/skip_3/sac_config_10hz_128_128_2.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/skip_3/sac_config_10hz_128_128_3.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/skip_3/sac_config_10hz_128_128_4.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/skip_3/sac_config_10hz_128_128_5.json

# skip5 1e6
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/skip_5/sac_config_10hz_128_128_1.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/skip_5/sac_config_10hz_128_128_2.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/skip_5/sac_config_10hz_128_128_3.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/skip_5/sac_config_10hz_128_128_4.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/skip_5/sac_config_10hz_128_128_5.json

# skip 3 to skip 1  1e6
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/two_stage_skip_3_skip_1/sac_config_10hz_128_128_1.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/two_stage_skip_3_skip_1/sac_config_10hz_128_128_2.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/two_stage_skip_3_skip_1/sac_config_10hz_128_128_3.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/two_stage_skip_3_skip_1/sac_config_10hz_128_128_4.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/two_stage_skip_3_skip_1/sac_config_10hz_128_128_5.json


# skip 5 to skip 1  1e6
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/two_stage_skip_5_skip_1/sac_config_10hz_128_128_1.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/two_stage_skip_5_skip_1/sac_config_10hz_128_128_2.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/two_stage_skip_5_skip_1/sac_config_10hz_128_128_3.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/two_stage_skip_5_skip_1/sac_config_10hz_128_128_4.json
CUDA_VISIBLE_DEVICES=0 python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/F2F/relative_hard/b_05/two_stage_skip_5_skip_1/sac_config_10hz_128_128_5.json