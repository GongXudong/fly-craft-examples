#!/bin/bash

python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name configs/train/D2D/sac/medium2/b_05/sac_config_10hz_128_128_1.json
python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name configs/train/D2D/sac/medium2/b_05/sac_config_10hz_128_128_2.json
python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name configs/train/D2D/sac/medium2/b_05/sac_config_10hz_128_128_3.json
python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name configs/train/D2D/sac/medium2/b_05/sac_config_10hz_128_128_4.json
python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name configs/train/D2D/sac/medium2/b_05/sac_config_10hz_128_128_5.json



# relative_hard her two_stage  b=2 --> b=0.5
python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name configs/train/D2D/relative_sac/two_stage_relative_hard_b_2_b_05/sac_config_10hz_128_128_1.json