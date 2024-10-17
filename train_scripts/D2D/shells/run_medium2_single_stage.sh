#!/bin/bash

python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name configs/train/D2D/sac/medium2/b_05/sac_config_10hz_128_128_1.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name configs/train/D2D/sac/medium2/b_05/sac_config_10hz_128_128_2.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name configs/train/D2D/sac/medium2/b_05/sac_config_10hz_128_128_3.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name configs/train/D2D/sac/medium2/b_05/sac_config_10hz_128_128_4.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name configs/train/D2D/sac/medium2/b_05/sac_config_10hz_128_128_5.json
