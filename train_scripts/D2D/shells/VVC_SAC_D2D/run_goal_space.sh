#!/bin/bash


# goal space easy to medium 
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/sac_config_10hz_128_128_1.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/sac_config_10hz_128_128_2.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/sac_config_10hz_128_128_3.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/sac_config_10hz_128_128_4.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/sac_config_10hz_128_128_5.json


# easy sac 
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/easy_sac/b_1/sac_config_10hz_128_128_1.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/easy_sac/b_1/sac_config_10hz_128_128_2.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/easy_sac/b_1/sac_config_10hz_128_128_3.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/easy_sac/b_1/sac_config_10hz_128_128_4.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/easy_sac/b_1/sac_config_10hz_128_128_5.json