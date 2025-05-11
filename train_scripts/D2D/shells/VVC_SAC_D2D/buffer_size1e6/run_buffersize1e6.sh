#!/bin/bash


# buffersize 1e6!!!! 
#baseline  b = 1   2e6   buffersize 1e6  242
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/2e6/buffer_size1e6/sac_config_10hz_128_128_1.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/2e6/buffer_size1e6/sac_config_10hz_128_128_2.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/2e6/buffer_size1e6/sac_config_10hz_128_128_3.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/2e6/buffer_size1e6/sac_config_10hz_128_128_4.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/2e6/buffer_size1e6/sac_config_10hz_128_128_5.json


# F2F skip 3 to skip 1 b=1 120
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_info_buferr.py --config-file-name=configs/train/D2D/F2F/medium/b_1/two_stage_skip_3_skip_1/2e6/buffer_size_1e6/sac_config_10hz_128_128_1.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_info_buferr.py --config-file-name=configs/train/D2D/F2F/medium/b_1/two_stage_skip_3_skip_1/2e6/buffer_size_1e6/sac_config_10hz_128_128_2.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_info_buferr.py --config-file-name=configs/train/D2D/F2F/medium/b_1/two_stage_skip_3_skip_1/2e6/buffer_size_1e6/sac_config_10hz_128_128_3.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_info_buferr.py --config-file-name=configs/train/D2D/F2F/medium/b_1/two_stage_skip_3_skip_1/2e6/buffer_size_1e6/sac_config_10hz_128_128_4.json
python train_scripts/D2D/train_with_rl_sac_her_multi_stages_info_buferr.py --config-file-name=configs/train/D2D/F2F/medium/b_1/two_stage_skip_3_skip_1/2e6/buffer_size_1e6/sac_config_10hz_128_128_5.json


#baseline  b = 1   2e6   buffersize 1e6 gamma 0.95 to 0.995 166
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/gamma/buffer_size1e6/sac_config_10hz_128_128_1.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/gamma/buffer_size1e6/sac_config_10hz_128_128_2.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/gamma/buffer_size1e6/sac_config_10hz_128_128_3.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/gamma/buffer_size1e6/sac_config_10hz_128_128_4.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/gamma/buffer_size1e6/sac_config_10hz_128_128_5.json


## easy 2 medium 2e6 buffer_size = 1e6  b = 1  10
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py  --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_1/2e6/easy2medium_buffer_size_1e6/sac_config_10hz_128_128_1.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py  --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_1/2e6/easy2medium_buffer_size_1e6/sac_config_10hz_128_128_2.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py  --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_1/2e6/easy2medium_buffer_size_1e6/sac_config_10hz_128_128_3.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py  --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_1/2e6/easy2medium_buffer_size_1e6/sac_config_10hz_128_128_4.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py  --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_1/2e6/easy2medium_buffer_size_1e6/sac_config_10hz_128_128_5.json


## reward devide b= 1  buffer_size = 1e6  140
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/reward_devide/medium/b_1/2e6/buffer_szie1e6/sac_config_10hz_128_128_1.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/reward_devide/medium/b_1/2e6/buffer_szie1e6/sac_config_10hz_128_128_2.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/reward_devide/medium/b_1/2e6/buffer_szie1e6/sac_config_10hz_128_128_3.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/reward_devide/medium/b_1/2e6/buffer_szie1e6/sac_config_10hz_128_128_4.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/reward_devide/medium/b_1/2e6/buffer_szie1e6/sac_config_10hz_128_128_5.json


## easy 2 medium b=0.5 buffersize 1e6 warmup 300
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_05/2e6/easy2medium_buffer_size_1e6_warmup_300/sac_config_10hz_128_128_1.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_05/2e6/easy2medium_buffer_size_1e6_warmup_300/sac_config_10hz_128_128_2.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_05/2e6/easy2medium_buffer_size_1e6_warmup_300/sac_config_10hz_128_128_3.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_05/2e6/easy2medium_buffer_size_1e6_warmup_300/sac_config_10hz_128_128_4.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_05/2e6/easy2medium_buffer_size_1e6_warmup_300/sac_config_10hz_128_128_5.json


## E2D buffer_size 1e6   aug4  transition 2e5  110
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_05_aug4/buffer_size1e6/num_transition2e5/sac_config_10hz_128_128_1.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_05_aug4/buffer_size1e6/num_transition2e5/sac_config_10hz_128_128_2.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_05_aug4/buffer_size1e6/num_transition2e5/sac_config_10hz_128_128_3.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_05_aug4/buffer_size1e6/num_transition2e5/sac_config_10hz_128_128_4.json
python  train_scripts/D2D/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_05_aug4/buffer_size1e6/num_transition2e5/sac_config_10hz_128_128_5.json