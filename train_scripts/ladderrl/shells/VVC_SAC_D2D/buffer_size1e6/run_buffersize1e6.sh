#!/bin/bash


# buffersize 1e6!!!! 
#baseline  b = 1   2e6   buffersize 1e6  242
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/2e6/buffer_size1e6/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/2e6/buffer_size1e6/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/2e6/buffer_size1e6/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/2e6/buffer_size1e6/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/2e6/buffer_size1e6/sac_config_10hz_128_128_5.json


# F2F skip 3 to skip 1 b=1 120 shutdown   seed 1 2 3 4 5 on 242
python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages_info_buferr.py --config-file-name=configs/train/D2D/F2F/medium/b_1/two_stage_skip_3_skip_1/2e6/buffer_size_1e6/sac_config_10hz_128_128_1.json
python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages_info_buferr.py --config-file-name=configs/train/D2D/F2F/medium/b_1/two_stage_skip_3_skip_1/2e6/buffer_size_1e6/sac_config_10hz_128_128_2.json
python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages_info_buferr.py --config-file-name=configs/train/D2D/F2F/medium/b_1/two_stage_skip_3_skip_1/2e6/buffer_size_1e6/sac_config_10hz_128_128_3.json
python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages_info_buferr.py --config-file-name=configs/train/D2D/F2F/medium/b_1/two_stage_skip_3_skip_1/2e6/buffer_size_1e6/sac_config_10hz_128_128_4.json
python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages_info_buferr.py --config-file-name=configs/train/D2D/F2F/medium/b_1/two_stage_skip_3_skip_1/2e6/buffer_size_1e6/sac_config_10hz_128_128_5.json


#baseline  b = 1   2e6   buffersize 1e6 gamma 0.95 to 0.995 166
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/gamma/buffer_size1e6/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/gamma/buffer_size1e6/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/gamma/buffer_size1e6/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/gamma/buffer_size1e6/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/gamma/buffer_size1e6/sac_config_10hz_128_128_5.json


## easy 2 medium 2e6 buffer_size = 1e6  b = 1  10
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py  --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_1/2e6/easy2medium_buffer_size_1e6/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py  --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_1/2e6/easy2medium_buffer_size_1e6/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py  --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_1/2e6/easy2medium_buffer_size_1e6/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py  --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_1/2e6/easy2medium_buffer_size_1e6/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py  --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_1/2e6/easy2medium_buffer_size_1e6/sac_config_10hz_128_128_5.json


## reward devide b= 1  buffer_size = 1e6  140
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/reward_devide/medium/b_1/2e6/buffer_szie1e6/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/reward_devide/medium/b_1/2e6/buffer_szie1e6/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/reward_devide/medium/b_1/2e6/buffer_szie1e6/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/reward_devide/medium/b_1/2e6/buffer_szie1e6/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/reward_devide/medium/b_1/2e6/buffer_szie1e6/sac_config_10hz_128_128_5.json


## easy 2 medium b=0.5 buffersize 1e6 warmup 300
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_05/2e6/easy2medium_buffer_size_1e6_warmup_300/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_05/2e6/easy2medium_buffer_size_1e6_warmup_300/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_05/2e6/easy2medium_buffer_size_1e6_warmup_300/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_05/2e6/easy2medium_buffer_size_1e6_warmup_300/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_05/2e6/easy2medium_buffer_size_1e6_warmup_300/sac_config_10hz_128_128_5.json


## E2D buffer_size 1e6   aug4  transition 2e5  110
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition2e5/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition2e5/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition2e5/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition2e5/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition2e5/sac_config_10hz_128_128_5.json


## E2D buffer_size 1e6   aug4  transition 1e6  251
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition1e6/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition1e6/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition1e6/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition1e6/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition1e6/sac_config_10hz_128_128_5.json

## D2D b=0.5 to b = 1  2e6   buffer_size 1e6   first on 251 second on 110
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/two_stage_b_05_b_1/buffer_size1e6/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/two_stage_b_05_b_1/buffer_size1e6/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/two_stage_b_05_b_1/buffer_size1e6/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/two_stage_b_05_b_1/buffer_size1e6/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/two_stage_b_05_b_1/buffer_size1e6/sac_config_10hz_128_128_5.json



# medium 2e6 without reset buffersize1e6 251
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/b_1/2e6/buffer_size1e6/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/b_1/2e6/buffer_size1e6/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/b_1/2e6/buffer_size1e6/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/b_1/2e6/buffer_size1e6/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/b_1/2e6/buffer_size1e6/sac_config_10hz_128_128_5.json

## easy2medium2hard b=1 buffersize 1e6  166
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_hard/evaluate_medium_hard_on_hard/b_1/3e6/buffer_size1e6/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_hard/evaluate_medium_hard_on_hard/b_1/3e6/buffer_size1e6/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_hard/evaluate_medium_hard_on_hard/b_1/3e6/buffer_size1e6/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_hard/evaluate_medium_hard_on_hard/b_1/3e6/buffer_size1e6/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_hard/evaluate_medium_hard_on_hard/b_1/3e6/buffer_size1e6/sac_config_10hz_128_128_5.json

## medium 2 medium b=1 warmup  300  buffersize_1e6 242
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/2e6/buffer_size1e6_warmup_300/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/2e6/buffer_size1e6_warmup_300/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/2e6/buffer_size1e6_warmup_300/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/2e6/buffer_size1e6_warmup_300/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/baseline/two_stage_b_1_b_1/2e6/buffer_size1e6_warmup_300/sac_config_10hz_128_128_5.json


## easy2medium2medium2 b=1 buffersize 1e6  166
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_medium2/evaluate_medium_medium2_on_medium2/b_1/3e6/buffer_size1e6/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_medium2/evaluate_medium_medium2_on_medium2/b_1/3e6/buffer_size1e6/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_medium2/evaluate_medium_medium2_on_medium2/b_1/3e6/buffer_size1e6/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_medium2/evaluate_medium_medium2_on_medium2/b_1/3e6/buffer_size1e6/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_medium2/evaluate_medium_medium2_on_medium2/b_1/3e6/buffer_size1e6/sac_config_10hz_128_128_5.json


## one stage b = 1 2e6 buffersize 1e6 10
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/b_1/2e6/one_stage_buffer_szie1e6/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/b_1/2e6/one_stage_buffer_szie1e6/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/b_1/2e6/one_stage_buffer_szie1e6/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/b_1/2e6/one_stage_buffer_szie1e6/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/b_1/2e6/one_stage_buffer_szie1e6/sac_config_10hz_128_128_5.json


## reward_devide buffersize 1e6 warmup 300  140
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/reward_devide/medium/b_1/2e6/buffer_size1e6_warmup300/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/reward_devide/medium/b_1/2e6/buffer_size1e6_warmup300/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/reward_devide/medium/b_1/2e6/buffer_size1e6_warmup300/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/reward_devide/medium/b_1/2e6/buffer_size1e6_warmup300/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/reward_devide/medium/b_1/2e6/buffer_size1e6_warmup300/sac_config_10hz_128_128_5.json


# F2F skip 3 to skip 1 b=1  warmup 300  166
python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages_info_buferr.py --config-file-name=configs/train/D2D/F2F/medium/b_1/two_stage_skip_3_skip_1/2e6/buffer_size_1e6_warmup300/sac_config_10hz_128_128_1.json
python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages_info_buferr.py --config-file-name=configs/train/D2D/F2F/medium/b_1/two_stage_skip_3_skip_1/2e6/buffer_size_1e6_warmup300/sac_config_10hz_128_128_2.json
python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages_info_buferr.py --config-file-name=configs/train/D2D/F2F/medium/b_1/two_stage_skip_3_skip_1/2e6/buffer_size_1e6_warmup300/sac_config_10hz_128_128_3.json
python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages_info_buferr.py --config-file-name=configs/train/D2D/F2F/medium/b_1/two_stage_skip_3_skip_1/2e6/buffer_size_1e6_warmup300/sac_config_10hz_128_128_4.json
python train_scripts/ladderrl/train_with_rl_sac_her_multi_stages_info_buferr.py --config-file-name=configs/train/D2D/F2F/medium/b_1/two_stage_skip_3_skip_1/2e6/buffer_size_1e6_warmup300/sac_config_10hz_128_128_5.json


# easy2medium  buffer_size 1e6 warmup300  110
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_1/2e6/easy2medium_buffer_size_1e6_warmup300/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_1/2e6/easy2medium_buffer_size_1e6_warmup300/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_1/2e6/easy2medium_buffer_size_1e6_warmup300/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_1/2e6/easy2medium_buffer_size_1e6_warmup300/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium/evaluate_medium/b_1/2e6/easy2medium_buffer_size_1e6_warmup300/sac_config_10hz_128_128_5.json


## D2D b= 0.5  to b = 1 buffersize  1e6   warmup 300   100
CUDA_VISIBLE_DEVICES=0 python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/two_stage_b_05_b_1/buffer_size1e6_warmup300/sac_config_10hz_128_128_1.json
CUDA_VISIBLE_DEVICES=0 python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/two_stage_b_05_b_1/buffer_size1e6_warmup300/sac_config_10hz_128_128_2.json
CUDA_VISIBLE_DEVICES=0 python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/two_stage_b_05_b_1/buffer_size1e6_warmup300/sac_config_10hz_128_128_3.json
CUDA_VISIBLE_DEVICES=1 python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/two_stage_b_05_b_1/buffer_size1e6_warmup300/sac_config_10hz_128_128_4.json
CUDA_VISIBLE_DEVICES=1 python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/two_stage_b_05_b_1/buffer_size1e6_warmup300/sac_config_10hz_128_128_5.json

## E2D WARMUP 300  NUMTRANSITIONS 2e5   100
CUDA_VISIBLE_DEVICES=2 python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition2e5_warmup_300/sac_config_10hz_128_128_1.json
CUDA_VISIBLE_DEVICES=2 python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition2e5_warmup_300/sac_config_10hz_128_128_2.json
CUDA_VISIBLE_DEVICES=3 python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition2e5_warmup_300/sac_config_10hz_128_128_3.json
CUDA_VISIBLE_DEVICES=3 python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition2e5_warmup_300/sac_config_10hz_128_128_4.json
CUDA_VISIBLE_DEVICES=3 python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition2e5_warmup_300/sac_config_10hz_128_128_5.json


## E2D WARMUP 300  NUMTRANSITIONS 1e6  120
CUDA_VISIBLE_DEVICES=1 python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition1e6_warmup_300/sac_config_10hz_128_128_1.json
CUDA_VISIBLE_DEVICES=1 python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition1e6_warmup_300/sac_config_10hz_128_128_2.json
CUDA_VISIBLE_DEVICES=1 python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition1e6_warmup_300/sac_config_10hz_128_128_3.json
CUDA_VISIBLE_DEVICES=1 python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition1e6_warmup_300/sac_config_10hz_128_128_4.json
CUDA_VISIBLE_DEVICES=1 python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/VVC/medium/sac/E2D_medium_b_1_aug4/buffer_size1e6/num_transition1e6_warmup_300/sac_config_10hz_128_128_5.json


## easy2medium (warmup)  2medium2 166
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_medium2/evaluate_medium_medium2_on_medium2/b_1/3e6/buffer_size1e6_warmup300/sac_config_10hz_128_128_1.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_medium2/evaluate_medium_medium2_on_medium2/b_1/3e6/buffer_size1e6_warmup300/sac_config_10hz_128_128_2.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_medium2/evaluate_medium_medium2_on_medium2/b_1/3e6/buffer_size1e6_warmup300/sac_config_10hz_128_128_3.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_medium2/evaluate_medium_medium2_on_medium2/b_1/3e6/buffer_size1e6_warmup300/sac_config_10hz_128_128_4.json
python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_medium2/evaluate_medium_medium2_on_medium2/b_1/3e6/buffer_size1e6_warmup300/sac_config_10hz_128_128_5.json


python  train_scripts/ladderrl/train_with_rl_sac_her_multi_stages.py --config-file-name=configs/train/D2D/goal_sapce/easy_to_medium_to_medium2/evaluate_medium_medium2_on_medium2/b_1/3e6/buffer_size1e6_warmup300/sac_config_10hz_128_128_5test.json