#!/bin/bash

# baseline
python train_scripts/reward_norm/train/train_with_rl_ppo.py --config-file-name configs/train/reward_norm/ppo/medium_end_to_end/128_128_seed_1.json
python train_scripts/reward_norm/train/train_with_rl_ppo.py --config-file-name configs/train/reward_norm/ppo/medium_end_to_end/128_128_seed_2.json
python train_scripts/reward_norm/train/train_with_rl_ppo.py --config-file-name configs/train/reward_norm/ppo/medium_end_to_end/128_128_seed_3.json
python train_scripts/reward_norm/train/train_with_rl_ppo.py --config-file-name configs/train/reward_norm/ppo/medium_end_to_end/128_128_seed_4.json
python train_scripts/reward_norm/train/train_with_rl_ppo.py --config-file-name configs/train/reward_norm/ppo/medium_end_to_end/128_128_seed_5.json
