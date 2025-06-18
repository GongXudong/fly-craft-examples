#!/bin/bash

# step_frequence = 2hz
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_2hz/128_128_seed_1.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_2hz/128_128_seed_2.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_2hz/128_128_seed_3.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_2hz/128_128_seed_4.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_2hz/128_128_seed_5.json

# step_frequence = 10hz, frame_skip = 5
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_10hz_frame_skip_5/128_128_seed_1.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_10hz_frame_skip_5/128_128_seed_2.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_10hz_frame_skip_5/128_128_seed_3.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_10hz_frame_skip_5/128_128_seed_4.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_10hz_frame_skip_5/128_128_seed_5.json

# step_frequence = 10hz, no frame_skip
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_10hz/128_128_seed_1.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_10hz/128_128_seed_2.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_10hz/128_128_seed_3.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_10hz/128_128_seed_4.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_10hz/128_128_seed_5.json
