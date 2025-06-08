#!/bin/bash

# step_frequence = 2hz

# 242
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_2hz/128_128_seed_1.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_2hz/128_128_seed_2.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_2hz/128_128_seed_3.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_2hz/128_128_seed_4.json
python train_scripts/msr/train/train_ppo_baseline.py --config-file-name configs/train/msr/baseline/ppo/medium/step_freq_2hz/128_128_seed_5.json

