# fly-craft-examples

Demonstrations generation and training scripts for [fly-craft](https://github.com/gongxudong/fly-craft).

## Generating Demonstrations

### generating with PID controller

```bash
# sample trajectories single-processing
python demonstrations/rollout_trajs/rollout_by_pid.py --data-dir-suffix v5 --step-frequence 10 --v-min 100 --v-max 110 --v-interval 10 --mu-min -5 --mu-max 5 --mu-interval 5 --chi-min -5 --chi-max 5 --chi-interval 5

# sample trajectories multi-processing with Ray
python demonstrations/rollout_trajs/rollout_by_pid_parallel.py --data-dir-suffix v4 --step-frequence 10 --v-min 100 --v-max 110 --v-interval 10 --mu-min -5 --mu-max 5 --mu-interval 5 --chi-min -5 --chi-max 5 --chi-interval 5
```

### updating demonstrations with policy

```bash
python demonstrations/rollout_trajs/rollout_by_policy_and_update_demostrations.py --policy-ckp-dir checkpoints/sac_her/best_model --env-config-dir configs/env/env_config_for_sac.json --demos-dir demonstrations/data/10hz_10_5_5_v2
```

## Training policies with Stable-baselines3

### BC
```bash
python train_scripts/train_with_bc_ppo.py --config-file-name configs/train/ppo_bc_config_10hz_128_128_3.json
```

### PPO
```bash
python train_scripts/train_with_rl_ppo.py --config-file-name configs/train/ppo_bc_config_10hz_128_128_2.json
```

### PPO fine-tuning a BC-pre-trained policy
```bash
python train_scripts/train_with_rl_bc_ppo.py --config-file-name configs/train/ppo_bc_config_10hz_128_128_2.json
```

### SAC with HER
```bash
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac_config_10hz_128_128_1.json
```
