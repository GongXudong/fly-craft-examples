# fly-craft-examples

Demonstrations generation and training scripts for [fly-craft](https://github.com/gongxudong/fly-craft).

## Generating Demonstrations

### generating with PID controller

sample from $ V \times \Mu \times \Chi = [v_{min}:v_{max}:v_{interval}] \times [mu_{min}:mu_{max}:mu_{interval}] \times [chi_{min}:chi_{max}:chi_{interval}] $ with PID controller and save sampled trajectories in demonstrations/data/\{step-frequence\}hz\_\{$v_{interval}$\}\_\{$mu_{interval}$\}\_\{$chi_{interval}$\}\_\{data-dir-suffix\}

```bash
# sample trajectories single-processing
python demonstrations/rollout_trajs/rollout_by_pid.py --data-dir-suffix v5 --step-frequence 10 --v-min 100 --v-max 110 --v-interval 10 --mu-min -5 --mu-max 5 --mu-interval 5 --chi-min -5 --chi-max 5 --chi-interval 5

# sample trajectories multi-processing with Ray
python demonstrations/rollout_trajs/rollout_by_pid_parallel.py --data-dir-suffix v4 --step-frequence 10 --v-min 100 --v-max 110 --v-interval 10 --mu-min -5 --mu-max 5 --mu-interval 5 --chi-min -5 --chi-max 5 --chi-interval 5
```

### updating demonstrations with policy

update demonstrations in {demos-dir} with policy in {policy-ckpt-dir}

```bash
python demonstrations/rollout_trajs/rollout_by_policy_and_update_demostrations.py --policy-ckpt-dir checkpoints/sac_her/best_model --env-config-dir configs/env/env_config_for_sac.json --demos-dir demonstrations/data/10hz_10_5_5_v2
```

### augment demonstrations

augment trajectories based on $\chi$'s symmetry

```bash
python demonstrations/utils/augment_trajs.py --demos-dir demonstrations/data/10hz_10_5_5_v2
```

### label demonstrations with rewards (**support for Offline RL**)

label demonstrations in {demos-dir} with rewards (--traj-prefix is the csv filename's prefix in the demonstration direction)

```bash
python demonstrations/utils/label_transitions_with_rewards.py --demos-dir demonstrations/data/10hz_10_5_5_test --traj-prefix my_f16trace
```

## Training policies with Stable-baselines3

### BC

```bash
python train_scripts/train_with_bc_ppo.py --config-file-name configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json
```

### PPO

```bash
python train_scripts/train_with_rl_ppo.py --config-file-name configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json
```

### PPO fine-tuning a BC-pre-trained policy

```bash
python train_scripts/train_with_rl_bc_ppo.py --config-file-name configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json
```

### SAC

```bash
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/sac_without_her/sac_config_10hz_128_128_1.json
```

### SAC with HER

```bash
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/sac_her/sac_config_10hz_128_128_1.json
```

### NMR (Non-Markovian Reward Problem)

```bash
# test SAC on NMR(last 10 observations)
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/easy_her_sparse_negative_non_markov_reward_persist_1_sec/sac_config_10hz_128_128_1.json

# test SAC on NMR(last 20 observations)
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/easy_her_sparse_negative_non_markov_reward_persist_2_sec/sac_config_10hz_128_128_1.json

# test SAC on NMR(last 30 observations)
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/easy_her_sparse_negative_non_markov_reward_persist_3_sec/sac_config_10hz_128_128_1.json

# try solve NMR with framestack
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/hard_her_framestack_sparse_negative_non_markov_reward_persist_1_sec/sac_config_10hz_128_128_1.json
```

## Citation

Cite as

```bib
@misc{gong2024flycraftexamples,
  title        = {fly-craft-examples},
  author       = {Gong, Xudong},
  year         = 2024,
  note         = {\url{https://github.com/GongXudong/fly-craft-examples} [Accessed: (2024-07-01)]},
}
```
