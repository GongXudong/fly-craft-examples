# fly-craft-examples

Demonstrations generation and training scripts for [fly-craft](https://github.com/gongxudong/fly-craft).

## Research Papers Based on This Repository

### 1.Improving the Continuity of Goal-Achievement Ability via Policy Self-Regularization for Goal-Conditioned Reinforcement Learning [ICML 2025]

This research proposed a margin-based policy self-regularization approach to improve the continuity of goal-achievement ability for goal-conditioned reinforcement learning ([paper link]()). Please refer to [train_scripts/MSR](train_scripts/MSR) for the training scripts.

### 2.VVC-Gym: A Fixed-Wing UAV Reinforcement Learning Environment for Multi-Goal Long-Horizon Problems [ICLR 2025]

This research provided a novel fixed-wing UAV RL environment, demonstrations, and baselines for multi-goal long-horizon problem research([paper link](https://openreview.net/forum?id=5xSRg3eYZz)). Please refer to [train_scripts/VVCGym](train_scripts/VVCGym) for the training scripts.

### 3.Iterative Regularized Policy Optimization with Imperfect Demonstrations [ICML 2024]

This research proposed Iterative Regularized Policy Optimization to solve the over-constrained exploration problem and the primacy bias problem in offline-to-online learning ([paper link](https://openreview.net/pdf?id=Gp5F6qzwGK)). Please refer to [train_scripts/IRPO](train_scripts/IRPO) for the training scripts.

## Generating Demonstrations

### 1.Generating with PID controller

Sample from $ V \times \Mu \times \Chi = [v_{min}:v_{max}:v_{interval}] \times [mu_{min}:mu_{max}:mu_{interval}] \times [chi_{min}:chi_{max}:chi_{interval}] $ with PID controller and save sampled trajectories in demonstrations/data/\{step-frequence\}hz\_\{$v_{interval}$\}\_\{$mu_{interval}$\}\_\{$chi_{interval}$\}\_\{data-dir-suffix\}

```bash
# sample trajectories single-processing
python demonstrations/rollout_trajs/rollout_by_pid.py --data-dir-suffix v5 --step-frequence 10 --v-min 100 --v-max 110 --v-interval 10 --mu-min -5 --mu-max 5 --mu-interval 5 --chi-min -5 --chi-max 5 --chi-interval 5

# sample trajectories multi-processing with Ray
python demonstrations/rollout_trajs/rollout_by_pid_parallel.py --data-dir-suffix v4 --step-frequence 10 --v-min 100 --v-max 110 --v-interval 10 --mu-min -5 --mu-max 5 --mu-interval 5 --chi-min -5 --chi-max 5 --chi-interval 5
```

### 2.Updating demonstrations with trained RL policy

Update demonstrations in {demos-dir} with policy in {policy-ckpt-dir}

```bash
python demonstrations/rollout_trajs/rollout_by_policy_and_update_demostrations.py --policy-ckpt-dir checkpoints/sac_her/best_model --env-config-dir configs/env/env_config_for_sac.json --demos-dir demonstrations/data/10hz_10_5_5_v2
```

### 3.Augment demonstrations

Augment trajectories based on $\chi$'s symmetry

```bash
python demonstrations/utils/augment_trajs.py --demos-dir demonstrations/data/10hz_10_5_5_v2
```

### 4.Label demonstrations with rewards (**support for Offline RL**)

Label demonstrations in {demos-dir} with rewards (--traj-prefix is the csv filename's prefix in the demonstration direction)

```bash
python demonstrations/utils/label_transitions_with_rewards.py --demos-dir demonstrations/data/10hz_10_5_5_test --traj-prefix my_f16trace
```

### 5.Process demonstrations (normarlize observations and actions, and concat all csv files) and cache the processed np.ndarray objects

```bash
python demonstrations/utils/load_dataset.py --demo-dir demonstrations/data/10hz_10_5_5_iter_1_aug --demo-cache-dir demonstrations/cache/10hz_10_5_5_iter_1_aug
```

**Note**: The cache directory should be consistent with the "data_cache_dir" in the training configurations.

## Training policies with Stable-baselines3

### 1.Behavioral Cloning (BC)

```bash
python train_scripts/IRPO/train_with_bc_ppo.py --config-file-name configs/train/IRPO/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json
```

***Note***: This repo depends on imitation (Version 1.0.0). There is a bug in the behavioral cloning (BC) algorithm of this version. Before running BC-related algorithms, it is necessary to modify line 494 of algorithms/bc.py in the imitation library from:

```python
acts = util.safe_to_tensor(batch["acts"], device=self.policy.device)
```

to:

```python
acts = util.safe_to_tensor(batch["acts"]).to(device=self.policy.device)
```

### 2.Proximal Policy Optimization (PPO)

```bash
python train_scripts/IRPO/train_with_rl_ppo.py --config-file-name configs/train/IRPO/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json
```

### 3.PPO fine-tuning a BC-pre-trained policy

```bash
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json
```

### 4.Soft Actor-Critic (SAC)

```bash
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/sac_without_her/sac_config_10hz_128_128_1.json
```

### 5.SAC with Hindsight Experience Replay (HER)

```bash
python train_scripts/train_with_rl_sac_her.py --config-file-name configs/train/sac/sac_her/sac_config_10hz_128_128_1.json
```

### 6.Non-Markovian Reward Problem (NMR)

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

## Evaluating policies

### 1.Visualization

The script _train_scripts/IRPO/evaluate/rollout_one_trajectory.py_ can be used to generate _.acmi_ files, which can be used to visualize the flight trajectory with the help of [**Tacview**](https://www.tacview.net/).

```bash
python train_scripts/IRPO/evaluate/rollout_one_trajectory.py --config-file-name configs/train/IRPO/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json --algo rl_bc --save-acmi --use-fixed-target --target-v 210 --target-mu 5 --target-chi 10 --save-dir train_scripts/IRPO/evaluate/rolled_out_trajs/
```

### 2.Statistical Evaluation

The script _train_scripts/IRPO/evaluate/evaluate_policy_by_success_rate.py_ can be used to evaluate trained policies statistically, which will obtain information such as the success rate, cumulative rewards, and trajectory length of the policy.

```bash
python train_scripts/IRPO/evaluate/evaluate_policy_by_success_rate.py --config-file-name configs/train/IRPO/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json --algo rl_bc --seed 11 --n-envs 8 --n-eval-episode 100
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
