# This directory is mitigrated from [IRPO](https://github.com/GongXudong/IRPO)

## 1.Generate Demonstrations

Please refer to the [README.md](https://github.com/GongXudong/fly-craft-examples/blob/main/README.md) of this repo.

## 2.Train

### 2.1 Train policy with Behavioral Cloning (BC)

```bash
python train_scripts/IRPO/train_with_bc_ppo.py --config-file-name configs/train/IRPO/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json
```

### 2.2 Train policy from scratch with PPO

```bash
python train_scripts/IRPO/train_with_rl_ppo.py --config-file-name configs/train/IRPO/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json
```

### 2.3 Fine-tune a BC-pre-trained policy with PPO

```bash
python train_scripts/IRPO/train_with_rl_bc_ppo.py --config-file-name configs/train/IRPO/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json
```

## 3.Evaluation

### 3.1 Visualize the rolled-out trajectory

Evaluate the model trained with config file specified by "--config-file-name" on a customized desired goal $(v, \mu, \chi)$.

```bash
python train_scripts/IRPO/evaluate/rollout_one_trajectory.py --config-file-name configs/train/IRPO/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json --algo rl_bc --save-acmi --use-fixed-target --target-v 210 --target-mu 5 --target-chi 10 --save-dir train_scripts/IRPO/evaluate/rolled_out_trajs/
```

The above command will generate a "tackview_data_{v}_{mu}_{chi}.txt.acmi" file in the directory specified by "--save-dir". You can drop the acmi file in the [Tackview](https://www.tacview.net/) to visualize the trajectory.

### 3.2 Evaluate policy on mean reward and success rate

Evaluate the model trained with config file specified by "--config-file-name" on desired goals sampled randomly from the desired goal space.

```bash
python train_scripts/IRPO/evaluate/evaluate_policy_by_success_rate.py --config-file-name configs/train/IRPO/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json --algo rl_bc --seed 11 --n-envs 8 --n-eval-episode 100
```

The above command will evaluate the rl_bc model (training configuration specified by "configs/train/IRPO/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json") with 100 desired goals with 8 parallel environments.
