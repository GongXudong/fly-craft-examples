# VVCGym

Official code for "VVC-Gym: A Fixed-Wing UAV Reinforcement Learning Environment for Multi-Goal Long-Horizon Problems" (ICLR 2025). ([paper link](https://openreview.net/forum?id=5xSRg3eYZz))

This repository provides training scripts for the experiments in the VVCGym paper. For the source code of the RL environment, please refer to [flycraft](https://github.com/GongXudong/fly-craft).

## prepare python environment

```bash
# in the root direction of fly-craft-examples
conda create --name VVCGym python=3.12
conda activate VVCGym
conda install --file requirements.txt
```

## training scripts

```bash
# in the root direction of fly-craft-examples

# run PPO-based trainings
bash train_scripts/VVCGym/shell_scripts/ppo.sh

# run HER-based trainings
bash train_scripts/VVCGym/shell_scripts/her.sh
```

## Citation

Cite as

```bib
@inproceedings{gong2025vvcgym,
  title        = {VVC-Gym: A Fixed-Wing UAV Reinforcement Learning Environment for Multi-Goal Long-Horizon Problems},
  author       = {Gong, Xudong and Feng, Dawei and Xu, kele and Wang, Weijia and Sun, Zhangjun and Zhou, Xing and Ding, Bo and Wang, Huaimin},
  booktitle    = {International Conference on Learning Representations},
  year         = {2025}
}
```
