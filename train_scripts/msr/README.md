# Margin-Based Policy Self-Regularization

Official code for "Improving the Continuity of Goal-Achievement Ability via Policy Self-Regularization for Goal-Conditioned Reinforcement Learning" (ICML 2025).

## prepare python environment

```bash
# in the root direction of fly-craft-examples
conda create --name MSR python=3.12
conda activate MSR
conda install --file requirements.txt
```

## training scripts

```bash
# in the root direction of fly-craft-examples

# run PPO-based trainings
bash train_scripts/msr/shell_scripts/train/train_ppo_on_flycraft_medium.sh

# run SAC-based trainings
bash train_scripts/msr/shell_scripts/train/train_sac_on_flycraft_medium.sh

# run HER-based trainings
bash train_scripts/msr/shell_scripts/train/train_her_on_flycraft_medium.sh
```

## Citation

Cite as

```bib
@inproceedings{gong2025improving,
  title        = {Improving the Continuity of Goal-Achievement Ability via Policy Self-Regularization for Goal-Conditioned Reinforcement Learning},
  author       = {Gong, Xudong and Yang, Sen and Feng, Dawei and Xu, kele and Ding, Bo and Wang, Huaimin and Dou, Yong},
  booktitle    = {Forty-second International Conference on Machine Learning},
  year         = {2025}
}
```
