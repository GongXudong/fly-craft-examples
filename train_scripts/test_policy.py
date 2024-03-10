import gymnasium as gym
import numpy as np
from pathlib import Path
import logging
import torch as th
import argparse
from copy import deepcopy
import sys

from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.sac import MlpPolicy
from imitation.algorithms import bc
from imitation.util.logger import HierarchicalLogger
from imitation.util import util
from imitation.data import types
from imitation.data.types import TransitionsMinimal
from imitation.data import rollout

import flycraft
from flycraft.env import FlyCraftEnv
from flycraft.utils.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.models.ppo_with_bc_loss import PPOWithBCLoss
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate
from demonstrations.utils.load_dataset import load_data_from_cache
from utils_my.sb3.my_schedule import linear_schedule
from utils_my.sb3.vec_env_helper import get_vec_env


if __name__ == "__main__":

    policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "bc" / "10hz_128_128_300epochs_loss_1"
    algo_ppo = PPOWithBCLoss.load(str((policy_save_dir / "bc_checkpoint").absolute()))

    vec_env = get_vec_env(
        num_process=4,
        config_file=PROJECT_ROOT_DIR / "configs" / "env" /"env_config_for_sac.json",
        custom_config={"debug_mode": False}
    )

    res = evaluate_policy_with_success_rate(
        model=algo_ppo.policy,
        env=vec_env,
        n_eval_episodes=100
    )

    print(res)