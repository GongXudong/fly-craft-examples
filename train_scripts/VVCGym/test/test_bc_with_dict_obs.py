import gymnasium as gym
import numpy as np
from pathlib import Path
import logging
import torch as th
import argparse
from copy import deepcopy
import sys

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.sac import MlpPolicy
from imitation.algorithms import bc
from imitation.util.logger import HierarchicalLogger
from imitation.util import util
from imitation.data.types import TransitionsMinimal
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

import flycraft
from flycraft.env import FlyCraftEnv
from flycraft.utils_common.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate
from demonstrations.utils.load_dataset import load_data_from_cache
from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper


def make_env(rank: int, seed: int = 0, **kwargs):
    """
    Utility function for multiprocessed env.

    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = FlyCraftEnv(
            config_file=kwargs["config_file"],
            custom_config=kwargs.get("custom_config", {})
        )
        env = RolloutInfoWrapper(ScaledActionWrapper(ScaledObservationWrapper(env)))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_vec_env(num_process: int=10, seed: int=0, **kwargs):
    return SubprocVecEnv([make_env(rank=i, seed=seed, **kwargs) for i in range(num_process)])



def test_dict_space():

    vec_env = get_vec_env(
        num_process=1,
        config_file=PROJECT_ROOT_DIR / "configs" / "env" / "VVCGym" / "env_config_for_sac.json",
        custom_config={"debug_mode": False}
    )

    # multi-input policy to accept dict observations
    assert isinstance(vec_env.observation_space, gym.spaces.Dict)
    policy = MultiInputActorCriticPolicy(
        vec_env.observation_space,
        vec_env.action_space,
        lambda _: 0.001,
    )
    rng = np.random.default_rng()

    # sample random transitions
    rollouts = rollout.rollout(
        policy=None,
        venv=vec_env,
        sample_until=rollout.make_sample_until(min_timesteps=None, min_episodes=5),
        rng=rng,
        unwrap=True,
    )
    transitions = rollout.flatten_trajectories(rollouts)
    # minimal_transition = TransitionsMinimal(obs=transitions.obs, acts=transitions.acts, infos=transitions.infos)
    bc_trainer = bc.BC(
        observation_space=vec_env.observation_space,
        policy=policy,
        action_space=vec_env.action_space,
        rng=rng,
        demonstrations=transitions,
    )
    # confirm that training works
    bc_trainer.train(n_epochs=1)



def test_rollout():

    vec_env = get_vec_env(
        num_process=1,
        config_file=PROJECT_ROOT_DIR / "configs" / "env" / "VVCGym" / "env_config_for_sac.json",
        custom_config={"debug_mode": False}
    )

    rollouts = rollout.rollout(
        policy=None,
        venv=vec_env,
        sample_until=rollout.make_sample_until(min_timesteps=None, min_episodes=2),
        rng=np.random.default_rng(),
        unwrap=True,
        exclude_infos=True,
    )
    print("finish rollout!!!!!!!!!!")

    transitions = rollout.flatten_trajectories(rollouts)

    print(transitions)


if __name__ == "__main__":
    test_dict_space()
