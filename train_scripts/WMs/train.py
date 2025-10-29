import numpy as np
from ray import tune
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from flycraft.env import FlyCraftEnv
import gymnasium as gym
from gymnasium.wrappers import TransformObservation


# customize flycraft，vector obs + init(config)
class CustomizeInitFlyCraftEnv(FlyCraftEnv):

    def __init__(self, config: dict):
        custom_config = {ky: config[ky] for ky in ["custom_config", "custom_file"] if ky in config.keys()}
        super().__init__(**custom_config)


def env_creator(config):
    env = CustomizeInitFlyCraftEnv(config)

    obs_space = gym.spaces.Box(
        low=np.concatenate([env.observation_space["observation"].low, env.observation_space["desired_goal"].low]),
        high=np.concatenate([env.observation_space["observation"].high, env.observation_space["desired_goal"].high]),
    )

    env = TransformObservation(
        env=env, 
        func=lambda obs: np.concatenate([obs["observation"], obs["desired_goal"]]),
        observation_space=obs_space
    )
    return env


# Register the customized_flycraft env including necessary wrappers via the `tune.register_env()` API.
tune.register_env("customized_flycraft", env_creator)

# Define the `config` variable to use for training.
config = (
    DreamerV3Config()
    # set the env to the pre-registered string
    .environment(
        env="customized_flycraft",
    )
    # play around with the insanely high number of hyperparameters for DreamerV3 ;)
    .training(
        model_size="S",
        training_ratio=1024,
    )
)

# Run the tuner job.
results = tune.Tuner(
    trainable="DreamerV3",
    param_space=config,
).fit()
