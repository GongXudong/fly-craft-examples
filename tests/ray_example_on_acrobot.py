from pathlib import Path
import ray
import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from flycraft.env import FlyCraftEnv

PROJECT_ROOT_DIR = Path(__file__).parent.parent

# def env_creator(env_config):
#     gym.make("Ant-v4")

# register_env("Ant-v5", env_creator)


algo = (
    PPOConfig(
    )
    .rollouts(num_rollout_workers=128)
    .resources(num_gpus=1)
    .environment(env="Acrobot-v1")
    .build()
)

for i in range(50):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")