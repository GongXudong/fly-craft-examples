from pathlib import Path
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from flycraft.env import FlyCraftEnv
import gymnasium as gym

PROJECT_ROOT_DIR = Path(__file__).parent.parent

def env_creator(env_config):
    try:
        return FlyCraftEnv(
            config_file=PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_ppo_easy.json"
        )  # return an env instance
    except:
        return FlyCraftEnv(
            config_file=Path.home() / "pythonprojects" / "fly-craft-examples" / "configs" / "env" / "env_config_for_ppo_easy.json"
        )

register_env("FlyCraft-v1", env_creator)


algo = (
    PPOConfig(
    )
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=1)
    .environment(env="FlyCraft-v1")
    .build()
)

for i in range(250):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")