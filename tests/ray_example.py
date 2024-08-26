from pprint import pprint
import flycraft
import ray
from ray.rllib.algorithms.ppo import PPOConfig

# using ray 2.34.0

# reference to https://docs.ray.io/en/latest/cluster/getting-started.html for deploying a ray cluster.
ray.init()  # This will autodetect an existing Ray cluster or start a new Ray instance if no existing cluster is found
# ray.init(address="ray://xx.xx.xx.xx:xxxx")  # connect to an existing remote cluster

config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment("FlyCraft-v0")
    .env_runners(
        num_env_runners=256  #  Number of EnvRunner actors to create for parallel sampling.
    )
)

algo = config.build()

for i in range(10000):
    result = algo.train()
    result.pop("config")
    pprint(result)

    if i % 1000 == 0:
        checkpoint_dir = algo.save_to_path()
        print(f"Checkpoint saved in directory {checkpoint_dir}")