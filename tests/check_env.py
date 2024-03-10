from pathlib import Path
import sys
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

import flycraft

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

if __name__ == "__main__":
    env = gym.make(
        id="FlyCraft-v0",
        config_file=str(PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_sac.json")
    )
    check_env(env)

    # AssertionError: The reward was not computed with `compute_reward()
    # ingore the above error, as is comes from the float precision
    # we test the reward with scipt in flycraft/test/test_reward.py