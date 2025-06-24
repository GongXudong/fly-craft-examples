import gymnasium as gym 
import sys
from pathlib import Path
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.env_utils.register_env import register_all_with_default_dense_params, register_all_with_default_sparse_params
register_all_with_default_dense_params()
env = gym.make("my-reach")
print(env.observation_space)
print(env.action_space)