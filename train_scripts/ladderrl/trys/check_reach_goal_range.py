import gymnasium as gym
import panda_gym
import sys
from pathlib import Path
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))
from utils_my.env_utils.register_env import register_all_with_default_dense_params, register_all_with_default_sparse_params
register_all_with_default_dense_params() 

env = gym.make('my-reach', render_mode="human")

observation, info = env.reset()

for _ in range(10000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()