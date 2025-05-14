import gymnasium as gym

import sys
from pathlib import Path
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.ladderrl.utils.wrappers import PowerRewardWrapper

env = gym.make('my-reach-sparse-5')
env = PowerRewardWrapper(env = env, b= 0.5,reward_type="sparse",distance_threshold=0.01)
obs,info = env.reset()
for episode in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()