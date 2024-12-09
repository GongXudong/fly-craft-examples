import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import SACPolicy

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper

def get_original_goal(env: gym.Env, goal: np.array):

    tmp_env = env
    tmp_obs = env.observation_space.sample()
    tmp_obs["desired_goal"] = goal

    while True:
        if isinstance(tmp_env, ScaledObservationWrapper):
            tmp_obs = tmp_env.inverse_scale_state(tmp_obs)
            return tmp_obs["desired_goal"]

        if not isinstance(tmp_env, gym.Wrapper):
            return goal
        
        tmp_env = tmp_env.env