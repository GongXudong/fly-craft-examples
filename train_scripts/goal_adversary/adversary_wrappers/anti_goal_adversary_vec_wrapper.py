import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
from copy import deepcopy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper, VecEnvStepReturn
import flycraft

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper

class VecAntiGoalAdversaryWrapper(VecEnvWrapper):

    def __init__(
        self, 
        venv: VecEnv, 
        goal_adversary_policy: BasePolicy,
        noise_min: np.ndarray, 
        noise_max: np.ndarray,
        env_config: Path = None,
    ):
        super().__init__(venv)
        self.goal_adversary_policy: BasePolicy = goal_adversary_policy
        
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space

    def step_async(self, actions):
        # Call the step_async method of the underlying vectorized environment
        self.venv.step_async(actions)
    
    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        
        # TODO: 是否需要根据noised_desired_goal来重新计算reward和done？

        return self.get_noised_desired_goal_obs(obs), reward, done, info
    
    def reset(self):
        obs = self.venv.reset()
        return self.get_noised_desired_goal_obs(obs)

    def get_noised_desired_goal_obs(self, observations: np.ndarray):
        
        # Generate desired goal noise using the goal adversary policy
        desired_goal_noise, _ = self.goal_adversary_policy.predict(
            observation=observations, 
            deterministic=True
        )

        # Create a copy of the observations and add the noise to the desired goal
        noised_obs = deepcopy(observations)
        noised_obs["desired_goal"] += desired_goal_noise

        # Clip the noised desired goal to the valid range
        noised_obs["desired_goal"] = np.clip(
            noised_obs["desired_goal"],
            self.observation_space["desired_goal"].low,
            self.observation_space["desired_goal"].high,
        )

        return noised_obs