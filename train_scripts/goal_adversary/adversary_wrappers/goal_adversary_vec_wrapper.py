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

class VecGoalAdversaryWrapper(VecEnvWrapper):

    def __init__(
        self, 
        venv: VecEnv, 
        policy: BasePolicy,
        noise_min: np.ndarray, 
        noise_max: np.ndarray,
        env_config: Path = None,
    ):
        super().__init__(venv)
        self.policy: BasePolicy = policy
        
        self.observation_space = venv.observation_space
        self.action_space = gym.spaces.Box(
            low=noise_min,
            high=noise_max,
            dtype=np.float32,
        )

        self.obs = None

        self.helper_env = gym.make(
            "FlyCraft-v0",
            config_file = env_config,
            custom_config = {
                "debug_mode": True,
            },
        )
        self.helper_env = ScaledActionWrapper(ScaledObservationWrapper(self.helper_env))

    def step_async(self, actions):

        noised_obs = deepcopy(self.obs)
        # print(actions.shape, noised_obs["desired_goal"].shape)
        noised_obs["desired_goal"] += actions

        # Clip noised desired goal to the valid range
        noised_obs["desired_goal"] = np.clip(
            noised_obs["desired_goal"],
            self.observation_space["desired_goal"].low,
            self.observation_space["desired_goal"].high,
        )
        # print(f"noised obs: {noised_obs}")

        # get actions from the policy under the noised observation
        true_actions, _ = self.policy.predict(noised_obs, deterministic=True)

        # Call the step_async method of the underlying vectorized environment
        self.venv.step_async(true_actions)
    
    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        self.obs = obs

        new_reward = []
        for i in range(len(reward)):
            if done[i]:
                # If the original environment succeeded, the adversary fails and gets a large penalty
                if info[i].get("is_success", False):
                    tmp_reward = self.helper_env.unwrapped.task.termination_funcs[0].get_penalty_base_on_steps_left(steps_cnt=info[i]["step"])
                    new_reward.append(tmp_reward)
                # If the original environment failed, the adversary succeeds and gets a reward
                else:
                    new_reward.append(0)
            else:
                new_reward.append(-reward[i] - 1)

        return obs, np.array(new_reward), done, info
    
    def reset(self):
        obs = self.venv.reset()
        self.obs = obs
        return obs

