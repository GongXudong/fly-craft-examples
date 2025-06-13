import numpy as np
import gymnasium as gym
from copy import deepcopy
from stable_baselines3.common.policies import BasePolicy

class GoalAdversaryWrapper(gym.Wrapper):
    """
    A wrapper for the environment that adds a goal adversary.
    This wrapper modifies the environment to include a goal adversary that can be used during training.
    """

    def __init__(
        self, 
        env: gym.Env, 
        policy: BasePolicy,
        noise_min: np.ndarray,
        noise_max: np.ndarray,
    ):
        super().__init__(env)
        self.policy: BasePolicy = policy

        self.observation_space = env.observation_space
        self.action_space = gym.spaces.Box(
            low=noise_min,
            high=noise_max,
            dtype=np.float32,
        )

        self.obs = None

    def step(self, action):

        noised_obs = deepcopy(self.obs)
        noised_obs["desired_goal"] += action

        # clipe noised desired goal to the valid range
        noised_obs["desired_goal"] = np.clip(
            noised_obs["desired_goal"],
            self.observation_space["desired_goal"].low,
            self.observation_space["desired_goal"].high,
        )
        # print(f"noised obs: {noised_obs}")

        true_action, _ = self.policy.predict(noised_obs, deterministic=True)

        obs, reward, terminated, truncated, info = self.env.step(true_action)
        self.obs = deepcopy(obs)

        if terminated or truncated:
            # 如果原环境成功了，则干扰失败，给大的惩罚
            if info.get("is_success", False):
                new_reward = self.env.unwrapped.task.termination_funcs[0].get_penalty_base_on_steps_left(steps_cnt = info["step"])
            # 如果原环境失败了，则干扰成功，给奖励
            else:
                new_reward = 0
        else:
            new_reward = - reward - 1

        return obs, new_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs = deepcopy(obs)
        return obs, info
