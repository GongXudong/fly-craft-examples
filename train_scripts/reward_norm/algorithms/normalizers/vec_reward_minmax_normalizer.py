from typing import List
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import gymnasium as gym

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.env_util import unwrap_wrapper

PROJECT_ROOT_DIR = Path(__file__).absolute().parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledObservationWrapper


class VecRewardMinMaxNormalize(VecEnvWrapper):

    def __init__(
        self,
        venv: VecEnv,
        radius: float = 1.,
        default_max_return: float = -100.,
        helper_env: gym.Env = None,
        key_names: List[str] = ["dg_v", "dg_mu", "dg_chi"],
        value_name: str = "cumulative_reward",
        is_success_name: str = "is_success",
    ):
        super().__init__(venv)

        print(f"Init normalizer, radius: {radius}, default: {default_max_return}!!!!!!!!!!!")

        self.radius: float = radius
        self.default_max_return: float = default_max_return
        self.helper_env: gym.Env = helper_env
        self.key_names: List[str] = key_names
        self.value_name: str = value_name
        self.is_success_name: str = is_success_name

        self.data: pd.DataFrame = None
        self.nn: NearestNeighbors = NearestNeighbors(radius=self.radius)

        self.max_value_of_current_episode = np.ones(shape=(venv.num_envs, ), dtype=np.float32) * self.default_max_return

    def _update_data(self, data_df: Path):

        # 检查csv文件包含的列名是否覆盖了self.key_names的所有列名
        assert all([item in data_df.columns for item in self.key_names]), f"dataframe must contains columns: {self.key_names}!"

        # 仅保留成功的轨迹数据
        tmp_data = data_df[data_df[self.is_success_name] == True]

        if self.data is None:
            self.data = tmp_data
        else:
            self.data = pd.concat([self.data, tmp_data])

    def update_data_from_csv_file(self, data_file_dir: Path):
        data_df = pd.read_csv(data_file_dir)
        self._update_data(data_df=data_df)
    
    def update_data_from_dataframe(self, data_df: pd.DataFrame):
        self._update_data(data_df=data_df)
    
    def fit_data(self, ):
        assert self.data is not None, "must laod data before calling fit_data()!"

        tmp_data = self.data.loc[:][self.key_names].to_numpy()
        self.nn.fit(tmp_data)

    def get_neighbours(self, desired_goals: np.ndarray):
        distances, indices = self.nn.radius_neighbors(desired_goals, radius=5.0, return_distance=True)
        return distances, indices

    def reset(self):
        "In reset: "
        obs = self.venv.reset()
        assert isinstance(obs, (np.ndarray, dict))

        self.max_value_of_current_episode = np.ones(shape=(self.venv.num_envs, ), dtype=np.float32) * self.default_max_return

        # 处理scale obs的情况
        tmp_env = unwrap_wrapper(self.helper_env, ScaledObservationWrapper)
        if tmp_env is not None:
            tmp_dgs = []
            for i in range(len(obs["desired_goal"])):
                tmp_dgs.append(tmp_env.inverse_scale_state({"observation": obs["observation"][i], "achieved_goal": obs["achieved_goal"][i], "desired_goal": obs["desired_goal"][i]})["desired_goal"])
            desired_goals = np.array(tmp_dgs)
        else:
            desired_goals = obs["desired_goal"]
        
        distances, indices = self.nn.radius_neighbors(desired_goals, return_distance=True)
        
        for index, (pt, dst, ind) in enumerate(zip(desired_goals, distances, indices)):
            if ind.size > 0:
                values = self.data.iloc[ind][self.value_name].to_numpy()

                # method to select a value
                self.max_value_of_current_episode[index] = np.abs(values.max())
                print(f"reset env to dg: {pt}, max return: {self.max_value_of_current_episode[index]}")
            else:
                self.max_value_of_current_episode[index] = np.abs(self.default_max_return)
                print(f"reset env to dg: {pt}, set max return to default value: {self.max_value_of_current_episode[index]}")
        
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        for idx, done in enumerate(dones):
            if not done:
                continue
        
            # VecEnv的设计中，如果done=True，则把next_obs修改为一个新环境的初始状态，且把真正的最后一步状态存贮在info["terminal_observation"]中
            

            tmp_env = unwrap_wrapper(self.helper_env, ScaledObservationWrapper)
            if tmp_env is not None:
                tmp_dgs = []
                this_dg = tmp_env.inverse_scale_state({"observation": obs["observation"][idx], "achieved_goal": obs["achieved_goal"][idx], "desired_goal": obs["desired_goal"][idx]})["desired_goal"]
            else:
                this_dg = obs["desired_goal"][idx]

            # print(obs["desired_goal"][idx], this_dg)
            # print(infos[idx])

            distances, indices = self.nn.radius_neighbors(this_dg.reshape((1, -1)), return_distance=True)

            distance_arr, indice_arr = distances[0], indices[0]

            if indice_arr.size > 0:
                values = self.data.iloc[indice_arr][self.value_name].to_numpy()

                # method to select a value
                self.max_value_of_current_episode[idx] = np.abs(values.max())
                print(f"reset env to dg: {this_dg}, max return: {self.max_value_of_current_episode[idx]}")
            else:
                self.max_value_of_current_episode[idx] = np.abs(self.default_max_return)
                print(f"reset env to dg: {this_dg}, set max return to default value: {self.max_value_of_current_episode[idx]}")
            
            # Normalize reward
            rewards = rewards / self.max_value_of_current_episode

        return obs, rewards, dones, infos