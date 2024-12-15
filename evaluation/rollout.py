import sys
from pathlib import Path
from abc import ABC
import gymnasium as gym
import numpy as np
import pandas as pd

from stable_baselines3.common.base_class import BaseAlgorithm

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper


class Rollout(ABC):

    def __init__(self, env: gym.Env, algo: BaseAlgorithm, debug_mode: bool=True):
        self.env = env
        self.algo = algo
        self.debug_mode = debug_mode

    def rollout_one_trajectory(self, save_acmi: bool=True, save_dir: Path=Path(__file__).parent):

        obs_list, act_list, reward_list, ve_vn_vh_list, timestamps = [], [], [], [], []

        tackview_data = {
            'ISO time': [],
            'id': [],
            'Longitude': [],
            'Latitude': [],
            'Altitude': [],
            'Roll': [],
            'Pitch': [],
            'Yaw': [],
        }

        obs, info = self.env.reset()
        # ve_vn_vh_list.append([0., 200., 0.])
        # reward_list.append(0.)
        terminated, truncated = False, False
        s_index = 0
        while not (terminated or truncated):
            timestamps.append(s_index * 1. / self.env.unwrapped.plane.step_frequence)

            action, _ = self.algo.predict(observation=obs, deterministic=True)

            is_wrappered_by_scale_action_wrapper = isinstance(self.env, gym.Wrapper) and isinstance(self.env, ScaledActionWrapper)
            if is_wrappered_by_scale_action_wrapper:
                tmp_act = self.env.inverse_scale_action(action)
            else:
                tmp_act = action

            is_wrappered_by_scale_obs_wrapper = hasattr(self.env, "env") and isinstance(self.env.env, gym.Wrapper) and isinstance(self.env.env, ScaledObservationWrapper)
            if is_wrappered_by_scale_obs_wrapper:
                tmp_obs = self.env.env.inverse_scale_state(obs)
            else:
                tmp_obs = obs

            if self.debug_mode:
                print(f"state = {tmp_obs["observation"]}, action = {tmp_act}")
            
            obs_list.append(tmp_obs["observation"])
            act_list.append(tmp_act)
            obs, reward, terminated, truncated, info = self.env.step(action=action)
            reward_list.append(reward)
            ve_vn_vh_list.append([info["plane_state"]["ve"], info["plane_state"]["vn"], info["plane_state"]["vh"]])
            
            tackview_data["ISO time"].append(s_index * 1. / self.env.unwrapped.plane.step_frequence)
            tackview_data["id"].append('101')
            tackview_data["Longitude"].append(info["plane_state"]["lon"])
            tackview_data["Latitude"].append(info["plane_state"]["lat"])
            tackview_data["Altitude"].append(info["plane_state"]["h"])
            tackview_data["Roll"].append(info["plane_state"]["phi"])
            tackview_data["Pitch"].append(info["plane_state"]["theta"])
            tackview_data["Yaw"].append(info["plane_state"]["psi"])
            
            s_index += 1
        
        obs_list = np.array(obs_list)
        act_list = np.array(act_list)
        ve_vn_vh_list = np.array(ve_vn_vh_list)
        df = pd.DataFrame(data={
            "time": timestamps,
            "s_phi": obs_list[:, 0], 
            "s_theta": obs_list[:, 1], 
            "s_psi": obs_list[:, 2], 
            "s_v": obs_list[:, 3], 
            "s_mu": obs_list[:, 4], 
            "s_chi": obs_list[:, 5], 
            "s_p": obs_list[:, 6], 
            "s_h": obs_list[:, 7], 
            "s_ve": ve_vn_vh_list[:, 0],
            "s_vn": ve_vn_vh_list[:, 1],
            "s_vh": ve_vn_vh_list[:, 2],
            "a_p": act_list[:, 0], 
            "a_nz": act_list[:, 1], 
            "a_pla": act_list[:, 2],
            "reward": reward_list,
            "target_v": [info["desired_goal"][0]] * len(obs_list), 
            "target_mu": [info["desired_goal"][1]] * len(obs_list), 
            "target_chi": [info["desired_goal"][2]] * len(obs_list),
        })
        df.to_csv(save_dir / 'test.csv', index=False)

        tackview_df = pd.DataFrame(data=tackview_data)
        tackview_file = save_dir / f"tackview_data_{info["desired_goal"][0]:.2f}_{info["desired_goal"][1]:.2f}_{info["desired_goal"][2]:.2f}.txt.acmi"

        if save_acmi:
            with open(tackview_file, mode='w', encoding='utf-8-sig') as f:
                f.write("FileType=text/acmi/tacview\n")
                f.write("FileVersion=2.1\n")
                f.write("0,ReferenceTime=2023-05-01T00:00:00Z\n")

            with open(tackview_file, mode='a', encoding='utf-8-sig') as f:
                for index, row in tackview_df.iterrows():
                    f.write(f"#{row['ISO time']:.2f}\n")
                    out_str = f"{row['id']},T={row['Longitude']}|{row['Latitude']}|{row['Altitude']}|{row['Roll']}|{row['Pitch']}|{row['Yaw']},Name=F16,Color=Red\n"
                    f.write(out_str)

        tackview_df.to_csv(save_dir / f"tackview_data_{info["desired_goal"][0]:.2f}_{info["desired_goal"][1]:.2f}_{info["desired_goal"][2]:.2f}.csv", index=False)

