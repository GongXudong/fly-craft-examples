from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import MultiInputPolicy
from imitation.data.types import DictObs
import numpy as np
import pandas as pd

import gymnasium as gym
from flycraft.env import FlyCraftEnv

import sys
from pathlib import Path
from time import time
import logging
from tqdm import tqdm

PROJECT_ROOT_DIR = Path(__file__).absolute().parent.parent.parent.parent
print(PROJECT_ROOT_DIR)
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.vec_env_helper import make_env
from utils_my.sb3.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper


def load_random_transitions_from_csv_files(
        data_dir: Path, 
        cache_data: bool,
        cache_data_dir: Path,
        trajectory_save_prefix: str="traj",
        env_config_file: Path=PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_sac.json",
        my_logger: logging.Logger=None, 
        select_transition_num: int=100_000,
        n_env = 1
    ):
    """加载数据集：从所有trajectories中随机挑选transitions（最终返回的数据集里的transitions是打乱顺序的）
    """
    
    start_time = time()
    res_file = data_dir / "res.csv"
    res_df = pd.read_csv(res_file)

    obs = []
    next_obs = []
    acts = []
    rewards = []
    dones = []
    infos = []

    traj_cnt = 0
    traj_file_cnt = 0
    transitions_cnt = 0

    origin_env = FlyCraftEnv(config_file=env_config_file)
    
    for index, row in tqdm(res_df.iterrows(), total=res_df.shape[0]):
        target_v, target_mu, target_chi, cur_length = row["v"], row["mu"], row["chi"], row["length"]   # TODO: 注意，有的res.csv保存了序号，每保存的话，使用row[:4]

        # 过滤掉desired goal范围之外的专家轨迹
        if not (
            (origin_env.env_config["goal"]["v_min"] <= target_v <= origin_env.env_config["goal"]["v_max"])
            and
            (origin_env.env_config["goal"]["mu_min"] <= target_mu <= origin_env.env_config["goal"]["mu_max"])
            and
            (origin_env.env_config["goal"]["chi_min"] <= target_chi <= origin_env.env_config["goal"]["chi_max"])
        ):
            continue

        if cur_length > 0:
            cur_length -=1 #丢掉最后一个不完整的transition
            # 能够生成轨迹的目标速度矢量
            cur_filename = f"{trajectory_save_prefix}_{int(target_v)}_{int(target_mu)}_{int(target_chi)}.csv"
            cur_file_path = data_dir / cur_filename
            transitions_cnt += cur_length 
            traj_cnt += 1
            if cur_file_path.exists():
                if my_logger is not None:
                    my_logger.info(f"process file: {cur_filename}")
                else:
                    print(f"process file: {cur_filename}")
                traj_file_cnt += 1
                cur_traj = pd.read_csv(cur_file_path.absolute())

                # 9.960(s) for 13 files with pd.iterrows
                # for index, row in cur_traj.iterrows():
                #     obs.append([*row[1:9], target_v, target_mu, target_chi])
                #     acts.append([*row[9:12]])
                #     infos.append(None)

                # 1.742(s) for 13 files without pd.iterrows
                obs.extend([{
                    "observation": np.array(item[0:8], dtype=np.float32),
                    "achieved_goal": np.array(item[3:6], dtype=np.float32),
                    "desired_goal": np.array(item[8:11], dtype=np.float32)
                } for item in zip(
                    cur_traj['s_phi'].tolist()[:-1],
                    cur_traj['s_theta'].tolist()[:-1],
                    cur_traj['s_psi'].tolist()[:-1],
                    cur_traj['s_v'].tolist()[:-1],
                    cur_traj['s_mu'].tolist()[:-1],
                    cur_traj['s_chi'].tolist()[:-1],
                    cur_traj['s_p'].tolist()[:-1],
                    cur_traj['s_h'].tolist()[:-1],
                    [target_v] * (cur_traj.count()['time'] - 1),
                    [target_mu] * (cur_traj.count()['time'] - 1),
                    [target_chi] * (cur_traj.count()['time'] - 1),
                )])

                next_obs.extend([{
                    "observation": np.array(item[0:8], dtype=np.float32),
                    "achieved_goal": np.array(item[3:6], dtype=np.float32),
                    "desired_goal": np.array(item[8:11], dtype=np.float32)
                } for item in zip(
                    cur_traj['s_phi'].tolist()[1:],
                    cur_traj['s_theta'].tolist()[1:],
                    cur_traj['s_psi'].tolist()[1:],
                    cur_traj['s_v'].tolist()[1:],
                    cur_traj['s_mu'].tolist()[1:],
                    cur_traj['s_chi'].tolist()[1:],
                    cur_traj['s_p'].tolist()[1:],
                    cur_traj['s_h'].tolist()[1:],
                    [target_v] * (cur_traj.count()['time'] - 1),
                    [target_mu] * (cur_traj.count()['time'] - 1),
                    [target_chi] * (cur_traj.count()['time'] - 1),
                )])

                acts.extend(zip(
                    cur_traj['a_p'].tolist()[:-1],
                    cur_traj['a_nz'].tolist()[:-1],
                    cur_traj['a_pla'].tolist()[:-1],
                ))
                rewards.extend([0.] * (cur_traj.count()['time']-1))
                dones.extend([False] * (cur_traj.count()['time']-2) + [True])
                infos.extend([None] * (cur_traj.count()['time']-1))

    # 数据标准化. 这里的标准化最耗时.
    scaled_obs_env = ScaledObservationWrapper(origin_env)
    scaled_act_env = ScaledActionWrapper(scaled_obs_env)

    # 先挑选select_transition_num个transition
    indices = np.arange(len(obs))
    selected_indices = np.random.choice(indices, select_transition_num * n_env, replace=False)

    selected_obs = np.array(obs)[selected_indices]
    selected_next_obs = np.array(next_obs)[selected_indices]
    selected_actions = np.array(acts)[selected_indices]
    selected_rewards = np.array(rewards)[selected_indices].astype(np.float32)
    selected_dones = np.array(dones)[selected_indices].astype(np.float32)
    selected_infos = np.array(infos)[selected_indices]



    scaled_selected_obs = np.array([scaled_obs_env.scale_state(item) for item in selected_obs])
    scaled_selected_next_obs = np.array([scaled_obs_env.scale_state(item) for item in selected_next_obs])
    scaled_selected_acts = np.array([scaled_act_env.scale_action(np.array(item)) for item in selected_actions], dtype=np.float32)
    
    #new_obs = {"observation":[item for item in scaled_selected_obs['observation']  ],"achieved_goal":[item for item in scaled_selected_obs['achieved_goal']],"desired_goal":[item for item in scaled_selected_obs['desired_goal']]} 

    # new_obs = {
    #     # "observation": np.array([item['observation'] for item in scaled_selected_obs], dtype=np.float32),
    #     # "achieved_goal": np.array([item['achieved_goal'] for item in scaled_selected_obs], dtype=np.float32),
    #     # "desired_goal": np.array([item['desired_goal'] for item in scaled_selected_obs], dtype=np.float32),
        
    #     "observation": np.array([np.expand_dims(item['observation'], axis=0) for item in scaled_selected_obs], dtype=np.float32),
    #     "achieved_goal": np.array([np.expand_dims(item['achieved_goal'], axis=0) for item in scaled_selected_obs], dtype=np.float32),
    #     "desired_goal": np.array([np.expand_dims(item['desired_goal'], axis=0) for item in scaled_selected_obs],dtype=np.float32)
    # }

    # new_next_obs = {
    #     # "observation": np.array([item['observation'] for item in scaled_selected_next_obs], dtype=np.float32),
    #     # "achieved_goal": np.array([item['achieved_goal'] for item in scaled_selected_next_obs], dtype=np.float32),
    #     # "desired_goal": np.array([item['desired_goal'] for item in scaled_selected_next_obs], dtype=np.float32),
    #     "observation": np.array([np.expand_dims(item['observation'], axis=0) for item in scaled_selected_next_obs], dtype=np.float32),
    #     "achieved_goal": np.array([np.expand_dims(item['achieved_goal'], axis=0) for item in scaled_selected_next_obs], dtype=np.float32),
    #     "desired_goal": np.array([np.expand_dims(item['desired_goal'], axis=0) for item in scaled_selected_next_obs],dtype=np.float32)
    # }

    if cache_data:
    # 缓存标准化后的数据
        if not cache_data_dir.exists():
            cache_data_dir.mkdir()

        if not cache_data_dir.exists():
            cache_data_dir.mkdir()

        np.save(str((cache_data_dir / "normalized_obs").absolute()), scaled_selected_obs)
        np.save(str((cache_data_dir / "normalized_next_obs").absolute()), scaled_selected_next_obs)
        np.save(str((cache_data_dir / "normalized_acts").absolute()), scaled_selected_acts)
        np.save(str((cache_data_dir / "rewards").absolute()), selected_rewards)
        np.save(str((cache_data_dir / "dones").absolute()), selected_dones)
        np.save(str((cache_data_dir / "infos").absolute()), np.array(selected_infos))

    # 输出统计量
    if my_logger is not None:
        my_logger.info(f"traj cnt: {traj_file_cnt}, transition(from *.csv) cnt: {len(obs)}, average traj length: {len(obs) / traj_file_cnt}")
        my_logger.info(f"traj cnt: {traj_cnt}, transition(from res.csv) cnt: {transitions_cnt}, average traj length: {transitions_cnt / traj_cnt}")
        my_logger.info(f"process time: {time() - start_time}(s).")
    else:
        print(f"traj cnt: {traj_file_cnt}, transition(from *.csv) cnt: {len(obs)}, average traj length: {len(obs) / traj_file_cnt}")
        print(f"traj cnt: {traj_cnt}, transition(from res.csv) cnt: {transitions_cnt}, average traj length: {transitions_cnt / traj_cnt}")
        print(f"process time: {time() - start_time}(s).")
    
    print(f"划分集合后总时间：{time() - start_time}(s).")

    # scaled_selected_obs = scaled_selected_obs.reshape(select_transition_num, n_env, -1)
    # scaled_selected_next_obs = scaled_selected_next_obs.reshape(select_transition_num, n_env, -1)
    # scaled_selected_acts = scaled_selected_acts.reshape(select_transition_num, n_env, -1)
    # selected_rewards = selected_rewards.reshape(select_transition_num, n_env, -1)
    # selected_dones = selected_dones.reshape(select_transition_num, n_env, -1)
    # selected_infos = selected_infos.reshape(select_transition_num, n_env, -1)
    # scaled_selected_acts = np.expand_dims(scaled_selected_acts, axis=1)
    # selected_rewards = np.expand_dims(selected_rewards, axis=1)
    # selected_dones = np.expand_dims(selected_dones, axis=1)
    # selected_infos = np.expand_dims(selected_infos, axis=1)
    # for key,input_shape in scaled_selected_obs[0].items():
    #     scaled_selected_obs.reshape((select_transition_num,n_env,*input_shape))
    # for key,input_shape in scaled_selected_next_obs[0].items():
    #     scaled_selected_next_obs.reshape((select_transition_num,n_env,*input_shape))
    # scaled_selected_obs.reshape((select_transition_num,n_env))
    # scaled_selected_next_obs.reshape((select_transition_num,n_env))
    scaled_selected_obs =  np.expand_dims(scaled_selected_obs, axis=1).tolist()
    scaled_selected_next_obs =  np.expand_dims(scaled_selected_next_obs, axis=1).tolist()
    scaled_selected_acts = np.expand_dims(scaled_selected_acts, axis=1).tolist()
    #scaled_selected_acts.reshape((select_transition_num,n_env,scaled_selected_acts[0].shape(-1)))
    scaled_selected_obs.reshape((select_transition_num,n_env))
    scaled_selected_next_obs.reshape((select_transition_num,n_env))
    scaled_selected_acts.reshape((select_transition_num,n_env))
    selected_rewards.reshape((select_transition_num,n_env))
    selected_dones.reshape((select_transition_num,n_env)) 
    selected_infos.reshape((select_transition_num,n_env)) 

    return DictObs.from_obs_list(scaled_selected_obs), DictObs.from_obs_list(scaled_selected_next_obs), scaled_selected_acts, selected_rewards, selected_dones, selected_infos
    # return new_obs, new_next_obs, scaled_selected_acts, selected_rewards, selected_dones, selected_infos

def load_random_trajectories_from_csv_files(
        data_dir: Path, 
        cache_data: bool,
        cache_data_dir: Path,
        trajectory_save_prefix: str="traj",
        env_config_file: Path=PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_sac.json",
        my_logger: logging.Logger=None, 
        select_transition_num: int=100_000,
        random_state: int=42
    ):
    """加载数据集：随机挑选完整的trajectories，transion总数 <= select_transition_num
    """
    """需改维度 变换为select_transition_num * n_env * obs的维度"""
    start_time = time()
    res_file = data_dir / "res.csv"
    res_df = pd.read_csv(res_file)

    obs = []
    next_obs = []
    acts = []
    rewards = []
    dones = []
    infos = []

    traj_cnt = 0
    traj_file_cnt = 0
    transitions_cnt = 0

    origin_env = FlyCraftEnv(config_file=env_config_file)

    tmp_transition_cnt = 0
    
    for index, row in tqdm(res_df.sample(frac=1, random_state=random_state).iterrows(), total=res_df.shape[0]):

        target_v, target_mu, target_chi, cur_length = row["v"], row["mu"], row["chi"], row["length"]   # TODO: 注意，有的res.csv保存了序号，每保存的话，使用row[:4]

        # 过滤掉desired goal范围之外的专家轨迹
        if not (
            (origin_env.env_config["goal"]["v_min"] <= target_v <= origin_env.env_config["goal"]["v_max"])
            and
            (origin_env.env_config["goal"]["mu_min"] <= target_mu <= origin_env.env_config["goal"]["mu_max"])
            and
            (origin_env.env_config["goal"]["chi_min"] <= target_chi <= origin_env.env_config["goal"]["chi_max"])
        ):
            continue

        if cur_length > 0:
            cur_length -=1 #丢掉最后一个不完整的transition
            # 保证选择完整的轨迹
            if tmp_transition_cnt + cur_length > select_transition_num:
                break

            tmp_transition_cnt += (cur_length-1)

            # 能够生成轨迹的目标速度矢量
            cur_filename = f"{trajectory_save_prefix}_{int(target_v)}_{int(target_mu)}_{int(target_chi)}.csv"
            cur_file_path = data_dir / cur_filename
            transitions_cnt += cur_length
            traj_cnt += 1
            if cur_file_path.exists():
                if my_logger is not None:
                    my_logger.info(f"process file: {cur_filename}")
                else:
                    print(f"process file: {cur_filename}")
                traj_file_cnt += 1
                cur_traj = pd.read_csv(cur_file_path.absolute())

                # 9.960(s) for 13 files with pd.iterrows
                # for index, row in cur_traj.iterrows():
                #     obs.append([*row[1:9], target_v, target_mu, target_chi])
                #     acts.append([*row[9:12]])
                #     infos.append(None)

                # 1.742(s) for 13 files without pd.iterrows
                obs.extend([{
                    "observation": np.array(item[0:8], dtype=np.float32),
                    "achieved_goal": np.array(item[3:6], dtype=np.float32),
                    "desired_goal": np.array(item[8:11], dtype=np.float32)
                } for item in zip(
                    cur_traj['s_phi'].tolist()[:-2],
                    cur_traj['s_theta'].tolist()[:-2],
                    cur_traj['s_psi'].tolist()[:-2],
                    cur_traj['s_v'].tolist()[:-2],
                    cur_traj['s_mu'].tolist()[:-2],
                    cur_traj['s_chi'].tolist()[:-2],
                    cur_traj['s_p'].tolist()[:-2],
                    cur_traj['s_h'].tolist()[:-2],
                    [target_v] * (cur_traj.count()['time'] - 1),
                    [target_mu] * (cur_traj.count()['time'] - 1),
                    [target_chi] * (cur_traj.count()['time'] - 1),
                )])
                """obs和acts 应为[:-1] 除去最后一个都要取 参考load_random_transitions_from_csv_files"""
                next_obs.extend([{
                    "observation": np.array(item[0:8], dtype=np.float32),
                    "achieved_goal": np.array(item[3:6], dtype=np.float32),
                    "desired_goal": np.array(item[8:11], dtype=np.float32)
                } for item in zip(
                    cur_traj['s_phi'].tolist()[1:],
                    cur_traj['s_theta'].tolist()[1:],
                    cur_traj['s_psi'].tolist()[1:],
                    cur_traj['s_v'].tolist()[1:],
                    cur_traj['s_mu'].tolist()[1:],
                    cur_traj['s_chi'].tolist()[1:],
                    cur_traj['s_p'].tolist()[1:],
                    cur_traj['s_h'].tolist()[1:],
                    [target_v] * (cur_traj.count()['time'] - 1),
                    [target_mu] * (cur_traj.count()['time'] - 1),
                    [target_chi] * (cur_traj.count()['time'] - 1),
                )])

                acts.extend(zip(
                    cur_traj['a_p'].tolist()[:-2],
                    cur_traj['a_nz'].tolist()[:-2],
                    cur_traj['a_pla'].tolist()[:-2],
                ))
                rewards.extend([0.] * (cur_traj.count()['time']-1))
                dones.extend([False] * (cur_traj.count()['time']-2) + [True])
                infos.extend([None] * (cur_traj.count()['time']-1))

    # 数据标准化. 这里的标准化最耗时.
    scaled_obs_env = ScaledObservationWrapper(origin_env)
    scaled_act_env = ScaledActionWrapper(scaled_obs_env)

    selected_obs = np.array(obs)
    selected_next_obs = np.array(obs)
    selected_actions = np.array(acts)
    selected_rewards = np.array(rewards)
    selected_dones = np.array(dones)
    selected_infos = np.array(infos)

    scaled_selected_obs = np.array([scaled_obs_env.scale_state(item) for item in selected_obs])
    scaled_selected_next_obs = np.array([scaled_obs_env.scale_state(item) for item in selected_next_obs])
    scaled_selected_acts = np.array([scaled_act_env.scale_action(np.array(item)) for item in selected_actions])

    if cache_data:
    # 缓存标准化后的数据
        if not cache_data_dir.exists():
            cache_data_dir.mkdir()

        if not cache_data_dir.exists():
            cache_data_dir.mkdir()

        np.save(str((cache_data_dir / "normalized_obs").absolute()), scaled_selected_obs)
        np.save(str((cache_data_dir / "normalized_next_obs").absolute()), scaled_selected_next_obs)
        np.save(str((cache_data_dir / "normalized_acts").absolute()), scaled_selected_acts)
        np.save(str((cache_data_dir / "rewards").absolute()), selected_rewards)
        np.save(str((cache_data_dir / "dones").absolute()), selected_dones)
        np.save(str((cache_data_dir / "infos").absolute()), np.array(selected_infos))

    # 输出统计量
    if my_logger is not None:
        my_logger.info(f"traj cnt: {traj_file_cnt}, transition(from *.csv) cnt: {len(obs)}, average traj length: {len(obs) / traj_file_cnt}")
        my_logger.info(f"traj cnt: {traj_cnt}, transition(from res.csv) cnt: {transitions_cnt}, average traj length: {transitions_cnt / traj_cnt}")
        my_logger.info(f"process time: {time() - start_time}(s).")
    else:
        print(f"traj cnt: {traj_file_cnt}, transition(from *.csv) cnt: {len(obs)}, average traj length: {len(obs) / traj_file_cnt}")
        print(f"traj cnt: {traj_cnt}, transition(from res.csv) cnt: {transitions_cnt}, average traj length: {transitions_cnt / traj_cnt}")
        print(f"process time: {time() - start_time}(s).")
    
    print(f"划分集合后总时间：{time() - start_time}(s).")
    
   
    return DictObs.from_obs_list(scaled_selected_obs), DictObs.from_obs_list(scaled_selected_next_obs), scaled_selected_acts, selected_rewards, selected_dones, selected_infos


