import os
import sys
from pathlib import Path
from typing import Tuple
import argparse
from copy import deepcopy

import numpy as np

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.sac import SAC

from flycraft.env import FlyCraftEnv
from flycraft.utils.load_config import load_config

PROJECT_ROOT_DIR: Path = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper

def rollout(
    algo: BaseAlgorithm, 
    env_config_file: Path, 
    rollout_transition_num: int=1000000,
    save_success_traj: bool=False,
    debug: bool=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """采样指定数量的transitions。使用单进程。返回的shape为[rollout_transition_num, *obs_shape], [rollout_transition_num, *action_shape], [rollout_transition_num, *obs_shape], [rollout_transition_num, ], [rollout_transition_num, ], [rollout_transition_num, ].

    Args:
        algo (BaseAlgorithm): SB3算法checkpoint路径
        env_config_file (Path): 环境配置文件路径
        rollout_transition_num (int, optional): 采样transition的数量. Defaults to 1000000.
        save_success_traj (bool, optional): 是否仅保存成功的轨迹. Defaults to False.
        debug (bool, optional): debug模式，输出统计量. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: observations, actions, next_observations, rewards, dones, infos
    """
    
    helper_env: FlyCraftEnv = FlyCraftEnv(config_file=env_config_file)
    scaled_obs_env = ScaledObservationWrapper(helper_env)
    scaled_act_env = ScaledActionWrapper(scaled_obs_env)

    algo.policy.set_training_mode(False)

    has_collected_transition_num = 0
    obs_arr, action_arr, next_obs_arr, reward_arr, done_arr, info_arr = [], [], [], [], [], []
    traj_length_log, success_traj_length_log = [], []
    
    while has_collected_transition_num < rollout_transition_num:
        
        tmp_obs_arr, tmp_action_arr, tmp_next_obs_arr, tmp_reward_arr, tmp_done_arr, tmp_info_arr = [], [], [], [], [], []

        # 采样
        obs, info = scaled_act_env.reset()
        terminate = False

        while not terminate:
            action, _ = algo.predict(observation=obs, deterministic=True)
            
            tmp_obs_arr.append(deepcopy(obs))
            tmp_action_arr.append(deepcopy(action))
            
            obs, reward, terminate, truncated, info = scaled_act_env.step(action=action)

            tmp_next_obs_arr.append(deepcopy(obs))
            tmp_reward_arr.append(reward)
            tmp_done_arr.append(terminate or truncated)
            tmp_info_arr.append(deepcopy(info))


        if (not save_success_traj) or (save_success_traj and info["is_success"]):
            
            if debug:
                if info["is_success"]:
                    print(f"\033[32m新增{helper_env.target_v}, {helper_env.target_mu}, {helper_env.target_chi}, length: {len(tmp_obs_arr)}!!!\033[0m")
                else:
                    print(f"\033[31m新增{helper_env.target_v}, {helper_env.target_mu}, {helper_env.target_chi}, length: {len(tmp_obs_arr)}!!!\033[0m")

            obs_arr.extend(tmp_obs_arr)
            action_arr.extend(tmp_action_arr)
            next_obs_arr.extend(tmp_next_obs_arr)
            reward_arr.extend(tmp_reward_arr)
            done_arr.extend(tmp_done_arr)
            info_arr.extend(tmp_info_arr)

            traj_length_log.append(len(tmp_obs_arr))
            has_collected_transition_num += len(tmp_obs_arr)
            if info["is_success"]:
                success_traj_length_log.append(len(tmp_obs_arr))

    print(f"一共记录了{len(traj_length_log)}条轨迹，轨迹平均长度：{np.array(traj_length_log).mean()}. 其中，成功轨迹{len(success_traj_length_log)}条，平均长度：{np.array(success_traj_length_log).mean()}")
    
    # 仅返回rollout_transition_num个transitions
    return (
        np.array(obs_arr[:rollout_transition_num]), 
        np.array(action_arr[:rollout_transition_num]), 
        np.array(next_obs_arr[:rollout_transition_num]), 
        np.array(reward_arr[:rollout_transition_num]), 
        np.array(done_arr[:rollout_transition_num]),
        np.array(info_arr[:rollout_transition_num])
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--policy-ckpt-dir", type=str, help="policy checkpoints dir")
    parser.add_argument("--env-config-dir", type=str, help="environment config dir")
    parser.add_argument("--rollout-transition-num", type=int, help="", default=1000)
    parser.add_argument("--only-save-success-traj", action="store_true", help="")
    args = parser.parse_args()

    algo_save_dir = Path(os.getcwd()) / args.policy_ckpt_dir
    env_config_file = Path(os.getcwd()) / args.env_config_dir
    cur_demonstration_dir = Path(os.getcwd()) / args.demos_dir

    env = FlyCraftEnv(config_file=env_config_file)

    sac_algo = SAC.load(
        algo_save_dir, 
        env=env,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space
        }    
    )
    
    rollout(
        algo=sac_algo, 
        env_config_file=env_config_file, 
        cur_expert_data_dir=cur_demonstration_dir,
        rollout_transition_num=args.rollout_transition_num,
        save_success_traj=args.only_save_success_traj,
        debug=True,
    )