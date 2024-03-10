from pathlib import Path
import pandas as pd
from tqdm import tqdm
import itertools
import logging
import sys

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
    debug: bool=False, 
    trajectory_save_prefix: str="traj",
    cur_expert_data_dir: Path=PROJECT_ROOT_DIR / "data" / "10hz_10_5_5_iter_1"
):
    """
    思路: 使用多个种子训练出来的策略更新专家轨迹.
    
    先把专家数据复制一份, 文件夹名称为{cur_expert_data_dir}, 然后对于每个策略, 运行一次本函数, 更新专家轨迹

    Args:
        policy_dir (Path): _description_
        debug (bool, optional): _description_. Defaults to False.
        model_save_name (str, optional): _description_. Defaults to "best_model".
        prev_expert_data_dir (str, optional): _description_. Defaults to "10hz_10_5_5_iter_1".
        cur_expert_data_dir (str, optional): _description_. Defaults to "10hz_10_5_5_iter_2".
    """
    # sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "runF16" / "algorithm" / "rollout" / "logs" / cur_expert_data_dir).absolute()), format_strings=['stdout', 'log', 'csv'])
    
    helper_env: FlyCraftEnv = FlyCraftEnv(config_file=env_config_file)
    scaled_obs_env = ScaledObservationWrapper(helper_env)
    scaled_act_env = ScaledActionWrapper(scaled_obs_env)
    env_config = load_config(env_config_file)

    algo.policy.set_training_mode(False)

    res_file = cur_expert_data_dir / "res.csv"
    res_df = pd.read_csv(res_file)

    # 枚举任务

    # 记录更新的轨迹数
    traj_renew_cnt = 0
    traj_add_cnt = 0
    
    for index, target in tqdm(res_df.iterrows(), total=res_df.shape[0]):
        # 为环境设置任务
        # target_v, target_mu, target_chi, expert_length = target["v"], target["mu"], target["chi"], target["length"]
        helper_env.task.goal_sampler.use_fixed_target = True
        helper_env.target_v = target["v"]
        helper_env.target_mu = target["mu"]
        helper_env.target_chi = target["chi"]
        helper_env.task.goal_sampler.goal_expert_length = target["length"]

        # 专家轨迹包括的字段: time,s_phi,s_theta,s_psi,s_v,s_mu,s_chi,s_p,s_h,a_p,a_nz,a_pla,a_rud
        traj = {
            "time": [],
            "s_phi": [],
            "s_theta": [],
            "s_psi": [],
            "s_v": [],
            "s_mu": [],
            "s_chi": [],
            "s_p": [],
            "s_h": [],
            "a_p": [],
            "a_nz": [],
            "a_pla": [],
            "a_rud": []
        }

        # 采样
        obs, info = scaled_act_env.reset()
        terminate = False
        s_index = 0
        while not terminate:
            action, _ = algo.predict(observation=obs, deterministic=True)
            obs, reward, terminate, truncated, info = scaled_act_env.step(action=action)
            
            # info包括的内容
            # {
            #     'step': 1, 
            #     'is_success': False, 
            #     'rewards': {'<runF16.env.rewards.dense_reward_based_on_angle_and_velocity.DenseRewardBasedOnAngleAndVelocity object at 0x7f7123446d90>': -0.8627592673161013}, 
            #     'plane_state': {'lef': 0.0, 'npos': 20.000000000000004, 'epos': 0.0, 'h': 5000.0, 'alpha': 2.6230275032725485, 'beta': 0.021475745218794213, 'phi': 0.0, 'theta': 2.6132205230940393, 'psi': 0.0, 'p': -74.3690767847601, 'q': 0.8627042874352823, 'r': -3.113576280240488, 'v': 199.87261389568465, 'vn': 199.87259692759682, 've': 0.07491674375388822, 'vh': -0.03421101226306433, 'nx': -0.08571581188146875, 'ny': 0.07641640436443489, 'nz': 0.9700209914318426, 'ele': -2.2219767026598327, 'ail': 21.5, 'rud': 0.0, 'thrust': 0.0, 'lon': 122.425, 'lat': 31.425180164618112, 'mu': 5.09213082360928e-16, 'chi': 0.0}, 
            #     'action': {'p': -180.0, 'nz': 9.0, 'pla': 0.0, 'rud': 0.0}
            # }

            traj["time"].append(s_index * 1. / env_config["task"].get("step_frequence", 10))
            traj["s_phi"].append(info["plane_state"]["phi"])
            traj["s_theta"].append(info["plane_state"]["theta"])
            traj["s_psi"].append(info["plane_state"]["psi"])
            traj["s_v"].append(info["plane_state"]["v"])
            traj["s_mu"].append(info["plane_state"]["mu"])
            traj["s_chi"].append(info["plane_state"]["chi"])
            traj["s_p"].append(info["plane_state"]["p"])
            traj["s_h"].append(info["plane_state"]["h"])
            traj["a_p"].append(info["action"]["p"])
            traj["a_nz"].append(info["action"]["nz"])
            traj["a_pla"].append(info["action"]["pla"])
            traj["a_rud"].append(info["action"]["rud"])

            s_index += 1

        # 对于能完成的轨迹,保存并更新res.csv中记录的轨迹长度
        if info["is_success"]:
            prev_length = (res_df.length[(res_df.v == helper_env.target_v) & (res_df.mu == helper_env.target_mu) & (res_df.chi == helper_env.target_chi)]).iloc[0]
            if prev_length == 0 or s_index < prev_length:
                traj_df = pd.DataFrame(data=traj, columns=["time", "s_phi", "s_theta", "s_psi", "s_v", "s_mu", "s_chi", "s_p", "s_h", "a_p", "a_nz", "a_pla", "a_rud"])
                traj_df.to_csv(cur_expert_data_dir / f"{trajectory_save_prefix}_{int(helper_env.target_v)}_{int(helper_env.target_mu)}_{int(helper_env.target_chi)}.csv", index=False)
                res_df.length[(res_df.v == helper_env.target_v) & (res_df.mu == helper_env.target_mu) & (res_df.chi == helper_env.target_chi)] = s_index
                print(f"\033[33m更新{helper_env.target_v}, {helper_env.target_mu}, {helper_env.target_chi}, length: from {prev_length} to {s_index}!!!\033[0m")
                if prev_length == 0:
                    traj_add_cnt += 1
                else:
                    traj_renew_cnt += 1
                print(f"新增了{traj_add_cnt}条轨迹，更新了{traj_renew_cnt}条轨迹")

    res_df.to_csv(res_file, index=False)
    print(f"一共新增了{traj_add_cnt}条轨迹，更新了{traj_renew_cnt}条轨迹.")


if __name__ == "__main__":
    algo_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "sac_her" / "best_model"
    env_config_file = PROJECT_ROOT_DIR / "configs" / "env" / "env_config_for_sac.json"
    cur_demonstration_dir = PROJECT_ROOT_DIR / "demonstrations" / "data" / "10hz_10_5_5_v2"

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
        cur_expert_data_dir=cur_demonstration_dir
    )