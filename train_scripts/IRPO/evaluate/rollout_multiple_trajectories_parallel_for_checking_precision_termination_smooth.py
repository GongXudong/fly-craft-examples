from pathlib import Path
import sys
import argparse
from copy import deepcopy
import pandas as pd
import numpy as np
from tqdm import tqdm
from ray.util.multiprocessing import Pool

from stable_baselines3 import SAC
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecCheckNan

from flycraft.env import FlyCraftEnv
from flycraft.utils.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.models.ppo_with_bc_loss import PPOWithBCLoss
from utils_my.sb3.vec_env_helper import get_vec_env, make_env
from utils_my.smoothness.smoothness_measure import smoothness_measure_by_delta, smoothness_measure_by_fft
from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper


STEP_FREQUENCE = 10


def termination_shotcut(termination_str: str):
    if termination_str.find("reach_target") != -1:
        return "reach target"
    if termination_str.find("timeout") != -1:
        return "timeout"
    if termination_str.find("move_away") != -1:
        return "continuous move away"
    if termination_str.find("roll") != -1:
        return "continuous roll"
    if termination_str.find("crash") != -1:
        return "crash"
    if termination_str.find("extreme") != -1:
        return "extreme state"
    if termination_str.find("negative") != -1:
        return "negative overload"

def rollout(
    policy_dir_str: str, 
    algo_str: str,
    env_config_path: str,
    target_goals_v: list,
    target_goals_mu: list,
    target_goals_chi: list,
    debug: bool=False, 
    model_save_name: str="best_model",
    seed: int=42,
):
    policy_dir = Path(policy_dir_str)
    target_goals = pd.DataFrame({
        "v": target_goals_v,
        "mu": target_goals_mu,
        "chi": target_goals_chi,
    })

    env = FlyCraftEnv(
        config_file=env_config_path,
        custom_config={}
    )
    scaled_obs_env = ScaledObservationWrapper(env)
    scaled_act_env = ScaledActionWrapper(scaled_obs_env)
    scaled_act_env.reset(seed=seed)

    scaled_act_env.unwrapped.task.goal_sampler.use_fixed_goal = True

    if algo_str == "ppo":
        algo_class = PPOWithBCLoss
    elif algo_str =="sac":
        algo_class = SAC

    algo = algo_class.load(
        str((policy_dir / model_save_name).absolute()), 
        env=scaled_act_env,
        custom_objects={
            "observation_space": scaled_act_env.observation_space,
            "action_space": scaled_act_env.action_space
        }
    )
    algo.policy.set_training_mode(False)

    res_dict = {
        "v": [],
        "mu": [],
        "chi": [],
        "length": [],
        "termination": [],
        "achieved v": [],
        "achieved mu": [],
        "achieved chi": [],
        "smooth_a_ail": [],
        "smooth_a_ele": [],
        "smooth_a_rud": [],
        "smooth_a_pla": [],
        "smooth2_a_ail": [],
        "smooth2_a_ele": [],
        "smooth2_a_rud": [],
        "smooth2_a_pla": [],
        "smooth_s_phi": [],
        "smooth_s_theta": [],
        "smooth_s_psi": [],
        "smooth_s_v": [],
        "smooth_s_mu": [],
        "smooth_s_chi": [],
        "smooth_s_p": [],
        "smooth_s_h": [],

    }

    # 枚举任务
    for index, target in tqdm(target_goals.iterrows(), total=target_goals.shape[0]):
        # 为环境设置任务
        # target_v, target_mu, target_chi, expert_length = target["v"], target["mu"], target["chi"], target["length"]
        scaled_act_env.unwrapped.task.goal_sampler.goal_v = target["v"]
        scaled_act_env.unwrapped.task.goal_sampler.goal_mu = target["mu"]
        scaled_act_env.unwrapped.task.goal_sampler.goal_chi = target["chi"]

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
            "a_ail": [],
            "a_ele": [],
            "a_rud": [],
            "a_pla": [],
        }

        # 采样
        obs, info = scaled_act_env.reset()

        tmp_obs = scaled_obs_env.inverse_scale_state(obs)

        terminate = False
        s_index = 0
        while not terminate:
            action, _ = algo.predict(observation=obs, deterministic=True)
            obs, reward, terminate, truncated, info = scaled_act_env.step(action=action)
            
            traj["time"].append(s_index * 1. / STEP_FREQUENCE)
            traj["s_phi"].append(info["plane_state"]["phi"])
            traj["s_theta"].append(info["plane_state"]["theta"])
            traj["s_psi"].append(info["plane_state"]["psi"])
            traj["s_v"].append(info["plane_state"]["v"])
            traj["s_mu"].append(info["plane_state"]["mu"])
            traj["s_chi"].append(info["plane_state"]["chi"])
            traj["s_p"].append(info["plane_state"]["p"])
            traj["s_h"].append(info["plane_state"]["h"])

            # traj["a_p"].append(info["action"]["p"])
            # traj["a_nz"].append(info["action"]["nz"])
            # traj["a_pla"].append(info["action"]["pla"])
            # traj["a_rud"].append(info["action"]["rud"])
            traj["a_ail"].append(info["plane_state"]["ail"])
            traj["a_ele"].append(info["plane_state"]["ele"])
            traj["a_rud"].append(info["plane_state"]["rud"])
            traj["a_pla"].append(info["plane_state"]["thrust"])
            
            s_index += 1

        traj_df = pd.DataFrame(data=traj, columns=["time", "s_phi", "s_theta", "s_psi", "s_v", "s_mu", "s_chi", "s_p", "s_h", "a_ail", "a_ele", "a_rud", "a_pla"])

        res_dict["v"].append(target["v"])
        res_dict["mu"].append(target["mu"])
        res_dict["chi"].append(target["chi"])
        res_dict["achieved v"].append(deepcopy(info["plane_next_state"]["v"]))
        res_dict["achieved mu"].append(deepcopy(info["plane_next_state"]["mu"]))
        res_dict["achieved chi"].append(deepcopy(info["plane_next_state"]["chi"]))
        res_dict["length"].append(s_index)
        res_dict["termination"].append(termination_shotcut(info["termination"]))
        
        res_dict["smooth_a_ail"].append(smoothness_measure_by_delta(traj_df, measure_columns=["a_ail"])[0])
        res_dict["smooth_a_ele"].append(smoothness_measure_by_delta(traj_df, measure_columns=["a_ele"])[0])
        res_dict["smooth_a_rud"].append(smoothness_measure_by_delta(traj_df, measure_columns=["a_rud"])[0])
        res_dict["smooth_a_pla"].append(smoothness_measure_by_delta(traj_df, measure_columns=["a_pla"])[0])

        res_dict["smooth2_a_ail"].append(smoothness_measure_by_fft(traj_df, measure_columns=["a_ail"])[0])
        res_dict["smooth2_a_ele"].append(smoothness_measure_by_fft(traj_df, measure_columns=["a_ele"])[0])
        res_dict["smooth2_a_rud"].append(smoothness_measure_by_fft(traj_df, measure_columns=["a_rud"])[0])
        res_dict["smooth2_a_pla"].append(smoothness_measure_by_fft(traj_df, measure_columns=["a_pla"])[0])
        
        res_dict["smooth_s_phi"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_phi"])[0])
        res_dict["smooth_s_theta"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_theta"])[0])
        res_dict["smooth_s_psi"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_psi"])[0])
        res_dict["smooth_s_v"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_v"])[0])
        res_dict["smooth_s_mu"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_mu"])[0])
        res_dict["smooth_s_chi"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_chi"])[0])
        res_dict["smooth_s_p"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_p"])[0])
        res_dict["smooth_s_h"].append(smoothness_measure_by_delta(traj_df, measure_columns=["s_h"])[0])

    return res_dict

# python evaluate/sb3_rollout_parallel_for_checking_precision_termination_smooth.py --config-file-name configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json --algo ppo --eval-traj-num 1000 --process-num 10 --seed 123 --save-file-name ppo_easy_1.csv

# python evaluate/sb3_rollout_parallel_for_checking_precision_termination_smooth.py --config-file-name configs/train/sac/easy_her/sac_config_10hz_128_128_1.json --algo sac --eval-traj-num 1000 --process-num 10 --seed 123 --save-file-name sac_easy_her_1.csv

# python evaluate/sb3_rollout_parallel_for_checking_precision_termination_smooth.py --config-file-name configs/train/sac/easy_her_end_to_end_mode/sac_config_10hz_128_128_1.json --algo sac --eval-traj-num 1000 --process-num 10 --seed 123 --save-file-name sac_easy_her_end2end_1.csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="传入配置文件")
    # parser.add_argument("--config-file-name", type=str, help="配置文件名", default="configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json")
    # parser.add_argument("--algo", type=str, help="算法(ppo, sac)", default="ppo")
    parser.add_argument("--eval-traj-num", type=int, help="测试使用的轨迹数", default=1000)
    parser.add_argument("--process-num", type=int, help="使用的采样进程数", default=10)
    parser.add_argument("--seed", type=int, help="随机种子", default=42)
    # parser.add_argument("--save-file-name", type=str, help="", default="res.csv")
    args = parser.parse_args()

    
    
    evaluation_goals = pd.DataFrame({
        "v": [np.random.random() * (250. - 150.) + 150. for i in range(args.eval_traj_num)],
        "mu": [np.random.random() * (10. - (-10.)) + (-10.) for i in range(args.eval_traj_num)],
        "chi": [np.random.random() * (30. - (-30.)) + (-30.) for i in range(args.eval_traj_num)],
    })

    # TODO: 在相同的任务集合上测试

    ppo_config_files = [
        "configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json",
        "configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_2.json",
        "configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_3.json",
        "configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_4.json",
        "configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_5.json",
    ]
    ppo_algo_strs = ["ppo"] * len(ppo_config_files)
    ppo_save_file_names= [
        "ppo_easy_guidance_law_mode_1.csv",
        "ppo_easy_guidance_law_mode_2.csv",
        "ppo_easy_guidance_law_mode_3.csv",
        "ppo_easy_guidance_law_mode_4.csv",
        "ppo_easy_guidance_law_mode_5.csv",
    ]

    ppo_e2e_config_files = [
        "configs/train/ppo/easy_end_to_end/ppo_bc_config_10hz_128_128_easy_1.json",
        "configs/train/ppo/easy_end_to_end/ppo_bc_config_10hz_128_128_easy_2.json",
        "configs/train/ppo/easy_end_to_end/ppo_bc_config_10hz_128_128_easy_3.json",
        "configs/train/ppo/easy_end_to_end/ppo_bc_config_10hz_128_128_easy_4.json",
        "configs/train/ppo/easy_end_to_end/ppo_bc_config_10hz_128_128_easy_5.json",
    ]
    ppo_e2e_algo_strs = ["ppo"] * len(ppo_e2e_config_files)
    ppo_e2e_save_file_names = [
        "ppo_easy_end_to_end_mode_1.csv",
        "ppo_easy_end_to_end_mode_2.csv",
        "ppo_easy_end_to_end_mode_3.csv",
        "ppo_easy_end_to_end_mode_4.csv",
        "ppo_easy_end_to_end_mode_5.csv",
    ]

    sac_config_files = [
        "configs/train/sac/easy_her/sac_config_10hz_128_128_1.json",
        "configs/train/sac/easy_her/sac_config_10hz_128_128_2.json",
        "configs/train/sac/easy_her/sac_config_10hz_128_128_3.json",
        "configs/train/sac/easy_her/sac_config_10hz_128_128_4.json",
        "configs/train/sac/easy_her/sac_config_10hz_128_128_5.json",
    ]
    sac_algo_strs = ["sac"] * len(sac_config_files)
    sac_save_file_names= [
        "sac_easy_guidance_law_mode_1.csv",
        "sac_easy_guidance_law_mode_2.csv",
        "sac_easy_guidance_law_mode_3.csv",
        "sac_easy_guidance_law_mode_4.csv",
        "sac_easy_guidance_law_mode_5.csv",
    ]

    sac_e2e_config_files = [
        "configs/train/sac/easy_her_end_to_end_mode/sac_config_10hz_128_128_1.json",
        "configs/train/sac/easy_her_end_to_end_mode/sac_config_10hz_128_128_2.json",
        "configs/train/sac/easy_her_end_to_end_mode/sac_config_10hz_128_128_3.json",
        "configs/train/sac/easy_her_end_to_end_mode/sac_config_10hz_128_128_4.json",
        "configs/train/sac/easy_her_end_to_end_mode/sac_config_10hz_128_128_5.json",
    ]
    sac_e2e_algo_strs = ["sac"] * len(sac_config_files)
    sac_e2e_save_file_names= [
        "sac_easy_control_law_mode_1.csv",
        "sac_easy_control_law_mode_2.csv",
        "sac_easy_control_law_mode_3.csv",
        "sac_easy_control_law_mode_4.csv",
        "sac_easy_control_law_mode_5.csv",
    ]

    for config_file_name, algo_str, save_file_name in zip(
        [
            *ppo_config_files, 
            *ppo_e2e_config_files,
            # *sac_config_files, 
            # *sac_e2e_config_files,
        ],
        [
            *ppo_algo_strs, 
            *ppo_e2e_algo_strs,
            # *sac_algo_strs, 
            # *sac_e2e_algo_strs,
        ],
        [
            *ppo_save_file_names, 
            *ppo_e2e_save_file_names,
            # *sac_save_file_names, 
            # *sac_e2e_save_file_names,
        ]
    ):

        print(f"processing: {config_file_name}")

        with Pool(processes=args.process_num) as pool:

            train_config = load_config(PROJECT_ROOT_DIR / config_file_name)
            env_config = load_config(PROJECT_ROOT_DIR / "configs" / "env" / train_config["env"].get("config_file", "env_config_for_sac.json"))

            v_min, v_max = env_config["goal"]["v_min"], env_config["goal"]["v_max"]
            mu_min, mu_max = env_config["goal"]["mu_min"], env_config["goal"]["mu_max"]
            chi_min, chi_max = env_config["goal"]["chi_min"], env_config["goal"]["chi_max"]

            # 设置要分成的份数
            n = args.process_num
            # 计算每份的行数
            chunk_size = len(evaluation_goals) // n
            # 分割DataFrame
            chunks = [evaluation_goals.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(n)]
            # 如果不能完全均分，处理剩余的数据
            if len(evaluation_goals) % n != 0:
                # 将剩余的数据分配到最后一个chunk
                last_chunk = evaluation_goals.iloc[n*chunk_size:]
                chunks[-1] = pd.concat([chunks[-1], last_chunk])

            model_name = "best_model"

            res = pool.starmap(
                rollout,
                [[
                    str(PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / train_config["rl"]["experiment_name"]),
                    algo_str,
                    str(PROJECT_ROOT_DIR / "configs" / "env" / train_config["env"]["config_file"]),
                    list(target.v),
                    list(target.mu),
                    list(target.chi),
                    False,
                    model_name,
                    args.seed
                ] for target in chunks]
            )

            res_df = pd.concat([pd.DataFrame(tmp) for tmp in res])
            res_df.to_csv(PROJECT_ROOT_DIR / "cache" / save_file_name, index=False)
