from pathlib import Path
import sys
import argparse
from copy import deepcopy
import pandas as pd
import numpy as np
from tqdm import tqdm
from ray.util.multiprocessing import Pool

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.logger import configure, Logger

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.algorithm.models.sb3_model import PPOWithBCLoss
from runF16.configs.load_config import load_config
from runF16.env.guidenv_dense_reward_filter_by_difficulty import GuideEnvDenseRewardFilteredByDifficulty
from runF16.env.utils.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper


def get_ppo_algo(env):
    policy_kwargs = dict(
        net_arch=dict(
            pi=NET_ARCH,
            vf=deepcopy(NET_ARCH)
        )
    )

    return PPOWithBCLoss(
        policy=MlpPolicy, 
        env=env, 
        seed=SEED,
        batch_size=PPO_BATCH_SIZE,
        gamma=GAMMA,
        n_steps=2048,  # 采样时每个环境采样的step数
        n_epochs=5,  # 采样的数据在训练中重复使用的次数
        policy_kwargs=policy_kwargs
    )


def rollout(
    policy_dir_str: str, 
    target_goals_v: list,
    target_goals_mu: list,
    target_goals_chi: list,
    debug: bool=False, 
    model_save_name: str="best_model", 
    cur_expert_data_dir: str="10hz_10_5_5_iter_2",
):
    """
    思路: 使用多个种子训练出来的策略更新专家轨迹.
    
    先把专家数据复制一份, 文件夹名称为{cur_expert_data_dir}, 然后对于每个策略, 运行一次本函数, 更新专家轨迹

    Args:
        policy_dir (str): _description_
        debug (bool, optional): _description_. Defaults to False.
        model_save_name (str, optional): _description_. Defaults to "best_model".
        prev_expert_data_dir (str, optional): _description_. Defaults to "10hz_10_5_5_iter_1".
        cur_expert_data_dir (str, optional): _description_. Defaults to "10hz_10_5_5_iter_2".
    """
    # sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "runF16" / "algorithm" / "rollout" / "logs" / cur_expert_data_dir).absolute()), format_strings=['stdout', 'log', 'csv'])

    policy_dir = Path(policy_dir_str)
    target_goals = pd.DataFrame({
        "v": target_goals_v,
        "mu": target_goals_mu,
        "chi": target_goals_chi
    })

    env_config = {
        "num_process": 1, 
        "logger": None, 
        "step_frequence": STEP_FREQUENCE,
        "max_simulate_time": MAX_SIMULATE_TIME,
        "gamma": GAMMA,
        "reward_scale_factor_b": 0.5,  # GuideEnvDenseReward中使用，取值小于1效果相当于在数值上放大了奖励，大于1相当于缩小了奖励
        "available_targets_dir": PROJECT_ROOT_DIR / "my_logs_for_parallel" / cur_expert_data_dir,
        "available_targets_filename": "res.csv",
        "use_fixed_target": False, 
        "sample_target_noise_std": [0., 0., 0.],
        "difficulty": -2  # 从所有任务上采样
    }
    env_class = GuideEnvDenseRewardFilteredByDifficulty
    
    helper_env = env_class(env_config=env_config)
    scaled_obs_env = ScaledObservationWrapper(helper_env)
    scaled_act_env = ScaledActionWrapper(scaled_obs_env)

    algo_ppo = PPOWithBCLoss.load(
        str((policy_dir / model_save_name).absolute()), 
        custom_objects={
            "observation_space": scaled_act_env.observation_space,
            "action_space": scaled_act_env.action_space
        }
    )
    algo_ppo.policy.set_training_mode(False)

    res_file = PROJECT_ROOT_DIR / "my_logs_for_parallel" / cur_expert_data_dir / "res.csv"

    # 记录更新的轨迹数
    traj_renew_cnt = 0
    traj_add_cnt = 0
    
    for index, target in tqdm(target_goals.iterrows(), total=target_goals.shape[0]):
        # 为环境设置任务
        # target_v, target_mu, target_chi, expert_length = target["v"], target["mu"], target["chi"], target["length"]
        helper_env.use_fixed_target = True
        tmp_noise = helper_env.sample_noise()
        helper_env.target_v = target["v"] + tmp_noise[0]
        helper_env.target_mu = target["mu"] + tmp_noise[1]
        helper_env.target_chi = target["chi"] + tmp_noise[2]

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
            action, _ = algo_ppo.predict(observation=obs, deterministic=True)
            tmp_obs = scaled_obs_env.inverse_scale_state(obs[-11:])
            obs, reward, terminate, truncated, info = scaled_act_env.step(action=action)

            traj["time"].append(s_index * 1. / STEP_FREQUENCE)
            traj["s_phi"].append(info["origin_obs"]["phi"])
            traj["s_theta"].append(info["origin_obs"]["theta"])
            traj["s_psi"].append(info["origin_obs"]["psi"])
            traj["s_v"].append(info["origin_obs"]["v"])
            traj["s_mu"].append(info["origin_obs"]["mu"])
            traj["s_chi"].append(info["origin_obs"]["chi"])
            traj["s_p"].append(info["origin_obs"]["p"])
            traj["s_h"].append(info["origin_obs"]["h"])
            traj["a_p"].append(info["action"]["p"])
            traj["a_nz"].append(info["action"]["nz"])
            traj["a_pla"].append(info["action"]["pla"])
            traj["a_rud"].append(info["action"]["rud"])

            s_index += 1

        # 对于能完成的轨迹，读取相应的轨迹文件，判断是否比已有轨迹更短，短的话，就替换
        if info["is_success"]:
            traj_df = pd.DataFrame(data=traj, columns=["time", "s_phi", "s_theta", "s_psi", "s_v", "s_mu", "s_chi", "s_p", "s_h", "a_p", "a_nz", "a_pla", "a_rud"])

            tmp_traj_file = PROJECT_ROOT_DIR / "my_logs_for_parallel" / cur_expert_data_dir / f"my_f16trace_{str(int(helper_env.target_v))}_{str(int(helper_env.target_mu))}_{str(int(helper_env.target_chi))}.csv"

            if tmp_traj_file.exists():
                prev_traj = pd.read_csv(tmp_traj_file)
                if prev_traj.shape[0] > traj_df.shape[0]:
                    traj_df.to_csv(tmp_traj_file, index=False)
                    traj_renew_cnt += 1
                    print(f"\033[33m更新{helper_env.target_v}, {helper_env.target_mu}, {helper_env.target_chi}, length: from {prev_traj.shape[0]} to {traj_df.shape[0]}!!!\033[0m")
                    print(f"新增了{traj_add_cnt}条轨迹，更新了{traj_renew_cnt}条轨迹")
            else:
                traj_df.to_csv(tmp_traj_file, index=False)
                traj_add_cnt += 1
                print(f"\033[33m新增{helper_env.target_v}, {helper_env.target_mu}, {helper_env.target_chi}, length: {traj_df.shape[0]}!!!\033[0m")
                print(f"新增了{traj_add_cnt}条轨迹，更新了{traj_renew_cnt}条轨迹")

    print(f"一共新增了{traj_add_cnt}条轨迹，更新了{traj_renew_cnt}条轨迹.")

# python runF16/algorithm/rollout/sb3_rollout_all_trajs_parallel.py --config-file-name no_framestack_configs/seed2/config_10hz_loss_128_128.json --cur-expert-data-dir 10hz_10_5_5_iter_2_rebuttal_pilot --process-num 64

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="传入配置文件")
    parser.add_argument("--config-file-name", type=str, help="配置文件名", default="no_framestack_configs/seed2/config_10hz_loss_128_128.json")
    parser.add_argument("--cur-expert-data-dir", type=str, help="当前专家数据目录", default="10hz_10_5_5_iter_2")
    parser.add_argument("--process-num", type=int, help="使用的采样进程数", default=10)
    args = parser.parse_args()

    custom_config = load_config(args.config_file_name)

    RL_EXPERIMENT_NAME = custom_config["rl_bc"]["experiment_name"]
    RL_SINGLE_EXPERIMENT_NAME = custom_config["rl"]["experiment_name"]
    BC_EXPERIMENT_NAME = custom_config["bc"]["experiment_name"]
    SEED = custom_config["rl_bc"]["seed"]
    SEED_FOR_LOAD_ALGO = custom_config["rl_bc"]["seed_for_load_algo"]
    STEP_FREQUENCE = custom_config["step_frequence"]
    BC_POLICY_FILE_NAME = custom_config["bc"]["policy_file_save_name"]
    EXPERT_DATA_DIR = custom_config["rollout"]["expert_data_dir"]
    NET_ARCH = custom_config["rl_bc"]["net_arch"]
    PPO_BATCH_SIZE = custom_config["rl_bc"]["batch_size"]
    GAMMA = custom_config["rl_bc"]["gamma"]
    ACTIVATE_VALUE_HEAD_TRAIN_STEPS = custom_config["rl_bc"]["activate_value_head_train_steps"]
    RL_TRAIN_STEPS = custom_config["rl_bc"]["train_steps"]
    ROLLOUT_PROCESS_NUM = custom_config["rl_bc"]["rollout_process_num"]

    MAX_SIMULATE_TIME = custom_config["env"]['max_simulate_time']

    rl_policy_save_dir = PROJECT_ROOT_DIR / "runF16" / "algorithm" / "checkpoints_sb3" / "rl" / RL_EXPERIMENT_NAME
    
    with Pool(processes=args.process_num) as pool:
        res_df = pd.read_csv(PROJECT_ROOT_DIR / "my_logs_for_parallel" / args.cur_expert_data_dir / "res.csv")
        # 设置要分成的份数
        n = args.process_num
        # 计算每份的行数
        chunk_size = len(res_df) // n
        # 分割DataFrame
        chunks = [res_df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(n)]
        # 如果不能完全均分，处理剩余的数据
        if len(res_df) % n != 0:
            # 将剩余的数据分配到最后一个chunk
            last_chunk = res_df.iloc[n*chunk_size:]
            chunks[-1] = pd.concat([chunks[-1], last_chunk])

        res = pool.starmap(
            rollout,
            [[
                str(rl_policy_save_dir),
                list(target.v),
                list(target.mu),
                list(target.chi),
                False,
                "best_model",
                args.cur_expert_data_dir
             ] for target in chunks]
        )

        print(res)
