import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from flycraft.utils_common.geometry_utils import angle_of_2_3d_vectors

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))


def get_reward(v, mu, chi, goal_v, goal_mu, goal_chi, angle_scale=180., velocity_scale=100., b=0.5) -> float:

    plane_current_velocity_vector = [
        v * np.cos(np.deg2rad(mu)) * np.sin(np.deg2rad(chi)), 
        v * np.cos(np.deg2rad(mu)) * np.cos(np.deg2rad(chi)),
        v * np.sin(np.deg2rad(mu)),
    ]
    target_velocity_vector = [
        goal_v * np.cos(np.deg2rad(goal_mu)) * np.sin(np.deg2rad(goal_chi)), 
        goal_v * np.cos(np.deg2rad(goal_mu)) * np.cos(np.deg2rad(goal_chi)),
        goal_v * np.sin(np.deg2rad(goal_mu)),
    ]
    angle = angle_of_2_3d_vectors(plane_current_velocity_vector, target_velocity_vector)
    angle_base_reward = - np.power(angle / angle_scale, b)

    # 奖励公式参考论文“Reinforcement learning for UAV attitude control”的4.3节
    velocity_error = np.abs(goal_v -v)
    cliped_velocity_error = np.clip(velocity_error / velocity_scale, a_min=0., a_max=1.)
    velocity_base_reward = - np.power(cliped_velocity_error, b)

    return 0.5 * (angle_base_reward + velocity_base_reward)


def process_one_file(file_path, goal_v, goal_mu, goal_chi):
    traj_df = pd.read_csv(file_path)
    tmp_rewards = []
    for index, row in traj_df.iterrows():    
        tmp_rewards.append(
            get_reward(
                v=row["s_v"],
                mu=row["s_mu"],
                chi= row["s_chi"],
                goal_v=goal_v,
                goal_mu= goal_mu,
                goal_chi= goal_chi
            )
        )     
    traj_df.insert(13,"reward", tmp_rewards)
    traj_df.to_csv(file_path, sep=',', index=False)

# python demonstrations/utils/label_transitions_with_rewards.py --demos-dir demonstrations/data/10hz_10_5_5_test --traj-prefix my_f16trace
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--demos-dir", type=str, help="demonstration direction")
    parser.add_argument("--traj-prefix", type=str, default="traj", help="trajectory prefix")
    args = parser.parse_args()

    start_time = time.time()
    print("Start")

    path = PROJECT_ROOT_DIR / args.demos_dir
    res_df = pd.read_csv(os.path.join(path,"res.csv"))
    for index, row in tqdm(res_df.iterrows(), total=res_df.shape[0]):
        goal_v, goal_mu, goal_chi = row["v"], row["mu"], row["chi"]
        if row["length"] > 0:
            file_name = os.path.join(path, f"{args.traj_prefix}_{int(goal_v)}_{int(goal_mu)}_{int(goal_chi)}.csv")
            process_one_file(file_name, goal_v, goal_mu, goal_chi)
    
    end_time  = time.time()
    print(f"Total running time: {end_time - start_time} seconds.")
    print("End")
