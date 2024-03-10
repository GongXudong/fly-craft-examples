from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))


def inverse_traj_df(traj_df: pd.DataFrame) -> pd.DataFrame:
    """将一条轨迹反转

    反转s_phi, s_psi, s_chi, s_p, a_p。

    Args:
        traj_df (pd.DataFrame): 飞行轨迹，轨迹包括列：time,s_phi,s_theta,s_psi,s_v,s_mu,s_chi,s_p,s_h,a_p,a_nz,a_pla,a_rud。

    Returns:
        pd.DataFrame: 反转s_phi, s_psi, s_chi, s_p, a_p后的轨迹
    """
    assert "s_phi" in traj_df.columns, "没有列：s_phi"
    assert "s_psi" in traj_df.columns, "没有列：s_psi"
    assert "s_chi" in traj_df.columns, "没有列：s_chi"
    assert "s_p" in traj_df.columns, "没有列：s_p"
    assert "a_p" in traj_df.columns, "没有列：a_p"

    traj_df["s_phi"] *= -1
    traj_df["s_psi"] *= -1
    traj_df["s_chi"] *= -1
    traj_df["s_p"] *= -1
    traj_df["a_p"] *= -1

    return traj_df

def update_traj(v: int, mu: int, chi: int, mirror_chi: int, expert_data_dir: Path, trajectory_save_prefix: str="traj"):
    """根据chi的轨迹更新mirror_chi的轨迹（调用此函数前，保证chi的轨迹比mirror_chi的轨迹短）

    Args:
        chi (_type_): _description_
        mirror_chi (_type_): _description_
    """

    chi_file_name = f"{trajectory_save_prefix}_{str(v)}_{str(mu)}_{str(chi)}.csv"
    mirror_chi_file_name = f"{trajectory_save_prefix}_{str(v)}_{str(mu)}_{str(mirror_chi)}.csv"
    
    assert (expert_data_dir / chi_file_name).exists(), f"{chi_file_name}不存在！！！"
    # assert (expert_data_dir / mirror_chi_file_name).exists(), f"{mirror_chi_file_name}不存在！！！"

    chi_df = pd.read_csv(expert_data_dir / chi_file_name)
    mirror_chi_df = inverse_traj_df(chi_df)
    # print(f"开始保存{mirror_chi_file_name}....................")
    mirror_chi_df.to_csv(expert_data_dir / mirror_chi_file_name, index=False)
    # print(f"完成保存{mirror_chi_file_name}!!!!!!!!!!!")

def process(
        expert_data_dir: Path,
        res_file_name: str="res.csv",
        trajectory_save_prefix: str="traj"
    ):
    expert_data_res_file = expert_data_dir / res_file_name

    res_df = pd.read_csv(expert_data_res_file)
    expert_trajs = res_df
    # expert_trajs = res_df[res_df.length > 0]
    # print(len(expert_trajs))
    
    update_traj_cnt = 0
    add_traj_cnt = 0

    for index, row in tqdm(expert_trajs.iterrows(), total=expert_trajs.shape[0]):

        cur_v, cur_mu, cur_chi, cur_length = int(row["v"]), int(row["mu"]), int(row["chi"]), int(row["length"])
        # print(cur_v, cur_mu, cur_chi, cur_length)

        if cur_length > 0:
            mirror_chi = -cur_chi
            mirror_traj = expert_trajs[(expert_trajs.v == cur_v) & (expert_trajs.mu == cur_mu) & (expert_trajs.chi == mirror_chi)].iloc[0]

            mirror_length = int(mirror_traj["length"])

            if cur_length < mirror_length or mirror_length == 0:
                print(f"更新{str(cur_v)}_{str(cur_mu)}_{str(-cur_chi)}, from {str(mirror_length)} to {str(cur_length)}.")
                # 更新轨迹csv文件
                update_traj(cur_v, cur_mu, cur_chi, mirror_chi, expert_data_dir)
                # 更新记录
                expert_trajs.length[(expert_trajs.v == cur_v) & (expert_trajs.mu == cur_mu) & (expert_trajs.chi == mirror_chi)] = cur_length
                
                if mirror_length == 0:
                    add_traj_cnt += 1
                else:
                    update_traj_cnt += 1

                print(f"新增了{add_traj_cnt}条轨迹，更新了{update_traj_cnt}条轨迹。")

    expert_trajs.to_csv(expert_data_res_file, index=False)
    print(f"一共新增了{add_traj_cnt}条轨迹，更新了{update_traj_cnt}条轨迹。")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--demos-dir", type=str, help="demonstration dir")
    parser.add_argument("--traj-prefix", type=str, default="traj", help="trajectory prefix")
    args = parser.parse_args()

    process(
        expert_data_dir=Path(os.getcwd()) / args.demos_dir,
        trajectory_save_prefix=args.traj_prefix
    )
