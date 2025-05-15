from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import matplotlib
from scipy.ndimage import gaussian_filter1d
# plt.style.use('ggplot')
sns.set_theme(context="notebook", style="darkgrid")
sns.set(font_scale=2.0)

PROJECT_ROOT_DIR = Path().absolute().parent


def load_data(algo: str, filename: str, seed_str: str, insert_no: int=19, algo_dir: str="rl", smooth_success_rate: bool=True):
    df = pd.read_csv(PROJECT_ROOT_DIR / "logs" / algo_dir / filename / "progress.csv")
    df = df[pd.notnull(df["eval/success_rate"])]
    df.insert(insert_no, "seed", [seed_str] * len(df))
    df.insert(insert_no+1, "algo", [algo] * len(df))

    # 平滑！！！
    if smooth_success_rate:
        # df["eval/success_rate"] = smooth(df["eval/success_rate"], 5)
        df["eval/success_rate"] = gaussian_filter1d(df["eval/success_rate"], sigma=1)

    return df

def load_two_stage_data(algo: str, first_stage_filename: str,second_stage_filename:str, seed_str: str, insert_no: int=19, algo_dir: str="rl", smooth_success_rate: bool=True):
    df_first_stage = pd.read_csv(PROJECT_ROOT_DIR / "logs" / algo_dir / first_stage_filename / "progress.csv")
    df_first_stage = df_first_stage[pd.notnull(df_first_stage["eval/success_rate"])]

    df_second_stage = pd.read_csv(PROJECT_ROOT_DIR / "logs" / algo_dir / second_stage_filename / "progress.csv")
    df_second_stage = df_second_stage[pd.notnull(df_second_stage["eval/success_rate"])]
    #df_second_stage = df_second_stage[df_second_stage["time/total_timesteps"]+500000]
    merge_df = pd.concat([df_first_stage,df_second_stage],ignore_index=True)
    merge_df.insert(insert_no, "seed", [seed_str] * len(merge_df))
    merge_df.insert(insert_no+1, "algo", [algo] * len(merge_df))


    # 平滑！！！
    if smooth_success_rate:
        # df["eval/success_rate"] = smooth(df["eval/success_rate"], 5)
        merge_df["eval/success_rate"] = gaussian_filter1d(merge_df["eval/success_rate"], sigma=1)

    return merge_df


SMOOTH = True

files = [
   "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_seed_1_singleRL",
   "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_seed_2_singleRL",
    "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_seed_3_singleRL",
    "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_seed_4_singleRL",
    "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_seed_5_singleRL",
]
seed_strs = [
    "seed 1",
    "seed 2",
    "seed 3",
    "seed 4",
    "seed 5"
]
df_hard_b_2_b05_first = pd.concat([load_data("b=2 to b=05  hard_first", filename, seed_str, insert_no=14, algo_dir="rl_single", smooth_success_rate=SMOOTH).iloc[::] for filename, seed_str in zip(files, seed_strs)])
#print(len(df_hard_b_2_b05_first))


files = [
   "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_b_05_5e5steps_seed_1_singleRL",
   "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_b_05_5e5steps_seed_2_singleRL",
    "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_b_05_5e5steps_seed_3_singleRL",
    "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_b_05_5e5steps_seed_4_singleRL",
    "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_b_05_5e5steps_seed_5_singleRL",
]
seed_strs = [
    "seed 1",
    "seed 2",
    "seed 3",
    "seed 4",
    "seed 5"
]
df_hard_b_2_b05_second = pd.concat([load_data("b=2 to b=05  hard_second", filename, seed_str, insert_no=14, algo_dir="rl_single", smooth_success_rate=SMOOTH).iloc[::] for filename, seed_str in zip(files, seed_strs)])
#print(len(df_hard_b_2_b05_second))
#print(df_hard_b_2_b05_second)

first_files = [
   "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_seed_1_singleRL",
   "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_seed_2_singleRL",
    "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_seed_3_singleRL",
    "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_seed_4_singleRL",
    "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_seed_5_singleRL",
]
second_files = [
   "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_b_05_5e5steps_seed_1_singleRL",
   "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_b_05_5e5steps_seed_2_singleRL",
    "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_b_05_5e5steps_seed_3_singleRL",
    "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_b_05_5e5steps_seed_4_singleRL",
    "D2D/hard_sac1e6/two_stage_hard_b2_b05/sac_her_10hz_128_128_b_2_5e5steps_b_05_5e5steps_seed_5_singleRL",
]
seed_strs = [
    "seed 1",
    "seed 2",
    "seed 3",
    "seed 4",
    "seed 5"
]
df_hard_b_2_b05 = pd.concat([load_two_stage_data("b=2 to b=05", first_filename,second_filename, seed_str, insert_no=14, algo_dir="rl_single", smooth_success_rate=SMOOTH).iloc[::] for first_filename, second_filename,seed_str in zip(first_files, second_files, seed_strs)])
print(len(df_hard_b_2_b05))
print(df_hard_b_2_b05)

data_plot = pd.concat([
    # df_hard_b_2_b05_first,
    # df_hard_b_2_b05_second,
    df_hard_b_2_b05
])

data_plot["eval/success_rate"] = data_plot["eval/success_rate"]

# 取部分数据做图
f, ax = plt.subplots(figsize=(10, 8))
ax = sns.lineplot(x="time/total_timesteps", y="eval/success_rate", hue="algo", data=data_plot, ax=ax)

ax.set_xlabel("env steps")
ax.set_ylabel("success rate")

sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 0.98), ncol=7, title=None, frameon=False, fontsize="x-small")
#sns.move_legend(ax, loc="upper right", title=None)

f.savefig("flycraft_vs_pyfly_SAC_in_rl_success_rate.pdf", format="pdf", bbox_inches="tight")