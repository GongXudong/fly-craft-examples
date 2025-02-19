from pathlib import Path
import pandas as pd
import numpy as np
import itertools
import tqdm
import time
from ray.util.multiprocessing import Pool
import sys
import argparse

from flycraft.utils.my_log import get_logger

PROJECT_ROOT_DIR: Path = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from demonstrations.rollout_trajs.rollout_by_pid import Rollout


class ParallelScheduleForRollout:

    def __init__(self, 
        rollout_class=Rollout, 
        v_range: list=[100, 300], 
        v_interval: int=10, 
        mu_range: list=[-85, 85], 
        mu_interval: int=5, 
        chi_range: list=[-180, 180], 
        chi_interval: int=5,
        step_frequence: int=10,
        pool_size: int=30,
        traj_save_dir: Path=PROJECT_ROOT_DIR / "demonstrations" / "data" / "tmp",
    ) -> None:
        self.rollout_class = rollout_class
        self.v_range = v_range
        self.v_interval = v_interval
        self.mu_range = mu_range
        self.mu_interval = mu_interval
        self.chi_range = chi_range
        self.chi_interval = chi_interval
        self.step_frequence = step_frequence
        self.pool_size = pool_size
        self.traj_save_dir = traj_save_dir
        
    def work(self):
        start_time = time.time()
        log = {
            "v": [],
            "mu": [],
            "chi": [],
            "length": []
        }

        if not self.traj_save_dir.exists():
            self.traj_save_dir.mkdir()
        
        my_logger = get_logger(logger_name="ucav", log_file_dir=str(self.traj_save_dir / 'my_sys_logs.log'))

        def rollout_func(target):
            rollout_worker = self.rollout_class(
                target_v=target[0], target_mu=target[1], target_chi=target[2], 
                log_state_keys=["phi", "theta", "psi", "v", "mu", "chi", "p", "q", "r", "h", "lon", "lat", "thrust", "nx", "ny", "nz", "alpha", "beta", "lef", "npos", "epos"],
                log_guidance_law_action_keys=["p", "nz", "pla", "rud"],
                log_control_law_action_keys=["ail", "ele", "rud", "pla"],
                my_logger=None,  # my_logger,
                traj_save_dir=self.traj_save_dir,
                step_frequence=self.step_frequence,
            )
            episode_length = rollout_worker.rollout()
            print(f"check: {target[0], target[1], target[2], episode_length}")
            return target[0], target[1], target[2], episode_length

        with Pool(processes=self.pool_size) as pool:
            
            # 所有v、mu、chi的组合
            all_iters = itertools.product(
                range(self.v_range[0], self.v_range[1]+1, self.v_interval), 
                range(self.mu_range[0], self.mu_range[1]+1, self.mu_interval), 
                range(self.chi_range[0], self.chi_range[1]+1, self.chi_interval)
            )
            
            all_iter_num = \
                ((self.v_range[1] - self.v_range[0]) / self.v_interval + 1) * \
                ((self.mu_range[1] - self.mu_range[0]) / self.mu_interval + 1)* \
                ((self.chi_range[1] - self.chi_range[0]) / self.chi_interval + 1)

            # 显示进度条
            res = list(tqdm.tqdm(pool.imap(
                rollout_func, 
                all_iters   
            ), total=all_iter_num, desc="采样进度"))

        # res = [[*item] for item in res] 
        print(type(res[0]))
        tmp_res = []
        for i in range(len(res)):
            print(res[i])
            # tmp_res.append([*res[i]])
        res = np.array(res)
        
        print(res)
        log["v"] = res[:, 0]
        log["mu"] = res[:, 1]
        log["chi"] = res[:, 2]
        log["length"] = res[:, 3]

        df = pd.DataFrame(log)
        df.to_csv(self.traj_save_dir / "res.csv", index=False)

        print("time used: ", time.time() - start_time)

# python demonstrations/rollout_trajs/rollout_by_pid_parallel.py --data-dir-suffix e2e_iter_1 --step-frequence 10 --v-min 100 --v-max 300 --v-interval 10 --mu-min -85 --mu-max 85 --mu-interval 5 --chi-min -180 --chi-max 175 --chi-interval 5
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data-dir-suffix", type=str, default="v1", help="suffix of data dir")
    parser.add_argument("--step-frequence", type=int, default=10, help="simulation frequence")
    parser.add_argument("--v-min", type=int, default=100, help="minimal value of speed")
    parser.add_argument("--v-max", type=int, default=300, help="maximum value of speed")
    parser.add_argument("--v-interval", type=int, default=10, help="sample interval of speed")
    parser.add_argument("--mu-min", type=int, default=-85, help="minimal value of flight path elevator angle")
    parser.add_argument("--mu-max", type=int, default=85, help="maximum value of flight path elevator angle")
    parser.add_argument("--mu-interval", type=int, default=5, help="sample interval of flight path elevator angle")
    parser.add_argument("--chi-min", type=int, default=-170, help="minimal value of flight path azimuth angle")
    parser.add_argument("--chi-max", type=int, default=170, help="maximum value of flight path azimuth angle")
    parser.add_argument("--chi-interval", type=int, default=5, help="sample interval of flight path azimuth angle")
    args = parser.parse_args()

    s = ParallelScheduleForRollout(
        rollout_class=Rollout, 
        v_range=[args.v_min, args.v_max], v_interval=args.v_interval, 
        mu_range=[args.mu_min, args.mu_max], mu_interval=args.mu_interval, 
        chi_range=[args.chi_min, args.chi_max], chi_interval=args.chi_interval,
        step_frequence=args.step_frequence,
        pool_size=10,
        traj_save_dir=PROJECT_ROOT_DIR / "demonstrations" / "data" / f"{args.step_frequence}hz_{args.v_interval}_{args.mu_interval}_{args.chi_interval}_{args.data_dir_suffix}"
    )
    s.work()
