from pathlib import Path
import pandas as pd
import numpy as np
import itertools
import tqdm
import time
from ray.util.multiprocessing import Pool
import sys

from flycraft.utils.my_log import get_logger

PROJECT_ROOT_DIR: Path = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

print(sys.path)

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
        traj_save_dir: Path=PROJECT_ROOT_DIR / "data" / "tmp",
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
        
        my_logger = get_logger(logger_name="ucav", log_file_name=str(self.traj_save_dir / 'my_sys_logs.log'))

        def rollout_func(target):
            rollout_worker = self.rollout_class(
                target_v=target[0], target_mu=target[1], target_chi=target[2], 
                my_logger=None,  # my_logger,
                traj_save_dir=self.traj_save_dir,
                step_frequence=self.step_frequence,
            )
            episode_length = rollout_worker.rollout()
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

        res = np.array(res)
        
        log["v"] = res[:, 0]
        log["mu"] = res[:, 1]
        log["chi"] = res[:, 2]
        log["length"] = res[:, 3]

        df = pd.DataFrame(log)
        df.to_csv(self.traj_save_dir / "res.csv", index=False)

        print("time used: ", time.time() - start_time)


if __name__ == "__main__":
    
    EXPERIMENT_NAME: str = 'v2'
    STEP_FREQUENCE: int = 10
    
    # s = ParallelScheduleForRollout(
    #     rollout_class=Rollout, 
    #     v_range=[100, 300], v_interval=10, 
    #     mu_range=[-85, 85], mu_interval=5, 
    #     chi_range=[-170, 170], chi_interval=5,
    #     step_frequence=STEP_FREQUENCE,
    #     pool_size=10,
    #     traj_save_dir=PROJECT_ROOT_DIR / "demonstrations" / "data" / f"10hz_{STEP_FREQUENCE}_5_5_{EXPERIMENT_NAME}"
    # )
    s = ParallelScheduleForRollout(
        rollout_class=Rollout, 
        v_range=[100, 110], v_interval=10, 
        mu_range=[-5, 5], mu_interval=5, 
        chi_range=[-5, 5], chi_interval=5,
        step_frequence=STEP_FREQUENCE,
        pool_size=10,
        traj_save_dir=PROJECT_ROOT_DIR / "demonstrations" / "data" / f"10hz_{STEP_FREQUENCE}_5_5_{EXPERIMENT_NAME}"
    )
    s.work()
