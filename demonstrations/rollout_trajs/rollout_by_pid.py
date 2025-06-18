from pathlib import Path
import pandas as pd
import itertools
import logging
import sys
import argparse

import flycraft
from flycraft.planes.f16_plane import F16Plane
from flycraft.planes.utils.f16Classes import Guide, ControlLaw, PlaneModel
from flycraft.env import FlyCraftEnv
from flycraft.utils_common.my_log import get_logger

PROJECT_ROOT_DIR: Path = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))


action_mins = F16Plane.get_action_lower_bounds()
action_maxs = F16Plane.get_action_higher_bounds()

P_MAX = action_maxs.p
P_MIN = action_mins.p
NZ_MAX = action_maxs.nz
NZ_MIN = action_mins.nz
PLA_MAX = action_maxs.pla
PLA_MIN = action_mins.pla

class Rollout:
    """给定目标速度矢量，利用Guide模型生成轨迹数据
    """

    def __init__(self, 
        target_v, target_mu, target_chi, 
        h0=5000, v0=200, 
        v_threshold=10., mu_threshold=1., chi_threshold=1., 
        integral_time_length=1, step_frequence=100, 
        max_rollout_time=120, 
        log_state_keys = ["phi", "theta", "psi", "v", "mu", "chi", "p", "h"],
        log_guidance_law_action_keys = ["p", "nz", "pla", "rud"],
        log_control_law_action_keys = ["ail", "ele", "rud", "pla"],
        traj_save_dir: Path=PROJECT_ROOT_DIR / "data" / "tmp",
        trajectory_save_prefix: str="traj",
        my_logger: logging.Logger=None
    ) -> None:
        """_summary_

        Args:
            target_v (_type_): 目标速度
            target_mu (_type_): 目标mu
            target_chi (_type_): 目标chi
            h0 (int, optional): 飞机初始高度. Defaults to 5000.
            v0 (int, optional): 飞机初始速度. Defaults to 200.
            v_threshold (_type_, optional): 判断速度达到目标速度所使用的误差. Defaults to 10..
            mu_threshold (_type_, optional): 判断mu达到目标mu所使用的误差. Defaults to 1..
            chi_threshold (_type_, optional): 判断chi达到目标chi所使用的误差. Defaults to 1..
            integral_time_length (int, optional): 积分时间窗口，积分值用来判断是否到达目标. Defaults to 1.
            step_frequence (int, optional): 控制频率（也是仿真频率）. Defaults to 100.
            max_rollout_time (int, optional): 一个episode的最大时长. Defaults to 120.
            my_log_dir (str, optional): 日志、数据存放目录. Defaults to 'my_logs'.
            trajectory_save_prefix (str, optional): 轨迹存储前缀
            my_logger (logging.Logger, optional): 使用的logger. Defaults to None.
        """

        self.target_v = target_v
        self.target_mu = target_mu
        self.target_chi = target_chi
        
        self.h0 = h0
        self.v0 = v0

        self.v_threshold = v_threshold
        self.mu_threshold = mu_threshold
        self.chi_threshold = chi_threshold
        self.integral_time_length = integral_time_length
        self.step_frequence = step_frequence

        self.max_rollout_time = max_rollout_time

        self.traj_save_dir = traj_save_dir
        self.trajectory_save_prefix = trajectory_save_prefix

        self.gtmp = Guide()
        self.f16cl = ControlLaw(stepTime=1./step_frequence)
        self.f16model = PlaneModel(h0, v0, stepTime=1./step_frequence)

        # log_name = f"f16trace_{target_v}_{target_mu}_{target_chi}.csv"
        # self.f16model.setLog(logName=log_name)  # 设置日志名

        self.log_state_keys = log_state_keys
        self.log_guidance_law_action_keys = log_guidance_law_action_keys
        self.log_control_law_action_keys = log_control_law_action_keys

        self.logs = {}
        
        self.init_log()

        self.logger = my_logger

    @property
    def wCmds(self) -> dict:
        """目标速度矢量

        Returns:
            dict: _description_
        """
        return {
            "v": self.target_v,
            "mu": self.target_mu,
            "chi": self.target_chi
        }
    
    @property
    def sim_interval(self) -> float:
        """单步仿真时长

        Returns:
            float: _description_
        """
        return 1. / self.step_frequence

    @property
    def integral_window_length(self):
        """v, mu, chi的积分窗口长度

        Returns:
            _type_: _description_
        """
        return self.integral_time_length * self.step_frequence
    
    @property
    def v_integral_threshold(self):
        """v的误差积分阈值，当v在最后self.integral_window_length上的积分小于该值时，认为v达到目标值

        Returns:
            _type_: _description_
        """
        return self.v_threshold * self.integral_window_length
    
    @property
    def mu_integral_threshold(self):
        """mu的误差积分阈值，当mu在最后self.integral_window_length上的积分小于该值时，认为mu达到目标值

        Returns:
            _type_: _description_
        """
        return self.mu_threshold * self.integral_window_length
    
    @property
    def chi_integral_threshold(self):
        """chi的误差积分阈值，当chi在最后self.integral_window_length上的积分小于该值时，认为chi达到目标值

        Returns:
            _type_: _description_
        """
        return self.chi_threshold * self.integral_window_length

    @property
    def max_rollout_length(self) -> int:
        """episode的最大长度

        Returns:
            _type_: _description_
        """
        return self.max_rollout_time * self.step_frequence
    
    def init_log(self):
        """初始化logs属性，logs属性存储状态与动作序列
        """
        self.logs = {}
        self.logs["time"] = []
        for k in self.log_state_keys:
            self.logs[f"s_{k}"] = []
        for k in self.log_guidance_law_action_keys:
            self.logs[f"a_{k}"] = []
        for k in self.log_control_law_action_keys:
            self.logs[f"a_end_{k}"] = []

    def log(self, state, guidance_law_action, control_law_action, time:float):
        """将state与action记入logs属性

        Args:
            state (_type_): _description_
            action (_type_): _description_
            time (float): _description_
        """
        self.logs["time"].append(round(time, 2))
        for k in self.log_state_keys:
            self.logs[f"s_{k}"].append(state[k])
        
        for k in self.log_guidance_law_action_keys:
            self.logs[f"a_{k}"].append(guidance_law_action[k])
        
        for k in self.log_control_law_action_keys:
            self.logs[f"a_end_{k}"].append(control_law_action[k])
            

    def save(self):
        """将logs属性存数的数据存成csv文件
        """
        df = pd.DataFrame(self.logs)
        df.to_csv(str((self.traj_save_dir / f"{self.trajectory_save_prefix}_{self.target_v}_{self.target_mu}_{self.target_chi}.csv").absolute()), index=False)

    def is_terminated(self) -> bool:
        """对于v, mu, chi，分别在给定窗口长度上积分，积分值均小于预设值返回True

        Returns:
            bool: 当前速度矢量是否达到目标速度矢量
        """
        if len(self.logs["time"]) < self.integral_window_length:
            return False
        else:
            v_flag, mu_flag, chi_flag = False, False, False
            if sum([abs(self.target_v - item) for item in self.logs["s_v"][-self.integral_window_length:]]) < self.v_integral_threshold:
                v_flag = True
            if sum([abs(self.target_mu - item) for item in self.logs["s_mu"][-self.integral_window_length:]]) < self.mu_integral_threshold:
                mu_flag = True
            if sum([abs(self.target_chi - item) for item in self.logs["s_chi"][-self.integral_window_length:]]) < self.chi_integral_threshold:
                chi_flag = True
            if v_flag and mu_flag and chi_flag:
                return True
            else:
                return False

    def rollout(self) -> int:
        """在最大时长内能飞到目标速度矢量，返回轨迹序列的长度；达到最大时长，返回0

        Returns:
            int: 轨迹序列的长度（0表示达到了最大仿真时长仍未到目标速度矢量，或者发生了坠机）
        """
        
        stsDict = self.f16model.getPlaneState()  # stsDict: lef, npos, epos, h, alpha, beta, phi, theta, psi, p, q, r, v, vn, ve, vh, nx, ny, nz, ele, ail, rud, thrust, lon, lat, mu, chi

        for i in range(self.max_rollout_length):
            self.gtmp.step(self.wCmds, stsDict)
            gout = self.gtmp.getOutputDict()

            # 检查gout的范围
            if not (P_MIN <= gout["p"] <= P_MAX):
                print('Invalid p: ', gout["p"])
            if not (NZ_MIN <= gout["nz"] <= NZ_MAX):
                print('Invalid nz: ', gout["nz"])
            if not (PLA_MIN <= gout["pla"] <= PLA_MAX):
                print("Invalid pla: ", gout["pla"])

            gout["p"] = P_MAX if gout["p"] > P_MAX else gout["p"]
            gout["p"] = P_MIN if gout["p"] < P_MIN else gout["p"]
            gout["nz"] = NZ_MAX if gout["nz"] > NZ_MAX else gout["nz"]
            gout["nz"] = NZ_MIN if gout["nz"] < NZ_MIN else gout["nz"]
            gout["pla"] = PLA_MAX if gout["pla"] > PLA_MAX else gout["pla"]
            gout["pla"] = PLA_MIN if gout["pla"] < PLA_MIN else gout["pla"]

            self.f16cl.step(gout, stsDict)
            clout = self.f16cl.getOutputDict()
            self.f16model.step(clout)
            # print(clout)

            self.log(state=stsDict, guidance_law_action=gout, control_law_action=clout, time=i * self.sim_interval)
            # print(f"v = {stsDict['v']:.1f}, mu = {stsDict['mu']:.1f}, chi = {stsDict['chi']:.1f}")

            # 高度小于0，直接结束！！！
            if stsDict["h"] < 0.:
                # print(f'\033[1;31mcrashed!!!!!!!!!!!\033[0m v = {self.target_v}, mu = {self.target_mu}, chi = {self.target_chi}')
                if self.logger is not None:
                    self.logger.info(f'\033[1;31mcrashed!!!!!!!!!!!\033[0m v = {self.target_v}, mu = {self.target_mu}, chi = {self.target_chi}')
                return 0

            # judge termination
            if self.is_terminated():
                # print(f"\033[1;32m Terminated. \033[0m v = {self.target_v}, mu = {self.target_mu}, chi = {self.target_chi}, len = {len(self.logs['time'])}")
                if self.logger is not None:
                    self.logger.warning(f"\033[1;32m Terminated. \033[0m v = {self.target_v}, mu = {self.target_mu}, chi = {self.target_chi}, len = {len(self.logs['time'])}")
                self.save()
                return len(self.logs["time"])

            stsDict = self.f16model.getPlaneState()

        # print(f'\033[1;31mreach max length!!!!!!!!!!!\033[0m v = {self.target_v}, mu = {self.target_mu}, chi = {self.target_chi}')
        
        if self.logger is not None:
            self.logger.info(f'\033[1;31mreach max length!!!!!!!!!!!\033[0m v = {self.target_v}, mu = {self.target_mu}, chi = {self.target_chi}')
        # self.save()
        return 0


class ScheduleForRollout:
    """采样的计划：枚举目标速度矢量，生成轨迹并存储成csv文件。
    """

    def __init__(self, 
            rollout_class=Rollout, 
            v_range:list=[100, 300], v_interval:int=10, 
            mu_range:list=[-85, 85], mu_interval:int=5, 
            chi_range:list=[-180, 180], chi_interval=5,
            step_frequence:int=10,
            save_dir: str="data/tmp"
        ) -> None:
        self.rollout_class = rollout_class
        self.v_range = v_range
        self.v_interval = v_interval
        self.mu_range = mu_range
        self.mu_interval = mu_interval
        self.chi_range = chi_range
        self.chi_interval = chi_interval
        self.step_frequence = step_frequence
        self.save_dir = PROJECT_ROOT_DIR / save_dir
        
    def work(self):
        log = {
            "v": [],
            "mu": [],
            "chi": [],
            "length": []
        }

        
        if not self.save_dir.exists():
            self.save_dir.mkdir()

        my_logger = get_logger(logger_name="ucav", log_file_dir=str(self.save_dir / 'my_sys_logs.log'))

        for v, mu, chi in itertools.product(
            range(self.v_range[0], self.v_range[1]+1, self.v_interval), 
            range(self.mu_range[0], self.mu_range[1]+1, self.mu_interval), 
            range(self.chi_range[0], self.chi_range[1]+1, self.chi_interval)):
            
            rollout_worker = self.rollout_class(
                target_v=v, target_mu=mu, target_chi=chi, 
                log_state_keys=["phi", "theta", "psi", "v", "mu", "chi", "p", "q", "r", "h", "lon", "lat", "thrust", "nx", "ny", "nz", "alpha", "beta", "lef", "npos", "epos"],
                log_guidance_law_action_keys=["p", "nz", "pla", "rud"],
                log_control_law_action_keys=["ail", "ele", "rud", "pla"],
                my_logger=my_logger,
                traj_save_dir=self.save_dir,
                step_frequence=self.step_frequence
            )
            episode_length = rollout_worker.rollout()
            
            log["v"].append(v)
            log["mu"].append(mu)
            log["chi"].append(chi)
            log["length"].append(episode_length)

        print(log)
        df = pd.DataFrame(log)
        df.to_csv(self.save_dir / "res.csv", index=False)


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
    
    s = ScheduleForRollout(
        rollout_class=Rollout, 
        v_range=[args.v_min, args.v_max], 
        v_interval=args.v_interval, 
        mu_range=[args.mu_min, args.mu_max], 
        mu_interval=args.mu_interval, 
        chi_range=[args.chi_min, args.chi_max], 
        chi_interval=args.chi_interval,
        step_frequence=args.step_frequence,
        save_dir=PROJECT_ROOT_DIR / "demonstrations" / "data" / f"{args.step_frequence}hz_{args.v_interval}_{args.mu_interval}_{args.chi_interval}_{args.data_dir_suffix}"
    )
    s.work()