from pathlib import Path
import sys
from typing import Union
import argparse
from copy import deepcopy
import gymnasium as gym

from stable_baselines3.ppo import PPO

import flycraft
from flycraft.utils_common.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.models.ppo_with_bc_loss import PPOWithBCLoss
from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from evaluation.rollout import Rollout

gym.register_envs(flycraft)


def work(train_config: dict, algo: str, use_fixed_target: bool, target_v: float, target_mu: float, target_chi: float, save_acmi: bool, save_dir: Union[str, Path]):
    
    if algo == 'bc':
        print("测试bc")
        policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "IRPO" / "bc" / BC_EXPERIMENT_NAME
        model_save_name="bc_checkpoint"
        policy_class = PPOWithBCLoss
    elif algo == 'rl':
        print("测试只用rl训练")
        policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "IRPO" / "rl_single" / RL_SINGLE_EXPERIMENT_NAME
        model_save_name="best_model"
        policy_class = PPO
    elif algo == 'rl_bc':
        print("测试bc后继续rl优化")
        policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "IRPO" / "rl" / RL_EXPERIMENT_NAME
        model_save_name="best_model"
        policy_class = PPOWithBCLoss
    else:
        print(f"脚本参数--algo只能是rl, bc, 或者rl_bc")
    
    helper_env = gym.make(
        "FlyCraft-v0",
        config_file=str(PROJECT_ROOT_DIR / "configs" / "env" / train_config["env"].get("config_file", "env_config_for_sac.json")),
        custom_config={
            "debug_mode": True, 
            "flag_str": "Train",
            "goal": {
                "use_fixed_goal": use_fixed_target,
                "goal_v": target_v,
                "goal_mu": target_mu,
                "goal_chi": target_chi,
            }
        }
    )
    scaled_obs_env = ScaledObservationWrapper(helper_env)
    scaled_act_env = ScaledActionWrapper(scaled_obs_env)

    algo_ppo = policy_class.load(
        str((policy_save_dir / model_save_name).absolute()), 
        custom_objects={
            "observation_space": scaled_act_env.observation_space,
            "action_space": scaled_act_env.action_space
        }
    )
    algo_ppo.policy.set_training_mode(False)

    # TODO: 初始化Rollout，并采样！！
    rollout = Rollout(
        env=scaled_act_env,
        algo=algo_ppo,
        debug_mode=True,
    )

    rollout.rollout_one_trajectory(save_acmi=save_acmi, save_dir=save_dir)

# python train_scripts/IRPO/evaluate/rollout_one_trajectory.py --config-file-name configs/train/IRPO/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json --algo rl_bc --save-acmi --use-fixed-target --target-v 210 --target-mu 5 --target-chi 10 --save-dir train_scripts/IRPO/evaluate/rolled_out_trajs/
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="pass configurations")
    parser.add_argument("--config-file-name", type=str, help="train configuration file", default="configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json")
    parser.add_argument("--algo", type=str, help="the algorithm model to be evaluated, can be rl, bc, or rl_bc", default="rl_bc")
    parser.add_argument("--save-acmi", action="store_true", help="whether to store the trajectory to acmi file")
    parser.add_argument("--use-fixed-target", action="store_true", help="whether to rollout trajectory on user customized goal")
    parser.add_argument("--target-v", type=float, default=200, help="user customized v (true airspeed)")
    parser.add_argument("--target-mu", type=float, default=20, help="user customized mu (flight path elevator angle)")
    parser.add_argument("--target-chi", type=float, default=30, help="user customized chi (flight path azimuth angle)")
    parser.add_argument("--save-dir", type=str, default="train_scripts/IRPO/evalate/rolled_out_trajs", help="the directory to save the sampled trajectory")
    args = parser.parse_args()

    custom_config = load_config(args.config_file_name)

    RL_EXPERIMENT_NAME = custom_config["rl_bc"]["experiment_name"]
    BC_EXPERIMENT_NAME = custom_config["bc"]["experiment_name"]
    RL_SINGLE_EXPERIMENT_NAME = custom_config["rl"]["experiment_name"]
    SEED = custom_config["rl_bc"]["seed"]
    ROLLOUT_PROCESS_NUM = custom_config["rl_bc"]["rollout_process_num"]

    work(
        train_config=custom_config, 
        algo=args.algo, 
        use_fixed_target=args.use_fixed_target, 
        target_v=args.target_v, 
        target_mu=args.target_mu, 
        target_chi=args.target_chi, 
        save_acmi=args.save_acmi, 
        save_dir=PROJECT_ROOT_DIR / args.save_dir
    )
