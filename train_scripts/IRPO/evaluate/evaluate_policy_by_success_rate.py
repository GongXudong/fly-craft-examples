from pathlib import Path
import sys
import argparse
import gymnasium as gym

from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import VecCheckNan

import flycraft
from flycraft.utils.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.models.ppo_with_bc_loss import PPOWithBCLoss
from utils_my.sb3.vec_env_helper import get_vec_env
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate

gym.register_envs(flycraft)


def work(train_config: dict, algo: str, seed: int=111, n_envs: int=8, n_eval_episodes: int=100):

    env_config_dict_in_training = {
        "num_process": n_envs,
        "seed": seed,
        "config_file": str(PROJECT_ROOT_DIR / "configs" / "env" / train_config["env"].get("config_file", "env_config_for_sac.json")),
        "custom_config": {"debug_mode": False, "flag_str": "Train"}
    }

    vec_env = VecCheckNan(get_vec_env(
        **env_config_dict_in_training
    ))

    if algo == 'bc':
        print("测试bc")
        policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "IRPO" / "bc" / train_config["bc"]["experiment_name"]
        model_save_name="bc_checkpoint"
        policy_class = PPOWithBCLoss
    elif algo == 'rl':
        print("测试只用rl训练")
        policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "IRPO" / "rl_single" / train_config["rl"]["experiment_name"]
        model_save_name="best_model"
        policy_class = PPO
    elif algo == 'rl_bc':
        print("测试bc后继续rl优化")
        policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "IRPO" / "rl" / train_config["rl_bc"]["experiment_name"]
        model_save_name="best_model"
        policy_class = PPOWithBCLoss
    else:
        print(f"脚本参数--algo只能是rl, bc, 或者rl_bc")

    algo_ppo = policy_class.load(
        str((policy_save_dir / model_save_name).absolute()), 
        custom_objects={
            "observation_space": vec_env.observation_space,
            "action_space": vec_env.action_space
        }
    )
    algo_ppo.policy.set_training_mode(False)

    mean_reward, mean_episode_length, success_rate = evaluate_policy_with_success_rate(
        model=algo_ppo.policy,
        env=vec_env,
        n_eval_episodes=n_eval_episodes
    )

    print(f"mean reward: {mean_reward}, mean episode length: {mean_episode_length}, success rate: {success_rate}")

# python train_scripts/IRPO/evaluate/evaluate_policy_by_success_rate.py --config-file-name configs/train/IRPO/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json --algo rl_bc --seed 11 --n-envs 8 --n-eval-episode 100
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="pass configurations")
    parser.add_argument("--config-file-name", type=str, help="train configuration file", default="configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json")
    parser.add_argument("--algo", type=str, help="the algorithm model to be evaluated, can be rl, bc, or rl_bc", default="rl_bc")
    parser.add_argument("--seed", type=int, default=11, help="the seed used in evaluation")
    parser.add_argument("--n-envs", type=int, default=8, help="the number of environments used in this evaluation")
    parser.add_argument("--n-eval-episodes", type=int, default=100, help="the number of episodes used in this evaluation")
    args = parser.parse_args()

    custom_config = load_config(args.config_file_name)

    work(
        train_config=custom_config, 
        algo=args.algo, 
        seed=args.seed,
        n_envs=args.n_envs,
        n_eval_episodes=args.n_eval_episodes,
    )
