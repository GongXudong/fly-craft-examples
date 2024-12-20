from pathlib import Path
import sys
import argparse
import gymnasium as gym

from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.common.vec_env import VecCheckNan

import flycraft
from flycraft.utils.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.disc.algorithms.smooth_goal_ppo import SmoothGoalPPO
from train_scripts.disc.algorithms.smooth_goal_sac import SmoothGoalSAC
from utils_my.sb3.vec_env_helper import get_vec_env
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate

gym.register_envs(flycraft)


def work(train_config: dict, env_config: Path, algo: str, seed: int=111, n_envs: int=8, n_eval_episodes: int=100):

    print(f"seed in eval: {seed}")

    env_config_dict_in_training = {
        "num_process": n_envs,
        "seed": seed,
        "config_file": str(env_config),
        "custom_config": {"debug_mode": False, "flag_str": "Train"}
    }

    vec_env = VecCheckNan(get_vec_env(
        **env_config_dict_in_training
    ))

    if algo == 'sac':
        print("测试sac")
        policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "disc" / RL_EXPERIMENT_NAME
        model_save_name="best_model"
        policy_class = SmoothGoalSAC
    elif algo == 'ppo':
        print("测试ppo")
        policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "disc" / RL_EXPERIMENT_NAME
        model_save_name="best_model"
        policy_class = SmoothGoalPPO
    elif algo == 'bc':
        print("测试bc")
        policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "disc" / RL_EXPERIMENT_NAME
        model_save_name="bc_checkpoint"
        policy_class = SmoothGoalPPO
    else:
        print(f"脚本参数--algo只能是sac, ppo, bc")

    algo_ppo = policy_class.load(
        str((policy_save_dir / model_save_name).absolute()), 
        env=vec_env,
        custom_objects={
            "seed": seed,
            "observation_space": vec_env.observation_space,
            "action_space": vec_env.action_space,
        }
    )
    algo_ppo.policy.set_training_mode(False)

    mean_reward, mean_episode_length, success_rate = evaluate_policy_with_success_rate(
        model=algo_ppo.policy,
        env=vec_env,
        n_eval_episodes=n_eval_episodes
    )

    print(f"mean reward: {mean_reward}, mean episode length: {mean_episode_length}, success rate: {success_rate}")

# python train_scripts/disc/evaluate/evaluate_policy_by_success_rate.py --config-file-name configs/train/disc/sac/medium/epsilon_0_1_reg_0_001/sac_config_10hz_128_128_seed_2.json --algo sac --seed 11 --n-envs 8 --n-eval-episode 100
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="pass configurations")
    parser.add_argument("--algo-config-file", type=str, help="train configuration file", default="configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json")
    parser.add_argument("--env-config-file", type=str, help="the configuration of environment used in evaluation", default="configs/env/env_hard_config_for_sac.json")
    parser.add_argument("--algo", type=str, help="the algorithm model to be evaluated, can be sac, or ppo", default="sac")
    parser.add_argument("--seed", type=int, default=11, help="the seed used in evaluation")
    parser.add_argument("--n-envs", type=int, default=8, help="the number of environments used in this evaluation")
    parser.add_argument("--n-eval-episodes", type=int, default=100, help="the number of episodes used in this evaluation")
    args = parser.parse_args()

    custom_config = load_config(PROJECT_ROOT_DIR / args.algo_config_file)

    RL_EXPERIMENT_NAME = custom_config["rl"]["experiment_name"] if args.algo != "bc" else custom_config["bc"]["experiment_name"]
    
    print(f"evaluate {RL_EXPERIMENT_NAME} on {args.env_config_file}................")

    work(
        train_config=custom_config,
        env_config=PROJECT_ROOT_DIR / args.env_config_file,
        algo=args.algo, 
        seed=args.seed,
        n_envs=args.n_envs,
        n_eval_episodes=args.n_eval_episodes,
    )
