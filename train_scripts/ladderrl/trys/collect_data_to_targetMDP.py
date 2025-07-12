from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
import sys
from pathlib import Path
import os
current_dir = os.getcwd()
print(current_dir)
PROJECT_ROOT_DIR = Path(current_dir)#.parent.parent.parent
print(PROJECT_ROOT_DIR)
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))
# from train_scripts.ladderrl.utils.wrappers import PowerRewardWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
# from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate
import gymnasium as gym
import flycraft
import warnings
warnings.filterwarnings("ignore")  # 过滤Gymnasium的UserWarning
gym.register_envs(flycraft)
from train_scripts.ladderrl.utils.get_vec_env import get_vec_env
from train_scripts.ladderrl.utils.InfoDictReplayBuffer import InfoDictReplayBuffer
import torch
import numpy as np
from flycraft.utils.load_config import load_config


def main(config_file_list,buffersize):
    index = 0
    for config_fiile in config_file_list:
        index= index + 1
        train_config = load_config(Path(os.getcwd()) / config_fiile)
        train_this_iter_config = train_config["rl_train"][-1]
        first_policy_location = train_config["rl_train"][0]["rl"]["experiment_name"]

        THIS_ITER_ENV_CONFIG_FILE = train_this_iter_config["env"]["config_file"] 
        THIS_ITER_SEED_IN_CALLBACK_ENV = train_this_iter_config["rl"].get("seed_in_callback_env")
        THIS_ITER_RL_EXPERIMENT_NAME = train_this_iter_config["rl"].get("experiment_name")
        THIS_ITER_WRAPPER_LIST = train_this_iter_config['rl'].get("wrappers", [])
        env_config = {
                "num_process": 1,
                "seed": THIS_ITER_SEED_IN_CALLBACK_ENV,
                "config_file":str(PROJECT_ROOT_DIR / "configs" / "env" / THIS_ITER_ENV_CONFIG_FILE),
                "custom_config": {"debug_mode": True, "flag_str": "Callback"}
            }
        
        for wrp in THIS_ITER_WRAPPER_LIST:
            if wrp["type"] == "frame_skip":
                env_config.update(frame_skip=True, skip=wrp.get("skip", 1))
    
            else:
                raise ValueError(f"Cann't process this type of wrapper: {wrp['type']}!")
        policy_path =str(PROJECT_ROOT_DIR / "checkpoints" /first_policy_location/"best_model")

        sac = SAC.load(policy_path)

        env =get_vec_env(
            
            **env_config
        )
        obs = env.reset()

        #buffersize = 200
        
        sac.replay_buffer = InfoDictReplayBuffer(
            buffer_size=buffersize,
            observation_space= env.observation_space,
            action_space= env.action_space,
            device = torch.device("cuda")
        )
        while sac.replay_buffer.size() < buffersize:
            action, _ = sac.predict(obs,deterministic=True)
            next_obs, reward,done,info = env.step(action)
            sac.replay_buffer.add(
                obs=obs,next_obs=next_obs,action=action,reward=reward,done=done,infos=info
            )
            obs = next_obs
            if done:
                obs = env.reset()
        sac.save_replay_buffer(str(PROJECT_ROOT_DIR / "checkpoints" / THIS_ITER_RL_EXPERIMENT_NAME / "replay_buffer_collect_on_target_mdp"))
        print(f"seed {index} has finished")


if __name__ == '__main__':
    # from multiprocessing import freeze_support
    # freeze_support()
    config_file_list = [
                    # "configs/train/D2D/F2F/medium/eval_on_skip_1/b_05/buffersize1e6/two_stage_skip_3_skip_1/sac_config_10hz_128_128_1.json",
                    # "configs/train/D2D/F2F/medium/eval_on_skip_1/b_05/buffersize1e6/two_stage_skip_3_skip_1/sac_config_10hz_128_128_2.json",
                    # "configs/train/D2D/F2F/medium/eval_on_skip_1/b_05/buffersize1e6/two_stage_skip_3_skip_1/sac_config_10hz_128_128_3.json",
                    "configs/train/D2D/F2F/medium/eval_on_skip_1/b_05/buffersize1e6/two_stage_skip_3_skip_1/sac_config_10hz_128_128_4.json",
                    "configs/train/D2D/F2F/medium/eval_on_skip_1/b_05/buffersize1e6/two_stage_skip_3_skip_1/sac_config_10hz_128_128_5.json",
                    ]
    buffersize = 1000000
    main(config_file_list,buffersize)