import gymnasium as gym
import numpy as np
from pathlib import Path
import os
import sys
import torch as th
import argparse

from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
import flycraft
from flycraft.utils.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_eval_callback import MyEvalCallback
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate
from train_scripts.D2D.utils.get_vec_env import get_vec_env
from train_scripts.D2D.utils.load_data_from_csv import load_random_trajectories_from_csv_files

import warnings
warnings.filterwarnings("ignore")  # 过滤Gymnasium的UserWarning
gym.register_envs(flycraft)


def train(train_config):

    # RL global config
    NET_ARCH = train_config["rl_common"]["net_arch"]
    GAMMA = train_config["rl_common"].get("gamma", 0.995)
    BUFFER_SIZE = train_config["rl_common"].get("buffer_size", 1e6)
    BATCH_SIZE = train_config["rl_common"].get("batch_size", 1024)
    RL_TRAIN_PROCESS_NUM = train_config["rl_common"].get("rollout_process_num", 32)
    RL_EVALUATE_PROCESS_NUM = train_config["rl_common"].get("evaluate_process_num", 32)
    CALLBACK_PROCESS_NUM = train_config["rl_common"].get("callback_process_num", 32)
    GRADIENT_STEPS = train_config["rl_common"].get("gradient_steps", 2)
    EVAL_FREQ = train_config["rl_common"].get("eval_freq", 1000)
    N_EVAL_EPISODES = train_config["rl_common"].get("n_eval_episodes", CALLBACK_PROCESS_NUM*10)
    USE_HER = train_config["rl_common"].get("use_her", True)

    for index, train_this_iter_config in enumerate(train_config["rl_train"]):
        THIS_ITER_ENV_CONFIG_FILE = train_this_iter_config["env"]["config_file"]
        THIS_ITER_ENV_CUSTOM_CONFIG = train_this_iter_config["env"].get("custom_config", {})

        THIS_ITER_SEED = train_this_iter_config["rl"].get("seed")
        THIS_ITER_SEED_IN_TRAINING_ENV = train_this_iter_config["rl"].get("seed_in_train_env")
        THIS_ITER_SEED_IN_CALLBACK_ENV = train_this_iter_config["rl"].get("seed_in_callback_env")
        THIS_ITER_RL_EXPERIMENT_NAME = train_this_iter_config["rl"]["experiment_name"]
        THIS_ITER_RL_TRAIN_STEPS = train_this_iter_config["rl"]["train_steps"]
        THIS_ITER_LEARNING_STARTS = train_this_iter_config["rl"].get("learning_starts", 10240)
        THIS_ITER_LEARNING_RATE = train_this_iter_config["rl"].get("learning_rate", 3e-4)
        THIS_ITER_RESET_POLICY = train_this_iter_config["rl"].get("reset_policy", False)
        THIS_ITER_WRAPPER_LIST = train_this_iter_config['rl'].get("wrappers", [])
        THIS_ITER_RESET_REPLAY_BUFFER = train_this_iter_config["rl"].get("reset_replay_buffer", False)
        THIS_ITER_RELABEL_REPLAY_BUFFER = train_this_iter_config["rl"].get("relabel_replay_buffer", False)
        THIS_ITER_HAS_TRAINED = train_this_iter_config["rl"].get("has_trained", False)
        THIS_ITER_STORE_INFO =  train_this_iter_config["rl"].get("store_info", False)
        THIS_ITER_PRE_FILL_REPLAY_BUFFER = train_this_iter_config["rl"].get("pre_fill_replay_buffer", False)
        THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS = train_this_iter_config["rl"].get("pre_fill_replay_buffer_kwargs", {})

        if THIS_ITER_HAS_TRAINED:
            continue
        
        # initialize env and algo
        env_config_in_training = {
            "num_process": RL_TRAIN_PROCESS_NUM,
            "seed": THIS_ITER_SEED_IN_TRAINING_ENV,
            "config_file": str(PROJECT_ROOT_DIR / "configs" / "env" / THIS_ITER_ENV_CONFIG_FILE),
            "custom_config": {"debug_mode": True, "flag_str": "Train"},
        }
        env_config_in_evaluation = {
            "num_process": RL_EVALUATE_PROCESS_NUM,
            "seed": THIS_ITER_SEED_IN_CALLBACK_ENV,
            "config_file": str(PROJECT_ROOT_DIR / "configs" / "env" / THIS_ITER_ENV_CONFIG_FILE),
            "custom_config": {"debug_mode": True, "flag_str": "Callback"}
        }
        
        for wrp in THIS_ITER_WRAPPER_LIST:
            if wrp["type"] == "frame_skip":
                env_config_in_training.update(frame_skip=True, skip=wrp.get("skip", 1))
                env_config_in_evaluation.update(frame_skip=True, skip=wrp.get("skip", 1))
            else:
                raise ValueError(f"Cann't process this type of wrapper: {wrp['type']}!")

        vec_env = get_vec_env(
            **env_config_in_training
        )
        eval_env_in_callback = get_vec_env(
            **env_config_in_evaluation
        )

        policy_save_dir = PROJECT_ROOT_DIR / "checkpoints"
        policy_save_name = "best_model"
        # policy_save_name = "final_model"
        replay_buffer_save_name = "replay_buffer"

        # prepare policy
        if (index == 0) or (index > 0 and THIS_ITER_RESET_POLICY):
            print(f'index= {index}',f'copy_info_dict= {THIS_ITER_STORE_INFO}')
            sac_algo = SAC(
                "MultiInputPolicy",
                vec_env,
                seed=THIS_ITER_SEED,
                replay_buffer_class=HerReplayBuffer if USE_HER else DictReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=4,
                    goal_selection_strategy="future",
                    copy_info_dict=THIS_ITER_STORE_INFO
                ) if USE_HER else None,
                verbose=1,
                buffer_size=int(BUFFER_SIZE),
                learning_starts=int(THIS_ITER_LEARNING_STARTS),
                gradient_steps=int(GRADIENT_STEPS),
                learning_rate=THIS_ITER_LEARNING_RATE,
                gamma=GAMMA,
                batch_size=int(BATCH_SIZE),
                policy_kwargs=dict(
                    net_arch=NET_ARCH,
                    activation_fn=th.nn.Tanh
                ),
            )
            print(f"Iter {index}: reset policy!!!!!")
        else:
            sac_algo = SAC.load(
                path=policy_save_dir / train_config["rl_train"][index-1]["rl"]["experiment_name"] / policy_save_name,
                env=vec_env
            )
            print(f"Iter {index}: load policy from {policy_save_dir / train_config['rl_train'][index-1]['rl']['experiment_name'] / policy_save_name}.")

        # prepare replay buffer
        if index > 0:
            # load replay buffer
            if not THIS_ITER_RESET_REPLAY_BUFFER:
                cut_N = train_config["rl_train"][index-1]["rl"].get("wrappers")[0].get("skip")
                print(f"cut_N:{cut_N}")
                save_name = "cut_"+f"{cut_N}"+"_cut_buffer"
                sac_algo.load_replay_buffer(policy_save_dir / train_config["rl_train"][index-1]["rl"]["experiment_name"] /save_name)
                print(f"Iter {index}: load replay buffer from {policy_save_dir / train_config['rl_train'][index-1]['rl']['experiment_name'] / replay_buffer_save_name}.")

                # # relabel rewards of transitions in the loaded replay buffer
                # if THIS_ITER_RELABEL_REPLAY_BUFFER:
                #     # sac_algo.replay_buffer.observations
                #     if not THIS_ITER_WRAPPER_LIST :
                #         loaded_replay_buffer_size = sac_algo.replay_buffer.size()
                #         new_rewards = vec_env.env_method(
                #             method_name="compute_reward",
                #             indices=[0],
                #             achieved_goal=sac_algo.replay_buffer.next_observations["achieved_goal"].squeeze()[:loaded_replay_buffer_size], 
                #             desired_goal=sac_algo.replay_buffer.observations["desired_goal"].squeeze()[:loaded_replay_buffer_size],
                #             info=sac_algo.replay_buffer.infos.squeeze()[:loaded_replay_buffer_size]
                #         )[0]
                #         tmp_reward = new_rewards.reshape(-1, 1)
                #         sac_algo.replay_buffer.rewards[:loaded_replay_buffer_size] = new_rewards.reshape(-1, 1)

                #         print(f"Iter {index}: reset rewards in replay buffer.")
                #     else:
                #         contains_frame_skip = any(wrapper.get("type") == "frame_skip" for wrapper in THIS_ITER_WRAPPER_LIST)
                #         print("compute relabel reward for skip wrapper")
                #         if contains_frame_skip:
                #             loaded_replay_buffer_size = sac_algo.replay_buffer.size()
                            
                #             new_rewards = []

                #             for info in sac_algo.replay_buffer.infos:
                #                 frame_skip_info = info[0].get('frame_skip_info')
                #                 if frame_skip_info is not None:
                #                     reward = frame_skip_info[0].get('reward')
                #                     new_rewards.append(reward)
                #                 else:
                #                     new_rewards.append(0.0)
 
                #             new_rewards = np.array(new_rewards).reshape(-1, 1)
                #             sac_algo.replay_buffer.rewards[:loaded_replay_buffer_size] = new_rewards.reshape(-1, 1)
                #             if not THIS_ITER_STORE_INFO:
                #                 sac_algo.replay_buffer.infos = np.array([[{} for _ in range(sac_algo.replay_buffer.n_envs)] for _ in range(sac_algo.replay_buffer.buffer_size)])
                #                 sac_algo.replay_buffer.copy_info_dict = False
                #             # for info in sac_algo.replay_buffer.infos:
                #             #     info=np.array([{}])

                        
            else:
                print(f"Iter {index}: reset replay buffer.")
        else:
            # check whether to fill replay buffer with expert demonstrations
            if THIS_ITER_PRE_FILL_REPLAY_BUFFER:
                loaded_obs, loaded_next_obs, loaded_action, loaded_reward, loaded_done, loaded_info = load_random_trajectories_from_csv_files(
                    data_dir=PROJECT_ROOT_DIR / THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["data_dir"],
                    cache_data=THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["cache_data"],
                    cache_data_dir=PROJECT_ROOT_DIR / THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["cache_data_dir"],
                    trajectory_save_prefix=THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["trajectory_save_prefix"],
                    env_config_file=PROJECT_ROOT_DIR / "configs" / "env" / THIS_ITER_ENV_CONFIG_FILE,
                    select_transition_num=THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["selected_transition_num"],
                    random_state=THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["random_state"]
                )

                sac_algo.replay_buffer.extend(
                    obs=loaded_obs,
                    next_obs=loaded_next_obs,
                    action=loaded_action,
                    reward=loaded_reward,
                    done=loaded_done,
                    infos=loaded_info,
                )

                print(f"Iter {index}: pre-fill replay buffer.")

                # relabel rewards of transitions in the loaded replay buffer
                if THIS_ITER_RELABEL_REPLAY_BUFFER:
                    # sac_algo.replay_buffer.observations
                    loaded_replay_buffer_size = sac_algo.replay_buffer.size()
                    new_rewards = vec_env.env_method(
                        method_name="compute_reward",
                        indices=[0],
                        achieved_goal=sac_algo.replay_buffer.next_observations["achieved_goal"].squeeze()[:loaded_replay_buffer_size], 
                        desired_goal=sac_algo.replay_buffer.observations["desired_goal"].squeeze()[:loaded_replay_buffer_size],
                        info=sac_algo.replay_buffer.infos.squeeze()[:loaded_replay_buffer_size]
                    )[0]

                    sac_algo.replay_buffer.rewards[:loaded_replay_buffer_size] = new_rewards.reshape(-1, 1)

                    print(f"Iter {index}: reset rewards in replay buffer.")

        
        sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / THIS_ITER_RL_EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])
        sac_algo.set_logger(sb3_logger)

        # callback: evaluate, save best
        eval_callback = MyEvalCallback(
            eval_env_in_callback, 
            best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / THIS_ITER_RL_EXPERIMENT_NAME).absolute()),
            log_path=str((PROJECT_ROOT_DIR / "logs" / THIS_ITER_RL_EXPERIMENT_NAME).absolute()), 
            eval_freq=EVAL_FREQ,  # 多少次env.step()评估一次，此处设置为1000，因为VecEnv有72个并行环境，所以实际相当于72*1000次step，评估一次
            n_eval_episodes=N_EVAL_EPISODES,  # 每次评估使用多少条轨迹
            deterministic=True, 
            render=False,
        )

        checkpoint_on_event = CheckpointCallback(save_freq=1, save_path=str((PROJECT_ROOT_DIR / "checkpoints" / THIS_ITER_RL_EXPERIMENT_NAME).absolute()))
        event_callback = EveryNTimesteps(n_steps=50000, callback=checkpoint_on_event)

        sac_algo.learn(
            total_timesteps=int(THIS_ITER_RL_TRAIN_STEPS),
            callback=[eval_callback, event_callback]
        )

        sac_algo.save(str(PROJECT_ROOT_DIR / "checkpoints" / THIS_ITER_RL_EXPERIMENT_NAME / "final_model"))
        sac_algo.save_replay_buffer(str(PROJECT_ROOT_DIR / "checkpoints" / THIS_ITER_RL_EXPERIMENT_NAME / replay_buffer_save_name))

        eval_reward, _, eval_success_rate = evaluate_policy_with_success_rate(sac_algo.policy, eval_env_in_callback, 1000)
        sb3_logger.info(f"Reward after RL: {eval_reward}")
        sb3_logger.info(f"Success rate after RL: {eval_success_rate}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="传入配置文件")
    parser.add_argument("--config-file-name", type=str, help="配置文件名", default="sac_config_10hz_128_128_1.json")
    args = parser.parse_args()

    train_config = load_config(Path(os.getcwd()) / args.config_file_name)

    train(train_config)
