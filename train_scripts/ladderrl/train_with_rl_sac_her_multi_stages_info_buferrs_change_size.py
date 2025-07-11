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
from train_scripts.ladderrl.utils.get_vec_env import get_vec_env
from train_scripts.ladderrl.utils.load_data_from_csv import load_random_trajectories_from_csv_files,load_random_transitions_from_csv_files
from train_scripts.ladderrl.utils.InfoDictReplayBuffer import InfoDictReplayBuffer
from utils_my.sb3.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
import pathlib
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
        ThIS_ITER_ENV_EVAL_CONFIG_FILE = train_this_iter_config["env"].get("evaluate_config","")

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
        GAMMA = train_this_iter_config["rl"].get("gamma", 0.995)
        THIS_ITER_WARMUP_EPOCHS=train_this_iter_config["rl"].get("warmup_epochs", 0)
        if THIS_ITER_HAS_TRAINED:
            continue
        
        # initialize env and algo
        env_config_in_training = {
            "num_process": RL_TRAIN_PROCESS_NUM,
            "seed": THIS_ITER_SEED_IN_TRAINING_ENV,
            "config_file": str(PROJECT_ROOT_DIR / "configs" / "env" / THIS_ITER_ENV_CONFIG_FILE),
            "custom_config": {"debug_mode": True, "flag_str": "Train"},
        }

        env_config_in_evaluation = {}

        if ThIS_ITER_ENV_EVAL_CONFIG_FILE =="":
            env_config_in_evaluation = {
                "num_process": RL_EVALUATE_PROCESS_NUM,
                "seed": THIS_ITER_SEED_IN_CALLBACK_ENV,
                "config_file": str(PROJECT_ROOT_DIR / "configs" / "env" / THIS_ITER_ENV_CONFIG_FILE),
                "custom_config": {"debug_mode": True, "flag_str": "Callback"}
            }
        else:
            env_config_in_evaluation = {
                "num_process": RL_EVALUATE_PROCESS_NUM,
                "seed": THIS_ITER_SEED_IN_CALLBACK_ENV,
                "config_file": str(PROJECT_ROOT_DIR / "configs" / "env" / ThIS_ITER_ENV_EVAL_CONFIG_FILE),
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
                replay_buffer_class=HerReplayBuffer if USE_HER else InfoDictReplayBuffer,
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
            print(f"buffer_size={int(BUFFER_SIZE)}")
            print(f"sac_algo.replay_buffer.buffer_size = {sac_algo.replay_buffer.buffer_size}")
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
                import pickle
                from stable_baselines3.common.save_util import open_path
                path = policy_save_dir / train_config["rl_train"][index-1]["rl"]["experiment_name"] / replay_buffer_save_name
                file = open_path(path, "r", suffix="pkl")
                tmp_buffer = pickle.load(file)
                if isinstance(path, (str, pathlib.Path)):
                    file.close()
                
                tmp_size = tmp_buffer.buffer_size

                # obs: [batch_size, obs_shape], action: [batch_size, action_shape], reward: [batch_size, 1], done: [batch_size, 1], info: [batch_size]
                tmp_sample = tmp_buffer._get_samples(np.arange(0,tmp_size))
                
                # tmp_sample = tmp_buffer.sample(tmp_size)
                
                for i in range(tmp_size):
                    
                    tmp_obs = {"observation":tmp_sample.observations["observation"][i].cpu().numpy(),
                            "achieved_goal":tmp_sample.observations["achieved_goal"][i].cpu().numpy(),
                            "desired_goal":tmp_sample.observations["desired_goal"][i].cpu().numpy(),
                            }
                    tmp_next_obs ={
                            "observation":tmp_sample.next_observations["observation"][i].cpu().numpy(),
                            "achieved_goal":tmp_sample.next_observations["achieved_goal"][i].cpu().numpy(),
                            "desired_goal":tmp_sample.next_observations["desired_goal"][i].cpu().numpy(),
                    }
                    tmp_reward = tmp_sample.rewards[i].cpu().numpy()
                    tmp_action = tmp_sample.actions[i].cpu().numpy()
                    tmp_done = tmp_sample.dones[i].cpu().numpy()
                    tmp_infos = [tmp_sample.infos[i]]
                     

                    # [env_inds, obs_shape]

                    for key in tmp_obs.keys():
                        tmp_obs[key] = tmp_obs[key].reshape((RL_TRAIN_PROCESS_NUM , tmp_obs[key].shape[-1]))
                    
                    for key in tmp_next_obs.keys():
                        tmp_next_obs[key] = tmp_next_obs[key].reshape((RL_TRAIN_PROCESS_NUM , tmp_next_obs[key].shape[-1]))
                    
                    tmp_action = tmp_action.reshape((RL_TRAIN_PROCESS_NUM, tmp_action.shape[-1]))
                    # tmp_obs = np.array(tmp_obs).reshape((RL_TRAIN_PROCESS_NUM,-1))

                    # tmp_next_obs = np.array(tmp_next_obs).reshape((RL_TRAIN_PROCESS_NUM,-1))
                    # tmp_action = np.array(tmp_action).reshape((RL_TRAIN_PROCESS_NUM,-1))
                    sac_algo.replay_buffer.add(obs=tmp_obs,next_obs=tmp_next_obs,action=tmp_action,reward=tmp_reward,done=tmp_done,infos=tmp_infos)

                    # for tmp_obs, tmp_next_obs, tmp_action, tmp_reward, tmp_done, tmp_info in zip(loaded_obs, loaded_next_obs, loaded_action, loaded_reward, loaded_done, loaded_info):
                    #     for key in tmp_obs:
                    #         tmp_obs[key] = tmp_obs[key].reshape(RL_TRAIN_PROCESS_NUM, tmp_obs[key].shape[-1])
                    #     for key in tmp_next_obs:
                    #         tmp_next_obs[key] = tmp_next_obs[key].reshape(RL_TRAIN_PROCESS_NUM ,tmp_next_obs[key].shape[-1])
                    #     sac_algo.replay_buffer.add(obs=tmp_obs,next_obs=tmp_next_obs,action=tmp_action,reward=tmp_reward,done=tmp_done,infos=tmp_info)
                # all_tmp_obs= np.array(tmp_buffer["observations"])
                # all_tmp_next_obs = np.array(tmp_buffer["next_observations"])
                # all_tmp_action = np.array(tmp_buffer["actions"])
                # all_tmp_reward = np.array(tmp_buffer["rewards"])
                # all_tmp_done = np.array(tmp_buffer["dones"])
                # all_tmp_info = np.array(tmp_buffer["infos"])

                # for i in range(len(tmp_buffer.dones)):
                #     tmp_obs= all_tmp_obs[i] 
                #     tmp_next_obs = all_tmp_next_obs[i]
                #     tmp_action = all_tmp_action[i]
                #     tmp_reward = all_tmp_reward[i]
                #     tmp_done = all_tmp_done[i]
                #     tmp_info = all_tmp_info[i]
                #     sac_algo.replay_buffer.add(obs=tmp_obs,next_obs=tmp_next_obs,action=tmp_action,reward=tmp_reward,done=tmp_done,infos=tmp_info)
                #sac_algo.load_replay_buffer(policy_save_dir / train_config["rl_train"][index-1]["rl"]["experiment_name"] / replay_buffer_save_name)


                # sac_algo.replay_buffer.observations["observation"][:len(tmp_buffer.dones)] = tmp_buffer.observations["observation"][:]
                # sac_algo.replay_buffer.observations["achieved_goal"][:len(tmp_buffer.dones)] = tmp_buffer.observations["achieved_goal"][:]
                # sac_algo.replay_buffer.observations["desired_goal"][:len(tmp_buffer.dones)] = tmp_buffer.observations["desired_goal"][:]
                # sac_algo.replay_buffer.next_observations["observation"][:len(tmp_buffer.dones)] = tmp_buffer.next_observations["observation"][:]
                # sac_algo.replay_buffer.next_observations["achieved_goal"][:len(tmp_buffer.dones)] = tmp_buffer.next_observations["achieved_goal"][:]
                # sac_algo.replay_buffer.next_observations["desired_goal"][:len(tmp_buffer.dones)] = tmp_buffer.next_observations["desired_goal"][:]
                # sac_algo.replay_buffer.actions[:len(tmp_buffer.dones)] = tmp_buffer.actions[:]
                # sac_algo.replay_buffer.rewards[:len(tmp_buffer.dones)] = tmp_buffer.rewards[:]
                # sac_algo.replay_buffer.dones[:len(tmp_buffer.dones)] = tmp_buffer.dones[:]
                # sac_algo.replay_buffer.infos[:len(tmp_buffer.dones)] = tmp_buffer.infos[:]
                # sac_algo.replay_buffer.pos=len(tmp_buffer.actions)

                print(f"Iter {index}: load replay buffer from {policy_save_dir / train_config['rl_train'][index-1]['rl']['experiment_name'] / replay_buffer_save_name}.")
                print(f"sac_algo.replay_buffer.buffer_size = {sac_algo.replay_buffer.buffer_size}")
                # relabel rewards of transitions in the loaded replay buffer
                if THIS_ITER_RELABEL_REPLAY_BUFFER:
                    # sac_algo.replay_buffer.observations
                    if not THIS_ITER_WRAPPER_LIST :
                        loaded_replay_buffer_size = sac_algo.replay_buffer.size()
                        new_rewards = vec_env.env_method(
                            method_name="compute_reward",
                            indices=[0],
                            achieved_goal=sac_algo.replay_buffer.next_observations["achieved_goal"].squeeze()[:loaded_replay_buffer_size], 
                            desired_goal=sac_algo.replay_buffer.observations["desired_goal"].squeeze()[:loaded_replay_buffer_size],
                          #  info=sac_algo.replay_buffer.infos.squeeze()[:loaded_replay_buffer_size]
                        )[0]
                        tmp_reward = new_rewards.reshape(-1, 1)
                        sac_algo.replay_buffer.rewards[:loaded_replay_buffer_size] = new_rewards.reshape(-1, 1)

                        print(f"Iter {index}: reset rewards in replay buffer.")
                    else:
                        contains_frame_skip = any(wrapper.get("type") == "frame_skip" for wrapper in THIS_ITER_WRAPPER_LIST)
                        print("compute relabel reward for skip wrapper")
                        if contains_frame_skip:
                            
                            # helper_env: flycraft.env.FlyCraftEnv = flycraft.env.FlyCraftEnv(config_file=env_config_in_training)
                            # scaled_obs_env = ScaledObservationWrapper(helper_env)
                            # scaled_act_env = ScaledActionWrapper(scaled_obs_env)
                            
                            loaded_replay_buffer_size = sac_algo.replay_buffer.size()
                            
                            new_rewards = []

                            for info in sac_algo.replay_buffer.infos:
                                frame_skip_info = info[0].get('frame_skip_info')
                                if frame_skip_info is not None:
                                    reward = frame_skip_info[-1].get('reward')
                                    new_rewards.append(reward)                                                                             
                                else:
                                    new_rewards.append(0.0) 
 
                            new_rewards = np.array(new_rewards).reshape(-1, 1)
                            sac_algo.replay_buffer.rewards[:len(new_rewards)] = new_rewards.reshape(-1, 1)
                            if not THIS_ITER_STORE_INFO:
                                sac_algo.replay_buffer.infos = np.array([[{} for _ in range(sac_algo.replay_buffer.n_envs)] for _ in range(sac_algo.replay_buffer.buffer_size)])
                                sac_algo.replay_buffer.copy_info_dict = False
                            # for info in sac_algo.replay_buffer.infos:
                            #     info=np.array([{}])

                        
            else:
                print(f"Iter {index}: reset replay buffer.")
        else:
            # check whether to fill replay buffer with expert demonstrations
            if THIS_ITER_PRE_FILL_REPLAY_BUFFER:
                if USE_HER:
                    loaded_obs, loaded_next_obs, loaded_action, loaded_reward, loaded_done, loaded_info = load_random_trajectories_from_csv_files(
                        data_dir=PROJECT_ROOT_DIR / THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["data_dir"],
                        cache_data=THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["cache_data"],
                        cache_data_dir=PROJECT_ROOT_DIR / THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["cache_data_dir"],
                        trajectory_save_prefix=THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["trajectory_save_prefix"],
                        env_config_file=PROJECT_ROOT_DIR / "configs" / "env" / THIS_ITER_ENV_CONFIG_FILE,
                        select_transition_num=THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["selected_transition_num"],
                        random_state=THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["random_state"]
                    )
                    # sac_algo.replay_buffer.extend(
                    #     obs=loaded_obs,
                    #     next_obs=loaded_next_obs,
                    #     action=loaded_action,
                    #     reward=loaded_reward,
                    #     done=loaded_done,
                    #     infos=loaded_info,
                    # )
                    sac_algo.replay_buffer.extend(
                        observations=loaded_obs,
                        next_observations=loaded_next_obs,
                        actions=loaded_action,
                        rewards=loaded_reward,
                        dones=loaded_done,
                        infos=loaded_info,
                    )
                else:
                    loaded_obs, loaded_next_obs, loaded_action, loaded_reward, loaded_done, loaded_info = load_random_transitions_from_csv_files(
                    data_dir=PROJECT_ROOT_DIR / THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["data_dir"],
                    cache_data=THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["cache_data"],
                    cache_data_dir=PROJECT_ROOT_DIR / THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["cache_data_dir"],
                    trajectory_save_prefix=THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["trajectory_save_prefix"],
                    env_config_file=PROJECT_ROOT_DIR / "configs" / "env" / THIS_ITER_ENV_CONFIG_FILE,
                    select_transition_num=THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS["selected_transition_num"],
                    n_env = RL_TRAIN_PROCESS_NUM
                    )
                    # print(f"loaded_obs shape = {loaded_obs.shape}")
                    # print(f"loaded_next_obs shape = {loaded_next_obs.shape}")
                    # sac_algo.replay_buffer.extend(
                    #     obs=loaded_obs,
                    #     next_obs=loaded_next_obs,
                    #     action=loaded_action,
                    #     reward=loaded_reward,
                    #     done=loaded_done,
                    #     infos=loaded_info,
                    # )
                    # sac_algo.replay_buffer.extend(
                    #     # observations=loaded_obs,
                    #     # next_observations=loaded_next_obs,
                    #     # obs=loaded_obs,         
                    #     # next_obs=loaded_next_obs,
                    #     # action=loaded_action,
                    #     # reward=loaded_reward,
                    #     # done=loaded_done,
                    #     # infos=loaded_info,
                    #     loaded_obs,         
                    #     loaded_next_obs,
                    #     loaded_action,
                    #     loaded_reward,
                    #     loaded_done,
                    #     loaded_info,
                    # )
                    for tmp_obs, tmp_next_obs, tmp_action, tmp_reward, tmp_done, tmp_info in zip(loaded_obs, loaded_next_obs, loaded_action, loaded_reward, loaded_done, loaded_info):
                        for key in tmp_obs:
                            tmp_obs[key] = tmp_obs[key].reshape(RL_TRAIN_PROCESS_NUM, tmp_obs[key].shape[-1])
                        for key in tmp_next_obs:
                            tmp_next_obs[key] = tmp_next_obs[key].reshape(RL_TRAIN_PROCESS_NUM ,tmp_next_obs[key].shape[-1])
                        sac_algo.replay_buffer.add(obs=tmp_obs,next_obs=tmp_next_obs,action=tmp_action,reward=tmp_reward,done=tmp_done,infos=tmp_info)
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
                        #info=sac_algo.replay_buffer.infos.squeeze()[:loaded_replay_buffer_size]
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

        sac_algo.train(gradient_steps=int(THIS_ITER_WARMUP_EPOCHS * sac_algo.replay_buffer.size() / BATCH_SIZE ), batch_size=BATCH_SIZE)

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
