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
import gymnasium_robotics


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_eval_callback import MyEvalCallback
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate
from train_scripts.ladderrl.utils.get_vec_env import get_vec_env
from train_scripts.ladderrl.utils.load_data_from_csv import load_random_trajectories_from_csv_files,load_random_transitions_from_csv_files
from utils_my.sb3.my_replay_buffer_utils import fill_replay_buffer

from train_scripts.ladderrl.trys.unified_indicator_callback_change_type import  UnifiedNetworkMonitorCallback_TYPE
import warnings
warnings.filterwarnings("ignore")  # 过滤Gymnasium的UserWarning
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from hand.PowerRewardWrapper import PowerRewardWrapper


def reset_actor(sac_algo):
    actor = sac_algo.policy.actor

    # 1. reset parameters in-place
    for m in actor.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    # 2. reset optimizer
    old_lr = actor.optimizer.param_groups[0]["lr"]
    actor.optimizer = type(actor.optimizer)(
        actor.parameters(), lr=old_lr
    )

def reset_critic(sac_algo):
    critic = sac_algo.policy.critic
    critic_target = sac_algo.policy.critic_target

    # 1. reset critic
    for m in critic.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    # 2. sync target critic
    critic_target.load_state_dict(critic.state_dict())

    # 3. reset optimizer
    old_lr = critic.optimizer.param_groups[0]["lr"]
    critic.optimizer = type(critic.optimizer)(
        critic.parameters(), lr=old_lr
    )

def get_param_stats(module):
    params = th.cat([p.data.flatten() for p in module.parameters()])
    return {
        "mean": params.mean().item(),
        "std": params.std().item(),
        "norm": params.norm().item(),
        "max": params.abs().max().item(),
    }

def compare_critic_parameter(sac_algo):           
    diff = 0.0
    for p, tp in zip(sac_algo.policy.critic.parameters(), sac_algo.policy.critic_target.parameters()):
        diff += (p - tp).abs().mean().item()
    return diff
    

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
    NON_LINEARITY = train_config["rl_common"].get("non_linear", "tanh") 
    

    for index, train_this_iter_config in enumerate(train_config["rl_train"]):

        This_ITER_ENV_ID = train_this_iter_config["env"]["id"]
        This_ITER_ENV_BETA = train_this_iter_config["env"]["b"]
        This_ITER_ENV_EVAL_BETA = train_this_iter_config["env"]["evaluate_b"]
        This_ITER_ENV_MIN_ANGLE = train_this_iter_config["env"]["min_angle_rad"]
        This_ITER_ENV_MAX_ANGLE = train_this_iter_config["env"]["max_angle_rad"]
        This_ITER_ENV_EVAL_MIN_ANGLE = train_this_iter_config["env"]["eval_min_angle_rad"]
        This_ITER_ENV_EVAL_MAX_ANGLE = train_this_iter_config["env"]["eval_max_angle_rad"]

        THIS_ITER_SEED = train_this_iter_config["rl"].get("seed")
        THIS_ITER_SEED_IN_TRAINING_ENV = train_this_iter_config["rl"].get("seed_in_train_env")
        THIS_ITER_SEED_IN_CALLBACK_ENV = train_this_iter_config["rl"].get("seed_in_callback_env")
        THIS_ITER_RL_EXPERIMENT_NAME = train_this_iter_config["rl"]["experiment_name"]
        THIS_ITER_RL_TRAIN_STEPS = train_this_iter_config["rl"]["train_steps"]
        THIS_ITER_LEARNING_STARTS = train_this_iter_config["rl"].get("learning_starts", 10240)
        THIS_ITER_LEARNING_RATE = train_this_iter_config["rl"].get("learning_rate", 3e-4)
        THIS_ITER_RESET_POLICY = train_this_iter_config["rl"].get("reset_policy", False)
        THIS_ITER_RESET_SCOPE = train_this_iter_config["rl"].get("reset_scope", "all")
        THIS_ITER_WRAPPER_LIST = train_this_iter_config['rl'].get("wrappers", [])
        THIS_ITER_RESET_REPLAY_BUFFER = train_this_iter_config["rl"].get("reset_replay_buffer", False)
        THIS_ITER_RELABEL_REPLAY_BUFFER = train_this_iter_config["rl"].get("relabel_replay_buffer", False)
        THIS_ITER_HAS_TRAINED = train_this_iter_config["rl"].get("has_trained", False)
        THIS_ITER_CHECK_FRQ = train_this_iter_config["rl"].get("check_freq", 10000)
        THIST_ITER_INDICATOR_BATCH = train_this_iter_config["rl"].get("indicator_batch", 2048)
        THIS_ITER_TAU = train_this_iter_config["rl"].get("tau", 0.3)

        THIS_ITER_STORE_INFO =  train_this_iter_config["rl"].get("store_info", False)
        THIS_ITER_PRE_FILL_REPLAY_BUFFER = train_this_iter_config["rl"].get("pre_fill_replay_buffer", False)
        THIS_ITER_PRE_FILL_REPLAY_BUFFER_KWARGS = train_this_iter_config["rl"].get("pre_fill_replay_buffer_kwargs", {})
        THIS_ITER_WARMUP_EPOCHS=train_this_iter_config["rl"].get("warmup_epochs", 0)
        THIST_ITER_BUFFER_SAVE_NAME = train_this_iter_config["rl"].get("replay_buffer_save_name","replay_buffer")
        if THIS_ITER_HAS_TRAINED:
            continue

        # prepare env
        vec_env = make_vec_env(
            env_id=This_ITER_ENV_ID,
            n_envs=RL_TRAIN_PROCESS_NUM,
            seed=THIS_ITER_SEED_IN_TRAINING_ENV,
            vec_env_cls=SubprocVecEnv, 
            wrapper_class=PowerRewardWrapper, 
            wrapper_kwargs={"b": This_ITER_ENV_BETA},
            env_kwargs={
                "min_angle_rad":np.radians(This_ITER_ENV_MIN_ANGLE),
                "max_angle_rad":np.radians(This_ITER_ENV_MAX_ANGLE)
            }
        )

        eval_env_in_callback = make_vec_env(
            env_id= This_ITER_ENV_ID,
            n_envs= CALLBACK_PROCESS_NUM,
            seed=THIS_ITER_SEED_IN_CALLBACK_ENV,
            wrapper_class=PowerRewardWrapper, 
            wrapper_kwargs={"b": This_ITER_ENV_EVAL_BETA,},
            vec_env_cls=SubprocVecEnv, 
            env_kwargs={
                "min_angle_rad":np.radians(This_ITER_ENV_EVAL_MIN_ANGLE),
                "max_angle_rad":np.radians(This_ITER_ENV_EVAL_MAX_ANGLE)
            }
        )




        policy_save_dir = PROJECT_ROOT_DIR / "checkpoints"
        policy_save_name = "best_model"
        # policy_save_name = "final_model"
        #replay_buffer_save_name = "replay_buffer"
        replay_buffer_save_name = THIST_ITER_BUFFER_SAVE_NAME

        # prepare policy
        if (index == 0): 
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
                    activation_fn=th.nn.Tanh if NON_LINEARITY=="tanh" else th.nn.ReLU
                ),
            )
            print(f"Iter {index}: Full Reset - Scope: {THIS_ITER_RESET_SCOPE}")

        if (index > 0 and THIS_ITER_RESET_POLICY):
            print(f'index= {index}',f'copy_info_dict= {THIS_ITER_STORE_INFO}')

            if THIS_ITER_RESET_SCOPE == "all":
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
                        activation_fn=th.nn.Tanh if NON_LINEARITY=="tanh" else th.nn.ReLU
                    ),
                )
                print(f"Iter {index}: Full Reset - Scope: {THIS_ITER_RESET_SCOPE}")

            elif THIS_ITER_RESET_SCOPE in ["actor","critic"]:
                sac_algo = SAC.load(path=policy_save_dir / train_config["rl_train"][index-1]["rl"]["experiment_name"] / policy_save_name,env=vec_env)              
                print(f"Iter {index}: load policy from {policy_save_dir / train_config['rl_train'][index-1]['rl']['experiment_name'] / policy_save_name}.")

                if THIS_ITER_RESET_SCOPE == "actor":
                    print(f"Iter {index}: >>> Resetting ACTOR network and optimizer <<<")
                    # print("Optimizer state size BEFORE:", len(sac_algo.policy.actor.optimizer.state))
                    # print("Actor BEFORE reset:", get_param_stats(sac_algo.policy.actor))
                    
                    reset_actor(sac_algo)
                    # print("Actor AFTER reset:", get_param_stats(sac_algo.policy.actor))
                    # print("Optimizer state size After:", len(sac_algo.policy.actor.optimizer.state))

                
                elif THIS_ITER_RESET_SCOPE == "critic":
                    print(f"Iter {index}: >>> Resetting CRITIC network and optimizer <<<")

                    # print("Critic BEFORE reset:", get_param_stats(sac_algo.policy.critic))
                    # print("Critic-target diff before reset:", compare_critic_parameter(sac_algo))    
                    # print("Optimizer state size BEFORE:", len(sac_algo.policy.critic.optimizer.state))
                    reset_critic(sac_algo)
                    # print("Optimizer state size After:", len(sac_algo.policy.critic.optimizer.state))
                    # print("Critic-target diff after reset:", compare_critic_parameter(sac_algo)) 
                    # print("Critic AFTER reset:", get_param_stats(sac_algo.policy.critic))

                

        elif  (index > 0)  and (not THIS_ITER_RESET_POLICY) :
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
                import pathlib
                path = policy_save_dir / train_config["rl_train"][index-1]["rl"]["experiment_name"] / replay_buffer_save_name
                file = open_path(path, "r", suffix="pkl")
                tmp_buffer = pickle.load(file)
                if isinstance(path, (str, pathlib.Path)):
                    file.close()
                
                tmp_size = tmp_buffer.buffer_size
                print(f"tmp load  buffer_size = {tmp_size} ",f"tmp_buffer.size() = {tmp_buffer.size()}")
                if tmp_size != BUFFER_SIZE:
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
                        tmp_infos = [{}]
                        

                        # [env_inds, obs_shape]

                        for key in tmp_obs.keys():
                            tmp_obs[key] = tmp_obs[key].reshape((RL_TRAIN_PROCESS_NUM , tmp_obs[key].shape[-1]))
                        
                        for key in tmp_next_obs.keys():
                            tmp_next_obs[key] = tmp_next_obs[key].reshape((RL_TRAIN_PROCESS_NUM , tmp_next_obs[key].shape[-1]))
                        
                        tmp_action = tmp_action.reshape((RL_TRAIN_PROCESS_NUM, tmp_action.shape[-1]))
                        sac_algo.replay_buffer.add(obs=tmp_obs,next_obs=tmp_next_obs,action=tmp_action,reward=tmp_reward,done=tmp_done,infos=tmp_infos)
                else:
                    sac_algo.load_replay_buffer(policy_save_dir / train_config["rl_train"][index-1]["rl"]["experiment_name"] / replay_buffer_save_name)
                print(f"Iter {index}: load replay buffer from {policy_save_dir / train_config['rl_train'][index-1]['rl']['experiment_name'] / replay_buffer_save_name}.")

                # relabel rewards of transitions in the loaded replay buffer
                if THIS_ITER_RELABEL_REPLAY_BUFFER and USE_HER:
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
                elif THIS_ITER_RELABEL_REPLAY_BUFFER and not USE_HER:
                    # sac_algo.replay_buffer.observations
                    loaded_replay_buffer_size = sac_algo.replay_buffer.size()
                    new_rewards = vec_env.env_method(
                        method_name="compute_reward",
                        indices=[0],
                        achieved_goal=sac_algo.replay_buffer.next_observations["achieved_goal"].squeeze()[:loaded_replay_buffer_size], 
                        desired_goal=sac_algo.replay_buffer.observations["desired_goal"].squeeze()[:loaded_replay_buffer_size],
                        info=[{} for _ in range(loaded_replay_buffer_size)]
                        
                    )[0]

                    sac_algo.replay_buffer.rewards[:loaded_replay_buffer_size] = new_rewards.reshape(-1, 1)
                    print(f"Iter {index}: reset rewards in replay buffer.")
                
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
                    fill_replay_buffer(replay_buffer=sac_algo.replay_buffer, observations=loaded_obs, actions=loaded_action, next_observations=loaded_next_obs, rewards=loaded_reward, dones=loaded_done, infos=loaded_info, n_envs=RL_TRAIN_PROCESS_NUM)
                    # for tmp_obs, tmp_next_obs, tmp_action, tmp_reward, tmp_done, tmp_info in zip(loaded_obs, loaded_next_obs, loaded_action, loaded_reward, loaded_done, loaded_info):
                    #     for key in tmp_obs:
                    #         tmp_obs[key] = tmp_obs[key].reshape(RL_TRAIN_PROCESS_NUM, tmp_obs[key].shape[-1])
                    #     for key in tmp_next_obs:
                    #         tmp_next_obs[key] = tmp_next_obs[key].reshape(RL_TRAIN_PROCESS_NUM ,tmp_next_obs[key].shape[-1])
                    #     sac_algo.replay_buffer.add(obs=tmp_obs,next_obs=tmp_next_obs,action=tmp_action,reward=tmp_reward,done=tmp_done,infos=tmp_info)
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

        # dormant callback
        unified_indicator_callback = UnifiedNetworkMonitorCallback_TYPE(check_freq=THIS_ITER_CHECK_FRQ,batch_size=THIST_ITER_INDICATOR_BATCH,dormant_tau=THIS_ITER_TAU,verbose=1,non_linearity=NON_LINEARITY)

        
        sac_algo.train(gradient_steps=int(THIS_ITER_WARMUP_EPOCHS * sac_algo.replay_buffer.size() / BATCH_SIZE ), batch_size=BATCH_SIZE)
        sac_algo.learn(
            total_timesteps=int(THIS_ITER_RL_TRAIN_STEPS),
            callback=[eval_callback, event_callback,unified_indicator_callback]
        )

        sac_algo.save(str(PROJECT_ROOT_DIR / "checkpoints" / THIS_ITER_RL_EXPERIMENT_NAME / "final_model"))
        replay_buffer_save_name = "replay_buffer"
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
