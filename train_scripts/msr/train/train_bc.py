import gymnasium as gym
import numpy as np
from pathlib import Path
import logging
import torch as th
import argparse
from copy import deepcopy
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.sac import MlpPolicy
from imitation.util.logger import HierarchicalLogger
from imitation.util import util
from imitation.data import types
from imitation.data.types import TransitionsMinimal
from imitation.data import rollout

import flycraft
from flycraft.utils_common.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.msr.algorithms.smooth_goal_bc import SmoothGoalBC, BCTrainingMetrics, BehaviorCloningLossCalculator
from train_scripts.msr.algorithms.smooth_goal_ppo import SmoothGoalPPO
from utils_my.sb3.vec_env_helper import get_vec_env
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate
from demonstrations.utils.load_dataset import load_data_from_cache
from utils_my.sb3.my_schedule import linear_schedule
from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper


def get_rl_algo(env):
    policy_kwargs = dict(
        full_std=True,  # 使用state dependant exploration
        # squash_output=True,  # 使用state dependant exploration
        net_arch=dict(
            pi=NET_ARCH,
            vf=deepcopy(NET_ARCH)
        ),
        activation_fn=th.nn.Tanh,
        ortho_init=True,
        optimizer_class=th.optim.Adam,
        optimizer_kwargs={
            "eps": 1e-5
        }
    )

    return SmoothGoalPPO(
        policy=MultiInputPolicy, 
        env=env, 
        seed=SEED,
        batch_size=PPO_BATCH_SIZE,  # PPO Mini Batch Size, PPO每次更新使用的数据量
        gamma=GAMMA,
        ent_coef=RL_ENT_COEF,
        n_steps=2048,  # 采样时每个环境采样的step数，PPO每次训练收集的数据量是n_steps * num_envs
        n_epochs=5,  # 采样的数据在训练中重复使用的次数
        policy_kwargs=policy_kwargs,
        use_sde=True,  # 使用state dependant exploration,
        normalize_advantage=True,
        device=DEVICE,
        learning_rate=linear_schedule(3e-4),
    )


# strategy for save policy：根据最小的loss保存。
# 因为bc.train()方法的on_batch_end是没有参数的回调函数，所以这里使用闭包，通过一个外部变量记录最高的prob_true_act
def on_best_loss_save(algo: BaseAlgorithm, validation_transitions: TransitionsMinimal, loss_calculator: BehaviorCloningLossCalculator, sb3_logger: Logger):
    min_loss = LOSS_THRESHOLD  # 预估一个初始值，不然训练开始阶段会浪费大量的时间在存储模型上！！！
    def calc_func():
        algo.policy.set_training_mode(mode=False)
        
        nonlocal min_loss
        
        obs = types.map_maybe_dict(
                lambda x: util.safe_to_tensor(x, device=DEVICE),
                types.maybe_unwrap_dictobs(validation_transitions.obs),
            )
        # obs = util.safe_to_tensor(validation_transitions.obs, device=DEVICE)
        acts = util.safe_to_tensor(validation_transitions.acts, device=DEVICE)
        
        metrics: BCTrainingMetrics = loss_calculator(policy=algo.policy, obs=obs, acts=acts)
        cur_loss = metrics.loss
        if cur_loss < min_loss:
            sb3_logger.info(f"update loss from {min_loss} to {cur_loss}!")
            min_loss = cur_loss

            # save policy
            checkpoint_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "msr" / EXPERIMENT_NAME
            if not checkpoint_save_dir.exists():
                os.makedirs(checkpoint_save_dir)

            algo.save(str(checkpoint_save_dir / POLICY_FILE_SAVE_NAME))

        algo.policy.set_training_mode(mode=True)
    return calc_func


def train():

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / "msr" / EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    vec_env = get_vec_env(
        num_process=RL_TRAIN_PROCESS_NUM,
        seed=RL_SEED,
        config_file=str(PROJECT_ROOT_DIR / "configs" / "env" / train_config["env"].get("config_file", "env_config_for_sac.json")),
        custom_config={"debug_mode": False}
    )

    helper_env = ScaledActionWrapper(
        ScaledObservationWrapper(
            gym.make(
                "FlyCraft-v0", 
                config_file=str(PROJECT_ROOT_DIR / "configs" / "env" / train_config["env"].get("config_file", "env_config_for_sac.json"))
            )
        )
    )
    
    algo_ppo = get_rl_algo(vec_env)
    sb3_logger.info(str(algo_ppo.policy))
    # print(algo_ppo.policy)

    rng = np.random.default_rng(SEED)

    # train_transitions, validation_transitions, test_transitions = load_data(
    #     data_dir=EXPERT_DATA_DIR, 
    #     my_logger=sb3_logger,
    #     train_size=0.9,
    #     validation_size=0.05,
    #     test_size=0.05
    # )  # 10hz的数据划分结果，train: 745643, validation: 41425, test: 41425

    sb3_logger.info("load data from: " + str(PROJECT_ROOT_DIR / "demonstrations" / EXPERT_DATA_CACHE_DIR))

    train_transitions, validation_transitions, test_transitions = load_data_from_cache(
        PROJECT_ROOT_DIR / "demonstrations" / EXPERT_DATA_CACHE_DIR,
        train_size=0.96,
        validation_size=0.02,
        test_size=0.02,
        shuffle=True,
    )

    sb3_logger.info(f"train_set: obs size, {train_transitions.obs.shape}, act size, {train_transitions.acts.shape}")
    sb3_logger.info(f"validation_set: obs size, {validation_transitions.obs.shape}, act size, {validation_transitions.acts.shape}")
    sb3_logger.info(f"test_set: obs size, {test_transitions.obs.shape}, act size, {test_transitions.acts.shape}")

    # begin：使用随机策略的采样训练
    # rollouts = rollout.rollout(
    #     policy=None,
    #     venv=vec_env,
    #     sample_until=rollout.make_sample_until(min_timesteps=None, min_episodes=5),
    #     rng=rng,
    #     unwrap=True,
    # )
    # train_transitions = rollout.flatten_trajectories(rollouts)
    # end：使用随机策略的采样训练

    # 注意: 需要把imitation(版本1.0.0)框架bc算法的494行，由acts = util.safe_to_tensor(batch["acts"]， device=self.policy.device)改为acts = util.safe_to_tensor(batch["acts"]).to(device=self.policy.device)！！！！！

    bc_trainer = SmoothGoalBC(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        policy=algo_ppo.policy,
        batch_size=BC_BATCH_SIZE,
        ent_weight=BC_ENT_WEIGHT,
        l2_weight=BC_L2_WEIGHT,
        noise_num_for_each_goal=NOISE_NUM_FOR_EACH_GOAL,
        policy_distance_measure_func=POLICY_DISTANCE_MEASURE_FUNC,
        ng_weight=NOISE_GOAL_LOSS_WEIGHT,
        demonstrations=train_transitions,
        rng=rng,
        device=DEVICE,
        custom_logger=HierarchicalLogger(sb3_logger)
    )

    sb3_logger.log(f"Check algo configs, epsilon: {GOAL_NOISE_EPSILON}, reg: {bc_trainer.loss_calculator.ng_weight}, noise num: {bc_trainer.loss_calculator.noise_num_for_each_goal}")

    bc_trainer.loss_calculator.init_desired_goal_params(helper_env, goal_noise_epsilon=np.array(GOAL_NOISE_EPSILON), device=algo_ppo.device)

    # train
    bc_trainer.train(
        n_epochs=TRAIN_EPOCHS,
        # on_batch_end=on_best_act_prob_save(algo_ppo, validation_transitions, sb3_logger),
        on_batch_end=on_best_loss_save(algo_ppo, validation_transitions, bc_trainer.loss_calculator, sb3_logger),
    )

    # evaluate with environment
    reward, _, success_rate = evaluate_policy_with_success_rate(algo_ppo.policy, vec_env, n_eval_episodes=1000)
    sb3_logger.info("Reward after BC: ", reward)
    sb3_logger.info("Success rate: ", success_rate)

    # 最终的policy在测试集上的prob_true_act / loss
    test_on_loss(algo_ppo.policy, test_transitions, bc_trainer.loss_calculator, sb3_logger, "训练结束时的策略", "测试集")

    return sb3_logger, validation_transitions, test_transitions, bc_trainer

def test_on_loss(
        policy: MlpPolicy, 
        test_transitions: TransitionsMinimal, 
        loss_calculator: BehaviorCloningLossCalculator, 
        sb3_logger: Logger, 
        policy_descreption: str, dataset_descreption: str
    ):
    policy.set_training_mode(mode=False)

    obs = types.map_maybe_dict(
                lambda x: util.safe_to_tensor(x, device=DEVICE),
                types.maybe_unwrap_dictobs(test_transitions.obs),
            )
    acts = util.safe_to_tensor(test_transitions.acts, device=DEVICE)
    
    metrics: BCTrainingMetrics = loss_calculator(policy=policy, obs=obs, acts=acts)
    sb3_logger.info(f"{policy_descreption}在{dataset_descreption}上的loss: {metrics.loss}.")


# python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/medium/128_128_seed_1.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="传入配置文件")
    parser.add_argument("--config-file-name", type=str, help="配置文件名", default="configs/train/msr/bc/medium/128_128_seed_1.json")
    args = parser.parse_args()

    train_config = load_config(Path(os.getcwd()) / args.config_file_name)

    EXPERIMENT_NAME = train_config["bc"]["experiment_name"]
    SEED = train_config["bc"]["seed"]
    POLICY_FILE_SAVE_NAME = train_config["bc"]["policy_file_save_name"]
    TRAIN_EPOCHS = train_config["bc"]["train_epochs"]
    BC_BATCH_SIZE = train_config["bc"]["batch_size"]
    BC_L2_WEIGHT = train_config["bc"].get("l2_weight", 0.0)
    BC_ENT_WEIGHT = train_config["bc"].get("ent_weight", 1e-3)
    EXPERT_DATA_CACHE_DIR = train_config["bc"]["data_cache_dir"]
    GOAL_NOISE_EPSILON = train_config["bc"].get("goal_noise_epsilon", [10., 3., 3.])
    NOISE_GOAL_LOSS_WEIGHT = train_config["bc"].get("noise_goal_loss_weight", 1e-3)
    LOSS_THRESHOLD = train_config["bc"]["loss_threshold"]
    NOISE_NUM_FOR_EACH_GOAL = train_config["bc"].get("noise_num_for_each_goal", 1)
    POLICY_DISTANCE_MEASURE_FUNC = train_config["bc"].get("policy_distance_measure_func", "KL")

    RL_SEED = train_config["rl"]["seed"]
    NET_ARCH = train_config["rl"]["net_arch"]
    PPO_BATCH_SIZE = train_config["rl"]["batch_size"]
    GAMMA = train_config["rl"]["gamma"]
    RL_ENT_COEF = train_config["rl"].get("ent_coef", 0.0)
    RL_TRAIN_PROCESS_NUM = train_config["rl"]["rollout_process_num"]
    DEVICE = train_config["rl"].get("device", "cpu")

    sb3_logger, validation_transitions, test_transitions, bc_trainer = train()

    # sb3_logger: Logger = configure(folder=str((PROJECT_TOOR_DIR / "my_rl_logs" / "bc" / EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv'])
    # train_transitions, validation_transitions, test_transitions = load_data(
    #     data_dir=EXPERT_DATA_DIR, 
    #     my_logger=sb3_logger,
    #     train_size=0.9,
    #     validation_size=0.05,
    #     test_size=0.05
    # )

    policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "msr"/ EXPERIMENT_NAME
    algo_ppo = SmoothGoalPPO.load(str((policy_save_dir / POLICY_FILE_SAVE_NAME).absolute()))

    test_on_loss(algo_ppo.policy, validation_transitions, bc_trainer.loss_calculator, sb3_logger, "最优策略", "验证集")
    test_on_loss(algo_ppo.policy, test_transitions, bc_trainer.loss_calculator, sb3_logger, "最优策略", "测试集")

    # model = PPO("MlpPolicy", vec_env, verbose=1)
    # model.learn(total_timesteps=25_000)

    # obs = vec_env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = vec_env.step(action)
    #     vec_env.render()