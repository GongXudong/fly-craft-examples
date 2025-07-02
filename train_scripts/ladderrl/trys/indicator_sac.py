import sys
from pathlib import Path
from abc import ABC
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
from copy import deepcopy

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.sac import SAC, MultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import copy

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))


from train_scripts.ladderrl.trys._functions import list_to_tensor, zero_optim_grads, weight_deviation, weight_hessian, feature_rank


class IndicatorSAC(SAC):
    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        
        self.global_step = 0

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            _init_setup_model=_init_setup_model,
        )


    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]


        self.recent_policy_state_dict = copy.deepcopy(self.actor.state_dict())
        self.recent_qf1_state_dict = copy.deepcopy(self.critic.state_dict())
        self.recent_qf2_state_dict = copy.deepcopy(self.critic_target.state_dict())


        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        # add zero_optim_grads, weight_deviation, weight_hessian, feature_rank
        # policy_zero_ratio = []
        # Qf1_zero_ration = []
        # Qf2_zero_ration = []

        # policy_feature_rank = []
        # Qf1_feature_rank = []
        # Qf2_feature_rank = []

        # policy_hessian_rank = []
        # Qf1_hessian_rank = []
        # Qf2_hessian_rank = []

        policy_weight_change = []
        Qf1_weight_change = []
        Qf2_weight_change = []

        for gradient_step in range(gradient_steps):
            self.global_step += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select aself.global_step = 0ction according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            

            # calculate indicator
            # Dormant neurons
            if self.global_step % 10000 == 0:

                # Dormant neurons
                # policy_zero_cnt = sum(self.actor._stats['zero_ratio'][-1000:]) / 1000
                # qf1_zero_cnt = sum(self._qf1._stats['zero_ratio'][-1000:]) / 1000
                # qf2_zero_cnt = sum(self._qf2._stats['zero_ratio'][-1000:]) / 1000

                # Feature rank, Hessian
                # eps = 0.001 # eps = 0.01 in Lyle et al. (2022); tried but the feature rank of policy is too small(below 10)
                # recent_obs = self.recent_trajectory.observation
                # recent_samples = self.recent_trajectory.samples


                # Weight deviation
                policy_state_dict = copy.deepcopy(self.actor.state_dict())
                qf1_state_dict = copy.deepcopy(self.critic.state_dict())
                qf2_state_dict = copy.deepcopy(self.critic_target.state_dict())


                policy_dev = weight_deviation(policy_state_dict, self.recent_policy_state_dict)
                qf1_dev = weight_deviation(qf1_state_dict, self.recent_qf1_state_dict)
                qf2_dev = weight_deviation(qf2_state_dict, self.recent_qf2_state_dict)

                self.recent_policy_state_dict = policy_state_dict
                self.recent_qf1_state_dict = qf1_state_dict
                self.recent_qf2_state_dict = qf2_state_dict

                # policy_weight_change.append()
                # Qf1_weight_change.append(qf1_dev.item())
                # Qf2_weight_change.append(qf2_dev.item())
                self.logger.record("train/policy_weight_change",policy_dev.item())
                self.logger.record("train/Qf1_weight_change",qf1_dev.item())
                self.logger.record("train/Qf2_weight_change",qf2_dev.item())
     
                
        

        self._n_updates += gradient_steps





        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        
    
        
