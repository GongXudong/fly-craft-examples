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

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from train_scripts.disc.utils.reset_env_utils import get_lower_bound_of_desired_goal, get_upper_bound_of_desired_goal


class SmoothGoalSAC(SAC):

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
        goal_noise_epsilon: np.ndarray = np.array([10., 3., 3.]),
        goal_regularization_strength: float = 1e-3,
        policy_distance_measure_func: str = "KL",
    ):
        
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

        self.goal_noise_epsilon = goal_noise_epsilon
        self.goal_regularization_strength = goal_regularization_strength
        self.policy_distance_measure_func = policy_distance_measure_func
    
    def init_desired_goal_params(self, helper_env: gym.Env=None):
        """_summary_

        Args:
            helper_env (gym.Env, optional): 使用ScaledActionWrapper, ScaledObservationWrapper包装过的env. Defaults to None.
        """

        assert isinstance(helper_env, ScaledActionWrapper), "需要使用ScaledActionWrapper包装环境"
        assert hasattr(helper_env, "env") and isinstance(helper_env.env, ScaledObservationWrapper), "需要使用ScaledObservationWrapper包装环境"

        self.desired_goal_max = get_upper_bound_of_desired_goal(helper_env)
        self.desired_goal_max = helper_env.env.goal_scalar.transform(self.desired_goal_max.reshape((1, -1))).reshape((-1))
        self.desired_goal_max = th.tensor(self.desired_goal_max, requires_grad=False, device=self.device)

        self.desired_goal_min = get_lower_bound_of_desired_goal(helper_env)
        self.desired_goal_min = helper_env.env.goal_scalar.transform(self.desired_goal_min.reshape((1, -1))).reshape((-1))
        self.desired_goal_min = th.tensor(self.desired_goal_min, requires_grad=False, device=self.device)

        self.noise_max = helper_env.env.goal_scalar.transform(self.goal_noise_epsilon.reshape((1, -1))).reshape((-1)) - np.array([0., 0.5, 0.5])
        self.noise_min = - self.noise_max.copy()
        self.noise_max = th.tensor(self.noise_max, requires_grad=False, device=self.device)
        self.noise_min = th.tensor(self.noise_min, requires_grad=False, device=self.device)
    
    def add_noise_to_desired_goals(self, observations: TensorDict) -> None:
        observations["desired_goal"] = th.clamp(
            input=observations["desired_goal"] + th.rand(size=observations["desired_goal"].shape, requires_grad=False, device=self.device) * (self.noise_max - self.noise_min) + self.noise_min,
            min=self.desired_goal_min,
            max=self.desired_goal_max,
        )
    
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        noised_goal_losses = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            
            # TODO: loc与scale是否需要detach？？？
            action_dist = th.distributions.Normal(
                loc=self.actor.action_dist.distribution.loc,
                scale=self.actor.action_dist.distribution.scale
            )
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
                # Select action according to policy
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

            # goal regularization loss !!!!!
            # 1.sample noise and add to obs
            noised_goal_obs = deepcopy(replay_data.observations)
            self.add_noise_to_desired_goals(observations=noised_goal_obs)

            # 2.get action dist
            noised_goal_actions_pi, noised_goal_log_prob = self.actor.action_log_prob(noised_goal_obs)
            noised_goal_action_dist = self.actor.action_dist.distribution
            # 3. calc KL or JS loss
            if self.policy_distance_measure_func == "KL":
                noised_goal_loss = th.distributions.kl_divergence(action_dist, noised_goal_action_dist).sum(axis=-1).mean()
            elif self.policy_distance_measure_func == "JS":
                noised_goal_loss = th.distributions.kl_divergence(action_dist, noised_goal_action_dist).sum(axis=-1) + th.distributions.kl_divergence(noised_goal_action_dist, action_dist).sum(axis=-1)
                noised_goal_loss = noised_goal_loss.mean()
            else:
                raise ValueError("policy_distance_measure_func must be either KL or JS!")
            
            actor_loss += self.goal_regularization_strength * noised_goal_loss
            noised_goal_losses.append(noised_goal_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/noised_goal_loss", np.mean(noised_goal_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

