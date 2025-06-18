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

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.ppo import PPO, MultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.distributions import kl_divergence

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
from train_scripts.msr.utils.reset_env_utils import get_lower_bound_of_desired_goal, get_upper_bound_of_desired_goal


class SmoothGoalPPO(PPO):
    """Add auxiliary loss for goal regularization to PPO.

    Loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss + goal_regularization_strength * noised_goal_loss

    noised_goal_loss = max{KL(policy_action_dist, noised_goal_action_dist) - goal_regularization_loss_threshold, 0}.mean()

    对于原来的一个obs，生成noise_num_for_each_goal个noised_goal_obs，每个noised_goal_obs都添加了一个随机噪声，
    然后计算每个noised_goal_obs对应的动作分布，最后计算这些动作分布和原来的动作分布之间的KL散度或JS散度。

    注意：使用noise_num_for_each_goal个KL的**平均值**作为noised_goal_loss。

    Args:
        PPO (_type_): _description_
    """
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        goal_noise_epsilon: np.ndarray = np.array([10., 3., 3.]),
        goal_regularization_strength: float = 1e-3,
        goal_regularization_loss_threshold: float = 0.0,
        noise_num_for_each_goal: int = 1,
        policy_distance_measure_func: str = "KL",
        regularize_state_value: bool = False,
        state_value_regularization_strength: float = 1e-3,
        state_value_regularization_loss_threshold: float = 0.0,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        self.goal_noise_epsilon = goal_noise_epsilon
        self.goal_regularization_strength = goal_regularization_strength
        self.goal_regularization_loss_threshold = goal_regularization_loss_threshold
        self.noise_num_for_each_goal = noise_num_for_each_goal
        self.policy_distance_measure_func = policy_distance_measure_func
        self.regularize_state_value = regularize_state_value
        self.state_value_regularization_strength = state_value_regularization_strength
        self.state_value_regularization_loss_threshold = state_value_regularization_loss_threshold

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

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        noised_goal_kls, noised_goal_losses = [], []
        noised_goal_value_dists, noised_goal_value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                # -----------------begin: 为了得到distribution，coyp了evaluation_actions的源码----------------------
                features = self.policy.extract_features(rollout_data.observations)
                if self.policy.share_features_extractor:
                    latent_pi, latent_vf = self.policy.mlp_extractor(features)
                else:
                    pi_features, vf_features = features
                    latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
                    latent_vf = self.policy.mlp_extractor.forward_critic(vf_features)
                distribution = self.policy._get_action_dist_from_latent(latent_pi)
                log_prob = distribution.log_prob(actions)
                values = self.policy.value_net(latent_vf)
                entropy = distribution.entropy()

                # 由于每个(state, goal)对应了noise_num_for_each_goal个noised_goal_obs，所以action_dist需要重复noise_num_for_each_goal次
                # action_dist仅用于计算noised_goal_loss
                action_dist = th.distributions.Normal(
                    loc=distribution.distribution.loc.repeat((self.noise_num_for_each_goal, 1)),
                    scale=distribution.distribution.scale.repeat((self.noise_num_for_each_goal, 1))
                )

                # -----------------end: 为了得到distribution，coyp了evaluation_actions的源码----------------------

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # ---------------------------------begin: goal regularization loss---------------------------------
                # 1.sample noise and add to obs
                noised_goal_obs = deepcopy(rollout_data.observations)
                for k in noised_goal_obs.keys():
                    noised_goal_obs[k] = noised_goal_obs[k].repeat((self.noise_num_for_each_goal, 1))
                self.add_noise_to_desired_goals(observations=noised_goal_obs)

                # 2.get action dist
                noised_goal_action_dist = self.policy.get_distribution(noised_goal_obs)

                # 3.calc KL or JS loss
                if self.policy_distance_measure_func == "KL":
                    noised_goal_kl = th.distributions.kl_divergence(action_dist, noised_goal_action_dist.distribution).sum(axis=-1)
                elif self.policy_distance_measure_func == "JS":
                    noised_goal_kl = th.distributions.kl_divergence(action_dist, noised_goal_action_dist.distribution).sum(axis=-1) + th.distributions.kl_divergence(noised_goal_action_dist.distribution, action_dist).sum(axis=-1)
                else:
                    raise ValueError("policy_distance_measure_func must be either KL or JS!")

                # 4.clip goal regularization loss
                if self.goal_regularization_loss_threshold > 0.0:
                    noised_goal_loss = th.maximum(noised_goal_kl - th.ones_like(noised_goal_kl) * self.goal_regularization_loss_threshold, th.zeros_like(noised_goal_kl)).mean()
                else:
                    noised_goal_loss = noised_goal_kl.mean()

                # 5.log
                noised_goal_kls.append(noised_goal_kl.mean().item())
                noised_goal_losses.append(noised_goal_loss.item())
                
                # 6.regularize state value
                # 6.1 get state values
                original_values = values_pred.unsqueeze(dim=1).repeat((self.noise_num_for_each_goal, 1))
                noised_goal_values = self.policy.predict_values(noised_goal_obs)
                
                # 6.2 calc loss
                noised_goal_value_dist = F.mse_loss(
                    input=original_values,
                    target=noised_goal_values,
                    reduction="none"
                )  # shape = (batch_size, 1)
                
                noised_goal_value_dists.append(noised_goal_value_dist.mean().item())

                if self.regularize_state_value:
                    # 6.3 clip state value regularization loss
                    if self.state_value_regularization_loss_threshold > 0.0:
                        noised_goal_value_loss = th.maximum(noised_goal_value_dist - th.ones_like(noised_goal_value_dist) * self.state_value_regularization_loss_threshold, th.zeros_like(noised_goal_value_dist)).mean()
                    else:
                        noised_goal_value_loss = noised_goal_value_dist.mean()
                    
                    # 6.4 log
                    noised_goal_value_losses.append(noised_goal_value_loss.item())
                
                # ---------------------------------end: goal regularization loss---------------------------------

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.goal_regularization_strength * noised_goal_loss
                if self.regularize_state_value:
                    loss += self.state_value_regularization_strength * noised_goal_value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/noised_goal_kl", np.mean(noised_goal_kls))
        self.logger.record("train/noised_goal_loss", np.mean(noised_goal_losses))
        self.logger.record("train/noised_goal_value_dist", np.mean(noised_goal_value_dists))
        if self.regularize_state_value:
            self.logger.record("train/noised_goal_value_loss", np.mean(noised_goal_value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
