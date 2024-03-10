from typing import Any, Dict, Optional, Type, TypeVar, Union, Callable
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from pathlib import Path
from pathlib import Path

from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.distributions import kl_divergence


class PPOWithBCLoss(PPO):

    def __init__(self, 
        policy: Union[str, Type[ActorCriticPolicy]], 
        env: Union[GymEnv, str], 
        bc_trained_algo: PPO = None,
        kl_coef_with_bc: Union[float, Callable] = 0.01,
        learning_rate: Union[float, Schedule] = 0.0003, 
        n_steps: int = 2048, 
        batch_size: int = 64, 
        n_epochs: int = 10, 
        gamma: float = 0.99, 
        gae_lambda: float = 0.95, 
        clip_range: Union[float, Schedule] = 0.2, 
        clip_range_vf: Union[None, float, Schedule] = None, 
        normalize_advantage: bool = True, 
        ent_coef: float = 0, 
        vf_coef: float = 0.5, 
        max_grad_norm: float = 0.5, 
        use_sde: bool = False, 
        sde_sample_freq: int = -1, 
        target_kl: Optional[float] = None, 
        stats_window_size: int = 100, 
        tensorboard_log: Optional[str] = None, 
        policy_kwargs: Optional[Dict[str, Any]] = None, 
        verbose: int = 0, 
        seed: Optional[int] = None, 
        device: Union[th.device, str] = "auto", 
        _init_setup_model: bool = True
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
            target_kl=target_kl, 
            stats_window_size=stats_window_size, 
            tensorboard_log=tensorboard_log, 
            policy_kwargs=policy_kwargs, 
            verbose=verbose, 
            seed=seed, 
            device=device, 
            _init_setup_model=_init_setup_model
        )

        self.kl_coef_with_bc = kl_coef_with_bc
        # copy一份BC学习好的模型
        self.bc_trained_algo = bc_trained_algo
        if self.bc_trained_algo is not None:
            self.logger.info("Successfully load the bc trained model.")
            self.bc_trained_algo.policy.set_training_mode(False)

    def train(self) -> None:
        """PPO loss + BC loss
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

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                
                # -----------------code from ActorCriticPolicy evaluate_actions()------------------------------
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
                # -----------------code from ActorCriticPolicy evaluate_actions()------------------------------

                # calculate kl loss with the bc-trained model
                bc_trained_algo_distribution = self.bc_trained_algo.policy.get_distribution(rollout_data.observations)
                kl_loss_with_bc_model = kl_divergence(
                    dist_true=bc_trained_algo_distribution, 
                    dist_pred=distribution
                )
                kl_loss = kl_loss_with_bc_model.mean()
                # TODO: check kl_loss!!!!
                # exit(0)

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

                # 根据self.num_timesteps计算新的KL系数
                if isinstance(self.kl_coef_with_bc, Callable):
                    tmp_kl_coef = self.kl_coef_with_bc(self._current_progress_remaining)
                else:
                    tmp_kl_coef = self.kl_coef_with_bc

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + tmp_kl_coef * kl_loss

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
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/kl_loss_with_bc_model", kl_loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)