import numpy as np
import torch
import torch.linalg as la
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

class FeatureRankCallback(BaseCallback):
    """
    A custom callback to monitor the Feature Rank (srank) of SAC policy and critic networks.
    Based on "Understanding and Preventing Capacity Loss in Reinforcement Learning" (Lyle et al., 2022).
    """

    def __init__(
        self,
        log_freq: int = 1000,
        batch_size: int = 2048,  # Large batch size for accurate rank estimation
        epsilon: float = 0.01,   # Threshold from the paper (without normalization)
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.batch_size = batch_size
        self.epsilon = epsilon

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            self._log_feature_rank()
        return True

    def _calculate_rank(self, features: torch.Tensor) -> float:
        """
        Computes the effective rank of the feature matrix.
        NOTE: We do NOT normalize by sqrt(N) to align with Lyle et al. (2022) & InFeR-DQN.
        """
        # SVD only needs singular values (S), so svdvals is faster than svd
        S = torch.linalg.svdvals(features)
        
        # Count singular values greater than epsilon
        rank = torch.sum(S > self.epsilon).item()
        return rank

    def _get_features(self, obs: torch.Tensor):
        """
        Extract features from Actor and Critic.
        SB3 SAC separates feature extraction and the final heads.
        """
        # We need to access the internal networks of the policy
        # SB3 SAC Policy structure:
        #   - actor: features_extractor -> latent_pi -> mu/log_std
        #   - critic: features_extractor -> qf1/qf2
        
        with torch.no_grad():
            # 1. Actor Features
            # Extract features (CNN/MLP output)
            actor_features = self.model.policy.actor.features_extractor(obs)
            # Pass through the MLP part of the actor (latent_pi)
            # Note: SB3 Actor's .forward() returns actions, we want the latent representation.
            # We manually pass it through the 'latent_pi' network (the MLP body).
            actor_latent = self.model.policy.actor.latent_pi(actor_features)

            # 2. Critic Features
            # Critics in SB3 share the feature extractor with the actor (usually) or have their own.
            # But the 'critic' object has its own qf1 and qf2 networks taking (obs, action).
            # However, for rank calculation, we usually just look at the representation of Observation.
            # SB3 Critic standard implementation concatenates obs+action. 
            # To measure "State Representation Rank", we can look at the output of the 'features_extractor' 
            # if it's shared, or the first layers of the Q-net.
            
            # Simplified approach for SB3's standard SAC (MlpPolicy/CnnPolicy):
            # The 'critic' usually takes cat(obs, action).
            # Measuring rank on (obs, action) is tricky because action depends on policy.
            # Lyle et al. usually measure the rank of the representation *before* the linear head.
            
            # Let's extract the Q-network's internal representation.
            # In SB3, model.policy.critic.qf1 is an MLP. We want the output of the 2nd to last layer.
            # We simulate a forward pass for Q1 using random actions (or policy actions)
            
            # Generate actions for the current observations to feed the critic
            actions, _ = self.model.policy.actor(obs) 
            q_input = torch.cat([obs, actions], dim=1)
            
            # SB3 doesn't easily expose the "embedding" of the Q-network because it's a `nn.Sequential`.
            # We can iterate through the layers of qf1 up to the last one.
            qf1_net = self.model.policy.critic.qf1
            
            q1_latent = q_input
            # Iterate over all layers except the last Linear layer
            for layer in qf1_net[:-1]:
                q1_latent = layer(q1_latent)
                
            # Now q1_latent is the input to the final Q-value head.
        
        return actor_latent, q1_latent

    def _log_feature_rank(self):
        # 1. Sample data from ReplayBuffer
        # SB3 ReplayBuffer.sample() returns a dict-like object
        if self.model.replay_buffer.size() < self.batch_size:
            return # Not enough data yet

        replay_data = self.model.replay_buffer.sample(
            self.batch_size, 
            env=self.model._vec_normalize_env
        )
        
        # Move to GPU if needed
        obs = replay_data.observations

        # 2. Extract features
        actor_embedding, critic_embedding = self._get_features(obs)

        # 3. Calculate Rank
        actor_rank = self._calculate_rank(actor_embedding)
        critic_rank = self._calculate_rank(critic_embedding)

        # 4. Log
        self.logger.record("rank/actor_rank", actor_rank)
        self.logger.record("rank/critic_rank", critic_rank)
        
        # Optional: Log normalized rank (Rank / Max_Possible_Rank)
        # max_rank = min(batch_size, feature_dim)
        # self.logger.record("rank/actor_rank_norm", actor_rank / actor_embedding.shape[1])

        if self.verbose > 0:
            print(f"Step {self.num_timesteps}: Actor Rank={actor_rank}, Critic Rank={critic_rank}")