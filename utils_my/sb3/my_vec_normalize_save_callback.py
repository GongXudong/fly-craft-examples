import os
import sys
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import unwrap_vec_wrapper


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.reward_norm.algorithms.normalizers.vec_normalize_goal_conditioned_reward_scaling import VecNormalizeGoalConditionedRewardScaling


class MyVecNormalizeSaveCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:

            # if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
            #     # If model has a replay buffer, save it too
            #     replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
            #     self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
            #     if self.verbose > 1:
            #         print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            my_vec_env_normalize = unwrap_vec_wrapper(self.model.env, VecNormalizeGoalConditionedRewardScaling)

            if my_vec_env_normalize is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path(checkpoint_type="vecnormalize_", extension="pkl")
                my_vec_env_normalize.save(vec_normalize_path)  # type: ignore[union-attr]
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True