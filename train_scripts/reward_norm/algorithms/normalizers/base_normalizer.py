from abc import ABC, abstractmethod
import numpy as np


class BaseNormalizer(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def get_cumulative_reward(self, desired_goal: np.ndarray) -> float:
        pass

    def normalize_reward(self, reward: float, desired_goal: np.ndarray) -> float:
        return reward / self.get_cumulative_reward(desired_goal=desired_goal)