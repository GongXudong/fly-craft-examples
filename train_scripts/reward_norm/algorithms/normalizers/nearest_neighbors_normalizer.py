import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT_DIR = Path(__file__).absolute().parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.reward_norm.algorithms.normalizers.base_normalizer import BaseNormalizer

class NearestNeighborsNormalizer(BaseNormalizer):

    def __init__(self, n_neighbors: int):
        super().__init__()
        self.data: pd.DataFrame = pd.DataFrame(data={
            "dg_v": [],
            "dg_mu": [],
            "dg_chi": [],
            "is_success": [],
            "termination": [],
            "cumulative_reward": [],
            "episode_length": [],
        })

        self.n_neighbors = 10
        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors)
    
    def load_data(self, file_path: Path):
        df = pd.read_csv(file_path)
        self.data = pd.concat([self.data, df])

        X_train = self.data[["dg_v", "dg_mu", "dg_chi"]].values
        self.nbrs.fix(X_train)

    def get_cumulative_reward(self, desired_goal: np.ndarray):
        if len(desired_goal.shape) == 1:
            distances, indices = self.nbrs.kneighbors(desired_goal.reshape((1, -1)))
            print(self.data.iloc[indices[0]])
            return self.data["cumulative_reward"].iloc[indices[0]].max()
