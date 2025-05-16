import unittest
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.reward_norm.algorithms.normalizers.nearest_neighbors_normalizer import NearestNeighborsNormalizer


class NearestNeighborsNormalizerTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.nearest_neighbors_normalizer = NearestNeighborsNormalizer(
            n_neighbors=10,
        )
        
    
    def test_sample_goal_noise_1(self):
        print("In test sample goal noise.............")
        for i in range(10):
            obs, _ = self.env_used_in_attacker.reset()
            tmp_noise = self.ppo_algo.sample_a_goal_noise(scaled_desired_goal=obs["desired_goal"])
            print(f"iter {i}, {tmp_noise}")
            self.assertTrue(th.all(th.less_equal(tmp_noise, self.ppo_algo.ppo_ga_attacker.noise_max)))
            self.assertTrue(th.all(th.less_equal(self.ppo_algo.ppo_ga_attacker.noise_min, tmp_noise)))



if __name__ == "__main__":
    unittest.main()
