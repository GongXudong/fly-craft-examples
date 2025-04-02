import unittest
import sys
from pathlib import Path
import numpy as np

import flycraft
from stable_baselines3.common.env_util import make_vec_env

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from train_scripts.reward_norm.algorithms.normalizers.vec_reward_minmax_normalizer import VecRewardMinMaxNormalize


class VecRewardMinMaxNormalizeTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.normalizer = VecRewardMinMaxNormalize(
            venv=make_vec_env(env_id="FlyCraft-v0", n_envs=4),
            radius=1.,
            default_max_return=-100.,
            key_names=["dg_v", "dg_mu", "dg_chi"],
            value_name="cumulative_reward",
            is_success_name="is_success",
        )
        
    
    def test_update_data_1(self):
        tmp_file = PROJECT_ROOT_DIR / "train_scripts/reward_norm/algorithms/normalizers/test/end2end_medium_success_model_128_128_2e8steps_lambda_1e-3_1.csv"
        self.normalizer.update_data_from_csv_file(tmp_file)
        print(len(self.normalizer.data))

    def test_update_data_2(self):
        tmp_file = PROJECT_ROOT_DIR / "train_scripts/reward_norm/algorithms/normalizers/test/end2end_medium_success_model_128_128_2e8steps_lambda_1e-3_1.csv"
        self.normalizer.update_data_from_csv_file(tmp_file)
        print(len(self.normalizer.data))

        tmp_file = PROJECT_ROOT_DIR / "train_scripts/reward_norm/algorithms/normalizers/test/end2end_medium_success_model_128_128_2e8steps_lambda_1e-3_1.csv"
        self.normalizer.update_data_from_csv_file(tmp_file)
        print(len(self.normalizer.data))

    def test_get_neighbours(self):
        print("In test_get_neighbours:")
        tmp_file = PROJECT_ROOT_DIR / "train_scripts/reward_norm/algorithms/normalizers/test/end2end_medium_success_model_128_128_2e8steps_lambda_1e-3_1.csv"
        self.normalizer.update_data_from_csv_file(tmp_file)
        
        self.normalizer.fit_data()

        points = np.array([
            [246.9183948144422, -20.26097251286599, 42.075240826812586],
            [229.61133044600447, 5.68530026045206, -10.970634372486728],
            [200, -10, 10],
        ])
        distances, indices = self.normalizer.get_neighbours(points)
        print("-----------------------")
        for pt, dst, ind in zip(points, distances, indices):
            if ind.size > 0:
                print(f"For point {pt}, find: {ind}, {self.normalizer.data.iloc[ind][["dg_v", "dg_mu", "dg_chi"]]}, distances: {dst}, values: {self.normalizer.data.iloc[ind]["cumulative_reward"].to_numpy()}, min_value: {self.normalizer.data.iloc[ind]["cumulative_reward"].min()}")
            else:
                print(f"For point {pt}, can't find nearest points within the specific radius!")

if __name__ == "__main__":
    unittest.main()
