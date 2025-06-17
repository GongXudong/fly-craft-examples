import unittest
import numpy as np
from pathlib import Path
import sys

import flycraft
from stable_baselines3.sac import SAC


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.sb3.my_rollout import rollout


class RolloutTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.env_config_path = PROJECT_ROOT_DIR / "configs/env/D2D/env_config_for_sac_medium_b_1.json"
        self.algo_checkpoint_path = PROJECT_ROOT_DIR / "checkpoints/D2D/goal_sapce/evaluate_medium/easy_to_medium/b_05/2e6/easy2medium_buffer_size_2e6/sac_10hz_128_128_b_05_easy_to_medium_1e6steps_seed_1_singleRL/best_model.zip"

        self.algo = SAC.load(
            path=self.algo_checkpoint_path,

        )
    
    def test_save_all_trajs(self):
        rollout_transition_num = 2000
        obss, actions, next_obss, rewards, dones, infos = rollout(self.algo, self.env_config_path, rollout_transition_num=rollout_transition_num, save_success_traj=False)

        self.assertEqual(rollout_transition_num, len(obss), "")
        
    def test_save_success_trajs(self):
        rollout_transition_num = 500
        obss, actions, next_obss, rewards, dones, infos = rollout(self.algo, self.env_config_path, rollout_transition_num=rollout_transition_num, save_success_traj=True)

        self.assertEqual(rollout_transition_num, len(obss), "")

    def test_transitions(self):
        rollout_transition_num = 1000
        obss, actions, next_obss, rewards, dones, infos = rollout(self.algo, self.env_config_path, rollout_transition_num=rollout_transition_num, save_success_traj=False)

        check_transition_num = 10
        print(obss[:check_transition_num])
        print(actions[:check_transition_num])
        print(next_obss[:check_transition_num])
        print(rewards[:check_transition_num])
        print(dones[:check_transition_num])


if __name__ == "__main__":
    unittest.main()