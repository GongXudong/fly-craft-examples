
from typing import Union
import numpy as np
from collections import namedtuple
from pathlib import Path
import logging
import sys
import flycraft

from flycraft.utils_common.geometry_utils import angle_of_2_3d_vectors


def get_reward(next_state,desired_goal) -> float:

    plane_current_velocity_vector = [
        next_state["v"] * np.cos(np.deg2rad(next_state["mu"])) * np.sin(np.deg2rad(next_state["chi"])), 
        next_state["v"] * np.cos(np.deg2rad(next_state["mu"])) * np.cos(np.deg2rad(next_state["chi"])), 
        next_state["v"] * np.sin(np.deg2rad(next_state["mu"]))
    ]
    target_velocity_vector = [
        desired_goal["goal_v"] * np.cos(np.deg2rad(desired_goal["goal_mu"])) * np.sin(np.deg2rad(desired_goal["goal_chi"])), 
        desired_goal["goal_v"] * np.cos(np.deg2rad(desired_goal["goal_mu"])) * np.cos(np.deg2rad(desired_goal["goal_chi"])),
        desired_goal["goal_v"] * np.sin(np.deg2rad(desired_goal["goal_mu"])),
    ]
    angle = angle_of_2_3d_vectors(plane_current_velocity_vector, target_velocity_vector)
    angle_base_reward = - np.power(angle / 180.0, 0.5)

    # 奖励公式参考论文“Reinforcement learning for UAV attitude control”的4.3节
    velocity_error = np.abs(desired_goal["goal_v"] - next_state["v"])
    cliped_velocity_error = np.clip(velocity_error / 100, a_min=0., a_max=1.)
    velocity_base_reward = - np.power(cliped_velocity_error, 0.5)

    return 0.5 *  angle_base_reward + (1. - 0.5) * velocity_base_reward


# achiev_goal = [
#   [0.2, 0.5, 0.5],
#   [0.2013272, 0.49991116, 0.4999833],
#   [0.20301437, 0.50083065, 0.5003389]              
#                 ]





next_state ={ 
    "v": 201.32721,#0.2013272,
    "mu": -0.01599294,#0.49991116,
    "chi": -0.0060072355,#0.4999833
    #[201.32721, -0.01599294, -0.0060072355]
    #[0.51300347, 0.51674414, 0.5005493, 0.2013272, 0.49991116, 0.4999833, 0.54618555, 0.2499997]
    # [2.8358245, 2.6812103, 0.12032899, 200.89601, -0.0007578539, -0.008174696, 18.415014, 4999.9995]
} 

desired_goal = {
    "goal_v" : 215.1554283620066, #0.21515542,
    "goal_mu" : 24.294495569218697, #0.6349694,
    "goal_chi": 41.601458243586194, #0.6155596
   #[0.21515542, 0.6349694, 0.6155596]
   #[215.1554283620066, 24.294495569218697, 41.601458243586194]
}

reward = get_reward(next_state,desired_goal)
print(f"reward = {reward}")