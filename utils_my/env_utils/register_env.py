from pathlib import Path
import sys
from gymnasium.envs.registration import register

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_my.env_utils.my_reach_env import MyPandaReachEnv
from utils_my.env_utils.my_push_env import MyPandaPushEnv
from utils_my.env_utils.my_slide_env import MyPandaSlideEnv
from utils_my.env_utils.my_pick_and_place_env import MyPandaPickAndPlaceEnv
from utils_my.env_utils.my_stack_env import MyPandaStackEnv


def register_my_reach_env(env_id: str, reward_type: str="dense", control_type: str="joints", goal_range: float=0.3, distance_threshold: float=0.01, max_episode_steps: int=50):
    register(
        id=env_id,
        entry_point=MyPandaReachEnv,
        # kwargs={"reward_type": "sparse", "control_type": "ee", "goal_range": goal_range, "distance_threshold": distance_threshold},
        kwargs={
            "reward_type": reward_type, 
            "control_type": control_type, 
            "goal_range": goal_range, 
            "distance_threshold": distance_threshold
        },
        max_episode_steps=max_episode_steps,
    )

def register_my_slide_env(env_id: str, reward_type: str="dense", control_type: str="joints", goal_xy_range: float=0.3, goal_x_offset: float=0.4, obj_xy_range: float=0.3, distance_threshold: float=0.01, max_episode_steps: int=50):
    register(
        id=env_id,
        entry_point=MyPandaSlideEnv,
        kwargs={
            "reward_type": reward_type, 
            "control_type": control_type, 
            "goal_xy_range": goal_xy_range, 
            "goal_x_offset": goal_x_offset,
            "obj_xy_range": obj_xy_range, 
            "distance_threshold": distance_threshold
        },
        max_episode_steps=max_episode_steps,
    )

def register_my_push_env(env_id: str, reward_type: str="dense", control_type: str="joints", goal_xy_range: float=0.3, obj_xy_range: float=0.3, distance_threshold: float=0.01, max_episode_steps: int=50):
    register(
        id=env_id,
        entry_point=MyPandaPushEnv,
        kwargs={
            "reward_type": reward_type, 
            "control_type": control_type, 
            "goal_xy_range": goal_xy_range, 
            "obj_xy_range": obj_xy_range, 
            "distance_threshold": distance_threshold
        },
        max_episode_steps=max_episode_steps,
    )

def register_my_pick_and_place_env(env_id: str, reward_type: str="dense", control_type: str="joints", goal_xy_range: float=0.3, goal_z_range: float=0.2, obj_xy_range: float=0.3, distance_threshold: float=0.01, max_episode_steps: int=50):
    register(
        id=env_id,
        entry_point=MyPandaPickAndPlaceEnv,
        kwargs={
            "reward_type": reward_type, 
            "control_type": control_type, 
            "goal_xy_range": goal_xy_range,
            "goal_z_range": goal_z_range, 
            "obj_xy_range": obj_xy_range, 
            "distance_threshold": distance_threshold
        },
        max_episode_steps=max_episode_steps,
    )

def register_my_stack_env(env_id: str, reward_type: str="dense", control_type: str="joints", goal_xy_range: float=0.3, obj_xy_range: float=0.3, distance_threshold: float=0.01, max_episode_steps: int=50):
    register(
        id=env_id,
        entry_point=MyPandaStackEnv,
        kwargs={
            "reward_type": reward_type, 
            "control_type": control_type, 
            "goal_xy_range": goal_xy_range, 
            "obj_xy_range": obj_xy_range, 
            "distance_threshold": distance_threshold
        },
        max_episode_steps=max_episode_steps,
    )

def register_all_with_default_dense_params():
    # reach
    register_my_reach_env(env_id="my-reach", reward_type="dense", control_type="joints", goal_range=0.5, distance_threshold=0.01, max_episode_steps=50)

    # push
    register_my_push_env(env_id="my-push-dense", reward_type="dense", control_type="joints", goal_xy_range=0.5, obj_xy_range=0.0, distance_threshold=0.05, max_episode_steps=50)

    # slide
    register_my_slide_env(env_id="my-slide-dense", reward_type="dense", control_type="joints", goal_xy_range=0.5, goal_x_offset=0.4, obj_xy_range=0.0, distance_threshold=0.05, max_episode_steps=50)

    # pick and place
    register_my_pick_and_place_env(env_id="my-pick-and-place-dense", reward_type="dense", control_type="ee", goal_xy_range=0.3, goal_z_range=0.2, obj_xy_range=0.0, distance_threshold=0.05, max_episode_steps=50)

    # stack
    register_my_stack_env(env_id="my-stack-dense", reward_type="dense", control_type="joints", goal_xy_range=0.3, obj_xy_range=0.0, distance_threshold=0.1, max_episode_steps=100)

def register_all_with_default_sparse_params():
    # reach
    register_my_reach_env(env_id="my-reach-sparse", reward_type="sparse", control_type="joints", goal_range=0.5, distance_threshold=0.01, max_episode_steps=50)

    # push
    register_my_push_env(env_id="my-push-sparse", reward_type="sparse", control_type="joints", goal_xy_range=0.5, obj_xy_range=0.0, distance_threshold=0.05, max_episode_steps=50)

    # slide
    register_my_slide_env(env_id="my-slide-sparse", reward_type="sparse", control_type="joints", goal_xy_range=0.5, goal_x_offset=0.4, obj_xy_range=0.0, distance_threshold=0.05, max_episode_steps=50)

    # pick and place
    register_my_pick_and_place_env(env_id="my-pick-and-place-sparse", reward_type="sparse", control_type="joints", goal_xy_range=0.5, goal_z_range=0.2, obj_xy_range=0.0, distance_threshold=0.05, max_episode_steps=50)

    # stack
    register_my_stack_env(env_id="my-stack-sparse", reward_type="sparse", control_type="joints", goal_xy_range=0.5, obj_xy_range=0.0, distance_threshold=0.1, max_episode_steps=100)
