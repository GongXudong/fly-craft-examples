import gymnasium as gym
import gymnasium_robotics

# 例子：创建一个 低频控制 (40 substeps) 但 快速收敛 的环境



# env_kwargs={
#                 "max_episode_steps":50,
#                 "n_substeps":20
#             }


env = gym.make(
    "HandManipulateBlockRotateZDense-v1", 
    max_episode_steps=120, 
    n_substeps=60  
        
)

print(f"当前环境 FrameSkip: {env.unwrapped.n_substeps}")
print(f"当前环境 MaxSteps: {env.spec.max_episode_steps}")



# register(
#     id=f"HandManipulateBlockRotateZ{suffix}-v0",
#     entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
#     kwargs=_merge(
#         {
#             "target_position": "ignore",
#             "target_rotation": "z",
#         },
#         kwargs,
#     ),
#     max_episode_steps=100,
# )