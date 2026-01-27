import numpy as np
import gymnasium as gym
from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block import MujocoHandBlockEnv
from gymnasium_robotics.utils import rotations
from gymnasium.envs.registration import register

class CustomRangeRotateZEnv(MujocoHandBlockEnv):
    def __init__(self, min_angle_rad=-np.pi, max_angle_rad=np.pi, **kwargs):
        """
        初始化函数，支持传入自定义的角度范围。
        
        Args:
            min_angle_rad (float): 最小角度（弧度），例如 -0.5
            max_angle_rad (float): 最大角度（弧度），例如 0.5
            **kwargs: 其他传递给 MujocoHandBlockEnv 的参数
        """
        # 保存用户定义的范围
        self.min_angle = min_angle_rad
        self.max_angle = max_angle_rad
        
        # 强制设置 env 核心参数：忽略位置，只转 Z 轴
        # 这样父类会帮我们处理很多基础设置
        super().__init__(
            target_position="ignore",
            target_rotation="z",
            reward_type="dense",
            **kwargs
        )

    def _sample_goal(self):
        # --------------------------------------------------------
        # 1. "借力"：先调用父类方法
        # --------------------------------------------------------
        # 父类会处理位置（position）和它默认的旋转（虽然那个旋转范围不对，我们稍后覆盖它）
        # 这样我们就不用手动去写 target_position 的那堆逻辑了
        goal = super()._sample_goal().copy()

        # --------------------------------------------------------
        # 2. "覆盖"：用你的范围生成新的 Z 轴旋转
        # --------------------------------------------------------
        # 使用你存下的 min_angle 和 max_angle
        angle = self.np_random.uniform(self.min_angle, self.max_angle)
        
        # 生成对应的四元数 (Z轴旋转)
        axis = np.array([0.0, 0.0, 1.0])
        # 注意：这里需要引入 gymnasium_robotics.utils.rotations 里的函数
        # 或者直接手写公式: w=cos(a/2), z=sin(a/2)
        new_target_quat = quat_from_angle_and_axis(angle, axis)
        
        # --------------------------------------------------------
        # 3. "注入"：把新生成的旋转塞回 goal 数组
        # --------------------------------------------------------
        # goal 的结构是 [x, y, z, w, qx, qy, qz]
        # 我们替换后 4 位
        goal[3:] = new_target_quat
        
        return goal

def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis])
    quat /= np.linalg.norm(quat)
    return quat