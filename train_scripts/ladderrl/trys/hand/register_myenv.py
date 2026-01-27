from gymnasium.envs.registration import register
import numpy as np
from hand  import mysamplez_env
# 给你的新环境起个 ID，比如 'MyRotateZ-v0'
register(
    id="MyRotateZ-v0",
    entry_point="hand.mysamplez_env:CustomRangeRotateZEnv", # 指向上面定义的那个类
    max_episode_steps=100, # 官方默认通常是 50 或 100
    kwargs={ # np.radians(45)
            "min_angle_rad":-np.pi,
            "max_angle_rad":np.pi
            } # 默认限制为 45 度
)


print("新环境注册成功！")