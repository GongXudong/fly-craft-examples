import numpy as np
from stable_baselines3.common.policies import BasePolicy
import torch as th


class RandomPolicy(BasePolicy):
    """
    一个输出随机动作的随机策略
    """
    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
    
    def _predict(self, observation, deterministic=False):
        """
        返回随机动作
        :param observation: 当前环境观察值(未使用)
        :param deterministic: 是否使用确定性动作(在此策略中忽略)
        :return: 随机动作
        """
        # print(f"RandomPolicy: observation shape: {observation["desired_goal"].shape}")
        tmp = [self.action_space.sample() for i in range(len(observation["desired_goal"]))]
        tmp = np.array(tmp)
        # print(f"RandomPolicy: action shape: {tmp.shape}")
        return th.as_tensor(tmp)
    
    def forward(self, *args, **kwargs):
        # 这个方法在PyTorch中是必须的，但对于随机策略我们不需要实现它
        raise NotImplementedError
    
    def _get_constructor_parameters(self):
        # 返回用于重建策略的参数
        return dict()