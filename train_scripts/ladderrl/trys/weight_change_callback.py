import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from functools import partial
import copy

def calculate_weight_deviation(param_dict_1, param_dict_2):
    """
    计算两个状态字典之间的均方误差 (MSE) 之和。
    """
    loss = 0.0
    mse_loss = nn.MSELoss(reduction='sum') # 使用 sum 累加所有参数的误差
    
    with torch.no_grad():
        for (name1, p1), (name2, p2) in zip(param_dict_1.items(), param_dict_2.items()):
            if name1 != name2:
                raise ValueError(f"Parameter mismatch: {name1} vs {name2}")
            # 计算该层参数的 MSE 并累加
            loss += mse_loss(p1, p2).item()
            
    return loss

class WeightChangeMonitorCallback(BaseCallback):
    """
    监控 Actor, Critic 和 Critic Target 的权重变化率 (Weight Deviation)。
    """
    def __init__(self, check_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        
        # 用于存储上一次的权重
        self.last_actor_params = None
        self.last_critic_params = None
        self.last_critic_target_params = None

    def _on_training_start(self) -> None:
        """
        训练开始时，初始化基准权重。
        """
        self.last_actor_params = copy.deepcopy(self.model.policy.actor.state_dict())
        self.last_critic_params = copy.deepcopy(self.model.policy.critic.state_dict())
        self.last_critic_target_params = copy.deepcopy(self.model.policy.critic_target.state_dict())

    def _on_step(self) -> bool:
        """
        每一步环境交互都会调用。在这里检查频率并记录日志。
        """
        if self.n_calls % self.check_freq == 0:
            self._calculate_and_log_deviations()
            
        return True

    def _calculate_and_log_deviations(self):
        # 1. 获取当前权重
        current_actor_params = self.model.policy.actor.state_dict()
        current_critic_params = self.model.policy.critic.state_dict()
        current_critic_target_params = self.model.policy.critic_target.state_dict()

        # 2. 计算偏差 (Current vs Last)
        actor_dev = calculate_weight_deviation(current_actor_params, self.last_actor_params)
        critic_dev = calculate_weight_deviation(current_critic_params, self.last_critic_params)
        target_dev = calculate_weight_deviation(current_critic_target_params, self.last_critic_target_params)

        # 3. 记录到 Logger (Tensorboard)
        self.logger.record("train/weight_change_actor", actor_dev)
        self.logger.record("train/weight_change_critic", critic_dev)
        self.logger.record("train/weight_change_critic_target", target_dev)

        # 4. 打印 (可选)
        if self.verbose > 0:
            print(f"Step {self.num_timesteps}: Actor Dev={actor_dev:.4f}, Critic Dev={critic_dev:.4f}")

        # 5. 更新“上次权重”为“当前权重”，以便计算下一个周期的变化
        # 注意：必须使用 deepcopy，否则只是引用
        self.last_actor_params = copy.deepcopy(current_actor_params)
        self.last_critic_params = copy.deepcopy(current_critic_params)
        self.last_critic_target_params = copy.deepcopy(current_critic_target_params)