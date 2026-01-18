import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from functools import partial

# ==========================================
# 1. 核心计算逻辑 (ReDo 算法的简化版)
# ==========================================

@torch.no_grad()
def _get_activation(name, activations):
    """钩子函数：捕获 ReLU 后的输出"""
    def hook(layer, input, output):
        # 核心逻辑：只统计 ReLU 激活后的非零值
        activations[name] = F.tanh(output)
    return hook

@torch.no_grad()
def _get_dormant_metrics(activations, tau, skip_last=False):
    """计算休眠神经元的比例"""
    total_neurons = 0
    dormant_count = 0
    zero_count = 0 # 统计完全死掉(等于0)的神经元
    layer_items = list(activations.items())
    # 根据参数决定是否切掉最后一层
    if skip_last and len(layer_items) > 0:
        layer_items = layer_items[:-1]

    # 遍历每一层 ([:-1] 跳过最后一层，因为最后一层是输出层，不应被计入)
    for name, activation in layer_items:
        # 1. 计算每个神经元的平均活跃度 (Score)
        if activation.ndim == 4:
            # Conv2d: (Batch, C, H, W) -> 在 (Batch, H, W) 上求均值，保留 C
            score = activation.abs().mean(dim=(0, 2, 3))
        else:
            # Linear: (Batch, Neurons) -> 在 Batch 上求均值
            score = activation.abs().mean(dim=0)

        total_neurons += score.numel()

        # 2. 归一化 (Normalization)
        # 避免除以0，加上极小值
        mean_score = score.mean()
        if mean_score == 0:
            # 如果整层都是0，那么所有神经元都是休眠的
            normalized_score = torch.zeros_like(score)
        else:
            normalized_score = score / (mean_score + 1e-9)

        # 3. 统计 Tau (休眠) 和 Zero (彻底死亡)
        # 彻底死亡 (Dead)
        zero_mask = torch.isclose(normalized_score, torch.zeros_like(normalized_score))
        zero_count += zero_mask.sum().item()

        # 休眠 (Dormant)
        if tau > 0.0:
            dormant_mask = normalized_score <= tau
            dormant_count += dormant_mask.sum().item()
        else:
            dormant_count += zero_mask.sum().item()

    # 计算百分比
    zero_ratio = (zero_count / total_neurons * 100) if total_neurons > 0 else 0.0
    dormant_ratio = (dormant_count / total_neurons * 100) if total_neurons > 0 else 0.0

    return zero_ratio, dormant_ratio


class DormantNeuronMonitorCallback(BaseCallback):
    def __init__(self, 
                 check_freq: int = 5000, 
                 tau: float = 0.025, 
                 batch_size: int = 256,
                 verbose: int = 0):
        """
        :param check_freq: 每多少步检查一次
        :param tau: 休眠阈值 (默认 0.025)
        :param batch_size: 从 ReplayBuffer 采样的样本数
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.tau = tau
        self.batch_size = batch_size

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self._check_and_log()
        return True

    def _check_and_log(self):
        # 1. 确保 Buffer 有足够数据
        if self.model.replay_buffer.size() < self.batch_size:
            return

        # 2. 采样数据
        replay_data = self.model.replay_buffer.sample(self.batch_size)
        obs = replay_data.observations
        # 注意：obs 已经在正确的 device 上了

        # ===========================
        # 3. 检查 Actor
        # ===========================
        with torch.no_grad():
            # 手动提取特征 
            actor_features = self.model.policy.actor.extract_features(
                obs, 
                self.model.policy.actor.features_extractor
            )
        
        actor_zero, actor_dormant = self._calculate_dormancy(
            network=self.model.policy.actor.latent_pi, 
            inputs=(actor_features,),                   
            prefix="Actor",
            skip_last=False
        )

        # ===========================
        # 4. 检查 Critic (基于源码逻辑修复)
        # ===========================
        with torch.no_grad():
            # A. 获取当前策略下的动作 (Action)
            current_actions = self.model.policy.actor(obs)

            critic_features = self.model.policy.critic.extract_features(
                obs, 
                self.model.policy.critic.features_extractor
            )
            
            # 这一步对应源码中的 qvalue_input = th.cat([features, actions], dim=1)
            q_input = torch.cat([critic_features, current_actions], dim=1)

        # C. 获取第一个 Q 网络 (qf0)
        critic_net = self.model.policy.critic.qf0 
        
        # D. 计算休眠率
        # 注意 inputs=(q_input,) 是一个单元素元组，因为 qf0 只接受一个张量输入
        critic_zero, critic_dormant = self._calculate_dormancy(
            network=critic_net,
            inputs=(q_input,), 
            prefix="Critic",
            skip_last=True
        )

        # ===========================
        # 5. 记录日志
        # ===========================
        self.logger.record("dormant/actor_dead_ratio", actor_zero)
        self.logger.record("dormant/actor_dormant_ratio", actor_dormant)
        self.logger.record("dormant/critic_dead_ratio", critic_zero)
        self.logger.record("dormant/critic_dormant_ratio", critic_dormant)

        if self.verbose > 0:
            print(f"Step {self.num_timesteps} | Actor Dormant: {actor_dormant:.2f}% | Critic Dormant: {critic_dormant:.2f}%")


    def _calculate_dormancy(self, network, inputs, prefix, skip_last=False):
        """通用辅助函数：负责注册 Hook -> 前向传播 -> 计算指标 -> 清理 Hook"""
        activations = {}
        activation_getter = partial(_get_activation, activations=activations)
        handles = []

        # 1. 注册 Hooks (只针对 Conv 和 Linear)
        for name, module in network.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(activation_getter(name)))

        # 2. 前向传播
        with torch.no_grad():
            try:
                # 自动解包参数：Actor传入(obs)，Critic传入(obs, action)
                network(*inputs)
            except Exception as e:
                print(f"[Warning] {prefix} forward pass failed: {e}")
        
        # 3. 计算指标
        zero_ratio, dormant_ratio = _get_dormant_metrics(activations, self.tau,skip_last=skip_last)

        # 4. 清理 Hooks (必须做！)
        for handle in handles:
            handle.remove()
            
        return zero_ratio, dormant_ratio