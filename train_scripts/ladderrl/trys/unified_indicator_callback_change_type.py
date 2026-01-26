import copy
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback

# ==========================================
# 1. 辅助函数 (保持不变)
# ==========================================

def calculate_weight_deviation(param_dict_1, param_dict_2):
    """计算两个状态字典之间的 MSE 偏差"""
    loss = 0.0
    mse_loss = nn.MSELoss(reduction='sum')
    with torch.no_grad():
        for (name1, p1), (name2, p2) in zip(param_dict_1.items(), param_dict_2.items()):
            if name1 != name2: 
                continue 
            loss += mse_loss(p1, p2).item()
    return loss

def _get_activation(name, activations,non_linearity):
    """Hook: 捕获 ReLU 后的输出"""
    
    def hook(layer, input, output):
        activations[name] = non_linearity(output)
    return hook

def _get_dormant_metrics(activations, tau, skip_last=False):
    """计算休眠神经元比例"""
    total_neurons = 0
    dormant_count = 0
    zero_count = 0
    
    layer_items = list(activations.items())
    if skip_last and len(layer_items) > 0:
        layer_items = layer_items[:-1]

    for name, activation in layer_items:
        if activation.ndim == 4: # Conv2d
            score = activation.abs().mean(dim=(0, 2, 3))
        else: # Linear
            score = activation.abs().mean(dim=0)

        total_neurons += score.numel()
        mean_score = score.mean()
        
        if mean_score == 0:
            normalized_score = torch.zeros_like(score)
        else:
            normalized_score = score / (mean_score + 1e-9)

        zero_mask = torch.isclose(normalized_score, torch.zeros_like(normalized_score))
        zero_count += zero_mask.sum().item()

        if tau > 0.0:
            dormant_mask = normalized_score <= tau
            dormant_count += dormant_mask.sum().item()
        else:
            dormant_count += zero_mask.sum().item()

    zero_ratio = (zero_count / total_neurons * 100) if total_neurons > 0 else 0.0
    dormant_ratio = (dormant_count / total_neurons * 100) if total_neurons > 0 else 0.0
    return zero_ratio, dormant_ratio
def _compute_ntk_metrics_from_grads(gradients_list, delta=0.01):
    """
    根据梯度列表计算 NTK 的多项指标：
    1. Approximate Rank (解释性秩)
    2. Diagonal Sum (梯度能量/幅度)
    3. Off-Diagonal Abs Sum (梯度干扰/相关性)
    """
    if not gradients_list:
        return 0.0, 0.0, 0.0
    
    # 1. 构建矩阵 G (num_batches, num_params)
    G = torch.stack(gradients_list)
    
    # 2. 构建 NTK Gram 矩阵 (num_batches, num_batches)
    # N[i, j] = g_i * g_j
    ntk_matrix = torch.matmul(G, G.T)
    
    # ==========================
    # 指标 A & B: 对角线与非对角线统计
    # ==========================
    # 获取对角线元素 (Abs)
    diag_vals = torch.diagonal(ntk_matrix).abs()
    diag_sum = diag_vals.sum().item()
    
    # 获取所有元素的绝对值之和
    total_abs_sum = ntk_matrix.abs().sum().item()
    
    # 非对角线和 = 总和 - 对角线和
    off_diag_sum = total_abs_sum - diag_sum

    # ==========================
    # 指标 C: Approximate Rank
    # ==========================
    eigenvalues = torch.linalg.eigvalsh(ntk_matrix).abs()
    vals_sorted, _ = torch.sort(eigenvalues, descending=True)
    
    total_energy = vals_sorted.sum()
    if total_energy < 1e-6:
        rank = 0.0
    else:
        target_energy = (1.0 - delta) * total_energy
        current_energy = 0.0
        k = 0
        for val in vals_sorted:
            current_energy += val
            k += 1
            if current_energy >= target_energy:
                break
        rank = float(k)
            
    return rank, diag_sum, off_diag_sum

# ==========================================
# 2. 统一监控 Callback (修改版)
# ==========================================

class UnifiedNetworkMonitorCallback_TYPE(BaseCallback):
    def __init__(
        self,
        check_freq: int = 10000,
        batch_size: int = 2048, 
        dormant_tau: float = 0.3,
        verbose: int = 0,
        non_linearity: str = "tanh",
        ntk_num_batches: int = 64,  # 构建 m x m 矩阵的 m
        ntk_batch_size: int = 32,   # 计算梯度时的小 batch
        ntk_delta: float = 0.01     # 能量阈值 (解释 99% 的方差)
       
        
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.batch_size = batch_size
        self.dormant_tau = dormant_tau
        self.non_linearity = non_linearity
        
        self.last_actor_params = None
        self.last_critic_params = None
        self.last_critic_target_params = None

        # NTK 参数
        self.ntk_num_batches = ntk_num_batches
        self.ntk_batch_size = ntk_batch_size
        self.ntk_delta = ntk_delta

    def _on_training_start(self) -> None:
        self.last_actor_params = copy.deepcopy(self.model.policy.actor.state_dict())
        self.last_critic_params = copy.deepcopy(self.model.policy.critic.state_dict())
        self.last_critic_target_params = copy.deepcopy(self.model.policy.critic_target.state_dict())

    def _on_step(self) -> bool:
        """主循环"""
        if self.n_calls % self.check_freq == 0:
            # 1. 统一采样
            if self.model.replay_buffer.size() < self.batch_size:
                return True
            
            replay_data = self.model.replay_buffer.sample(
                self.batch_size, 
                env=self.model._vec_normalize_env
            )
            obs = replay_data.observations
            
            # 2. 执行监控并获取返回值 (修改点：接收返回值)
            w_actor, w_critic = self._monitor_weights()
            d_actor, d_critic, z_actor, z_critic = self._monitor_dormant(obs)
            r_actor, r_critic = self._monitor_rank(obs)
            
            # 3. [新增] 执行 NTK 监控
            #ntk_r_actor, ntk_r_critic = self._monitor_ntk_approx_rank()


            # 3. 打印详细信息 (修改点：详细格式化输出)
            if self.verbose > 0:
                print("-" * 60)
                print(f"[UnifiedMonitor] Step {self.num_timesteps}")
                print(f"  > Rank    | Actor: {r_actor:<4} | Critic: {r_critic:<4}")
                print(f"  > Dormant | Actor: {d_actor:5.2f}% | Critic: {d_critic:5.2f}% (Dead: A={z_actor:.1f}%, C={z_critic:.1f}%)")
                print(f"  > Weights | Actor: {w_actor:.4f} | Critic: {w_critic:.4f}")
                #print(f"  > NTK Appx Rank| Actor: {ntk_r_actor:<4}  | Critic: {ntk_r_critic:<4} (Thresh: {1-self.ntk_delta:.2f})")
                
                print("-" * 60)
                
        return True

    # ------------------------------------------
    # 模块 A: 权重变化监控 (修改：添加 return)
    # ------------------------------------------
    def _monitor_weights(self):
        current_actor = self.model.policy.actor.state_dict()
        current_critic = self.model.policy.critic.state_dict()
        current_target = self.model.policy.critic_target.state_dict()

        actor_dev = calculate_weight_deviation(current_actor, self.last_actor_params)
        critic_dev = calculate_weight_deviation(current_critic, self.last_critic_params)
        target_dev = calculate_weight_deviation(current_target, self.last_critic_target_params)

        self.logger.record("monitor/weight_change_actor", actor_dev)
        self.logger.record("monitor/weight_change_critic", critic_dev)
        self.logger.record("monitor/weight_change_target", target_dev)

        # 更新快照
        self.last_actor_params = copy.deepcopy(current_actor)
        self.last_critic_params = copy.deepcopy(current_critic)
        self.last_critic_target_params = copy.deepcopy(current_target)
        
        # 返回用于打印的值
        return actor_dev, critic_dev

    # ------------------------------------------
    # 模块 B: 休眠神经元监控 (修改：添加 return)
    # ------------------------------------------
    def _monitor_dormant(self, obs: torch.Tensor):
        # 1. Actor
        with torch.no_grad():
            actor_features = self.model.policy.actor.extract_features(
                obs, self.model.policy.actor.features_extractor
            )
            if isinstance(actor_features, torch.Tensor):
                actor_features = actor_features.float()
        
        a_zero, a_dormant = self._calculate_dormancy_forward(
            network=self.model.policy.actor.latent_pi,
            inputs=(actor_features,),
            skip_last=False,
            non_linearity=self.non_linearity
        )

        # 2. Critic
        with torch.no_grad():
            current_actions = self.model.policy.actor(obs) 
            critic_features = self.model.policy.critic.extract_features(
                obs, self.model.policy.critic.features_extractor
            )
            q_input = torch.cat([critic_features, current_actions], dim=1)
            q_input = q_input.float()
        
        c_zero, c_dormant = self._calculate_dormancy_forward(
            network=self.model.policy.critic.qf0,
            inputs=(q_input,),
            skip_last=True,
            non_linearity=self.non_linearity
        )

        self.logger.record("monitor/dormant_actor_ratio", a_dormant)
        self.logger.record("monitor/dormant_critic_ratio", c_dormant)
        self.logger.record("monitor/dead_actor_ratio", a_zero)
        self.logger.record("monitor/dead_critic_ratio", c_zero)
        
        # 返回用于打印的值
        return a_dormant, c_dormant, a_zero, c_zero

    def _calculate_dormancy_forward(self, network, inputs, skip_last,non_linearity):
        activations = {}
        non_linearity = torch.tanh if non_linearity=="tanh" else torch.relu
        activation_getter = partial(_get_activation, activations=activations,non_linearity=non_linearity)
        handles = []
        for name, module in network.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(activation_getter(name)))
        with torch.no_grad():
            try:
                network(*inputs)
            except Exception as e:
                print(f"Warning: Dormant check failed: {e}")
        zero, dormant = _get_dormant_metrics(activations, self.dormant_tau, skip_last=skip_last)
        for handle in handles:
            handle.remove()
        return zero, dormant

    # ------------------------------------------
    # 模块 C: 特征秩监控 (修改：添加 return)
    # ------------------------------------------
    def _monitor_rank(self, obs: torch.Tensor):
        with torch.no_grad():
            # Actor
            actor_features = self.model.policy.actor.features_extractor(obs)
            if isinstance(actor_features, torch.Tensor):
                            actor_features = actor_features.float()

            actor_latent = self.model.policy.actor.latent_pi(actor_features)

            # Critic
            actions = self.model.policy.actor(obs)
            critic_features = self.model.policy.critic.extract_features(
                obs, self.model.policy.critic.features_extractor
            )
            q_input = torch.cat([critic_features, actions], dim=1)
            q_input = q_input.float()

            q1_latent = q_input
            qf1_net = self.model.policy.critic.qf1
            for layer in qf1_net[:-1]:
                q1_latent = layer(q1_latent)
        
        a_rank = self._calculate_rank_svd(actor_latent)
        c_rank = self._calculate_rank_svd(q1_latent)

        self.logger.record("monitor/rank_actor", a_rank)
        self.logger.record("monitor/rank_critic", c_rank)
        
        # 返回用于打印的值
        return a_rank, c_rank

    # def _calculate_rank_svd(self, features: torch.Tensor) -> float:
    #     S = torch.linalg.svdvals(features)
    #     rank = torch.sum(S > self.rank_epsilon).item()
    #     return rank
    

    def _calculate_rank_svd(self, features: torch.Tensor) -> float:
        if features.numel()==0:
            return 0.0
        features = features.float()
        
        # qu zhong xin hua 
        features = features - features.mean(dim=0,keepdim=True)


        # 1. 计算奇异值
        S = torch.linalg.svdvals(features)
        
        # 2. 获取最大奇异值
        max_s = S[0]
        
        # --- 这里的 1e-6 只是为了防止除以 0，不是调节阈值 ---
        # 如果 max_s 非常接近 0，说明整个网络没有任何输出，Rank 应为 0
        if max_s < 1e-6:
            return 0.0
            
        # 3. 计算 Stable Rank
        # 公式: sum(S^2) / max(S)^2
        numerator = torch.sum(S ** 2)
        denominator = max_s ** 2
        
        stable_rank = (numerator / denominator).item()
        
        return stable_rank
        
# ------------------------------------------
    # 模块 D: NTK 近似秩监控 (适配 MultiInputPolicy/HER)
    # ------------------------------------------
    def _monitor_ntk_approx_rank(self):
        """
        计算 Actor 和 Critic 的 Mini-batch NTK Approximate Rank。
        完美支持 MultiInputPolicy (Dict Observation) 和 HER。
        """
        total_samples = self.ntk_num_batches * self.ntk_batch_size
        
        # 1. 采样数据
        # sample() 返回的 observations 对于 MultiInputPolicy 是一个 Dict
        samples = self.model.replay_buffer.sample(total_samples, env=self.model._vec_normalize_env)
        
        # -------------------------------------------------
        # [核心修复 1] 处理观测数据的 Device 移动
        # -------------------------------------------------
        if isinstance(samples.observations, dict):
            # 如果是字典 (MultiInputPolicy / HER)，遍历每个 key 移动 tensor
            all_obs = {k: v.to(self.model.device) for k, v in samples.observations.items()}
        else:
            # 如果是普通 Tensor (MlpPolicy / CnnPolicy)
            all_obs = samples.observations.to(self.model.device)

        all_acts = samples.actions.to(self.model.device)
        
        actor = self.model.policy.actor
        critic = self.model.policy.critic.qf0 

        actor_grads = []
        critic_grads = []
        
        # 2. 循环计算 Mini-batch 梯度
        for i in range(self.ntk_num_batches):
            start = i * self.ntk_batch_size
            end = (i + 1) * self.ntk_batch_size
            
            # -------------------------------------------------
            # [核心修复 2] 处理观测数据的切片 (Slicing)
            # -------------------------------------------------
            if isinstance(all_obs, dict):
                # 对字典里的每个 Tensor 分别切片
                obs_batch = {k: v[start:end] for k, v in all_obs.items()}
            else:
                # 普通切片
                obs_batch = all_obs[start:end]
            
            act_batch = all_acts[start:end]

            # === Part A: Critic NTK ===
            # SB3 的 extract_features 会自动处理 Dict 输入并拼接
            c_features = self.model.policy.critic.extract_features(obs_batch, self.model.policy.critic.features_extractor)
            q_input = torch.cat([c_features, act_batch], dim=1)
            q_values = critic(q_input)
            
            c_loss = q_values.sum()
            
            self.model.policy.critic.zero_grad()
            c_loss.backward()
            
            c_g = []
            for param in critic.parameters():
                if param.grad is not None:
                    c_g.append(param.grad.view(-1))
            if c_g:
                critic_grads.append(torch.cat(c_g))

            # === Part B: Actor NTK ===
            # Actor 同样支持 Dict 输入
            if hasattr(actor, "get_action_dist_params"):
                # obs_batch 是 dict，直接传入即可
                dist_params = actor.get_action_dist_params(obs_batch)
                if isinstance(dist_params, tuple):
                    mean_actions = dist_params[0] 
                else:
                    mean_actions = dist_params.distribution.loc
            else:
                mean_actions = actor(obs_batch)
            
            a_loss = mean_actions.sum()
            
            self.model.policy.actor.zero_grad()
            a_loss.backward()
            
            a_g = []
            for param in actor.parameters():
                if param.grad is not None:
                    a_g.append(param.grad.view(-1))
            if a_g:
                actor_grads.append(torch.cat(a_g))
                
        # 3. 计算 Rank (确保引入了之前定义的辅助函数 _compute_ntk_metrics_from_grads)
        # 如果您只想要 Rank，不想统计对角线数据，可以用 _calculate_approx_rank_from_grads
        try:
            c_rank, c_diag, c_off = _compute_ntk_metrics_from_grads(critic_grads, self.ntk_delta)
            a_rank, a_diag, a_off = _compute_ntk_metrics_from_grads(actor_grads, self.ntk_delta)
            
            # 4. 记录日志
            self.logger.record("monitor/ntk_rank_actor", a_rank)
            self.logger.record("monitor/ntk_rank_critic", c_rank)
            self.logger.record("monitor/ntk_diag_sum_actor", a_diag)
            self.logger.record("monitor/ntk_diag_sum_critic", c_diag)
            self.logger.record("monitor/ntk_off_diag_sum_actor", a_off)
            self.logger.record("monitor/ntk_off_diag_sum_critic", c_off)
            
            return a_rank, c_rank
            
        except Exception as e:
            print(f"[Warning] NTK calculation failed: {e}")
            return 0.0, 0.0