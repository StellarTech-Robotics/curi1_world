"""
Actor-Critic 控制器
Actor: 策略网络，输出动作分布
Critic: 价值网络，估计状态价值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class Actor(nn.Module):
    """
    策略网络（Actor）
    输入世界模型状态，输出动作分布
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        action_range: Tuple[float, float] = (-1.0, 1.0),
        log_std_min: float = -10.0,
        log_std_max: float = 2.0
    ):
        """
        Args:
            state_dim: 状态维度 (deter_dim + stoch_dim)
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            num_layers: 网络层数
            action_range: 动作范围
            log_std_min: 对数标准差最小值
            log_std_max: 对数标准差最大值
        """
        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.action_range = action_range
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 构建网络
        layers = []
        input_dim = state_dim

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # 输出层：均值和对数标准差
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            state: 状态 [batch_size, state_dim] 或 [batch_size, seq_len, state_dim]
            deterministic: 是否使用确定性策略

        Returns:
            action: 动作
            mean: 均值（可选）
            log_std: 对数标准差（可选）
        """
        h = self.backbone(state)

        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        if deterministic:
            # 确定性策略：直接使用均值
            action = torch.tanh(mean)
            # 缩放到动作范围
            action = self._scale_action(action)
            return action, mean, log_std
        else:
            # 随机策略：从分布中采样
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            sample = dist.rsample()  # 重参数化采样

            # Tanh 挤压
            action = torch.tanh(sample)

            # 缩放到动作范围
            action = self._scale_action(action)

            return action, mean, log_std

    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        将 [-1, 1] 的动作缩放到指定范围

        Args:
            action: 原始动作

        Returns:
            缩放后的动作
        """
        low, high = self.action_range
        return low + (action + 1.0) * 0.5 * (high - low)

    def get_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        计算动作的对数概率

        Args:
            state: 状态
            action: 动作

        Returns:
            log_prob: 对数概率
        """
        h = self.backbone(state)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # 反向缩放动作
        action_unscaled = self._unscale_action(action)

        # 反向 tanh
        # 注意：tanh^{-1}(x) = 0.5 * log((1+x)/(1-x))
        action_unscaled = torch.clamp(action_unscaled, -0.999, 0.999)
        action_pre_tanh = 0.5 * torch.log((1 + action_unscaled) / (1 - action_unscaled))

        # 计算高斯分布的对数概率
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action_pre_tanh)

        # Tanh 变换的雅可比行列式修正
        log_prob = log_prob - torch.log(1 - action_unscaled.pow(2) + 1e-6)

        # 对所有动作维度求和
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return log_prob

    def _unscale_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        将动作从指定范围反向缩放到 [-1, 1]

        Args:
            action: 缩放后的动作

        Returns:
            原始动作
        """
        low, high = self.action_range
        return 2.0 * (action - low) / (high - low) - 1.0


class Critic(nn.Module):
    """
    价值网络（Critic）
    估计状态的价值函数
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        """
        Args:
            state_dim: 状态维度 (deter_dim + stoch_dim)
            hidden_dim: 隐藏层维度
            num_layers: 网络层数
        """
        super(Critic, self).__init__()

        # 构建网络
        layers = []
        input_dim = state_dim

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            state: 状态 [batch_size, state_dim] 或 [batch_size, seq_len, state_dim]

        Returns:
            value: 状态价值 [batch_size, 1] 或 [batch_size, seq_len, 1]
        """
        value = self.network(state)
        return value


class EnsembleCritic(nn.Module):
    """
    集成 Critic
    使用多个 Critic 网络来减少过估计偏差
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_critics: int = 2
    ):
        """
        Args:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
            num_layers: 网络层数
            num_critics: Critic 数量
        """
        super(EnsembleCritic, self).__init__()

        self.num_critics = num_critics
        self.critics = nn.ModuleList([
            Critic(state_dim, hidden_dim, num_layers)
            for _ in range(num_critics)
        ])

    def forward(self, state: torch.Tensor, return_all: bool = False):
        """
        前向传播

        Args:
            state: 状态
            return_all: 是否返回所有 Critic 的输出

        Returns:
            如果 return_all=False，返回最小值
            如果 return_all=True，返回所有值的列表
        """
        values = [critic(state) for critic in self.critics]

        if return_all:
            return values
        else:
            # 返回最小值（保守估计）
            return torch.min(torch.stack(values, dim=0), dim=0)[0]

    def get_min_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        获取最小价值估计

        Args:
            state: 状态

        Returns:
            最小价值
        """
        values = [critic(state) for critic in self.critics]
        return torch.min(torch.stack(values, dim=0), dim=0)[0]


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 参数
    batch_size = 4
    seq_len = 10
    state_dim = 288  # deter_dim (256) + stoch_dim (32)
    action_dim = 6

    # 创建模型
    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim).to(device)
    ensemble_critic = EnsembleCritic(state_dim, num_critics=2).to(device)

    # 创建数据
    state = torch.randn(batch_size, state_dim).to(device)
    state_seq = torch.randn(batch_size, seq_len, state_dim).to(device)

    # Actor 测试
    action, mean, log_std = actor(state, deterministic=False)
    print(f"Action shape: {action.shape}")
    print(f"Action range: [{action.min().item():.2f}, {action.max().item():.2f}]")

    action_det, _, _ = actor(state, deterministic=True)
    print(f"Deterministic action shape: {action_det.shape}")

    # Critic 测试
    value = critic(state)
    print(f"Value shape: {value.shape}")

    value_seq = critic(state_seq)
    print(f"Value sequence shape: {value_seq.shape}")

    # Ensemble Critic 测试
    ensemble_value = ensemble_critic(state)
    print(f"Ensemble value shape: {ensemble_value.shape}")

    all_values = ensemble_critic(state, return_all=True)
    print(f"Number of critics: {len(all_values)}")
    print(f"Each critic value shape: {all_values[0].shape}")
