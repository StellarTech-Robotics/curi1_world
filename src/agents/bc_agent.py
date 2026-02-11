"""
行为克隆智能体
基于监督学习的模仿学习
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import BaseAgent


class BCPolicy(nn.Module):
    """
    行为克隆策略网络
    从观察直接预测动作
    """

    def __init__(
        self,
        obs_channels: int = 3,
        action_dim: int = 6,
        hidden_dim: int = 256,
        use_tanh: bool = True
    ):
        super().__init__()

        self.use_tanh = use_tanh

        # CNN 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # MLP 预测头
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, C, H, W]

        Returns:
            action: [B, action_dim]
        """
        features = self.encoder(obs)
        action = self.mlp(features)

        if self.use_tanh:
            action = torch.tanh(action)

        return action


class BehaviorCloningAgent(BaseAgent):
    """
    行为克隆智能体

    特点:
    - 纯监督学习，从专家演示中学习
    - 无需奖励信号
    - 训练简单高效
    - 适合有大量高质量演示的场景
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
                - obs_channels: 观察通道数
                - action_dim: 动作维度
                - hidden_dim: 隐藏层维度
                - use_tanh: 是否使用tanh激活
                - learning_rate: 学习率
                - device: 设备
        """
        super().__init__(config)

        # 创建策略网络
        self.policy = BCPolicy(
            obs_channels=config.get('obs_channels', 3),
            action_dim=config.get('action_dim', 6),
            hidden_dim=config.get('hidden_dim', 256),
            use_tanh=config.get('use_tanh', True)
        )

        self.policy.to(self.device)

        # 优化器
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate
        )

    def reset(self):
        """重置状态（BC是无状态的）"""
        pass

    def select_action(
        self,
        observation: np.ndarray | torch.Tensor,
        deterministic: bool = True,  # BC 默认确定性
        **kwargs
    ) -> np.ndarray:
        """
        选择动作

        Args:
            observation: 观察 [C, H, W]
            deterministic: BC总是确定性的

        Returns:
            action: [action_dim]
        """
        self.policy.eval()

        # 转换为 tensor
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()

        # 确保格式为 [C, H, W]
        if observation.ndim == 3 and observation.shape[-1] in [1, 3]:
            observation = observation.permute(2, 0, 1)

        observation = observation.unsqueeze(0).to(self.device)  # [1, C, H, W]

        with torch.no_grad():
            action = self.policy(observation)

        return action.cpu().numpy()[0]

    def predict_action_chunk(
        self,
        observation: np.ndarray | torch.Tensor,
        chunk_size: int = 1,
        **kwargs
    ) -> np.ndarray:
        """
        预测动作序列
        注意：基础BC不支持时序建模，返回重复的动作

        Args:
            observation: 观察
            chunk_size: 序列长度

        Returns:
            actions: [chunk_size, action_dim]
        """
        action = self.select_action(observation)
        return np.repeat(action[np.newaxis, :], chunk_size, axis=0)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行一次训练步骤

        Args:
            batch: 批次数据
                - observations: [B, T, C, H, W] 或 [B, C, H, W]
                - actions: [B, T, action_dim] 或 [B, action_dim]

        Returns:
            losses: 损失字典
        """
        self.policy.train()

        # 移动到设备
        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)

        # 处理时序数据（展平）
        if observations.ndim == 5:  # [B, T, C, H, W]
            B, T = observations.shape[:2]
            observations = observations.view(B * T, *observations.shape[2:])
            actions = actions.view(B * T, -1)

        # 前向传播
        pred_actions = self.policy(observations)

        # 计算 MSE 损失
        loss = F.mse_loss(pred_actions, actions)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'bc_loss': loss.item(),
            'total_loss': loss.item()
        }

    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        self.policy.load_state_dict(state_dict['policy'])
        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def eval(self):
        """设置为评估模式"""
        self.policy.eval()

    def train(self):
        """设置为训练模式"""
        self.policy.train()


if __name__ == "__main__":
    # 测试代码
    config = {
        'obs_channels': 3,
        'action_dim': 6,
        'device': 'cuda'
    }

    agent = BehaviorCloningAgent(config)

    # 测试 select_action
    obs = np.random.randn(3, 64, 64).astype(np.float32)
    action = agent.select_action(obs)
    print(f"Action shape: {action.shape}")

    # 测试训练
    batch = {
        'observations': torch.randn(32, 3, 64, 64),
        'actions': torch.randn(32, 6)
    }

    losses = agent.train_step(batch)
    print(f"Losses: {losses}")
