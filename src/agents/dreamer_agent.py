"""
DreamerV4 智能体
实现基于世界模型的强化学习
"""

from typing import Dict, Tuple, Optional, Any
import torch
import numpy as np

from .base import BaseAgent
from ..models.dreamer import DreamerV4


class DreamerV4Agent(BaseAgent):
    """
    DreamerV4 强化学习智能体

    特点:
    - 基于世界模型的RL
    - 在想象中学习策略
    - 支持图像和状态输入
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
                - obs_channels: 观察通道数
                - action_dim: 动作维度
                - action_range: 动作范围
                - vae_latent_dim: VAE潜在维度
                - stoch_dim: RSSM随机维度
                - deter_dim: RSSM确定性维度
                - device: 设备
                - ... 其他DreamerV4参数
        """
        super().__init__(config)

        # 创建 DreamerV4 模型
        self.model = DreamerV4(
            obs_channels=config.get('obs_channels', 3),
            action_dim=config.get('action_dim', 6),
            action_range=config.get('action_range', (-1.0, 1.0)),
            vae_latent_dim=config.get('vae_latent_dim', 32),
            stoch_dim=config.get('stoch_dim', 32),
            deter_dim=config.get('deter_dim', 256),
            hidden_dim=config.get('hidden_dim', 256),
            actor_hidden_dim=config.get('actor_hidden_dim', 256),
            critic_hidden_dim=config.get('critic_hidden_dim', 256),
            num_critics=config.get('num_critics', 2),
            kl_weight=config.get('kl_weight', 1.0),
            free_nats=config.get('free_nats', 3.0),
            device=self.device
        )

        self.model.to(self.device)
        self._rssm_state = None  # RSSM 隐藏状态

        # 训练配置
        self.imagine_horizon = config.get('imagine_horizon', 15)
        self.gamma = config.get('gamma', 0.99)
        self.lambda_ = config.get('lambda_', 0.95)

    def reset(self):
        """重置智能体状态"""
        self._rssm_state = None

    def select_action(
        self,
        observation: np.ndarray | torch.Tensor,
        deterministic: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        选择动作

        Args:
            observation: 观察 [C, H, W] 或 [H, W, C]
            deterministic: 是否确定性策略

        Returns:
            action: [action_dim]
        """
        # 转换为 tensor
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()

        # 确保格式为 [C, H, W]
        if observation.ndim == 3 and observation.shape[-1] in [1, 3]:
            observation = observation.permute(2, 0, 1)

        observation = observation.to(self.device)

        # 调用模型
        action, self._rssm_state = self.model.select_action(
            observation,
            state=self._rssm_state,
            deterministic=deterministic
        )

        return action

    def predict_action_chunk(
        self,
        observation: np.ndarray | torch.Tensor,
        chunk_size: int = 1,
        **kwargs
    ) -> np.ndarray:
        """
        预测动作序列

        Args:
            observation: 观察
            chunk_size: 序列长度

        Returns:
            actions: [chunk_size, action_dim]
        """
        actions = []
        for _ in range(chunk_size):
            action = self.select_action(observation, deterministic=True)
            actions.append(action)

        return np.stack(actions, axis=0)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行一次训练步骤

        Args:
            batch: 批次数据
                - observations: [B, T, C, H, W]
                - actions: [B, T, action_dim]
                - rewards: [B, T]
                - dones: [B, T]

        Returns:
            losses: 损失字典
        """
        # 移动到设备
        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        dones = batch['dones'].to(self.device)

        # 计算世界模型损失
        wm_losses = self.model.compute_world_model_loss(
            observations, actions, rewards, dones
        )

        # 提取状态用于想象轨迹
        h_seq, z_seq = wm_losses['states']

        # 想象轨迹并训练 Actor-Critic
        # 使用序列中的状态
        states = torch.cat([h_seq, z_seq], dim=-1)  # [B, T, state_dim]

        # 从第一个时间步开始想象
        initial_state = (h_seq[:, 0], z_seq[:, 0])

        # 想象轨迹
        imagined_states, imagined_actions, predictions = self.model.imagine_trajectory(
            initial_state=initial_state,
            horizon=self.imagine_horizon,
            actor=self.model.actor
        )

        # 计算 Actor 损失
        actor_loss = self.model.compute_actor_loss(
            states,
            imagine_horizon=self.imagine_horizon,
            gamma=self.gamma,
            lambda_=self.lambda_
        )

        # 计算 Critic 损失
        critic_loss = self.model.compute_critic_loss(
            states,
            imagined_states,
            predictions,
            gamma=self.gamma,
            lambda_=self.lambda_
        )

        # 合并损失
        total_loss = wm_losses['total_loss'] + actor_loss + critic_loss

        # 构建完整的损失字典
        losses_dict = {
            'world_model_loss': wm_losses['total_loss'].item(),
            'recon_loss': wm_losses['recon_loss'].item(),
            'kl_loss': wm_losses['kl_loss'].item(),
            'reward_loss': wm_losses['reward_loss'].item(),
            'done_loss': wm_losses['done_loss'].item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': total_loss.item()
        }

        return losses_dict

    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'model': self.model.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        self.model.load_state_dict(state_dict['model'])

    def eval(self):
        """设置为评估模式"""
        self.model.eval()

    def train(self):
        """设置为训练模式"""
        self.model.train()


if __name__ == "__main__":
    # 测试代码
    config = {
        'obs_channels': 3,
        'action_dim': 6,
        'device': 'cuda'
    }

    agent = DreamerV4Agent(config)

    # 测试 select_action
    obs = np.random.randn(3, 64, 64).astype(np.float32)
    action = agent.select_action(obs)
    print(f"Action shape: {action.shape}")

    # 测试训练
    batch = {
        'observations': torch.randn(4, 10, 3, 64, 64),
        'actions': torch.randn(4, 10, 6),
        'rewards': torch.randn(4, 10),
        'dones': torch.zeros(4, 10)
    }

    losses = agent.train_step(batch)
    print(f"Losses: {losses}")
