"""
DreamerV4 主模型
整合 VAE, RSSM, Actor, Critic 实现完整的世界模型强化学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from .vae import VAE
from .rnn import RSSM
from .controller import Actor, EnsembleCritic


class DreamerV4(nn.Module):
    """
    DreamerV4 完整模型

    组件：
    1. VAE: 学习观察的紧凑表示
    2. RSSM: 学习世界模型（状态转移和预测）
    3. Actor: 策略网络
    4. Critic: 价值网络
    """

    def __init__(
        self,
        # 观察和动作空间
        obs_channels: int = 3,
        action_dim: int = 6,
        action_range: Tuple[float, float] = (-1.0, 1.0),

        # VAE 配置
        vae_latent_dim: int = 32,

        # RSSM 配置
        stoch_dim: int = 32,
        deter_dim: int = 256,
        hidden_dim: int = 256,

        # Actor-Critic 配置
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 256,
        num_critics: int = 2,

        # 训练配置
        kl_weight: float = 1.0,
        free_nats: float = 3.0,  # KL 自由位（避免后验坍塌）

        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            obs_channels: 观察图像通道数
            action_dim: 动作维度
            action_range: 动作范围
            vae_latent_dim: VAE 潜在维度
            stoch_dim: RSSM 随机状态维度
            deter_dim: RSSM 确定性状态维度
            hidden_dim: RSSM 隐藏层维度
            actor_hidden_dim: Actor 隐藏层维度
            critic_hidden_dim: Critic 隐藏层维度
            num_critics: Critic 集成数量
            kl_weight: KL 散度权重
            free_nats: KL 自由位
            device: 设备
        """
        super(DreamerV4, self).__init__()

        self.obs_channels = obs_channels
        self.action_dim = action_dim
        self.action_range = action_range
        self.vae_latent_dim = vae_latent_dim
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.kl_weight = kl_weight
        self.free_nats = free_nats
        self.device = device

        # 1. VAE：观察编码器/解码器
        self.vae = VAE(
            input_channels=obs_channels,
            latent_dim=vae_latent_dim
        )

        # 2. RSSM：世界模型
        self.rssm = RSSM(
            action_dim=action_dim,
            latent_dim=vae_latent_dim,
            hidden_dim=hidden_dim,
            stoch_dim=stoch_dim,
            deter_dim=deter_dim
        )

        # 3. Actor：策略网络
        state_dim = deter_dim + stoch_dim
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            action_range=action_range
        )

        # 4. Critic：价值网络
        self.critic = EnsembleCritic(
            state_dim=state_dim,
            hidden_dim=critic_hidden_dim,
            num_critics=num_critics
        )

        self.to(device)

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """
        使用 VAE 编码观察

        Args:
            observations: [batch_size, seq_len, C, H, W] 或 [batch_size, C, H, W]

        Returns:
            embeddings: [batch_size, seq_len, latent_dim] 或 [batch_size, latent_dim]
        """
        is_sequence = observations.dim() == 5

        if is_sequence:
            batch_size, seq_len, C, H, W = observations.shape
            obs_flat = observations.view(batch_size * seq_len, C, H, W)
            embeddings = self.vae.encode(obs_flat)
            embeddings = embeddings.view(batch_size, seq_len, -1)
        else:
            embeddings = self.vae.encode(observations)

        return embeddings

    def decode_observations(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        使用 VAE 解码嵌入

        Args:
            embeddings: [batch_size, seq_len, latent_dim] 或 [batch_size, latent_dim]

        Returns:
            reconstructions: [batch_size, seq_len, C, H, W] 或 [batch_size, C, H, W]
        """
        is_sequence = embeddings.dim() == 3

        if is_sequence:
            batch_size, seq_len, latent_dim = embeddings.shape
            emb_flat = embeddings.view(batch_size * seq_len, latent_dim)
            reconstructions = self.vae.decode(emb_flat)
            _, C, H, W = reconstructions.shape
            reconstructions = reconstructions.view(batch_size, seq_len, C, H, W)
        else:
            reconstructions = self.vae.decode(embeddings)

        return reconstructions

    def imagine_trajectory(
        self,
        initial_state: Tuple[torch.Tensor, torch.Tensor],
        horizon: int,
        actor: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        在想象中展开轨迹（用于策略学习）

        Args:
            initial_state: (h_0, z_0)
            horizon: 想象步数
            actor: 使用的 Actor，默认使用 self.actor

        Returns:
            states: 想象的状态序列
            actions: 采取的动作序列
            predictions: RSSM 预测
        """
        if actor is None:
            actor = self.actor

        batch_size = initial_state[0].shape[0]
        device = initial_state[0].device

        h, z = initial_state
        states_h = []
        states_z = []
        actions = []
        reward_preds = []
        done_preds = []

        for t in range(horizon):
            # 当前状态
            state = torch.cat([h, z], dim=-1)

            # 使用策略选择动作
            action, _, _ = actor(state, deterministic=False)

            # 状态转移（使用先验，因为是想象）
            (h, z), info = self.rssm.step((h, z), action, obs_embed=None)

            # 存储
            states_h.append(h)
            states_z.append(z)
            actions.append(action)
            reward_preds.append(info['reward_pred'])
            done_preds.append(info['done_pred'])

        # 堆叠
        states_h = torch.stack(states_h, dim=1)  # [batch, horizon, deter_dim]
        states_z = torch.stack(states_z, dim=1)  # [batch, horizon, stoch_dim]
        states = torch.cat([states_h, states_z], dim=-1)  # [batch, horizon, state_dim]
        actions = torch.stack(actions, dim=1)  # [batch, horizon, action_dim]

        predictions = {
            'reward': torch.stack(reward_preds, dim=1),
            'done': torch.stack(done_preds, dim=1)
        }

        return states, actions, predictions

    def compute_world_model_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算世界模型损失（VAE + RSSM）

        Args:
            observations: [batch_size, seq_len, C, H, W]
            actions: [batch_size, seq_len, action_dim]
            rewards: [batch_size, seq_len]
            dones: [batch_size, seq_len]

        Returns:
            损失字典
        """
        batch_size, seq_len = observations.shape[:2]

        # 1. VAE 编码观察
        obs_embeds = self.encode_observations(observations)

        # 2. RSSM 前向传播
        initial_state = self.rssm.get_initial_state(batch_size, self.device)
        (h_seq, z_seq), predictions = self.rssm.rollout(
            initial_state,
            actions,
            obs_embeds
        )

        # 3. 计算损失

        # 3.1 观察重构损失（VAE）
        obs_recon = self.decode_observations(predictions['obs_pred'])
        recon_loss = F.mse_loss(obs_recon, observations, reduction='mean')

        # 3.2 KL 散度损失（带自由位）
        prior_dist = self.rssm.get_dist(predictions['prior_mu'], predictions['prior_logstd'])
        post_dist = self.rssm.get_dist(predictions['post_mu'], predictions['post_logstd'])
        kl_loss = torch.distributions.kl_divergence(post_dist, prior_dist)
        kl_loss = torch.maximum(kl_loss, torch.ones_like(kl_loss) * self.free_nats)
        kl_loss = kl_loss.mean()

        # 3.3 奖励预测损失
        reward_loss = F.mse_loss(predictions['reward_pred'], rewards, reduction='mean')

        # 3.4 终止预测损失
        done_loss = F.binary_cross_entropy_with_logits(
            predictions['done_pred'],
            dones.float(),
            reduction='mean'
        )

        # 总损失
        total_loss = (
            recon_loss +
            self.kl_weight * kl_loss +
            reward_loss +
            done_loss
        )

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'reward_loss': reward_loss,
            'done_loss': done_loss,
            'states': (h_seq, z_seq)
        }

    def compute_actor_loss(
        self,
        states: torch.Tensor,
        imagine_horizon: int = 15,
        gamma: float = 0.99,
        lambda_: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """
        计算 Actor 损失（策略梯度）

        Args:
            states: 起始状态 [batch_size, state_dim]
            imagine_horizon: 想象轨迹长度
            gamma: 折扣因子
            lambda_: GAE lambda

        Returns:
            损失字典
        """
        # 从状态中分离出 h 和 z
        h = states[..., :self.deter_dim]
        z = states[..., self.deter_dim:]

        # 想象轨迹
        with torch.no_grad():
            imagined_states, imagined_actions, predictions = self.imagine_trajectory(
                (h, z),
                horizon=imagine_horizon,
                actor=self.actor
            )

            # 计算价值和优势
            values = self.critic(imagined_states).squeeze(-1)  # [batch, horizon]
            rewards = predictions['reward']  # [batch, horizon]

            # Lambda 回报（用于减少偏差）
            returns = self._compute_lambda_returns(
                rewards, values, gamma, lambda_
            )

        # 计算策略损失（最大化回报）
        # 重新计算动作（需要梯度）
        imagined_actions_with_grad, _, _ = self.actor(imagined_states)

        # 策略梯度损失
        actor_loss = -returns.mean()

        return {
            'actor_loss': actor_loss,
            'mean_return': returns.mean(),
            'mean_value': values.mean()
        }

    def compute_critic_loss(
        self,
        states: torch.Tensor,
        imagine_horizon: int = 15,
        gamma: float = 0.99,
        lambda_: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """
        计算 Critic 损失（TD 学习）

        Args:
            states: 起始状态 [batch_size, state_dim]
            imagine_horizon: 想象轨迹长度
            gamma: 折扣因子
            lambda_: GAE lambda

        Returns:
            损失字典
        """
        # 从状态中分离出 h 和 z
        h = states[..., :self.deter_dim]
        z = states[..., self.deter_dim:]

        # 想象轨迹
        with torch.no_grad():
            imagined_states, _, predictions = self.imagine_trajectory(
                (h, z),
                horizon=imagine_horizon,
                actor=self.actor
            )
            rewards = predictions['reward']

        # 计算目标值
        values = self.critic(imagined_states, return_all=True)  # List of [batch, horizon, 1]

        # 计算 lambda 回报作为目标
        with torch.no_grad():
            target_values = self.critic.get_min_value(imagined_states).squeeze(-1)
            targets = self._compute_lambda_returns(
                rewards, target_values, gamma, lambda_
            )

        # Critic 损失（所有 Critic 的 MSE）
        critic_losses = []
        for value in values:
            value = value.squeeze(-1)
            critic_loss = F.mse_loss(value, targets, reduction='mean')
            critic_losses.append(critic_loss)

        total_critic_loss = sum(critic_losses)

        return {
            'critic_loss': total_critic_loss,
            'mean_target': targets.mean()
        }

    def _compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float,
        lambda_: float
    ) -> torch.Tensor:
        """
        计算 Lambda 回报（TD(λ)）

        Args:
            rewards: [batch, horizon]
            values: [batch, horizon]
            gamma: 折扣因子
            lambda_: lambda 参数

        Returns:
            returns: [batch, horizon]
        """
        batch_size, horizon = rewards.shape
        returns = torch.zeros_like(rewards)

        # 从后向前计算
        last_value = values[:, -1]
        for t in reversed(range(horizon)):
            if t == horizon - 1:
                returns[:, t] = rewards[:, t] + gamma * last_value
            else:
                delta = rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
                returns[:, t] = values[:, t] + delta + gamma * lambda_ * (returns[:, t + 1] - values[:, t + 1])

        return returns

    def select_action(
        self,
        observation: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
        """
        选择动作（推理时使用）

        Args:
            observation: 当前观察 [C, H, W]
            state: 当前 RSSM 状态 (h, z)，如果为 None 则初始化
            deterministic: 是否使用确定性策略

        Returns:
            action: numpy 数组
            next_state: 下一个 RSSM 状态
        """
        with torch.no_grad():
            # 编码观察
            obs = observation.unsqueeze(0).to(self.device)  # [1, C, H, W]
            obs_embed = self.vae.encode(obs)  # [1, latent_dim]

            # 初始化状态
            if state is None:
                state = self.rssm.get_initial_state(1, self.device)

            # 状态转移（使用后验）
            # 注意：这里假设我们有前一步的动作，实际使用中需要存储
            # 这里简化处理，使用零动作
            prev_action = torch.zeros(1, self.action_dim).to(self.device)
            next_state, _ = self.rssm.step(state, prev_action, obs_embed)

            # 选择动作
            h, z = next_state
            model_state = torch.cat([h, z], dim=-1)
            action, _, _ = self.actor(model_state, deterministic=deterministic)

            action = action.cpu().numpy()[0]

        return action, next_state


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    dreamer = DreamerV4(
        obs_channels=3,
        action_dim=6,
        vae_latent_dim=32,
        stoch_dim=32,
        deter_dim=256,
        device=device
    )

    print(f"DreamerV4 模型已创建")
    print(f"总参数量: {sum(p.numel() for p in dreamer.parameters()):,}")

    # 测试数据
    batch_size = 4
    seq_len = 10

    observations = torch.randn(batch_size, seq_len, 3, 64, 64).to(device)
    actions = torch.randn(batch_size, seq_len, 6).to(device)
    rewards = torch.randn(batch_size, seq_len).to(device)
    dones = torch.zeros(batch_size, seq_len).to(device)

    # 测试世界模型损失
    wm_losses = dreamer.compute_world_model_loss(observations, actions, rewards, dones)
    print(f"\n世界模型损失:")
    print(f"  总损失: {wm_losses['total_loss'].item():.4f}")
    print(f"  重构损失: {wm_losses['recon_loss'].item():.4f}")
    print(f"  KL 损失: {wm_losses['kl_loss'].item():.4f}")
    print(f"  奖励损失: {wm_losses['reward_loss'].item():.4f}")

    # 测试 Actor-Critic 损失
    h_seq, z_seq = wm_losses['states']
    states = torch.cat([h_seq[:, 0], z_seq[:, 0]], dim=-1)

    actor_losses = dreamer.compute_actor_loss(states, imagine_horizon=15)
    print(f"\nActor 损失: {actor_losses['actor_loss'].item():.4f}")

    critic_losses = dreamer.compute_critic_loss(states, imagine_horizon=15)
    print(f"Critic 损失: {critic_losses['critic_loss'].item():.4f}")

    # 测试推理
    single_obs = torch.randn(3, 64, 64).to(device)
    action, state = dreamer.select_action(single_obs, deterministic=True)
    print(f"\n推理测试:")
    print(f"  动作: {action}")
    print(f"  动作范围: [{action.min():.2f}, {action.max():.2f}]")
