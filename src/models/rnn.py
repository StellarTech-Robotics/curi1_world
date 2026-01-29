"""
RSSM (Recurrent State Space Model)
DreamerV4 的世界模型核心组件，用于序列建模和预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class RSSM(nn.Module):
    """
    循环状态空间模型 (Recurrent State Space Model)

    状态包含两部分：
    - 确定性状态 h_t (deterministic state): 通过 GRU 更新
    - 随机状态 z_t (stochastic state): 通过潜在变量采样

    状态转移：
    h_t = f(h_{t-1}, z_{t-1}, a_{t-1})  # 确定性状态更新
    z_t ~ p(z_t | h_t)                   # 随机状态采样
    """

    def __init__(
        self,
        action_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        stoch_dim: int = 32,
        deter_dim: int = 256,
        num_layers: int = 1
    ):
        """
        Args:
            action_dim: 动作维度
            latent_dim: VAE 潜在维度（观察编码维度）
            hidden_dim: 隐藏层维度
            stoch_dim: 随机状态维度
            deter_dim: 确定性状态维度
            num_layers: GRU 层数
        """
        super(RSSM, self).__init__()

        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim

        # GRU 用于更新确定性状态
        # 输入：[z_{t-1}, a_{t-1}]
        self.gru = nn.GRU(
            input_size=stoch_dim + action_dim,
            hidden_size=deter_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # 先验网络：p(z_t | h_t)
        self.prior_fc = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)  # 输出均值和对数标准差
        )

        # 后验网络：q(z_t | h_t, o_t)
        # o_t 是观察的 VAE 编码
        self.posterior_fc = nn.Sequential(
            nn.Linear(deter_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)  # 输出均值和对数标准差
        )

        # 观察预测器：p(o_t | h_t, z_t)
        self.obs_predictor = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # 奖励预测器：p(r_t | h_t, z_t)
        self.reward_predictor = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 终止预测器：p(done_t | h_t, z_t)
        self.done_predictor = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧

        Args:
            mu: 均值
            logstd: 对数标准差

        Returns:
            采样的随机状态
        """
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_dist(self, mu: torch.Tensor, logstd: torch.Tensor):
        """
        获取分布（用于计算 KL 散度）

        Args:
            mu: 均值
            logstd: 对数标准差

        Returns:
            正态分布对象
        """
        std = torch.exp(logstd)
        return torch.distributions.Normal(mu, std)

    def prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        先验分布：p(z_t | h_t)

        Args:
            h: 确定性状态 [batch_size, deter_dim]

        Returns:
            z: 采样的随机状态
            mu: 均值
            logstd: 对数标准差
        """
        out = self.prior_fc(h)
        mu, logstd = torch.chunk(out, 2, dim=-1)
        logstd = torch.clamp(logstd, -10, 2)  # 稳定训练
        z = self.reparameterize(mu, logstd)
        return z, mu, logstd

    def posterior(
        self,
        h: torch.Tensor,
        obs_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        后验分布：q(z_t | h_t, o_t)

        Args:
            h: 确定性状态 [batch_size, deter_dim]
            obs_embed: 观察编码 [batch_size, latent_dim]

        Returns:
            z: 采样的随机状态
            mu: 均值
            logstd: 对数标准差
        """
        x = torch.cat([h, obs_embed], dim=-1)
        out = self.posterior_fc(x)
        mu, logstd = torch.chunk(out, 2, dim=-1)
        logstd = torch.clamp(logstd, -10, 2)
        z = self.reparameterize(mu, logstd)
        return z, mu, logstd

    def step(
        self,
        prev_state: Tuple[torch.Tensor, torch.Tensor],
        action: torch.Tensor,
        obs_embed: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        单步状态转移

        Args:
            prev_state: (h_{t-1}, z_{t-1})
            action: a_{t-1} [batch_size, action_dim]
            obs_embed: o_t [batch_size, latent_dim], 如果提供则使用后验，否则使用先验

        Returns:
            next_state: (h_t, z_t)
            info: 包含预测和分布信息的字典
        """
        prev_h, prev_z = prev_state

        # 更新确定性状态：h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        gru_input = torch.cat([prev_z, action], dim=-1).unsqueeze(1)  # [batch, 1, dim]
        h, _ = self.gru(gru_input, prev_h.unsqueeze(0))  # [1, batch, deter_dim]
        h = h.squeeze(1)  # [batch, deter_dim]

        # 采样随机状态
        if obs_embed is not None:
            # 训练时：使用后验 q(z_t | h_t, o_t)
            z, post_mu, post_logstd = self.posterior(h, obs_embed)
            # 同时计算先验用于 KL 散度
            _, prior_mu, prior_logstd = self.prior(h)
        else:
            # 推理时：使用先验 p(z_t | h_t)
            z, prior_mu, prior_logstd = self.prior(h)
            post_mu = post_logstd = None

        # 预测
        state = torch.cat([h, z], dim=-1)
        obs_pred = self.obs_predictor(state)
        reward_pred = self.reward_predictor(state).squeeze(-1)
        done_pred = self.done_predictor(state).squeeze(-1)

        info = {
            'obs_pred': obs_pred,
            'reward_pred': reward_pred,
            'done_pred': done_pred,
            'prior_mu': prior_mu,
            'prior_logstd': prior_logstd,
            'post_mu': post_mu,
            'post_logstd': post_logstd,
        }

        return (h, z), info

    def rollout(
        self,
        initial_state: Tuple[torch.Tensor, torch.Tensor],
        actions: torch.Tensor,
        obs_embeds: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        序列展开

        Args:
            initial_state: (h_0, z_0)
            actions: [batch_size, seq_len, action_dim]
            obs_embeds: [batch_size, seq_len, latent_dim], 可选

        Returns:
            states: (h_seq, z_seq)
            predictions: 包含所有预测的字典
        """
        batch_size, seq_len, _ = actions.shape
        device = actions.device

        h, z = initial_state

        # 存储序列
        h_seq = []
        z_seq = []
        obs_pred_seq = []
        reward_pred_seq = []
        done_pred_seq = []
        prior_mu_seq = []
        prior_logstd_seq = []
        post_mu_seq = []
        post_logstd_seq = []

        for t in range(seq_len):
            obs_t = obs_embeds[:, t] if obs_embeds is not None else None
            (h, z), info = self.step((h, z), actions[:, t], obs_t)

            h_seq.append(h)
            z_seq.append(z)
            obs_pred_seq.append(info['obs_pred'])
            reward_pred_seq.append(info['reward_pred'])
            done_pred_seq.append(info['done_pred'])
            prior_mu_seq.append(info['prior_mu'])
            prior_logstd_seq.append(info['prior_logstd'])
            if info['post_mu'] is not None:
                post_mu_seq.append(info['post_mu'])
                post_logstd_seq.append(info['post_logstd'])

        # 堆叠序列
        h_seq = torch.stack(h_seq, dim=1)  # [batch, seq_len, deter_dim]
        z_seq = torch.stack(z_seq, dim=1)  # [batch, seq_len, stoch_dim]

        predictions = {
            'obs_pred': torch.stack(obs_pred_seq, dim=1),
            'reward_pred': torch.stack(reward_pred_seq, dim=1),
            'done_pred': torch.stack(done_pred_seq, dim=1),
            'prior_mu': torch.stack(prior_mu_seq, dim=1),
            'prior_logstd': torch.stack(prior_logstd_seq, dim=1),
        }

        if post_mu_seq:
            predictions['post_mu'] = torch.stack(post_mu_seq, dim=1)
            predictions['post_logstd'] = torch.stack(post_logstd_seq, dim=1)

        return (h_seq, z_seq), predictions

    def get_initial_state(self, batch_size: int, device: torch.device):
        """
        获取初始状态

        Args:
            batch_size: 批量大小
            device: 设备

        Returns:
            (h_0, z_0): 初始状态
        """
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, z


class RecurrentStateSpaceModel(RSSM):
    """
    RSSM 的别名，保持接口一致性
    """
    pass


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 参数
    batch_size = 4
    seq_len = 10
    action_dim = 6
    latent_dim = 32

    # 创建模型
    rssm = RSSM(
        action_dim=action_dim,
        latent_dim=latent_dim,
        stoch_dim=32,
        deter_dim=256
    ).to(device)

    # 创建数据
    actions = torch.randn(batch_size, seq_len, action_dim).to(device)
    obs_embeds = torch.randn(batch_size, seq_len, latent_dim).to(device)

    # 获取初始状态
    initial_state = rssm.get_initial_state(batch_size, device)

    # 展开
    (h_seq, z_seq), predictions = rssm.rollout(initial_state, actions, obs_embeds)

    print(f"Deterministic state sequence: {h_seq.shape}")
    print(f"Stochastic state sequence: {z_seq.shape}")
    print(f"Observation predictions: {predictions['obs_pred'].shape}")
    print(f"Reward predictions: {predictions['reward_pred'].shape}")
    print(f"Done predictions: {predictions['done_pred'].shape}")
