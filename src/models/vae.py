"""
VAE (Variational Autoencoder) 模型
用于学习观察空间的低维潜在表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class Encoder(nn.Module):
    """
    VAE 编码器：将高维观察（如图像）编码为低维潜在表示
    """
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 32,
        hidden_dims: list = None
    ):
        super(Encoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        self.latent_dim = latent_dim

        # 构建卷积层
        modules = []
        in_channels = input_channels

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # 计算展平后的维度（假设输入为 64x64）
        self.flatten_dim = hidden_dims[-1] * 4 * 4

        # 均值和方差分支
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入观察 [batch_size, channels, height, width]

        Returns:
            mu: 潜在空间均值 [batch_size, latent_dim]
            logvar: 潜在空间对数方差 [batch_size, latent_dim]
        """
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class Decoder(nn.Module):
    """
    VAE 解码器：从潜在表示重构观察
    """
    def __init__(
        self,
        latent_dim: int = 32,
        output_channels: int = 3,
        hidden_dims: list = None
    ):
        super(Decoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # 从潜在向量到特征图
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * 4 * 4)

        # 构建转置卷积层
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        # 最后一层输出
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Sigmoid()  # 将输出限制在 [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            z: 潜在表示 [batch_size, latent_dim]

        Returns:
            reconstruction: 重构的观察 [batch_size, channels, height, width]
        """
        h = self.fc(z)
        h = h.view(-1, self.hidden_dims[0], 4, 4)
        h = self.decoder(h)
        reconstruction = self.final_layer(h)

        return reconstruction


class VAE(nn.Module):
    """
    完整的 VAE 模型
    """
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 32,
        encoder_hidden_dims: list = None,
        decoder_hidden_dims: list = None
    ):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        # 编码器和解码器
        self.encoder = Encoder(input_channels, latent_dim, encoder_hidden_dims)
        self.decoder = Decoder(latent_dim, input_channels, decoder_hidden_dims)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧：z = mu + sigma * epsilon

        Args:
            mu: 均值
            logvar: 对数方差

        Returns:
            z: 采样的潜在向量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入观察 [batch_size, channels, height, width]

        Returns:
            字典包含：
                - reconstruction: 重构的观察
                - mu: 潜在空间均值
                - logvar: 潜在空间对数方差
                - z: 采样的潜在向量
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)

        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码观察到潜在空间（使用均值，不采样）

        Args:
            x: 输入观察

        Returns:
            z: 潜在表示
        """
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        从潜在表示解码

        Args:
            z: 潜在向量

        Returns:
            reconstruction: 重构的观察
        """
        return self.decoder(z)

    def loss_function(
        self,
        reconstruction: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        VAE 损失函数 = 重构损失 + KL 散度

        Args:
            reconstruction: 重构的观察
            x: 原始观察
            mu: 潜在空间均值
            logvar: 潜在空间对数方差
            kl_weight: KL 散度权重

        Returns:
            损失字典
        """
        # 重构损失（MSE）
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum')

        # KL 散度：-0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # 总损失
        total_loss = recon_loss + kl_weight * kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    vae = VAE(input_channels=3, latent_dim=32).to(device)

    # 创建随机输入
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64).to(device)

    # 前向传播
    output = vae(x)

    # 计算损失
    losses = vae.loss_function(
        output['reconstruction'],
        x,
        output['mu'],
        output['logvar']
    )

    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {output['reconstruction'].shape}")
    print(f"Latent shape: {output['z'].shape}")
    print(f"Total loss: {losses['loss'].item():.4f}")
    print(f"Recon loss: {losses['recon_loss'].item():.4f}")
    print(f"KL loss: {losses['kl_loss'].item():.4f}")
