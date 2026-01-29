"""
VAE 预训练脚本
用于预训练 VAE 编码器和解码器
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.vae import VAE
from src.utils.data_loader import ObservationDataset
from src.utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description='训练 VAE')
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                        help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='数据目录')
    parser.add_argument('--exp_name', type=str, default='vae_pretrain',
                        help='实验名称')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, kl_weight=1.0):
    """
    训练一个 epoch

    Args:
        model: VAE 模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        kl_weight: KL 散度权重

    Returns:
        平均损失字典
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        observations = batch['observation'].to(device)

        # 前向传播
        output = model(observations)

        # 计算损失
        losses = model.loss_function(
            output['reconstruction'],
            observations,
            output['mu'],
            output['logvar'],
            kl_weight=kl_weight
        )

        # 反向传播
        optimizer.zero_grad()
        losses['loss'].backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 累积损失
        total_loss += losses['loss'].item()
        total_recon_loss += losses['recon_loss'].item()
        total_kl_loss += losses['kl_loss'].item()
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({
            'loss': losses['loss'].item(),
            'recon': losses['recon_loss'].item(),
            'kl': losses['kl_loss'].item()
        })

    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches
    }


def validate(model, dataloader, device, kl_weight=1.0):
    """
    验证模型

    Args:
        model: VAE 模型
        dataloader: 数据加载器
        device: 设备
        kl_weight: KL 散度权重

    Returns:
        平均损失字典
    """
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            observations = batch['observation'].to(device)

            # 前向传播
            output = model(observations)

            # 计算损失
            losses = model.loss_function(
                output['reconstruction'],
                observations,
                output['mu'],
                output['logvar'],
                kl_weight=kl_weight
            )

            total_loss += losses['loss'].item()
            total_recon_loss += losses['recon_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
            num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches
    }


def main():
    args = parse_args()

    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"配置文件 {args.config} 不存在，使用默认配置")
        config = {
            'vae': {
                'latent_dim': 32,
                'input_channels': 3
            },
            'training': {
                'batch_size': 32,
                'num_epochs': 100,
                'learning_rate': 1e-4,
                'kl_weight': 1.0,
                'save_interval': 10
            }
        }

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = VAE(
        input_channels=config['vae'].get('input_channels', 3),
        latent_dim=config['vae'].get('latent_dim', 32)
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建数据加载器
    try:
        train_dataset = ObservationDataset(
            data_dir=os.path.join(args.data_dir, 'train'),
            transform=None
        )
        val_dataset = ObservationDataset(
            data_dir=os.path.join(args.data_dir, 'val'),
            transform=None
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=4
        )

        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")

    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请确保数据目录正确并包含数据")
        return

    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    # 创建日志目录
    log_dir = project_root / 'experiments' / 'logs' / args.exp_name
    checkpoint_dir = project_root / 'experiments' / 'checkpoints' / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir)

    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"从 epoch {start_epoch} 恢复训练")

    # 训练循环
    num_epochs = config['training']['num_epochs']
    kl_weight = config['training'].get('kl_weight', 1.0)

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # 训练
        train_losses = train_epoch(model, train_loader, optimizer, device, kl_weight)

        # 验证
        val_losses = validate(model, val_loader, device, kl_weight)

        # 记录日志
        writer.add_scalar('Loss/train', train_losses['loss'], epoch)
        writer.add_scalar('Loss/val', val_losses['loss'], epoch)
        writer.add_scalar('ReconLoss/train', train_losses['recon_loss'], epoch)
        writer.add_scalar('ReconLoss/val', val_losses['recon_loss'], epoch)
        writer.add_scalar('KLLoss/train', train_losses['kl_loss'], epoch)
        writer.add_scalar('KLLoss/val', val_losses['kl_loss'], epoch)

        print(f"训练损失: {train_losses['loss']:.4f}, 验证损失: {val_losses['loss']:.4f}")

        # 调整学习率
        scheduler.step(val_losses['loss'])

        # 保存检查点
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses['loss'],
                'val_loss': val_losses['loss'],
                'best_val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")

        # 保存最佳模型
        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            best_model_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['loss'],
                'config': config
            }, best_model_path)
            print(f"保存最佳模型: {best_model_path}")

    writer.close()
    print("\n训练完成！")


if __name__ == '__main__':
    main()
