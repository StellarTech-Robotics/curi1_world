"""
数据加载器
用于加载观察数据和序列数据
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import pickle


class ObservationDataset(Dataset):
    """
    观察数据集
    用于 VAE 预训练，加载单个观察图像
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        extension: str = '.png'
    ):
        """
        Args:
            data_dir: 数据目录路径
            transform: 数据变换（可选）
            extension: 图像文件扩展名
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.extension = extension

        # 查找所有图像文件
        self.image_paths = sorted(list(self.data_dir.glob(f'*{extension}')))

        if len(self.image_paths) == 0:
            print(f"警告: 在 {data_dir} 中未找到 {extension} 文件")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        获取一个样本

        Args:
            idx: 索引

        Returns:
            sample: 包含观察的字典
        """
        # 加载图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # 转换为 numpy 数组并归一化
        image = np.array(image).astype(np.float32) / 255.0

        # 转换为 [C, H, W]
        image = np.transpose(image, (2, 0, 1))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 转换为 tensor
        observation = torch.from_numpy(image).float()

        return {'observation': observation}


class SequenceDataset(Dataset):
    """
    序列数据集
    用于加载时间序列数据（观察序列、动作序列、奖励序列）
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 50,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            data_dir: 数据目录路径
            seq_len: 序列长度
            transform: 数据变换（可选）
        """
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.transform = transform

        # 查找所有 episode 文件
        self.episode_files = sorted(list(self.data_dir.glob('episode_*.pkl')))

        if len(self.episode_files) == 0:
            print(f"警告: 在 {data_dir} 中未找到 episode 文件")

        # 加载所有 episode
        self.episodes = []
        for ep_file in self.episode_files:
            with open(ep_file, 'rb') as f:
                episode = pickle.load(f)
                self.episodes.append(episode)

        # 构建索引：(episode_idx, start_idx)
        self.indices = []
        for ep_idx, episode in enumerate(self.episodes):
            ep_len = len(episode['observations'])
            for start_idx in range(ep_len - seq_len + 1):
                self.indices.append((ep_idx, start_idx))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """
        获取一个序列样本

        Args:
            idx: 索引

        Returns:
            sample: 包含观察序列、动作序列、奖励序列的字典
        """
        ep_idx, start_idx = self.indices[idx]
        episode = self.episodes[ep_idx]

        # 提取序列
        observations = episode['observations'][start_idx:start_idx + self.seq_len]
        actions = episode['actions'][start_idx:start_idx + self.seq_len]
        rewards = episode['rewards'][start_idx:start_idx + self.seq_len]
        dones = episode.get('dones', np.zeros(len(observations)))[start_idx:start_idx + self.seq_len]

        # 归一化观察（如果是图像）
        if observations.ndim == 4:  # [T, H, W, C]
            observations = observations.astype(np.float32) / 255.0
            observations = np.transpose(observations, (0, 3, 1, 2))  # [T, C, H, W]

        # 应用变换
        if self.transform:
            observations = self.transform(observations)

        # 转换为 tensor
        sample = {
            'observations': torch.from_numpy(observations).float(),
            'actions': torch.from_numpy(actions).float(),
            'rewards': torch.from_numpy(rewards).float(),
            'dones': torch.from_numpy(dones).float()
        }

        return sample


class EpisodeDataset(Dataset):
    """
    Episode 数据集
    加载完整的 episode 数据
    """

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)

        # 查找所有 episode 文件
        self.episode_files = sorted(list(self.data_dir.glob('episode_*.pkl')))

        if len(self.episode_files) == 0:
            print(f"警告: 在 {data_dir} 中未找到 episode 文件")

    def __len__(self) -> int:
        return len(self.episode_files)

    def __getitem__(self, idx: int) -> dict:
        """
        获取一个完整的 episode

        Args:
            idx: 索引

        Returns:
            episode: episode 数据字典
        """
        with open(self.episode_files[idx], 'rb') as f:
            episode = pickle.load(f)

        # 归一化观察（如果是图像）
        observations = episode['observations']
        if observations.ndim == 4:  # [T, H, W, C]
            observations = observations.astype(np.float32) / 255.0
            observations = np.transpose(observations, (0, 3, 1, 2))  # [T, C, H, W]

        episode_data = {
            'observations': torch.from_numpy(observations).float(),
            'actions': torch.from_numpy(episode['actions']).float(),
            'rewards': torch.from_numpy(episode['rewards']).float(),
            'dones': torch.from_numpy(episode.get('dones', np.zeros(len(observations)))).float()
        }

        return episode_data


def save_episode(
    episode_data: dict,
    save_dir: str,
    episode_idx: int
):
    """
    保存 episode 数据

    Args:
        episode_data: episode 数据字典
        save_dir: 保存目录
        episode_idx: episode 索引
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f'episode_{episode_idx:06d}.pkl'

    with open(save_path, 'wb') as f:
        pickle.dump(episode_data, f)

    print(f"Episode {episode_idx} 已保存到: {save_path}")


def load_episode(
    load_path: str
) -> dict:
    """
    加载 episode 数据

    Args:
        load_path: episode 文件路径

    Returns:
        episode_data: episode 数据字典
    """
    with open(load_path, 'rb') as f:
        episode_data = pickle.load(f)

    return episode_data


if __name__ == "__main__":
    # 测试代码

    # 创建模拟数据
    data_dir = Path("data/test")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 保存一些模拟 episode
    for i in range(5):
        episode = {
            'observations': np.random.randint(0, 255, (100, 64, 64, 3), dtype=np.uint8),
            'actions': np.random.randn(100, 6).astype(np.float32),
            'rewards': np.random.randn(100).astype(np.float32),
            'dones': np.zeros(100, dtype=np.float32)
        }
        save_episode(episode, data_dir, i)

    # 测试 SequenceDataset
    dataset = SequenceDataset(data_dir, seq_len=50)
    print(f"SequenceDataset 大小: {len(dataset)}")

    sample = dataset[0]
    print(f"观察形状: {sample['observations'].shape}")
    print(f"动作形状: {sample['actions'].shape}")
    print(f"奖励形状: {sample['rewards'].shape}")

    # 清理测试数据
    import shutil
    shutil.rmtree(data_dir)
