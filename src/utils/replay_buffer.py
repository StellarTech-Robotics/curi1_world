"""
回放缓冲区
用于存储和采样经验数据
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import random


class ReplayBuffer:
    """
    简单的回放缓冲区
    存储完整的 episode 数据
    """

    def __init__(self, capacity: int = 100000):
        """
        Args:
            capacity: 缓冲区容量（存储的转移数量）
        """
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        self.total_transitions = 0

    def add_episode(self, episode_data: Dict[str, np.ndarray]):
        """
        添加一个 episode

        Args:
            episode_data: episode 数据字典
                - observations: [T, ...]
                - actions: [T, action_dim]
                - rewards: [T]
                - dones: [T]
        """
        episode_len = len(episode_data['observations'])

        # 如果缓冲区满了，移除最早的 episode
        if self.total_transitions + episode_len > self.capacity:
            while self.episodes and self.total_transitions + episode_len > self.capacity:
                removed_episode = self.episodes.popleft()
                self.total_transitions -= len(removed_episode['observations'])

        # 添加新 episode
        self.episodes.append(episode_data)
        self.total_transitions += episode_len

    def sample(self, batch_size: int, seq_len: int) -> Dict[str, np.ndarray]:
        """
        采样批次序列

        Args:
            batch_size: 批量大小
            seq_len: 序列长度

        Returns:
            batch: 批次数据字典
        """
        if len(self.episodes) == 0:
            raise ValueError("回放缓冲区为空")

        batch_observations = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []

        for _ in range(batch_size):
            # 随机选择一个 episode
            episode = random.choice(self.episodes)
            episode_len = len(episode['observations'])

            # 如果 episode 长度小于序列长度，用零填充
            if episode_len < seq_len:
                # 填充
                pad_len = seq_len - episode_len

                obs = episode['observations']
                if obs.ndim == 4:  # 图像
                    obs_padded = np.concatenate([
                        obs,
                        np.zeros((pad_len, *obs.shape[1:]), dtype=obs.dtype)
                    ], axis=0)
                else:  # 状态向量
                    obs_padded = np.concatenate([
                        obs,
                        np.zeros((pad_len, *obs.shape[1:]), dtype=obs.dtype)
                    ], axis=0)

                actions_padded = np.concatenate([
                    episode['actions'],
                    np.zeros((pad_len, *episode['actions'].shape[1:]), dtype=episode['actions'].dtype)
                ], axis=0)

                rewards_padded = np.concatenate([
                    episode['rewards'],
                    np.zeros(pad_len, dtype=episode['rewards'].dtype)
                ], axis=0)

                dones_padded = np.concatenate([
                    episode['dones'],
                    np.ones(pad_len, dtype=episode['dones'].dtype)  # 填充部分标记为 done
                ], axis=0)

                batch_observations.append(obs_padded)
                batch_actions.append(actions_padded)
                batch_rewards.append(rewards_padded)
                batch_dones.append(dones_padded)

            else:
                # 随机选择起始位置
                start_idx = random.randint(0, episode_len - seq_len)

                batch_observations.append(episode['observations'][start_idx:start_idx + seq_len])
                batch_actions.append(episode['actions'][start_idx:start_idx + seq_len])
                batch_rewards.append(episode['rewards'][start_idx:start_idx + seq_len])
                batch_dones.append(episode['dones'][start_idx:start_idx + seq_len])

        # 堆叠批次
        return {
            'observations': np.stack(batch_observations, axis=0),
            'actions': np.stack(batch_actions, axis=0),
            'rewards': np.stack(batch_rewards, axis=0),
            'dones': np.stack(batch_dones, axis=0)
        }

    def __len__(self) -> int:
        """返回存储的转移数量"""
        return self.total_transitions

    def clear(self):
        """清空缓冲区"""
        self.episodes.clear()
        self.total_transitions = 0


class PrioritizedReplayBuffer:
    """
    优先级回放缓冲区
    基于 TD 误差进行优先级采样
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001
    ):
        """
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数（0=均匀采样，1=完全优先级采样）
            beta: 重要性采样权重指数
            beta_increment: beta 增量
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        self.max_priority = 1.0

    def add_episode(
        self,
        episode_data: Dict[str, np.ndarray],
        priority: Optional[float] = None
    ):
        """
        添加一个 episode

        Args:
            episode_data: episode 数据
            priority: 优先级（如果为 None，使用最大优先级）
        """
        if priority is None:
            priority = self.max_priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(episode_data)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = episode_data
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        seq_len: int
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        优先级采样

        Args:
            batch_size: 批量大小
            seq_len: 序列长度

        Returns:
            batch: 批次数据
            weights: 重要性采样权重
            indices: 采样的索引
        """
        if len(self.buffer) == 0:
            raise ValueError("回放缓冲区为空")

        # 计算采样概率
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 采样索引
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)

        # 计算重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化

        # 采样数据
        batch_observations = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []

        for idx in indices:
            episode = self.buffer[idx]
            episode_len = len(episode['observations'])

            if episode_len < seq_len:
                # 填充
                pad_len = seq_len - episode_len

                obs = episode['observations']
                if obs.ndim == 4:  # 图像
                    obs_padded = np.concatenate([
                        obs,
                        np.zeros((pad_len, *obs.shape[1:]), dtype=obs.dtype)
                    ], axis=0)
                else:
                    obs_padded = np.concatenate([
                        obs,
                        np.zeros((pad_len, *obs.shape[1:]), dtype=obs.dtype)
                    ], axis=0)

                actions_padded = np.concatenate([
                    episode['actions'],
                    np.zeros((pad_len, *episode['actions'].shape[1:]), dtype=episode['actions'].dtype)
                ], axis=0)

                rewards_padded = np.concatenate([
                    episode['rewards'],
                    np.zeros(pad_len, dtype=episode['rewards'].dtype)
                ], axis=0)

                dones_padded = np.concatenate([
                    episode['dones'],
                    np.ones(pad_len, dtype=episode['dones'].dtype)
                ], axis=0)

                batch_observations.append(obs_padded)
                batch_actions.append(actions_padded)
                batch_rewards.append(rewards_padded)
                batch_dones.append(dones_padded)

            else:
                start_idx = random.randint(0, episode_len - seq_len)

                batch_observations.append(episode['observations'][start_idx:start_idx + seq_len])
                batch_actions.append(episode['actions'][start_idx:start_idx + seq_len])
                batch_rewards.append(episode['rewards'][start_idx:start_idx + seq_len])
                batch_dones.append(episode['dones'][start_idx:start_idx + seq_len])

        # 增加 beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = {
            'observations': np.stack(batch_observations, axis=0),
            'actions': np.stack(batch_actions, axis=0),
            'rewards': np.stack(batch_rewards, axis=0),
            'dones': np.stack(batch_dones, axis=0)
        }

        return batch, weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        更新优先级

        Args:
            indices: 更新的索引
            priorities: 新的优先级
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.priorities.clear()
        self.position = 0
        self.max_priority = 1.0


if __name__ == "__main__":
    # 测试代码

    # 创建回放缓冲区
    buffer = ReplayBuffer(capacity=1000)

    print("测试 ReplayBuffer:")

    # 添加一些 episode
    for i in range(10):
        episode = {
            'observations': np.random.randn(100, 3, 64, 64).astype(np.float32),
            'actions': np.random.randn(100, 6).astype(np.float32),
            'rewards': np.random.randn(100).astype(np.float32),
            'dones': np.zeros(100, dtype=np.float32)
        }
        buffer.add_episode(episode)

    print(f"缓冲区大小: {len(buffer)}")

    # 采样
    batch = buffer.sample(batch_size=4, seq_len=50)

    print(f"批次观察形状: {batch['observations'].shape}")
    print(f"批次动作形状: {batch['actions'].shape}")
    print(f"批次奖励形状: {batch['rewards'].shape}")

    # 测试优先级回放缓冲区
    print("\n测试 PrioritizedReplayBuffer:")

    pr_buffer = PrioritizedReplayBuffer(capacity=1000)

    # 添加一些 episode
    for i in range(10):
        episode = {
            'observations': np.random.randn(100, 3, 64, 64).astype(np.float32),
            'actions': np.random.randn(100, 6).astype(np.float32),
            'rewards': np.random.randn(100).astype(np.float32),
            'dones': np.zeros(100, dtype=np.float32)
        }
        pr_buffer.add_episode(episode, priority=np.random.rand())

    print(f"优先级缓冲区大小: {len(pr_buffer)}")

    # 采样
    batch, weights, indices = pr_buffer.sample(batch_size=4, seq_len=50)

    print(f"批次观察形状: {batch['observations'].shape}")
    print(f"重要性采样权重: {weights}")
    print(f"采样索引: {indices}")
