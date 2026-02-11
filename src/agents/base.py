"""
基础智能体接口
定义统一的API规范，支持模仿学习和强化学习
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import torch
import numpy as np
from pathlib import Path


class BaseAgent(ABC):
    """
    智能体基类
    定义所有智能体必须实现的接口方法
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化智能体

        Args:
            config: 配置字典
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self._state = None  # 内部状态（用于recurrent模型）

    @abstractmethod
    def reset(self):
        """
        重置智能体状态
        在每个 episode 开始时调用
        """
        pass

    @abstractmethod
    def select_action(
        self,
        observation: np.ndarray | torch.Tensor,
        deterministic: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        选择动作（推理时使用）

        Args:
            observation: 观察，支持 numpy 或 torch 格式
            deterministic: 是否使用确定性策略
            **kwargs: 其他参数

        Returns:
            action: numpy 格式的动作 [action_dim]
        """
        pass

    @abstractmethod
    def predict_action_chunk(
        self,
        observation: np.ndarray | torch.Tensor,
        chunk_size: int = 1,
        **kwargs
    ) -> np.ndarray:
        """
        预测动作序列（用于action chunking策略）

        Args:
            observation: 观察
            chunk_size: 动作序列长度

        Returns:
            actions: numpy 格式的动作序列 [chunk_size, action_dim]
        """
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行一次训练步骤

        Args:
            batch: 批次数据
                - observations: [B, T, ...]
                - actions: [B, T, action_dim]
                - rewards: [B, T]
                - dones: [B, T]

        Returns:
            losses: 损失字典
        """
        pass

    def save(self, save_path: str | Path):
        """
        保存模型检查点

        Args:
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'config': self.config,
            'state_dict': self.state_dict(),
        }

        torch.save(checkpoint, save_path)
        print(f"模型已保存到: {save_path}")

    @classmethod
    def load(cls, load_path: str | Path, **kwargs):
        """
        从检查点加载模型

        Args:
            load_path: 检查点路径
            **kwargs: 其他参数

        Returns:
            agent: 加载的智能体实例
        """
        load_path = Path(load_path)
        checkpoint = torch.load(load_path, map_location='cpu')

        # 合并配置
        config = checkpoint['config']
        config.update(kwargs)

        # 创建实例
        agent = cls(config)
        agent.load_state_dict(checkpoint['state_dict'])

        print(f"模型已从 {load_path} 加载")
        return agent

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """
        获取模型状态字典

        Returns:
            state_dict: 状态字典
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        加载模型状态字典

        Args:
            state_dict: 状态字典
        """
        pass

    def eval(self):
        """设置为评估模式"""
        pass

    def train(self):
        """设置为训练模式"""
        pass
