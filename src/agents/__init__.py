"""
智能体模块

提供统一的智能体接口，支持：
- 强化学习（DreamerV4）
- 模仿学习（Behavioral Cloning）
"""

from .base import BaseAgent
from .dreamer_agent import DreamerV4Agent
from .bc_agent import BehaviorCloningAgent, BCPolicy
from .factory import (
    make_agent,
    load_agent,
    get_default_config,
    AGENT_REGISTRY
)

__all__ = [
    # 基类
    'BaseAgent',

    # 具体实现
    'DreamerV4Agent',
    'BehaviorCloningAgent',
    'BCPolicy',

    # 工厂函数
    'make_agent',
    'load_agent',
    'get_default_config',
    'AGENT_REGISTRY',
]
