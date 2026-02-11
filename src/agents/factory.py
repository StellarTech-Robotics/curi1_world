"""
智能体工厂
提供统一的创建接口
"""

from typing import Dict, Any, Type
from pathlib import Path

from .base import BaseAgent
from .dreamer_agent import DreamerV4Agent
from .bc_agent import BehaviorCloningAgent


# 注册所有可用的智能体类型
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    'dreamer': DreamerV4Agent,
    'dreamerv4': DreamerV4Agent,
    'bc': BehaviorCloningAgent,
    'behavioral_cloning': BehaviorCloningAgent,
}


def make_agent(agent_type: str, config: Dict[str, Any]) -> BaseAgent:
    """
    创建智能体

    Args:
        agent_type: 智能体类型
            - 'dreamer' / 'dreamerv4': DreamerV4 强化学习
            - 'bc' / 'behavioral_cloning': 行为克隆
        config: 配置字典

    Returns:
        agent: 智能体实例

    Examples:
        >>> # 创建 DreamerV4 agent
        >>> config = {
        ...     'obs_channels': 3,
        ...     'action_dim': 6,
        ...     'device': 'cuda'
        ... }
        >>> agent = make_agent('dreamer', config)

        >>> # 创建 BC agent
        >>> config = {
        ...     'obs_channels': 3,
        ...     'action_dim': 6,
        ...     'learning_rate': 1e-4
        ... }
        >>> agent = make_agent('bc', config)
    """
    agent_type = agent_type.lower()

    if agent_type not in AGENT_REGISTRY:
        available = ', '.join(AGENT_REGISTRY.keys())
        raise ValueError(
            f"未知的智能体类型: {agent_type}\n"
            f"可用类型: {available}"
        )

    agent_class = AGENT_REGISTRY[agent_type]
    return agent_class(config)


def load_agent(checkpoint_path: str | Path, agent_type: str | None = None, **kwargs) -> BaseAgent:
    """
    从检查点加载智能体

    Args:
        checkpoint_path: 检查点路径
        agent_type: 智能体类型（如果为None，从checkpoint中读取）
        **kwargs: 覆盖配置参数

    Returns:
        agent: 加载的智能体实例

    Examples:
        >>> # 从检查点加载
        >>> agent = load_agent('experiments/checkpoints/dreamer_best.pt')

        >>> # 指定类型加载
        >>> agent = load_agent('model.pt', agent_type='bc')

        >>> # 覆盖配置
        >>> agent = load_agent('model.pt', device='cpu')
    """
    import torch

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 获取智能体类型
    if agent_type is None:
        if 'agent_type' in checkpoint.get('config', {}):
            agent_type = checkpoint['config']['agent_type']
        else:
            raise ValueError(
                "无法从检查点中确定智能体类型，请手动指定 agent_type 参数"
            )

    # 合并配置
    config = checkpoint['config']
    config.update(kwargs)

    # 创建智能体
    agent = make_agent(agent_type, config)

    # 加载状态字典
    agent.load_state_dict(checkpoint['state_dict'])

    print(f"成功从 {checkpoint_path} 加载 {agent_type} 智能体")

    return agent


def get_default_config(agent_type: str) -> Dict[str, Any]:
    """
    获取智能体的默认配置

    Args:
        agent_type: 智能体类型

    Returns:
        config: 默认配置字典

    Examples:
        >>> config = get_default_config('dreamer')
        >>> print(config)
    """
    if agent_type.lower() in ['dreamer', 'dreamerv4']:
        return {
            'agent_type': 'dreamer',
            'obs_channels': 3,
            'action_dim': 6,
            'action_range': (-1.0, 1.0),
            'vae_latent_dim': 32,
            'stoch_dim': 32,
            'deter_dim': 256,
            'hidden_dim': 256,
            'actor_hidden_dim': 256,
            'critic_hidden_dim': 256,
            'num_critics': 2,
            'kl_weight': 1.0,
            'free_nats': 3.0,
            'imagine_horizon': 15,
            'gamma': 0.99,
            'lambda_': 0.95,
            'device': 'cuda'
        }

    elif agent_type.lower() in ['bc', 'behavioral_cloning']:
        return {
            'agent_type': 'bc',
            'obs_channels': 3,
            'action_dim': 6,
            'hidden_dim': 256,
            'use_tanh': True,
            'learning_rate': 1e-4,
            'device': 'cuda'
        }

    else:
        raise ValueError(f"未知的智能体类型: {agent_type}")


if __name__ == "__main__":
    # 测试工厂函数
    import numpy as np

    print("=" * 60)
    print("测试智能体工厂")
    print("=" * 60)

    # 测试 DreamerV4
    print("\n1. 创建 DreamerV4 智能体")
    config = get_default_config('dreamer')
    config['device'] = 'cpu'  # 测试用CPU
    dreamer = make_agent('dreamer', config)
    print(f"   创建成功: {type(dreamer).__name__}")

    obs = np.random.randn(3, 64, 64).astype(np.float32)
    action = dreamer.select_action(obs)
    print(f"   动作输出: shape={action.shape}")

    # 测试 BC
    print("\n2. 创建 BC 智能体")
    config = get_default_config('bc')
    config['device'] = 'cpu'
    bc = make_agent('bc', config)
    print(f"   创建成功: {type(bc).__name__}")

    action = bc.select_action(obs)
    print(f"   动作输出: shape={action.shape}")

    # 测试保存/加载
    print("\n3. 测试保存和加载")
    save_path = "test_agent.pt"
    bc.save(save_path)
    loaded_bc = load_agent(save_path, agent_type='bc', device='cpu')
    print(f"   加载成功: {type(loaded_bc).__name__}")

    # 清理
    import os
    os.remove(save_path)
    print(f"   清理测试文件")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
