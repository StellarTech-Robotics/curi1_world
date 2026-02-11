"""
智能体使用示例
展示如何使用 curi1_world 的统一接口
"""

import numpy as np
import torch
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents import (
    make_agent,
    load_agent,
    get_default_config,
    AGENT_REGISTRY
)


def example_1_create_dreamer_agent():
    """示例1: 创建 DreamerV4 强化学习智能体"""
    print("\n" + "=" * 60)
    print("示例 1: DreamerV4 强化学习智能体")
    print("=" * 60)

    # 获取默认配置
    config = get_default_config('dreamer')
    config['device'] = 'cpu'  # 使用CPU测试

    # 创建智能体
    agent = make_agent('dreamer', config)
    print(f"✓ 创建成功: {type(agent).__name__}")

    # 模拟环境交互
    obs = np.random.randn(3, 64, 64).astype(np.float32)

    # 重置状态
    agent.reset()

    # 选择动作（随机策略）
    action = agent.select_action(obs, deterministic=False)
    print(f"✓ 随机动作: {action}")

    # 选择动作（确定性策略）
    action = agent.select_action(obs, deterministic=True)
    print(f"✓ 确定性动作: {action}")

    # 预测动作序列
    actions = agent.predict_action_chunk(obs, chunk_size=5)
    print(f"✓ 动作序列: shape={actions.shape}")

    # 保存模型
    save_path = "dreamer_test.pt"
    agent.save(save_path)
    print(f"✓ 模型已保存到: {save_path}")

    # 清理
    Path(save_path).unlink()


def example_2_create_bc_agent():
    """示例2: 创建行为克隆智能体"""
    print("\n" + "=" * 60)
    print("示例 2: 行为克隆（模仿学习）智能体")
    print("=" * 60)

    # 获取默认配置
    config = get_default_config('bc')
    config['device'] = 'cpu'

    # 创建智能体
    agent = make_agent('bc', config)
    print(f"✓ 创建成功: {type(agent).__name__}")

    # 模拟观察
    obs = np.random.randn(3, 64, 64).astype(np.float32)

    # 选择动作（BC总是确定性的）
    action = agent.select_action(obs)
    print(f"✓ 预测动作: {action}")

    # 模拟训练
    batch = {
        'observations': torch.randn(32, 3, 64, 64),
        'actions': torch.randn(32, 6)
    }

    agent.train()
    losses = agent.train_step(batch)
    print(f"✓ 训练损失: {losses}")


def example_3_training_loop_dreamer():
    """示例3: DreamerV4 训练循环"""
    print("\n" + "=" * 60)
    print("示例 3: DreamerV4 训练循环（伪代码）")
    print("=" * 60)

    config = get_default_config('dreamer')
    config['device'] = 'cpu'

    agent = make_agent('dreamer', config)
    agent.train()

    # 模拟训练数据
    batch = {
        'observations': torch.randn(4, 10, 3, 64, 64),  # [B, T, C, H, W]
        'actions': torch.randn(4, 10, 6),               # [B, T, action_dim]
        'rewards': torch.randn(4, 10),                  # [B, T]
        'dones': torch.zeros(4, 10)                     # [B, T]
    }

    print("开始训练...")
    for epoch in range(3):
        losses = agent.train_step(batch)
        print(f"  Epoch {epoch + 1}: {losses}")

    print("✓ 训练完成")


def example_4_training_loop_bc():
    """示例4: BC 训练循环"""
    print("\n" + "=" * 60)
    print("示例 4: 行为克隆训练循环")
    print("=" * 60)

    config = get_default_config('bc')
    config['device'] = 'cpu'

    agent = make_agent('bc', config)
    agent.train()

    # 模拟专家演示数据
    demo_data = {
        'observations': torch.randn(64, 3, 64, 64),  # [B, C, H, W]
        'actions': torch.randn(64, 6)                # [B, action_dim]
    }

    print("从专家演示中学习...")
    for epoch in range(3):
        losses = agent.train_step(demo_data)
        print(f"  Epoch {epoch + 1}: BC Loss = {losses['bc_loss']:.4f}")

    print("✓ 训练完成")


def example_5_save_and_load():
    """示例5: 保存和加载模型"""
    print("\n" + "=" * 60)
    print("示例 5: 保存和加载模型")
    print("=" * 60)

    # 创建并保存 BC 智能体
    config = get_default_config('bc')
    config['device'] = 'cpu'

    agent = make_agent('bc', config)
    save_path = "bc_model.pt"
    agent.save(save_path)
    print(f"✓ 模型已保存到: {save_path}")

    # 加载模型
    loaded_agent = load_agent(save_path, agent_type='bc', device='cpu')
    print(f"✓ 模型已加载: {type(loaded_agent).__name__}")

    # 测试加载的模型
    obs = np.random.randn(3, 64, 64).astype(np.float32)
    action = loaded_agent.select_action(obs)
    print(f"✓ 加载的模型预测: {action}")

    # 清理
    Path(save_path).unlink()


def example_6_registry():
    """示例6: 查看所有可用的智能体"""
    print("\n" + "=" * 60)
    print("示例 6: 查看所有可用的智能体类型")
    print("=" * 60)

    print("可用的智能体类型:")
    for agent_type, agent_class in AGENT_REGISTRY.items():
        print(f"  - '{agent_type}': {agent_class.__name__}")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Curi1 World 智能体接口使用示例")
    print("=" * 60)

    try:
        example_1_create_dreamer_agent()
        example_2_create_bc_agent()
        example_3_training_loop_dreamer()
        example_4_training_loop_bc()
        example_5_save_and_load()
        example_6_registry()

        print("\n" + "=" * 60)
        print("✓ 所有示例运行成功！")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
