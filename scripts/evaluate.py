"""
评估脚本
用于评估训练好的 DreamerV4 模型
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.dreamer import DreamerV4
from src.envs.robot_env import RobotEnv


def parse_args():
    parser = argparse.ArgumentParser(description='评估 DreamerV4')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--env_name', type=str, default='Curi1-v0',
                        help='环境名称')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='评估 episode 数量')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--render', action='store_true',
                        help='是否渲染环境')
    parser.add_argument('--save_video', action='store_true',
                        help='是否保存视频')
    parser.add_argument('--deterministic', action='store_true',
                        help='使用确定性策略')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    return parser.parse_args()


def evaluate_episode(env, agent, max_steps=1000, render=False, deterministic=True):
    """
    评估一个 episode

    Args:
        env: 环境
        agent: DreamerV4 agent
        max_steps: 最大步数
        render: 是否渲染
        deterministic: 是否使用确定性策略

    Returns:
        episode_info: episode 信息字典
    """
    observations = []
    actions = []
    rewards = []

    obs = env.reset()
    state = None
    episode_reward = 0.0
    episode_length = 0

    for step in range(max_steps):
        if render:
            env.render()

        # 选择动作
        obs_tensor = torch.from_numpy(obs).float()
        action, state = agent.select_action(obs_tensor, state, deterministic=deterministic)

        # 执行动作
        next_obs, reward, done, info = env.step(action)

        # 存储
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

        episode_reward += reward
        episode_length += 1

        obs = next_obs

        if done:
            break

    return {
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards)
    }


def plot_episode_statistics(all_rewards, save_path=None):
    """
    绘制 episode 统计图

    Args:
        all_rewards: 所有 episode 的奖励列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Episode 奖励
    axes[0].plot(all_rewards, marker='o')
    axes[0].axhline(y=np.mean(all_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(all_rewards):.2f}')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True)

    # 奖励分布
    axes[1].hist(all_rewards, bins=20, edgecolor='black')
    axes[1].axvline(x=np.mean(all_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(all_rewards):.2f}')
    axes[1].set_xlabel('Reward')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Reward Distribution')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"统计图已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载检查点
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 获取配置
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    # 创建环境
    try:
        env = RobotEnv(env_name=args.env_name)
        obs_shape = env.observation_space.shape
        action_dim = env.action_space.shape[0]
        print(f"环境: {args.env_name}")
        print(f"观察空间: {obs_shape}")
        print(f"动作空间: {action_dim}")
    except Exception as e:
        print(f"环境创建失败: {e}")
        return

    # 创建模型
    agent = DreamerV4(
        obs_channels=obs_shape[0] if len(obs_shape) == 3 else 3,
        action_dim=action_dim,
        vae_latent_dim=model_config.get('vae_latent_dim', 32),
        stoch_dim=model_config.get('stoch_dim', 32),
        deter_dim=model_config.get('deter_dim', 256),
        device=device
    )

    # 加载模型权重
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.eval()

    print(f"模型已加载，开始评估...")
    print(f"评估 {args.num_episodes} 个 episodes")
    print(f"策略类型: {'确定性' if args.deterministic else '随机性'}")

    # 评估
    all_rewards = []
    all_lengths = []

    for episode in tqdm(range(args.num_episodes), desc='评估'):
        episode_info = evaluate_episode(
            env,
            agent,
            render=args.render,
            deterministic=args.deterministic
        )

        all_rewards.append(episode_info['episode_reward'])
        all_lengths.append(episode_info['episode_length'])

        print(f"Episode {episode + 1}: Reward={episode_info['episode_reward']:.2f}, "
              f"Length={episode_info['episode_length']}")

    # 统计信息
    print("\n" + "=" * 60)
    print("评估结果统计:")
    print("=" * 60)
    print(f"平均奖励: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"最大奖励: {np.max(all_rewards):.2f}")
    print(f"最小奖励: {np.min(all_rewards):.2f}")
    print(f"平均长度: {np.mean(all_lengths):.2f} ± {np.std(all_lengths):.2f}")
    print("=" * 60)

    # 绘制统计图
    results_dir = project_root / 'experiments' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = Path(args.checkpoint).stem
    plot_path = results_dir / f'{checkpoint_name}_evaluation.png'
    plot_episode_statistics(all_rewards, save_path=plot_path)

    # 保存结果
    results_file = results_dir / f'{checkpoint_name}_results.txt'
    with open(results_file, 'w') as f:
        f.write("评估结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"检查点: {args.checkpoint}\n")
        f.write(f"环境: {args.env_name}\n")
        f.write(f"Episodes: {args.num_episodes}\n")
        f.write(f"策略: {'确定性' if args.deterministic else '随机性'}\n")
        f.write("=" * 60 + "\n")
        f.write(f"平均奖励: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}\n")
        f.write(f"最大奖励: {np.max(all_rewards):.2f}\n")
        f.write(f"最小奖励: {np.min(all_rewards):.2f}\n")
        f.write(f"平均长度: {np.mean(all_lengths):.2f} ± {np.std(all_lengths):.2f}\n")
        f.write("=" * 60 + "\n")
        f.write("\n各 Episode 详细结果:\n")
        for i, (reward, length) in enumerate(zip(all_rewards, all_lengths)):
            f.write(f"Episode {i + 1}: Reward={reward:.2f}, Length={length}\n")

    print(f"\n结果已保存到: {results_file}")

    env.close()


if __name__ == '__main__':
    main()
