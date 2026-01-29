"""
DreamerV4 主训练脚本
端到端训练世界模型和策略
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from collections import deque

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.dreamer import DreamerV4
from src.utils.replay_buffer import ReplayBuffer
from src.utils.logger import Logger
from src.envs.robot_env import RobotEnv


def parse_args():
    parser = argparse.ArgumentParser(description='训练 DreamerV4')
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                        help='配置文件路径')
    parser.add_argument('--env_name', type=str, default='Curi1-v0',
                        help='环境名称')
    parser.add_argument('--exp_name', type=str, default='dreamer_v4',
                        help='实验名称')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def collect_episode(env, agent, replay_buffer, max_steps=1000, random=False):
    """
    收集一个 episode 的数据

    Args:
        env: 环境
        agent: DreamerV4 agent
        replay_buffer: 回放缓冲区
        max_steps: 最大步数
        random: 是否使用随机动作

    Returns:
        episode_info: episode 信息字典
    """
    observations = []
    actions = []
    rewards = []
    dones = []

    obs = env.reset()
    state = None
    episode_reward = 0.0
    episode_length = 0

    for step in range(max_steps):
        if random:
            # 随机动作（探索）
            action = env.action_space.sample()
        else:
            # 使用策略选择动作
            obs_tensor = torch.from_numpy(obs).float()
            action, state = agent.select_action(obs_tensor, state, deterministic=False)

        # 执行动作
        next_obs, reward, done, info = env.step(action)

        # 存储
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        episode_reward += reward
        episode_length += 1

        obs = next_obs

        if done:
            break

    # 添加到回放缓冲区
    episode_data = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones)
    }
    replay_buffer.add_episode(episode_data)

    return {
        'episode_reward': episode_reward,
        'episode_length': episode_length
    }


def train_step(agent, replay_buffer, batch_size, seq_len, imagine_horizon, gamma, lambda_):
    """
    执行一次训练步骤

    Args:
        agent: DreamerV4 agent
        replay_buffer: 回放缓冲区
        batch_size: 批量大小
        seq_len: 序列长度
        imagine_horizon: 想象轨迹长度
        gamma: 折扣因子
        lambda_: GAE lambda

    Returns:
        losses: 损失字典
    """
    # 从回放缓冲区采样
    batch = replay_buffer.sample(batch_size, seq_len)

    observations = torch.from_numpy(batch['observations']).float().to(agent.device)
    actions = torch.from_numpy(batch['actions']).float().to(agent.device)
    rewards = torch.from_numpy(batch['rewards']).float().to(agent.device)
    dones = torch.from_numpy(batch['dones']).float().to(agent.device)

    # 1. 训练世界模型
    wm_losses = agent.compute_world_model_loss(observations, actions, rewards, dones)

    agent.world_model_optimizer.zero_grad()
    wm_losses['total_loss'].backward()
    torch.nn.utils.clip_grad_norm_(
        list(agent.vae.parameters()) + list(agent.rssm.parameters()),
        max_norm=100.0
    )
    agent.world_model_optimizer.step()

    # 2. 训练 Actor
    # 使用世界模型学到的状态
    h_seq, z_seq = wm_losses['states']
    batch_size_actual, seq_len_actual = h_seq.shape[:2]

    # 随机选择起始状态
    indices = torch.randint(0, seq_len_actual, (batch_size_actual,))
    h_start = h_seq[torch.arange(batch_size_actual), indices]
    z_start = z_seq[torch.arange(batch_size_actual), indices]
    states_start = torch.cat([h_start, z_start], dim=-1).detach()

    actor_losses = agent.compute_actor_loss(
        states_start,
        imagine_horizon=imagine_horizon,
        gamma=gamma,
        lambda_=lambda_
    )

    agent.actor_optimizer.zero_grad()
    actor_losses['actor_loss'].backward()
    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=100.0)
    agent.actor_optimizer.step()

    # 3. 训练 Critic
    critic_losses = agent.compute_critic_loss(
        states_start.detach(),
        imagine_horizon=imagine_horizon,
        gamma=gamma,
        lambda_=lambda_
    )

    agent.critic_optimizer.zero_grad()
    critic_losses['critic_loss'].backward()
    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=100.0)
    agent.critic_optimizer.step()

    # 返回所有损失
    return {
        **{f'wm_{k}': v.item() if torch.is_tensor(v) else v for k, v in wm_losses.items() if k != 'states'},
        **{f'actor_{k}': v.item() if torch.is_tensor(v) else v for k, v in actor_losses.items()},
        **{f'critic_{k}': v.item() if torch.is_tensor(v) else v for k, v in critic_losses.items()}
    }


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"配置文件 {args.config} 不存在，使用默认配置")
        config = {
            'env': {'name': 'Curi1-v0'},
            'model': {
                'vae_latent_dim': 32,
                'stoch_dim': 32,
                'deter_dim': 256,
                'action_dim': 6
            },
            'training': {
                'num_episodes': 10000,
                'prefill_episodes': 5,
                'batch_size': 16,
                'seq_len': 50,
                'imagine_horizon': 15,
                'train_steps_per_episode': 100,
                'gamma': 0.99,
                'lambda': 0.95,
                'learning_rate': 3e-4,
                'save_interval': 100
            }
        }

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

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
        print("使用模拟环境配置")
        obs_shape = (3, 64, 64)
        action_dim = 6

    # 创建模型
    agent = DreamerV4(
        obs_channels=obs_shape[0] if len(obs_shape) == 3 else 3,
        action_dim=action_dim,
        vae_latent_dim=config['model']['vae_latent_dim'],
        stoch_dim=config['model']['stoch_dim'],
        deter_dim=config['model']['deter_dim'],
        device=device
    )

    # 创建优化器
    agent.world_model_optimizer = optim.Adam(
        list(agent.vae.parameters()) + list(agent.rssm.parameters()),
        lr=config['training']['learning_rate']
    )
    agent.actor_optimizer = optim.Adam(
        agent.actor.parameters(),
        lr=config['training']['learning_rate']
    )
    agent.critic_optimizer = optim.Adam(
        agent.critic.parameters(),
        lr=config['training']['learning_rate']
    )

    print(f"模型参数量: {sum(p.numel() for p in agent.parameters()):,}")

    # 创建回放缓冲区
    replay_buffer = ReplayBuffer(capacity=100000)

    # 创建日志目录
    log_dir = project_root / 'experiments' / 'logs' / args.exp_name
    checkpoint_dir = project_root / 'experiments' / 'checkpoints' / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir)

    # 恢复训练
    start_episode = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        agent.world_model_optimizer.load_state_dict(checkpoint['wm_optimizer_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        start_episode = checkpoint['episode'] + 1
        print(f"从 episode {start_episode} 恢复训练")

    # 预填充回放缓冲区
    print(f"\n预填充回放缓冲区 ({config['training']['prefill_episodes']} episodes)...")
    try:
        for i in range(config['training']['prefill_episodes']):
            episode_info = collect_episode(env, agent, replay_buffer, random=True)
            print(f"Episode {i + 1}: Reward={episode_info['episode_reward']:.2f}, Length={episode_info['episode_length']}")
    except Exception as e:
        print(f"数据收集失败: {e}")
        print("跳过预填充步骤")

    # 训练循环
    num_episodes = config['training']['num_episodes']
    recent_rewards = deque(maxlen=100)

    print(f"\n开始训练...")
    for episode in range(start_episode, num_episodes):
        # 收集数据
        try:
            episode_info = collect_episode(env, agent, replay_buffer, random=False)
            recent_rewards.append(episode_info['episode_reward'])

            # 记录 episode 信息
            writer.add_scalar('Episode/Reward', episode_info['episode_reward'], episode)
            writer.add_scalar('Episode/Length', episode_info['episode_length'], episode)
            writer.add_scalar('Episode/Reward_Mean100', np.mean(recent_rewards), episode)
        except Exception as e:
            print(f"Episode {episode}: 数据收集失败 - {e}")
            continue

        # 训练步骤
        if len(replay_buffer) >= config['training']['batch_size']:
            for _ in range(config['training']['train_steps_per_episode']):
                try:
                    losses = train_step(
                        agent,
                        replay_buffer,
                        batch_size=config['training']['batch_size'],
                        seq_len=config['training']['seq_len'],
                        imagine_horizon=config['training']['imagine_horizon'],
                        gamma=config['training']['gamma'],
                        lambda_=config['training']['lambda']
                    )

                    # 记录损失（每个训练步骤）
                    for k, v in losses.items():
                        writer.add_scalar(f'Loss/{k}', v, episode * config['training']['train_steps_per_episode'] + _)

                except Exception as e:
                    print(f"训练步骤失败: {e}")
                    continue

        # 打印信息
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward (100): {avg_reward:.2f}, "
                  f"Buffer Size: {len(replay_buffer)}")

        # 保存检查点
        if (episode + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_episode_{episode + 1}.pt'
            torch.save({
                'episode': episode,
                'agent_state_dict': agent.state_dict(),
                'wm_optimizer_state_dict': agent.world_model_optimizer.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                'config': config
            }, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")

    writer.close()
    print("\n训练完成！")


if __name__ == '__main__':
    main()
