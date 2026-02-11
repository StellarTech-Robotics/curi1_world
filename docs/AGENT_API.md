# Curi1 World 智能体接口文档

## 📚 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [接口设计](#接口设计)
- [支持的智能体](#支持的智能体)
- [使用示例](#使用示例)
- [API 参考](#api-参考)

---

## 概述

Curi1 World 提供统一的智能体接口，支持：

- **强化学习**：DreamerV4（基于世界模型）
- **模仿学习**：Behavioral Cloning（监督学习）

所有智能体都继承自 `BaseAgent`，提供一致的 API。

---

## 快速开始

### 1. 创建 DreamerV4 智能体（强化学习）

```python
from src.agents import make_agent, get_default_config

# 获取默认配置
config = get_default_config('dreamer')

# 创建智能体
agent = make_agent('dreamer', config)

# 环境交互
obs = env.reset()
agent.reset()

for step in range(1000):
    action = agent.select_action(obs, deterministic=False)
    obs, reward, done, _ = env.step(action)

    if done:
        agent.reset()
        obs = env.reset()
```

### 2. 创建 BC 智能体（模仿学习）

```python
from src.agents import make_agent, get_default_config

# 获取默认配置
config = get_default_config('bc')

# 创建智能体
agent = make_agent('bc', config)

# 从专家演示中训练
for epoch in range(100):
    batch = load_demo_batch()  # 加载演示数据
    losses = agent.train_step(batch)
    print(f"Epoch {epoch}: Loss = {losses['bc_loss']:.4f}")

# 推理
obs = env.reset()
action = agent.select_action(obs)
```

---

## 接口设计

### BaseAgent 基类

所有智能体都必须实现以下接口：

```python
class BaseAgent(ABC):
    @abstractmethod
    def reset(self):
        """重置智能体状态（每个episode开始时调用）"""
        pass

    @abstractmethod
    def select_action(
        self,
        observation: np.ndarray | torch.Tensor,
        deterministic: bool = False,
        **kwargs
    ) -> np.ndarray:
        """选择动作（推理时使用）"""
        pass

    @abstractmethod
    def predict_action_chunk(
        self,
        observation: np.ndarray | torch.Tensor,
        chunk_size: int = 1,
        **kwargs
    ) -> np.ndarray:
        """预测动作序列"""
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """执行一次训练步骤"""
        pass

    def save(self, save_path: str | Path):
        """保存模型"""
        pass

    @classmethod
    def load(cls, load_path: str | Path, **kwargs):
        """加载模型"""
        pass
```

---

## 支持的智能体

### 1. DreamerV4Agent（强化学习）

基于世界模型的强化学习智能体。

**特点：**
- ✅ 在想象中学习策略
- ✅ 样本效率高
- ✅ 支持图像输入
- ✅ Actor-Critic 架构

**配置参数：**
```python
config = {
    'obs_channels': 3,          # 观察通道数
    'action_dim': 6,            # 动作维度
    'action_range': (-1.0, 1.0),# 动作范围
    'vae_latent_dim': 32,       # VAE潜在维度
    'stoch_dim': 32,            # RSSM随机维度
    'deter_dim': 256,           # RSSM确定性维度
    'imagine_horizon': 15,      # 想象轨迹长度
    'gamma': 0.99,              # 折扣因子
    'lambda_': 0.95,            # GAE lambda
    'device': 'cuda'
}
```

**训练数据格式：**
```python
batch = {
    'observations': torch.Tensor,  # [B, T, C, H, W]
    'actions': torch.Tensor,       # [B, T, action_dim]
    'rewards': torch.Tensor,       # [B, T]
    'dones': torch.Tensor          # [B, T]
}
```

### 2. BehaviorCloningAgent（模仿学习）

基于监督学习的行为克隆智能体。

**特点：**
- ✅ 从专家演示中学习
- ✅ 无需奖励信号
- ✅ 训练简单快速
- ✅ 适合有大量演示的场景

**配置参数：**
```python
config = {
    'obs_channels': 3,      # 观察通道数
    'action_dim': 6,        # 动作维度
    'hidden_dim': 256,      # 隐藏层维度
    'use_tanh': True,       # 使用tanh激活
    'learning_rate': 1e-4,  # 学习率
    'device': 'cuda'
}
```

**训练数据格式：**
```python
batch = {
    'observations': torch.Tensor,  # [B, C, H, W] 或 [B, T, C, H, W]
    'actions': torch.Tensor        # [B, action_dim] 或 [B, T, action_dim]
}
```

---

## 使用示例

### 示例 1: 完整的 RL 训练循环

```python
from src.agents import make_agent, get_default_config
from src.envs import make_env
from src.utils.replay_buffer import ReplayBuffer

# 创建环境和智能体
env = make_env('Curi1-v0')
config = get_default_config('dreamer')
agent = make_agent('dreamer', config)

# 创建回放缓冲区
buffer = ReplayBuffer(capacity=100000)

# 训练循环
for episode in range(1000):
    obs = env.reset()
    agent.reset()

    episode_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': []
    }

    # 收集一个episode
    for step in range(1000):
        action = agent.select_action(obs, deterministic=False)
        next_obs, reward, done, _ = env.step(action)

        episode_data['observations'].append(obs)
        episode_data['actions'].append(action)
        episode_data['rewards'].append(reward)
        episode_data['dones'].append(done)

        obs = next_obs

        if done:
            break

    # 添加到回放缓冲区
    buffer.add_episode(episode_data)

    # 训练
    if len(buffer) > 1000:
        batch = buffer.sample(batch_size=32, seq_len=50)
        losses = agent.train_step(batch)
        print(f"Episode {episode}: {losses}")

    # 保存检查点
    if episode % 100 == 0:
        agent.save(f'checkpoints/dreamer_ep{episode}.pt')
```

### 示例 2: 完整的 BC 训练循环

```python
from src.agents import make_agent, get_default_config
from torch.utils.data import DataLoader
from src.utils.data_loader import ObservationDataset

# 创建智能体
config = get_default_config('bc')
agent = make_agent('bc', config)

# 加载专家演示数据
dataset = ObservationDataset(data_dir='data/demonstrations')
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 训练循环
agent.train()
for epoch in range(100):
    epoch_losses = []

    for batch in dataloader:
        losses = agent.train_step(batch)
        epoch_losses.append(losses['bc_loss'])

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

    # 保存检查点
    if epoch % 10 == 0:
        agent.save(f'checkpoints/bc_epoch{epoch}.pt')
```

### 示例 3: 评估智能体

```python
from src.agents import load_agent
from src.envs import make_env

# 加载训练好的模型
agent = load_agent('checkpoints/dreamer_best.pt')
agent.eval()

# 创建环境
env = make_env('Curi1-v0')

# 评估
num_episodes = 10
episode_rewards = []

for episode in range(num_episodes):
    obs = env.reset()
    agent.reset()

    episode_reward = 0

    for step in range(1000):
        action = agent.select_action(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

        if done:
            break

    episode_rewards.append(episode_reward)
    print(f"Episode {episode}: Reward = {episode_reward:.2f}")

print(f"\nAverage Reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
```

### 示例 4: 切换智能体类型

```python
from src.agents import make_agent, get_default_config

# 先用 BC 预训练
bc_config = get_default_config('bc')
bc_agent = make_agent('bc', bc_config)

# ... BC 训练 ...

# 保存 BC 模型
bc_agent.save('pretrained_bc.pt')

# 切换到 RL 微调
rl_config = get_default_config('dreamer')
rl_agent = make_agent('dreamer', rl_config)

# 可选：加载 BC 的观察编码器（迁移学习）
# rl_agent.model.vae.encoder.load_state_dict(bc_agent.policy.encoder.state_dict())

# ... RL 训练 ...
```

---

## API 参考

### 工厂函数

#### `make_agent(agent_type: str, config: dict) -> BaseAgent`

创建智能体实例。

**参数：**
- `agent_type`: 智能体类型
  - `'dreamer'` 或 `'dreamerv4'`: DreamerV4
  - `'bc'` 或 `'behavioral_cloning'`: BC
- `config`: 配置字典

**返回：**
- `agent`: 智能体实例

#### `load_agent(checkpoint_path: str, agent_type: str = None, **kwargs) -> BaseAgent`

从检查点加载智能体。

**参数：**
- `checkpoint_path`: 检查点路径
- `agent_type`: 智能体类型（可选）
- `**kwargs`: 覆盖配置参数

**返回：**
- `agent`: 加载的智能体实例

#### `get_default_config(agent_type: str) -> dict`

获取默认配置。

**参数：**
- `agent_type`: 智能体类型

**返回：**
- `config`: 默认配置字典

---

## 与其他框架的对比

### vs LeRobot

| 特性 | **Curi1 World** | **LeRobot** |
|------|----------------|-------------|
| **RL 算法** | DreamerV4 | SAC, TDMPC, SARM |
| **IL 算法** | BC | ACT, Diffusion Policy |
| **世界模型** | ✅ RSSM | ✅ TDMPC |
| **接口设计** | BaseAgent | PreTrainedPolicy |
| **模型分发** | 本地 | HuggingFace Hub |

### vs UnifoLM-WMA

| 特性 | **Curi1 World** | **UnifoLM-WMA** |
|------|----------------|-----------------|
| **学习范式** | RL + IL | IL only |
| **世界模型** | RSSM | Video Diffusion |
| **动作生成** | Actor-Critic | Diffusion Policy |
| **数据需求** | 中等 | 大量 |

---

## 常见问题

### Q1: 如何选择使用 RL 还是 IL？

**使用 RL (DreamerV4) 当：**
- ✅ 有明确的奖励信号
- ✅ 需要探索和优化
- ✅ 想超越演示数据

**使用 IL (BC) 当：**
- ✅ 有大量高质量演示
- ✅ 没有奖励信号
- ✅ 需要快速部署

### Q2: 如何迁移到真实机器人？

参考 `src/envs/robot_env.py` 中的 TODO 部分，实现真实机器人接口：

```python
def _init_robot(self):
    # TODO: 初始化真实机器人连接
    # 例如：使用 ROS, SDK 等
    pass

def _get_camera_image(self):
    # TODO: 获取真实相机图像
    pass
```

### Q3: 如何调试模型？

使用评估脚本：

```bash
python scripts/evaluate.py --checkpoint checkpoints/model.pt --render
```

---

## 更多资源

- [完整示例](../examples/agent_usage.py)
- [训练脚本](../scripts/train_dreamer.py)
- [评估脚本](../scripts/evaluate.py)
- [环境接口](../src/envs/robot_env.py)
