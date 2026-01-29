# 数据目录

此目录用于存储训练数据和处理后的数据集。

## 目录结构

```
data/
├── raw/                  # 原始数据
│   ├── train/           # 训练集
│   ├── val/             # 验证集
│   └── test/            # 测试集
└── processed/           # 处理后的数据
    ├── train/
    ├── val/
    └── test/
```

## 数据格式

### Episode 数据格式

每个 episode 保存为 `.pkl` 文件，包含以下字段：

```python
{
    'observations': np.ndarray,  # [T, C, H, W] 或 [T, state_dim]
    'actions': np.ndarray,       # [T, action_dim]
    'rewards': np.ndarray,       # [T]
    'dones': np.ndarray          # [T]
}
```

### 图像数据格式

- 格式：PNG, JPG
- 尺寸：64x64 或 128x128（可配置）
- 通道：RGB (3 channels)
- 数值范围：0-255 (uint8)

## 数据收集

使用以下脚本收集机器人交互数据：

```python
from src.envs.robot_env import RobotEnv
from src.utils.data_loader import save_episode

env = RobotEnv(env_name="Curi1-v0")

for episode_idx in range(num_episodes):
    observations = []
    actions = []
    rewards = []
    dones = []

    obs = env.reset()
    done = False

    while not done:
        action = policy.select_action(obs)  # 或随机动作
        next_obs, reward, done, info = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        obs = next_obs

    # 保存 episode
    episode_data = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones)
    }

    save_episode(episode_data, 'data/raw/train', episode_idx)
```

## 数据预处理

数据预处理步骤（如果需要）：

1. 图像归一化：除以 255.0
2. 图像增强：随机裁剪、颜色抖动等
3. 序列分割：将长 episode 分割为固定长度序列

## 注意事项

- 确保有足够的磁盘空间存储数据
- 定期备份重要数据
- 使用 `.gitignore` 排除大型数据文件
- 考虑使用数据压缩以节省空间
