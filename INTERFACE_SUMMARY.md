# Curi1 World 智能体接口总结

## 🎯 接口位置

### **主要接口文件**

| 文件路径 | 说明 | 角色 |
|---------|------|------|
| [src/agents/base.py](src/agents/base.py) | 抽象基类 | 定义统一API |
| [src/agents/dreamer_agent.py](src/agents/dreamer_agent.py) | DreamerV4实现 | **强化学习** |
| [src/agents/bc_agent.py](src/agents/bc_agent.py) | BC实现 | **模仿学习** |
| [src/agents/factory.py](src/agents/factory.py) | 工厂函数 | 创建和加载 |

---

## 📦 快速使用

### **1. 强化学习（DreamerV4）**

```python
from src.agents import make_agent, get_default_config

# 创建智能体
config = get_default_config('dreamer')
agent = make_agent('dreamer', config)

# 环境交互
obs = env.reset()
agent.reset()  # 重置RSSM状态

for step in range(1000):
    # 选择动作
    action = agent.select_action(obs, deterministic=False)
    obs, reward, done, _ = env.step(action)

    if done:
        agent.reset()
        obs = env.reset()

# 训练
batch = {
    'observations': torch.Tensor,  # [B, T, C, H, W]
    'actions': torch.Tensor,       # [B, T, action_dim]
    'rewards': torch.Tensor,       # [B, T]
    'dones': torch.Tensor          # [B, T]
}
losses = agent.train_step(batch)
```

### **2. 模仿学习（BC）**

```python
from src.agents import make_agent, get_default_config

# 创建智能体
config = get_default_config('bc')
agent = make_agent('bc', config)

# 从专家演示训练
demo_batch = {
    'observations': torch.Tensor,  # [B, C, H, W]
    'actions': torch.Tensor        # [B, action_dim]
}
losses = agent.train_step(demo_batch)

# 推理
obs = env.reset()
action = agent.select_action(obs, deterministic=True)
```

---

## 🔌 统一接口

### **BaseAgent 基类**

所有智能体都必须实现以下方法：

```python
class BaseAgent(ABC):
    # === 核心方法 ===
    def reset(self):
        """重置智能体状态"""

    def select_action(
        self,
        observation: np.ndarray | torch.Tensor,
        deterministic: bool = False
    ) -> np.ndarray:
        """选择动作 [action_dim]"""

    def predict_action_chunk(
        self,
        observation: np.ndarray | torch.Tensor,
        chunk_size: int = 1
    ) -> np.ndarray:
        """预测动作序列 [chunk_size, action_dim]"""

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """训练步骤，返回损失字典"""

    # === 模型管理 ===
    def save(self, save_path: str):
        """保存检查点"""

    @classmethod
    def load(cls, load_path: str, **kwargs):
        """加载检查点"""

    def state_dict(self) -> Dict:
        """获取状态字典"""

    def load_state_dict(self, state_dict: Dict):
        """加载状态字典"""

    def eval(self):
        """评估模式"""

    def train(self):
        """训练模式"""
```

---

## 🆚 两种智能体对比

| 维度 | **DreamerV4Agent** | **BehaviorCloningAgent** |
|------|-------------------|-------------------------|
| **学习范式** | 强化学习 | 监督学习（模仿学习） |
| **数据需求** | 环境交互数据 + 奖励 | 专家演示数据 |
| **训练信号** | Reward | MSE损失 |
| **探索** | ✅ 支持 | ❌ 无探索 |
| **样本效率** | 高（模型辅助） | 中等 |
| **泛化能力** | 可超越演示 | 受限于演示 |
| **训练复杂度** | 高 | 低 |
| **推理速度** | 快 | 快 |
| **状态管理** | 有状态（RSSM） | 无状态 |

---

## 🏗️ 接口设计理念

### **1. 受 LeRobot 启发**

```python
# LeRobot风格
from lerobot.policies.factory import make_policy

policy = make_policy('sac', config)
action = policy.select_action(obs)

# Curi1 World风格
from src.agents import make_agent

agent = make_agent('dreamer', config)
action = agent.select_action(obs)
```

**关键相似点：**
- ✅ 工厂模式创建
- ✅ 统一的 `select_action` 接口
- ✅ 支持保存/加载
- ✅ 配置驱动

### **2. 支持两种学习范式**

```python
# RL模式：需要reward
rl_agent = make_agent('dreamer', config)
batch_rl = {
    'observations': ...,
    'actions': ...,
    'rewards': ...,  # RL需要reward
    'dones': ...
}
losses = rl_agent.train_step(batch_rl)

# IL模式：只需要(obs, action)对
il_agent = make_agent('bc', config)
batch_il = {
    'observations': ...,
    'actions': ...  # BC只需要演示
}
losses = il_agent.train_step(batch_il)
```

### **3. 灵活的观察格式**

```python
# 支持多种格式
obs_formats = [
    np.ndarray,  # NumPy数组
    torch.Tensor,  # PyTorch张量
    (C, H, W),  # 通道优先
    (H, W, C),  # 通道后置（自动转换）
]

# 自动处理
action = agent.select_action(obs_any_format)
```

---

## 📂 项目结构

```
curi1_world/
├── src/
│   ├── agents/              # 🎯 智能体接口（核心）
│   │   ├── __init__.py      # 导出所有接口
│   │   ├── base.py          # BaseAgent 抽象基类
│   │   ├── dreamer_agent.py # DreamerV4 RL实现
│   │   ├── bc_agent.py      # BC IL实现
│   │   └── factory.py       # 工厂函数
│   │
│   ├── models/              # 模型实现
│   │   ├── dreamer.py       # DreamerV4核心
│   │   ├── vae.py           # VAE编码器
│   │   ├── rnn.py           # RSSM世界模型
│   │   └── controller.py    # Actor-Critic
│   │
│   ├── envs/                # 环境接口
│   │   └── robot_env.py     # Curi1机器人环境
│   │
│   └── utils/               # 工具模块
│       ├── replay_buffer.py
│       ├── data_loader.py
│       └── logger.py
│
├── scripts/                 # 训练脚本
│   ├── train_dreamer.py     # RL训练
│   └── evaluate.py          # 评估
│
├── examples/                # 使用示例
│   └── agent_usage.py       # 完整示例
│
├── docs/                    # 文档
│   └── AGENT_API.md         # 详细API文档
│
└── configs/                 # 配置文件
    ├── default.yaml
    ├── train.yaml
    └── eval.yaml
```

---

## 🎓 使用建议

### **什么时候用 DreamerV4？**

✅ **推荐场景：**
- 有明确的奖励信号
- 需要探索和优化
- 样本效率很重要
- 想超越人类演示

❌ **不推荐场景：**
- 没有奖励信号
- 已有大量高质量演示
- 需要极快速部署

### **什么时候用 BC？**

✅ **推荐场景：**
- 有大量高质量演示
- 没有明确奖励
- 需要快速部署
- 任务相对简单

❌ **不推荐场景：**
- 演示数据不足
- 需要探索新策略
- 任务分布变化大

### **组合使用策略**

```python
# 阶段1: BC预训练
bc_agent = make_agent('bc', bc_config)
# ... 从演示中学习 ...
bc_agent.save('pretrained_bc.pt')

# 阶段2: RL微调
rl_agent = make_agent('dreamer', rl_config)
# 可选: 迁移BC的观察编码器
# rl_agent.model.vae.encoder.load_state_dict(
#     bc_agent.policy.encoder.state_dict()
# )
# ... RL训练 ...
```

---

## 📚 更多资源

- **完整示例**: [examples/agent_usage.py](examples/agent_usage.py)
- **详细API文档**: [docs/AGENT_API.md](docs/AGENT_API.md)
- **训练脚本**: [scripts/train_dreamer.py](scripts/train_dreamer.py)
- **评估脚本**: [scripts/evaluate.py](scripts/evaluate.py)

---

## 🔗 相关项目对比

| 项目 | 接口设计 | RL支持 | IL支持 |
|------|---------|--------|--------|
| **Curi1 World** | BaseAgent | DreamerV4 | BC |
| **LeRobot** | PreTrainedPolicy | SAC, TDMPC | ACT, Diffusion |
| **UnifoLM-WMA** | 无统一接口 | ❌ | Diffusion Policy |

**Curi1 World 的优势：**
- ✅ 统一的 RL + IL 接口
- ✅ 基于世界模型的高效RL
- ✅ 简单易用的 BC 实现
- ✅ 工厂模式灵活创建
