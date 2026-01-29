"""
日志记录器
用于训练过程的日志记录和可视化
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime


class Logger:
    """
    简单的日志记录器
    记录训练指标到文件和控制台
    """

    def __init__(
        self,
        log_dir: str,
        exp_name: str = "experiment"
    ):
        """
        Args:
            log_dir: 日志目录
            exp_name: 实验名称
        """
        self.log_dir = Path(log_dir) / exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"train_{timestamp}.log"

        # 指标存储
        self.metrics = {}

        # 写入初始日志
        self.log(f"实验名称: {exp_name}")
        self.log(f"日志目录: {self.log_dir}")
        self.log(f"开始时间: {timestamp}")
        self.log("-" * 60)

    def log(self, message: str, print_console: bool = True):
        """
        记录消息

        Args:
            message: 消息内容
            print_console: 是否打印到控制台
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        # 写入文件
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

        # 打印到控制台
        if print_console:
            print(log_message)

    def log_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
        prefix: str = ""
    ):
        """
        记录指标

        Args:
            step: 步数或 episode 数
            metrics: 指标字典
            prefix: 指标前缀
        """
        # 存储指标
        for key, value in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key

            if full_key not in self.metrics:
                self.metrics[full_key] = []

            self.metrics[full_key].append((step, value))

        # 记录到日志文件
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.log(f"Step {step} - {prefix}: {metrics_str}")

    def save_metrics(self, filename: str = "metrics.json"):
        """
        保存所有指标到 JSON 文件

        Args:
            filename: 文件名
        """
        save_path = self.log_dir / filename

        # 转换为可序列化的格式
        serializable_metrics = {}
        for key, values in self.metrics.items():
            serializable_metrics[key] = {
                'steps': [v[0] for v in values],
                'values': [v[1] for v in values]
            }

        with open(save_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        self.log(f"指标已保存到: {save_path}")

    def get_metric_history(self, metric_name: str):
        """
        获取指标历史

        Args:
            metric_name: 指标名称

        Returns:
            steps, values: 步数列表和值列表
        """
        if metric_name not in self.metrics:
            return [], []

        history = self.metrics[metric_name]
        steps = [h[0] for h in history]
        values = [h[1] for h in history]

        return steps, values


class WandbLogger(Logger):
    """
    Weights & Biases 日志记录器
    继承基础 Logger 并添加 W&B 支持
    """

    def __init__(
        self,
        log_dir: str,
        exp_name: str = "experiment",
        project_name: str = "curi1_world_model",
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            log_dir: 日志目录
            exp_name: 实验名称
            project_name: W&B 项目名称
            entity: W&B 实体名称
            config: 配置字典
        """
        super().__init__(log_dir, exp_name)

        # 导入 wandb
        try:
            import wandb
            self.wandb = wandb

            # 初始化 wandb
            self.wandb.init(
                project=project_name,
                entity=entity,
                name=exp_name,
                config=config,
                dir=str(self.log_dir)
            )

            self.use_wandb = True
            self.log("W&B 已启用")

        except ImportError:
            self.use_wandb = False
            self.log("警告: wandb 未安装，将仅使用本地日志")

    def log_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
        prefix: str = ""
    ):
        """
        记录指标（同时记录到本地和 W&B）

        Args:
            step: 步数
            metrics: 指标字典
            prefix: 指标前缀
        """
        # 本地记录
        super().log_metrics(step, metrics, prefix)

        # W&B 记录
        if self.use_wandb:
            wandb_metrics = {}
            for key, value in metrics.items():
                full_key = f"{prefix}/{key}" if prefix else key
                wandb_metrics[full_key] = value

            self.wandb.log(wandb_metrics, step=step)

    def log_video(
        self,
        video: np.ndarray,
        name: str = "video",
        step: Optional[int] = None,
        fps: int = 30
    ):
        """
        记录视频到 W&B

        Args:
            video: 视频数据 [T, H, W, C]
            name: 视频名称
            step: 步数
            fps: 帧率
        """
        if not self.use_wandb:
            return

        self.wandb.log({
            name: self.wandb.Video(video, fps=fps, format="mp4")
        }, step=step)

    def log_image(
        self,
        image: np.ndarray,
        name: str = "image",
        step: Optional[int] = None
    ):
        """
        记录图像到 W&B

        Args:
            image: 图像数据 [H, W, C] 或 [C, H, W]
            name: 图像名称
            step: 步数
        """
        if not self.use_wandb:
            return

        # 转换为 [H, W, C] 格式
        if image.shape[0] in [1, 3]:  # [C, H, W]
            image = np.transpose(image, (1, 2, 0))

        self.wandb.log({
            name: self.wandb.Image(image)
        }, step=step)

    def finish(self):
        """完成日志记录"""
        if self.use_wandb:
            self.wandb.finish()

        self.log("日志记录完成")


if __name__ == "__main__":
    # 测试代码
    import tempfile

    # 创建临时日志目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 测试基础 Logger
        logger = Logger(tmpdir, exp_name="test_exp")

        # 记录一些指标
        for step in range(10):
            metrics = {
                'loss': np.random.rand(),
                'reward': np.random.rand() * 100
            }
            logger.log_metrics(step, metrics, prefix="train")

        # 保存指标
        logger.save_metrics()

        print("\n基础 Logger 测试完成")

        # 获取指标历史
        steps, values = logger.get_metric_history("train/loss")
        print(f"Loss 历史: {len(values)} 个值")
