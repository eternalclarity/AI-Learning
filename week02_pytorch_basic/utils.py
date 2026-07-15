"""
训练脚本和评估脚本共同使用的工具函数

1. 设置随机种子；
2. 自动选择 CPU 或 GPU；
3. 定义 FashionMNIST 数据预处理；
4. 创建输出目录；
5. 统计模型参数数量；
6. 保存和加载模型检查点；
7. 保存训练历史；
8. 绘制训练曲线。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any      # Any 表示该变量可以是任意类型

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms  # transforms 用于定义图片预处理流程


# FashionMNIST 一共有 10 个类别
FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

FASHION_MNIST_MEAN = (0.2860,)  # FashionMNIST 图片像素的平均值
FASHION_MNIST_STD = (0.3530,)   # FashionMNIST 图片像素的标准差


def set_seed(seed: int = 42) -> None:
    """
    设置常见随机数生成器的随机种子， 使实验结果更容易复现。
    seed 默认值为 42.
     """

    random.seed(seed)       # 设置 Python 标准库 random 的随机种子
    np.random.seed(seed)    # 设置 NumPy 的随机种子
    torch.manual_seed(seed) # 设置 PyTorch CPU 随机数生成器的随机种子

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)    # 设置当前 GPU 的随机种子
        torch.cuda.manual_seed_all(seed)    # 设置所有 GPU 的随机种子

    # 检查 torch.backends 中是否存在 cudnn， cuDNN 是 NVIDIA 提供的深度神经网络计算库
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True   # 强制 cuDNN 尽量使用确定性算法
        torch.backends.cudnn.benchmark = False      # 关闭 cuDNN 自动寻找最快算法的功能


def get_device(disable_cuda: bool = False) -> torch.device:
    """如果 CUDA 可用并且用户没有主动禁用 CUDA， 就使用 GPU。 否则使用 CPU."""

    if not disable_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_fashion_mnist_transform() -> transforms.Compose:
    """返回训练和评估共同使用的图片预处理流程."""

    # transforms.Compose 用于按照顺序组合多个预处理操作, 图片会依次经过列表中的每一个 transform
    return transforms.Compose(
        [
            transforms.ToTensor(),  # 将 PIL 图片或 NumPy 图片转换成 PyTorch Tensor -> [1, 28, 28]
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),    # 对图片进行标准化, (原像素值 - 平均值) / 标准差
        ]
    )


def ensure_output_directories(output_dir: Path) -> dict[str, Path]:
    """创建输出目录、模型检查点目录和图片目录。 返回包含这些路径的字典."""

    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    plots_dir = output_dir / "plots"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output": output_dir,
        "checkpoints": checkpoints_dir,
        "plots": plots_dir,
    }


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """统计会被优化器更新的模型参数总数量."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _to_serializable(value: Any) -> Any:
    """把 Path 和嵌套容器转换成适合保存的基本类型."""

    # 判断 value 是否是 Path 对象, 将 Path 转换为普通字符串
    if isinstance(value, Path):
        return str(value)

    # 判断 value 是否是字典, 将键转换成字符串
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}

    # 判断 value 是否是列表或元组, 最终统一转换成列表
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def save_checkpoint(
    checkpoint_path: Path,
    *,      # 单独的 * 表示：从这里开始，后面的参数必须通过参数名传入。必须写：save_checkpoint(path, model=model, optimizer=optimizer, ...) 不能全部使用位置参数。
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    validation_accuracy: float,
    model_config: dict[str, Any],
    training_args: dict[str, Any],
) -> None:
    """保存一个可以用于恢复模型和训练状态的检查点."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)   # 创建检查点文件所在的文件夹

    # 创建检查点字典
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),         # 保存模型的状态字典
        "optimizer_state_dict": optimizer.state_dict(), # 保存优化器状态
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "model_config": _to_serializable(model_config), # 保存模型结构配置
        "training_args": _to_serializable(training_args),   # 保存训练参数
        "class_names": FASHION_MNIST_CLASSES,
    }

    # 将整个 checkpoint 字典保存到磁盘。如果 checkpoint_path 已经存在，torch.save 会直接覆盖原文件。真正创建 .pth 文件的是这一句。
    # torch.save() 会将对象序列化到文件；当前默认格式是基于 ZIP 的序列化格式，并使用 Python 的 pickle 机制保存对象结构，同时处理 Tensor 的底层存储
    # 保存的是一个字典，使用 torch.load()后会返回这个字典
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    """加载模型检查点."""

    checkpoint_path = Path(checkpoint_path)

    try:
        # map_location=device：把检查点中的 Tensor 映射到指定设备
        # weights_only=True：使用限制更严格、更安全的加载方式
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        # 如果当前 PyTorch 版本不支持 weights_only 参数，就会抛出 TypeError.
        return torch.load(checkpoint_path, map_location=device)


def save_history(history: dict[str, list[float]], save_path: Path) -> None:
    """将训练历史保存为容易阅读的 JSON 文件."""

    save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)     # 只创建文件所在的文件夹，不会创建文件。

    save_path.write_text(                                   # 文件不存在：自动创建文件
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_history(history: dict[str, list[float]], plots_dir: Path) -> None:
    """分别保存损失曲线和准确率曲线."""

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # 绘制损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    plt.plot(epochs, history["val_loss"], marker="s", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curve.png", dpi=200)
    plt.close()

    # 绘制准确率曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], marker="o", label="Train accuracy")
    plt.plot(epochs, history["val_accuracy"], marker="s", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and validation accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_curve.png", dpi=200)
    plt.close()
