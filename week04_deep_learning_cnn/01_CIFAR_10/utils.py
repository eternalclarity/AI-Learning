"""随机种子、设备、优化器、检查点、结果保存和绘图工具。"""

from __future__ import annotations

import csv  # 用于保存无需 pandas 的简单 CSV。
import json  # 用于保存配置和最终指标。
import random  # 用于设置 Python 随机种子。
from pathlib import Path  # 用于安全地管理输出路径。
from typing import Any  # 用于通用字典值类型。

import matplotlib.pyplot as plt  # 用于绘制训练曲线、混淆矩阵和预测样例。
import numpy as np  # 用于随机种子和矩阵处理。
import torch  # 用于模型、优化器、检查点和设备管理。
from sklearn.metrics import confusion_matrix  # 用于计算混淆矩阵。
from torch import nn  # 用于模型类型标注。

from data_utils import CIFAR10_CLASSES, denormalize_batch  # 导入类别名称和反标准化函数。


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 随机种子。"""

    random.seed(seed)  # 固定 Python 内置随机模块。
    np.random.seed(seed)  # 固定 NumPy 随机模块。
    torch.manual_seed(seed)  # 固定 PyTorch CPU 随机数。

    if torch.cuda.is_available():  # 只有存在 CUDA 时才调用 CUDA 随机种子接口。
        torch.cuda.manual_seed_all(seed)  # 固定全部 GPU 的随机数。

    torch.backends.cudnn.deterministic = True  # 尽量使用确定性 CuDNN 算法。
    torch.backends.cudnn.benchmark = False  # 关闭根据输入自动搜索最快算法，增强可复现性。


def resolve_device(device_name: str = "auto") -> torch.device:
    """根据用户输入选择 CPU、CUDA 或 MPS。"""

    normalized = device_name.strip().lower()  # 去空格并转成小写。

    if normalized == "auto":  # 自动模式优先选择 CUDA，其次 MPS，最后 CPU。
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    requested = torch.device(normalized)  # 将字符串转换为 torch.device。

    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 环境无法使用 CUDA。")

    if requested.type == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            raise RuntimeError("请求了 MPS，但当前环境不支持 MPS。")

    return requested  # 返回用户明确请求且可用的设备。


def create_output_directories(output_root: str | Path, experiment_name: str) -> dict[str, Path]:
    """为一个实验创建检查点、图像和结果目录。"""

    root = Path(output_root)  # 转换为 Path。
    directories = {
        "checkpoints": root / "checkpoints" / experiment_name,
        "plots": root / "plots" / experiment_name,
        "results": root / "results" / experiment_name,
    }

    for directory in directories.values():  # 逐个创建目录。
        directory.mkdir(parents=True, exist_ok=True)

    return directories  # 返回后续代码直接使用的路径字典。


def count_parameters(model: nn.Module) -> dict[str, int]:
    """统计模型总参数量与可训练参数量。"""

    total = sum(parameter.numel() for parameter in model.parameters())  # 统计全部参数元素个数。
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {"total_parameters": total, "trainable_parameters": trainable}


def build_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """创建 Adam 或 SGD 优化器。"""

    normalized = optimizer_name.strip().lower()  # 统一优化器名称格式。

    if normalized == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if normalized == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True,
        )

    raise ValueError("optimizer_name 只支持 'adam' 或 'sgd'。")


def build_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """创建学习率调度器；none 表示保持固定学习率。"""

    normalized = scheduler_name.strip().lower()  # 统一名称格式。

    if normalized == "none":
        return None

    if normalized == "step":
        step_size = max(epochs // 3, 1)  # 每训练约三分之一总轮数衰减一次。
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=0.1,
        )

    if normalized == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
        )

    raise ValueError("scheduler_name 只支持 'none'、'step' 或 'cosine'。")


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """将字典以 UTF-8 和缩进格式保存为 JSON。"""

    output_path = Path(path)  # 转换路径类型。
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在。
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_history_csv(history: list[dict[str, float | int]], path: str | Path) -> None:
    """保存每个 epoch 的训练历史。"""

    if not history:  # 没有记录时不写空文件。
        return

    output_path = Path(path)  # 转换路径类型。
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录。

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))  # 使用首行键作为表头。
        writer.writeheader()  # 写入 CSV 表头。
        writer.writerows(history)  # 写入全部 epoch 记录。


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_val_accuracy: float,
    history: list[dict[str, float | int]],
    experiment_config: dict[str, Any],
    split_config: dict[str, Any],
) -> None:
    """保存可恢复训练和可独立评估的完整检查点。"""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_accuracy": best_val_accuracy,
        "history": history,
        "experiment_config": experiment_config,
        "split_config": split_config,
        "class_names": list(CIFAR10_CLASSES),
    }

    output_path = Path(path)  # 转换成 Path。
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保检查点目录存在。
    torch.save(checkpoint, output_path)  # 将 Python 字典序列化为 .pth 文件。


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """读取检查点，并将张量映射到指定设备。"""

    checkpoint_path = Path(path)  # 转换路径类型。

    if not checkpoint_path.exists():  # 提前给出比 torch.load 更直观的错误。
        raise FileNotFoundError(f"找不到检查点：{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    return checkpoint


def plot_training_history(history: list[dict[str, float | int]], output_path: str | Path) -> None:
    """在同一张图中绘制损失曲线和准确率曲线。"""

    if not history:  # 防止空历史导致绘图错误。
        return

    epochs = [int(row["epoch"]) for row in history]  # 提取横轴 epoch。
    train_loss = [float(row["train_loss"]) for row in history]  # 提取训练损失。
    val_loss = [float(row["val_loss"]) for row in history]  # 提取验证损失。
    train_accuracy = [float(row["train_accuracy"]) for row in history]  # 提取训练准确率。
    val_accuracy = [float(row["val_accuracy"]) for row in history]  # 提取验证准确率。

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))  # 创建左右两个坐标轴。

    axes[0].plot(epochs, train_loss, label="Train Loss")  # 绘制训练损失。
    axes[0].plot(epochs, val_loss, label="Validation Loss")  # 绘制验证损失。
    axes[0].set_xlabel("Epoch")  # 设置横轴名称。
    axes[0].set_ylabel("Cross-Entropy Loss")  # 设置纵轴名称。
    axes[0].set_title("Loss Curves")  # 设置子图标题。
    axes[0].legend()  # 显示图例。
    axes[0].grid(alpha=0.3)  # 添加浅色网格便于读取。

    axes[1].plot(epochs, train_accuracy, label="Train Accuracy")  # 绘制训练准确率。
    axes[1].plot(epochs, val_accuracy, label="Validation Accuracy")  # 绘制验证准确率。
    axes[1].set_xlabel("Epoch")  # 设置横轴名称。
    axes[1].set_ylabel("Accuracy")  # 设置纵轴名称。
    axes[1].set_title("Accuracy Curves")  # 设置子图标题。
    axes[1].legend()  # 显示图例。
    axes[1].grid(alpha=0.3)  # 添加网格。

    figure.tight_layout()  # 自动调整边距，避免文字重叠。
    output = Path(output_path)  # 转换输出路径。
    output.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录。
    figure.savefig(output, dpi=160, bbox_inches="tight")  # 保存高清图片。
    plt.close(figure)  # 关闭图像，防止多次训练时积累内存。


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    output_path: str | Path,
) -> np.ndarray:
    """计算并绘制 10 类混淆矩阵。"""

    matrix = confusion_matrix(labels, predictions, labels=list(range(len(CIFAR10_CLASSES))))
    figure, axis = plt.subplots(figsize=(9, 8))  # 创建方形画布。
    image = axis.imshow(matrix, interpolation="nearest")  # 将矩阵显示为热力图。
    figure.colorbar(image, ax=axis)  # 添加颜色刻度条。

    axis.set(
        xticks=np.arange(len(CIFAR10_CLASSES)),
        yticks=np.arange(len(CIFAR10_CLASSES)),
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
        xlabel="Predicted label",
        ylabel="True label",
        title="CIFAR-10 Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0  # 决定标注文字的对比度。

    for row in range(matrix.shape[0]):  # 遍历真实类别。
        for column in range(matrix.shape[1]):  # 遍历预测类别。
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=8,
            )

    figure.tight_layout()  # 调整布局。
    output = Path(output_path)  # 转换输出路径。
    output.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录。
    figure.savefig(output, dpi=160, bbox_inches="tight")  # 保存图片。
    plt.close(figure)  # 释放图像资源。
    return matrix  # 返回矩阵，便于进一步保存或分析。


def plot_sample_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    output_path: str | Path,
    max_images: int = 16,
) -> None:
    """绘制少量测试图片及其真实标签和预测标签。"""

    if images.numel() == 0:  # 没有保存图片时直接返回。
        return

    image_count = min(max_images, images.size(0))  # 确定实际绘制数量。
    restored_images = denormalize_batch(images[:image_count]).permute(0, 2, 3, 1).numpy()
    columns = 4  # 每行显示 4 张图片。
    rows = int(np.ceil(image_count / columns))  # 根据图片数计算行数。
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))  # 创建网格画布。
    axes_array = np.atleast_1d(axes).reshape(-1)  # 无论行数多少，都统一展平成一维数组。

    for index, axis in enumerate(axes_array):  # 遍历全部子图位置。
        axis.axis("off")  # 默认隐藏坐标轴。
        if index >= image_count:  # 多余子图保持空白。
            continue

        true_name = CIFAR10_CLASSES[int(labels[index])]  # 获取真实类别名称。
        predicted_name = CIFAR10_CLASSES[int(predictions[index])]  # 获取预测类别名称。
        correctness = "✓" if true_name == predicted_name else "✗"  # 用符号标记是否正确。
        axis.imshow(restored_images[index])  # 显示图片。
        axis.set_title(f"{correctness} True: {true_name}\nPred: {predicted_name}", fontsize=9)

    figure.tight_layout()  # 自动调整间距。
    output = Path(output_path)  # 转换输出路径。
    output.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录。
    figure.savefig(output, dpi=160, bbox_inches="tight")  # 保存图片。
    plt.close(figure)  # 释放资源。
