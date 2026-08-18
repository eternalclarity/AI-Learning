"""CIFAR-10 数据集、数据增强、分层划分与 DataLoader 工具。"""

from __future__ import annotations

from dataclasses import dataclass  # 用于保存数据集划分信息。
from pathlib import Path  # 比字符串路径更安全、可读。
from typing import Final  # 标记不会改变的常量。

import numpy as np  # 用于可复现地生成分层划分索引。
import torch  # 用于 DataLoader 和 CUDA 状态判断。
from torch.utils.data import DataLoader, Subset  # DataLoader 负责按批读取数据；Subset 表示索引子集。
from torchvision import datasets, transforms  # torchvision 提供 CIFAR-10 和常见图像变换。


CIFAR10_CLASSES: Final[tuple[str, ...]] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# 这是项目统一采用的 CIFAR-10 三通道均值与标准差。
# 所有实验必须使用同一组统计量，才能保证比较公平。
CIFAR10_MEAN: Final[tuple[float, float, float]] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Final[tuple[float, float, float]] = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class SplitInfo:
    """记录训练集、验证集和测试集的样本数量。"""

    train_size: int  # 训练子集大小。
    val_size: int  # 验证子集大小。
    test_size: int  # 测试集大小。
    val_ratio: float  # 从官方训练集中划出的验证集比例。
    seed: int  # 用于生成划分的随机种子。


def build_transforms(use_augmentation: bool) -> tuple[transforms.Compose, transforms.Compose]:
    """创建训练变换和评估变换。

    训练集可以使用随机变换；验证集和测试集必须使用确定性变换。
    """

    train_steps: list[object] = []  # 先准备一个可动态追加步骤的列表。

    if use_augmentation:  # 只有实验明确要求数据增强时才添加随机操作。
        train_steps.extend(
            [
                transforms.RandomCrop(size=32, padding=4),  # 先补边再随机裁剪回 32×32。
                transforms.RandomHorizontalFlip(p=0.5),  # 以 50% 概率做水平翻转。
            ]
        )

    train_steps.extend(
        [
            transforms.ToTensor(),  # 把 PIL 图片转换为 [C,H,W] 且范围为 [0,1] 的张量。
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),  # 按通道标准化。
        ]
    )

    eval_steps = [
        transforms.ToTensor(),  # 验证和测试同样要转换为张量。
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),  # 使用与训练集相同的归一化规则。
    ]

    return transforms.Compose(train_steps), transforms.Compose(eval_steps)  # 返回两套独立变换。


def create_stratified_split_indices(
    targets: list[int] | np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """按类别比例生成训练索引和验证索引。

    与简单随机切分相比，分层切分能让训练集和验证集中的十个类别比例更稳定。
    """

    if not 0.0 < val_ratio < 1.0:  # 防止出现 0% 或 100% 验证集。
        raise ValueError("val_ratio 必须位于 (0, 1) 区间内。")

    target_array = np.asarray(targets, dtype=np.int64)  # 转成 NumPy 整数数组，便于条件筛选。
    random_generator = np.random.default_rng(seed)  # 创建独立随机数生成器，避免污染全局状态。
    train_indices: list[int] = []  # 保存训练样本下标。
    val_indices: list[int] = []  # 保存验证样本下标。

    for class_id in np.unique(target_array):  # 逐个类别处理，保证每个类别都按同一比例划分。
        class_indices = np.flatnonzero(target_array == class_id)  # 找到该类别的全部样本下标。
        random_generator.shuffle(class_indices)  # 在类别内部随机打乱。
        class_val_size = int(round(len(class_indices) * val_ratio))  # 计算该类别应进入验证集的数量。
        val_indices.extend(class_indices[:class_val_size].tolist())  # 前一部分放入验证集。
        train_indices.extend(class_indices[class_val_size:].tolist())  # 剩余部分放入训练集。

    random_generator.shuffle(train_indices)  # 再打乱训练索引，避免类别成块排列。
    random_generator.shuffle(val_indices)  # 再打乱验证索引。

    return train_indices, val_indices  # 返回两个互不重叠的索引列表。


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    val_ratio: float,
    seed: int,
    use_augmentation: bool,
    num_workers: int = 0,
    download: bool = True,
) -> tuple[dict[str, DataLoader], SplitInfo]:
    """创建训练、验证、测试三个 DataLoader。"""

    if batch_size <= 0:  # batch_size 必须是正整数。
        raise ValueError("batch_size 必须大于 0。")

    if num_workers < 0:  # 工作进程数量不能为负数。
        raise ValueError("num_workers 不能小于 0。")

    data_path = Path(data_dir)  # 将字符串路径转换成 Path。
    data_path.mkdir(parents=True, exist_ok=True)  # 如果目录不存在就递归创建。

    train_transform, eval_transform = build_transforms(use_augmentation)  # 获取两套图像变换。

    # 创建两个指向同一官方训练集的 Dataset 对象，是为了给训练子集和验证子集使用不同 transform。
    train_dataset_full = datasets.CIFAR10(
        root=data_path,
        train=True,
        transform=train_transform,
        download=download,
    )
    val_dataset_full = datasets.CIFAR10(
        root=data_path,
        train=True,
        transform=eval_transform,
        download=download,
    )
    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        transform=eval_transform,
        download=download,
    )

    train_indices, val_indices = create_stratified_split_indices(
        targets=train_dataset_full.targets,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_dataset = Subset(train_dataset_full, train_indices)  # 训练子集使用随机增强变换。
    val_dataset = Subset(val_dataset_full, val_indices)  # 验证子集使用确定性变换。

    pin_memory = torch.cuda.is_available()  # 使用 CUDA 时，固定内存可加快 CPU→GPU 拷贝。
    persistent_workers = num_workers > 0  # 只有存在工作进程时才能保持其常驻。

    common_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,  # 每个 epoch 都重新打乱训练样本顺序。
        drop_last=False,  # 保留最后一个不足 batch_size 的批次。
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,  # 评估时不需要打乱，结果更容易复现和排查。
        drop_last=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )

    split_info = SplitInfo(
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
        val_ratio=val_ratio,
        seed=seed,
    )

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    return loaders, split_info  # 同时返回 DataLoader 字典和划分信息。


def denormalize_batch(images: torch.Tensor) -> torch.Tensor:
    """将标准化后的图像还原到接近 [0,1]，便于可视化。"""

    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(1, 3, 1, 1)  # 变形成可广播形状。
    std = torch.tensor(CIFAR10_STD, device=images.device).view(1, 3, 1, 1)  # 变形成可广播形状。
    restored = images * std + mean  # 逆变换：x = normalized × std + mean。
    return restored.clamp(0.0, 1.0)  # 防止浮点误差让像素超出显示范围。
