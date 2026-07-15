"""
加载已经训练好的 FashionMNIST MLP 模型检查点，并在 FashionMNIST 官方测试集上评估模型。

1. 解析评估参数；
2. 加载 best_model.pth；
3. 根据检查点中的配置重新创建 MLP；
4. 加载训练好的模型参数；
5. 加载 FashionMNIST 官方测试集；
6. 计算测试集损失和整体准确率；
7. 计算每一个服装类别的准确率；
8. 打印部分测试图片的预测结果；
9. 将测试样本拼成图片并保存.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn    # 从 PyTorch 中导入神经网络模块 nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import make_grid     # make_grid 可以把多张小图片拼成一张网格大图。

from models import MLP
from utils import (
    FASHION_MNIST_CLASSES,
    get_device,
    get_fashion_mnist_transform,
    load_checkpoint,
)


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """ 解析运行 evaluate.py 时传入的命令行参数。 返回一个 argparse.Namespace 对象。"""

    parser = argparse.ArgumentParser(description="Evaluate a FashionMNIST checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BASE_DIR / "outputs" / "checkpoints" / "best_model.pth",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument(
        "--sample-image",
        type=Path,
        default=BASE_DIR / "outputs" / "plots" / "sample_predictions.png",
    )
    parser.add_argument("--sample-count", type=int, default=16)
    return parser.parse_args()


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_count: int,
) -> tuple[float, float, list[float]]:
    """
    在完整测试集上评估模型.
    返回 测试损失、整体准确率 以及 每个类别的准确率
    """

    model.eval()    # 将模型切换到评估模式

    loss_fn = nn.CrossEntropyLoss() # 创建 多分类交叉熵损失函数

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    class_correct = torch.zeros(class_count, dtype=torch.long)  # 创建一个长度为 class_count 的全零 Tensor。用来记录每个类别预测正确的数量。
    class_total = torch.zeros(class_count, dtype=torch.long)    # 用来记录测试集中每个类别的样本总数量

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = loss_fn(logits, labels)

            # 在每张图片的 10 个类别分数中,找到分数最大的类别下标。dim=1 表示沿着类别这一维寻找最大值。 predictions 的形状是：[batch_size]
            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            # 累计当前批次的损失总和。 loss.item() 是当前批次的平均损失。 乘以 batch_size 后， 得到当前批次所有样本的损失总和
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

            # 逐个统计每一个类别的数据
            for class_index in range(class_count):
                # 创建当前类别的布尔掩码 -> 布尔掩码：一串布尔值，用于 计数 (.sum()) 和 进一步相与(& [False ...]).sum()
                # 例如:# class_mask = # tensor([True, False, True, False, False]) -> # True 表示这个样本属于当前类别
                class_mask = labels == class_index

                class_total[class_index] += class_mask.sum().cpu()  # .cpu()：将统计结果移动回 CPU
                class_correct[class_index] += (
                    (predictions == labels) & class_mask    # & class_mask：进一步要求样本属于当前类别
                ).sum().cpu()

    # 计算整个测试集的平均损失
    average_loss = total_loss / total_samples
    # 计算整个测试集的整体准确率
    overall_accuracy = 100.0 * total_correct / total_samples

    # 用来计算每一个类别的准确
    per_class_accuracy = []
    for correct, total in zip(class_correct, class_total):
        accuracy = 100.0 * correct.item() / max(1, total.item())
        per_class_accuracy.append(accuracy)

    return average_loss, overall_accuracy, per_class_accuracy


def save_sample_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    sample_count: int,  # 要展示的图片数量
    save_path: Path,
) -> None:
    """保存测试图片网格，
    并在控制台打印它们的预测类别和真实类别.
    """

    model.eval()
    images, labels = next(iter(data_loader))    # iter(data_loader)：将 DataLoader 转换成迭代器。 next(...)： 只获取迭代器中的第一个批次
    images = images[:sample_count]  # 只保留前 sample_count 张图片
    labels = labels[:sample_count]

    with torch.no_grad():
        logits = model(images.to(device))
        predictions = logits.argmax(dim=1).cpu()

    print("\nSample predictions")
    for index, (prediction, label) in enumerate(zip(predictions, labels), start=1):
        # zip(predictions, labels)：把每个预测标签和真实标签组合起来
        # 打印当前图片的预测类别和真实类别
        print(
            f"{index:02d}. predicted={class_names[prediction.item()]:<12} | "
            f"actual={class_names[label.item()]}"
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    # 将多张测试图片拼成一个网格
    image_grid = make_grid(images, nrow=4, normalize=True, scale_each=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_grid.permute(1, 2, 0).numpy()) # 显示图片网格
    plt.axis("off") # 关闭坐标轴
    plt.title("FashionMNIST test samples; predictions are printed in the console")
    plt.tight_layout()  # 自动调整布局，防止标题或图片内容被裁剪
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    device = get_device(disable_cuda=args.no_cuda)

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Run train.py before evaluate.py."
        )

    # 加载模型检查点
    checkpoint = load_checkpoint(args.checkpoint, device)
    model_config = checkpoint.get("model_config", {})
    class_names = checkpoint.get("class_names", FASHION_MNIST_CLASSES)

    # 根据检查点中保存的配置重新创建 MLP
    model = MLP(**model_config)
    # 将检查点中保存的模型参数加载到刚创建的 MLP 中
    model.load_state_dict(checkpoint["model_state_dict"])
    # 将模型移动到指定设备
    model.to(device)

    # 加载 FashionMNIST 官方测试集
    test_dataset = datasets.FashionMNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=get_fashion_mnist_transform(),
    )

    # 为测试集创建 DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # 在完整测试集上进行评估
    test_loss, test_accuracy, per_class_accuracy = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        class_count=len(class_names),
    )

    print("=" * 70)
    print("FashionMNIST evaluation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation accuracy in checkpoint: {checkpoint.get('validation_accuracy', 0.0):.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2f}%")

    print("\nPer-class accuracy")
    for class_name, accuracy in zip(class_names, per_class_accuracy):
        print(f"{class_name:<12}: {accuracy:6.2f}%")

    # 保存测试样本图片, 并打印这些样本的预测类别。
    save_sample_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
        class_names=class_names,
        sample_count=args.sample_count,
        save_path=args.sample_image,
    )
    print(f"\nSample image saved to: {args.sample_image}")


if __name__ == "__main__":
    main()
