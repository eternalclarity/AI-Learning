"""
使用 MLP 模型训练 FashionMNIST 图像分类任务。

主要完成：
1. 解析命令行参数；
2. 下载并划分 FashionMNIST 数据集；
3. 创建训练集和验证集 DataLoader；
4. 创建 MLP 模型；
5. 创建损失函数和优化器；
6. 逐轮训练和验证；
7. 保存最佳模型和最后一轮模型；
8. 保存训练历史并绘制曲线。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split   # random_split：负责按照指定长度随机拆分数据集,用于拆分训练集和验证集
from torchvision import datasets    # torchvision.datasets 提供了很多常见图像数据集,  其中包括 FashionMNIST

from models import MLP
from utils import (
    count_trainable_parameters, # 统计模型中可以训练的参数数量
    ensure_output_directories,  # 创建输出目录
    get_device,                 # 自动选择 CPU 或 GPU
    get_fashion_mnist_transform,# 获取 FashionMNIST 的数据预处理操作
    plot_history,               # 根据训练历史绘制 loss 和 accuracy 曲线
    save_checkpoint,            # 保存模型检查点
    save_history,               # 将训练历史保存成 JSON 文件
    set_seed,                   # 设置随机种子，保证实验尽量可以复现
)


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """
    解析用户运行 train.py 时传入的命令行参数。 。
    """

    parser = argparse.ArgumentParser(description="Train an MLP on FashionMNIST")

    parser.add_argument("--epochs", type=int, default=10)   # 添加训练轮数参数
    parser.add_argument("--batch-size", type=int, default=64)   # 添加批次大小参数
    parser.add_argument("--lr", type=float, default=1e-3)       # 添加学习率参数
    parser.add_argument("--weight-decay", type=float, default=0.0)  # 添加权重衰减参数, weight_decay 通常用于 L2 正则化,可以在一定程度上减少模型过拟合

    """
    过拟合: 模型“学过头了” -> 把训练集噪声和偶然特征也学去了 -> 模型泛化能力差
        - 模型没有真正学会通用规律，而是把训练数据中的细节、噪声甚至偶然特征也记住了 -> 模型在训练集上更好了，但在验证集上反而变差了
    
    L2 正则化: 防止参数过大 ->会在原来的损失函数上，加一项“模型参数过大的惩罚” -> (不允许模型参数变得过于极端)
        - 如果权重特别大 (w1 = 1000,w2 = -800) 那么输入稍微变化一点，输出就可能产生巨大变化,这通常说明模型过于敏感，可能把训练数据中的微小噪声也当成了重要规律
        - weight_decay 越大 -> 对大权重的限制越强
    
    Dropout: 神经网络中的一种正则化方法 -> 在每次训练时，随机让一部分神经元暂时失效 -> (不允许神经元之间产生过强依赖)
        - 加入 Dropout 后，由于任何神经元都可能被随机关闭，每个神经元都不能过度依赖其他神经元，只能自己学习更有用、更稳定的特征
        - Dropout 只在训练时生效
    """

    parser.add_argument("--hidden-size1", type=int, default=256)    # 第一隐藏层的神经元数量
    parser.add_argument("--hidden-size2", type=int, default=128)    # 第二隐藏层的神经元数量
    parser.add_argument("--dropout", type=float, default=0.2)       # Dropout 丢弃概率
    parser.add_argument("--validation-ratio", type=float, default=0.1)  # 验证集占训练数据的比例
    parser.add_argument("--num-workers", type=int, default=0)   # DataLoader 使用的子进程数量
    parser.add_argument("--seed", type=int, default=42) # 随机种子
    parser.add_argument("--no-cuda", action="store_true")   # 添加是否禁用 CUDA 的参数
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data") # 数据集保存目录
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs") # 训练输出目录

    return parser.parse_args()


def create_dataloaders(
    data_dir: Path,         # FashionMNIST 数据保存目录
    batch_size: int,        # 每个批次的样本数量
    validation_ratio: float,    # 验证集比例
    num_workers: int,       # DataLoader 子进程数量
    seed: int,              # 随机种子
    use_pin_memory: bool,   # 是否使用锁页内存,使用 GPU 时通常设为 True,可以在某些情况下加快 CPU 到 GPU 的数据传输
) -> tuple[DataLoader, DataLoader]: # 返回两个 DataLoader：第一个是训练集 DataLoader, 第二个是验证集 DataLoader
    """
    下载 FashionMNIST 数据集，并将它原本的训练部分拆分成训练集和验证集.
        - Fashion-MNIST 是 服装图片分类 数据集，主要用于学习和测试机器学习、深度学习中的 图像分类
        - 给模型一张衣服图片，让模型判断这是 T 恤、裤子、鞋子、包，还是其他服装

        - 训练集： 计算损失、反向传播、修改模型参数(60,000 张)
        - 验证集： 训练过程中检查模型效果和过拟合 (在实际训练中，我们通常从 60,000 张训练图片里再划分一部分作为验证集)
        - 测试集： 模型全部训练完成后，进行最终评估(10,000 张)
    """

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")

    # 创建完整的 FashionMNIST 训练数据集
    full_training_dataset = datasets.FashionMNIST(
        root=data_dir,  # 数据集保存目录
        train=True,     # train=True 表示加载 FashionMNIST 官方训练集
        download=True,  # 如果本地没有数据，就自动从网络下载
        transform=get_fashion_mnist_transform(),    # 对每张图片执行数据预处理 -> 可输入MLP的tensor[1, 28, 28]
    )

    # 计算验证集样本数量
    validation_size = int(len(full_training_dataset) * validation_ratio)
    # 计算真正用于训练的样本数量
    training_size = len(full_training_dataset) - validation_size

    # 创建一个独立的 PyTorch 随机数生成器,manual_seed(seed)为该生成器设置随机种子
    split_generator = torch.Generator().manual_seed(seed)

    # 将完整训练集拆分成:training_dataset, validation_dataset
    training_dataset, validation_dataset = random_split(
        full_training_dataset,
        lengths=[training_size, validation_size],
        generator=split_generator,
    )

    # 再创建一个随机数生成器，专门控制训练 DataLoader 的随机打乱顺序
    loader_generator = torch.Generator().manual_seed(seed)

    # 创建训练集 DataLoader
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,               # 每个epoch 训练集需要打乱,避免模型总是按照完全相同的顺序看到数据
        num_workers=num_workers,    # 数据加载子进程数量
        pin_memory=use_pin_memory,  # 使用 GPU 时，可以使用锁页内存
        generator=loader_generator, # 控制 shuffle 的随机性
    )

    # 创建验证集 DataLoader
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,              # 验证集不需要随机打乱, 因为验证过程不会训练参数, 样本顺序不会影响最终准确率和损失
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return training_loader, validation_loader


def train_one_epoch(
    data_loader: DataLoader,    # 训练集 DataLoader -> 1. logits: tensor[batch_size, 10] 2: labels:[batch_size]
    model: nn.Module,           # 要训练的模型
    loss_fn: nn.Module,         # 损失函数
    optimizer: torch.optim.Optimizer,   # 优化器
    device: torch.device,       # 模型和数据所在的设备
) -> tuple[float, float]:       # 返回： 1. 当前 epoch 的平均损失； 2. 当前 epoch 的准确率
    """
    在完整训练集上训练一轮 epoch.
    """

    model.train()   # 将模型切换到训练模式 -> Dropout 会生效； BatchNorm 会更新统计信息

    total_loss = 0.0    # 用于累计当前 epoch 所有样本的损失总和
    total_correct = 0   # 用于累计预测正确的样本数量
    total_samples = 0   # 用于累计已经处理的样本总数量

    # 遍历训练集中的每个批次 -> 5 步 -> 一个epoch
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)   # non_blocking=True 在满足条件时，可以进行异步数据传输

        logits = model(images)          # 1. 前向传播: [batch_size, 10]: 每一行包含一张图片对应的 10 个类别分数。
        loss = loss_fn(logits, labels)  # 2. 损失函数: (SoftMax) CrossEntropyLoss (多分类交叉熵损失)

        optimizer.zero_grad(set_to_none=True)   # 3. 清空上一轮的梯度: set_to_none=True 表示：将梯度设置为 None，而不是设置成全 0 Tensor, 更加节省内存
        loss.backward()                 # 4. 反向传播: 计算参数梯度
        optimizer.step()                # 5. 更新参数: Optim.Adam

        batch_size = labels.size(0)     # 最后一批：获取当前批次的实际样本数量, labels.size(0) 就是当前批次包含的样本数
        total_loss += loss.item() * batch_size  # 累计当前批次的损失总和
        total_correct += (logits.argmax(dim=1) == labels).sum().item()  # logits.argmax(dim=1): 找到每张图片 10 个类别分数中最大值所在的下标
        total_samples += batch_size     # 累计当前 epoch 已处理的样本数量

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return average_loss, accuracy


def validate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    在验证集上评估模型
    验证过程不会更新模型参数.
     """
    model.eval()        # 将模型切换到评估模式

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():               # 关闭自动求导
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)   # GPU显存有限，常不建议把整个数据集一次性放到 GPU，而是每次只把一个 batch 搬到 GPU
            labels = labels.to(device, non_blocking=True)

            logits = model(images)              # 1. 前向传播
            loss = loss_fn(logits, labels)      # 2. 损失函数

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return average_loss, accuracy


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device(disable_cuda=args.no_cuda)
    output_paths = ensure_output_directories(args.output_dir)

    print("=" * 70)
    print("FashionMNIST MLP training")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. images, 2.labels
    training_loader, validation_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        use_pin_memory=device.type == "cuda",
    )

    # 3. model
    model = MLP(
        hidden_sizes=(args.hidden_size1, args.hidden_size2),
        dropout=args.dropout,
    ).to(device)

    # 4. loss函数
    loss_fn = nn.CrossEntropyLoss()     # CrossEntropyLoss 用于多分类任务 10个得分 -> 对应labels的概率p -> loss = -log(p)

    # 5. 优化器 optim.Adam
    optimizer = torch.optim.Adam(   # Adaptive Moment Estimation，适应性矩估计
        model.parameters(),         # 带动量的 SGD + 每个参数拥有自己的自适应学习率
        lr=args.lr,                 # 如果最近几次梯度方向都一致，Adam 就会更坚定地朝这个方向更新；如果梯度方向来回变化，更新就会受到抑制
        weight_decay=args.weight_decay,
    )

    print(model)
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Training samples:   {len(training_loader.dataset):,}")
    print(f"Validation samples: {len(validation_loader.dataset):,}")

    # 创建训练历史字典,用于记录每一轮的：训练损失；训练准确率； 验证损失；验证准确率。
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_validation_accuracy = float("-inf")    # 初始化最佳验证准确率
    best_checkpoint_path = output_paths["checkpoints"] / "best_model.pth"   # 最佳模型保存路径
    last_checkpoint_path = output_paths["checkpoints"] / "last_model.pth"   # 最后一轮模型保存路径

    # 开始 epoch 训练循环 -> 10轮训练,每轮都过一遍随机的 训练集 和 验证集
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=training_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = validate(
            model=model,
            data_loader=validation_loader,
            loss_fn=loss_fn,
            device=device,
        )

        # 保存本轮
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        history["val_loss"].append(validation_loss)
        history["val_accuracy"].append(validation_accuracy)

        # 输出本轮
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"train loss={train_loss:.4f}, train acc={train_accuracy:.2f}% | "
            f"val loss={validation_loss:.4f}, val acc={validation_accuracy:.2f}%"
        )

        # 判断当前 验证准确率 是否创造了新的最好成绩， 而非训练集
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            # 保存当前最佳模型检查点
            save_checkpoint(
                best_checkpoint_path,   # 保存路径
                model=model,            # 模型及其参数
                optimizer=optimizer,    # 优化器及其状态
                epoch=epoch,            # 当前训练到第几轮
                validation_loss=validation_loss,    # 当前验证损失
                validation_accuracy=validation_accuracy,    # 当前验证准确率
                model_config=model.get_config(),    # 模型结构配置
                training_args=vars(args),   # vars(args) 将 Namespace 转换成字典
            )
            print(f"  New best model saved: {best_checkpoint_path}")

    # 保存最后一轮模型
    save_checkpoint(
        last_checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        validation_loss=history["val_loss"][-1],
        validation_accuracy=history["val_accuracy"][-1],
        model_config=model.get_config(),
        training_args=vars(args),
    )

    # 将 history 字典保存为 JSON 文件
    history_path = output_paths["output"] / "history.json"
    save_history(history, history_path)
    plot_history(history, output_paths["plots"])

    print("=" * 70)
    print("Training complete")
    print(f"Best validation accuracy: {best_validation_accuracy:.2f}%")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Last checkpoint: {last_checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Plots: {output_paths['plots']}")


if __name__ == "__main__":
    main()
