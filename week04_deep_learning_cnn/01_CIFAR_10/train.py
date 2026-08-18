"""训练单个第四周实验。

推荐命令：
    python train.py --experiment exp1_basic_cnn
    python train.py --experiment exp4_cnn_bn_dropout_aug --epochs 20 --amp
"""

from __future__ import annotations

import argparse  # 用于解析命令行参数。
from dataclasses import asdict  # 将 dataclass 配置转换成普通字典。
from pathlib import Path  # 用于管理数据和输出目录。
from typing import Any  # 用于配置字典类型标注。

import torch  # 用于模型训练、损失函数和 AMP。
from torch import nn  # 用于创建交叉熵损失。

from config import EXPERIMENT_PRESETS, get_experiment_preset  # 导入实验预设。
from data_utils import build_dataloaders  # 导入 CIFAR-10 数据加载函数。
from engine import evaluate_one_epoch, train_one_epoch  # 导入单轮训练和验证函数。
from models import create_model  # 导入统一模型工厂。
from utils import (
    build_optimizer,
    build_scheduler,
    count_parameters,
    create_output_directories,
    load_checkpoint,
    plot_training_history,
    resolve_device,
    save_checkpoint,
    save_history_csv,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """定义并读取训练命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Train one CIFAR-10 experiment for Week 4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="exp1_basic_cnn",
        choices=list(EXPERIMENT_PRESETS.keys()),
        help="实验预设名称。",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="CIFAR-10 数据目录。")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="实验输出根目录。")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数。")
    parser.add_argument("--batch-size", type=int, default=128, help="每个批次的样本数量。")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="从官方训练集中划出的验证集比例。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 工作进程数；Windows 初学阶段建议 0。")
    parser.add_argument("--device", type=str, default="auto", help="auto、cpu、cuda 或 mps。")
    parser.add_argument("--amp", action="store_true", help="在 CUDA 上启用自动混合精度。")
    parser.add_argument("--resume", type=Path, default=None, help="从已有 last_model.pth 继续训练。")
    parser.add_argument("--no-download", action="store_true", help="不自动下载 CIFAR-10。")

    # 以下参数用于有意识地覆盖预设；不填写时保持预设值，便于公平比较。
    parser.add_argument("--learning-rate", type=float, default=None, help="覆盖实验预设的学习率。")
    parser.add_argument("--weight-decay", type=float, default=None, help="覆盖实验预设的权重衰减。")
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default=None, help="覆盖优化器。")
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "step", "cosine"],
        default=None,
        help="覆盖学习率调度器。",
    )

    return parser.parse_args()  # 返回解析后的命名空间。


def validate_args(args: argparse.Namespace) -> None:
    """在真正训练前检查常见参数错误。"""

    if args.epochs <= 0:  # 至少训练一个 epoch。
        raise ValueError("epochs 必须大于 0。")
    if args.batch_size <= 0:  # 批大小必须为正数。
        raise ValueError("batch_size 必须大于 0。")
    if not 0.0 < args.val_ratio < 1.0:  # 验证集比例必须合理。
        raise ValueError("val_ratio 必须位于 (0,1) 区间。")
    if args.num_workers < 0:  # 工作进程不能为负数。
        raise ValueError("num_workers 不能小于 0。")


def main() -> None:
    """组织完整训练流程。"""

    args = parse_args()  # 读取命令行参数。
    validate_args(args)  # 提前检查参数。
    preset = get_experiment_preset(args.experiment)  # 读取实验预设。
    set_seed(args.seed)  # 固定随机性，增强复现能力。
    device = resolve_device(args.device)  # 自动或手动选择运行设备。

    learning_rate = args.learning_rate if args.learning_rate is not None else preset.learning_rate
    weight_decay = args.weight_decay if args.weight_decay is not None else preset.weight_decay
    optimizer_name = args.optimizer if args.optimizer is not None else preset.optimizer_name
    scheduler_name = args.scheduler if args.scheduler is not None else preset.scheduler_name

    output_dirs = create_output_directories(args.output_dir, preset.name)  # 为当前实验创建输出目录。

    loaders, split_info = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_augmentation=preset.use_augmentation,
        num_workers=args.num_workers,
        download=not args.no_download,
    )

    model = create_model(
        model_name=preset.model_name,
        num_classes=10,
        use_batch_norm=preset.use_batch_norm,
        dropout=preset.dropout,
    )
    model = model.to(device)  # 将模型参数移动到选定设备。

    criterion = nn.CrossEntropyLoss()  # 多分类任务使用交叉熵损失。
    optimizer = build_optimizer(
        optimizer_name=optimizer_name,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = build_scheduler(
        scheduler_name=scheduler_name,
        optimizer=optimizer,
        epochs=args.epochs,
    )

    amp_enabled = args.amp and device.type == "cuda"  # 非 CUDA 环境自动关闭 AMP。
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)  # 创建梯度缩放器。

    parameter_info = count_parameters(model)  # 统计模型参数量。
    start_epoch = 1  # 默认从第一轮开始。
    best_val_accuracy = -1.0  # 用负数保证第一轮一定能保存最佳模型。
    history: list[dict[str, float | int]] = []  # 保存每轮训练和验证记录。

    experiment_config: dict[str, Any] = {
        **asdict(preset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer_name": optimizer_name,
        "scheduler_name": scheduler_name,
        "amp": amp_enabled,
        "seed": args.seed,
        "device": str(device),
        **parameter_info,
    }
    split_config = asdict(split_info)  # 将数据划分信息转换为可保存字典。

    if args.resume is not None:  # 用户指定检查点时恢复训练状态。
        checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model_state_dict"])  # 恢复模型参数。
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # 恢复优化器状态。
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # 恢复调度器状态。
        start_epoch = int(checkpoint["epoch"]) + 1  # 从下一轮继续。
        best_val_accuracy = float(checkpoint["best_val_accuracy"])  # 恢复历史最佳准确率。
        history = list(checkpoint.get("history", []))  # 恢复训练曲线记录。

    print("=" * 80)
    print("Week 4 - CIFAR-10 CNN Training")
    print("=" * 80)
    print(f"Experiment:        {preset.name}")
    print(f"Description:       {preset.description}")
    print(f"Device:            {device}")
    print(f"Model:             {preset.model_name}")
    print(f"BatchNorm:         {preset.use_batch_norm}")
    print(f"Dropout:           {preset.dropout}")
    print(f"Data augmentation: {preset.use_augmentation}")
    print(f"Train / Val / Test:{split_info.train_size} / {split_info.val_size} / {split_info.test_size}")
    print(f"Parameters:        {parameter_info['trainable_parameters']:,}")
    print(f"AMP enabled:       {amp_enabled}")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):  # 依次执行每个 epoch。
        current_learning_rate = optimizer.param_groups[0]["lr"]  # 记录该轮实际学习率。

        train_metrics = train_one_epoch(
            model=model,
            data_loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            scaler=scaler,
            use_amp=amp_enabled,
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            data_loader=loaders["val"],
            criterion=criterion,
            device=device,
            description=f"Val   [{epoch}/{args.epochs}]",
        )

        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": float(current_learning_rate),
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "train_seconds": train_metrics.seconds,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
            "val_seconds": val_metrics.seconds,
        }
        history.append(epoch_record)  # 将本轮结果加入历史记录。

        if scheduler is not None:  # 每个 epoch 结束后更新下一轮学习率。
            scheduler.step()

        improved = val_metrics.accuracy > best_val_accuracy  # 判断验证准确率是否刷新最佳值。
        if improved:
            best_val_accuracy = val_metrics.accuracy  # 更新历史最佳值。
            save_checkpoint(
                path=output_dirs["checkpoints"] / "best_model.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                history=history,
                experiment_config=experiment_config,
                split_config=split_config,
            )

        save_checkpoint(
            path=output_dirs["checkpoints"] / "last_model.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_accuracy=best_val_accuracy,
            history=history,
            experiment_config=experiment_config,
            split_config=split_config,
        )

        save_history_csv(history, output_dirs["results"] / "history.csv")  # 每轮覆盖保存最新历史。
        plot_training_history(history, output_dirs["plots"] / "training_curves.png")  # 更新训练曲线。

        marker = "  ← best" if improved else ""  # 用文本标记最佳轮次。
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train loss={train_metrics.loss:.4f}, acc={train_metrics.accuracy:.4f} | "
            f"val loss={val_metrics.loss:.4f}, acc={val_metrics.accuracy:.4f} | "
            f"lr={current_learning_rate:.6f}{marker}"
        )

    best_epoch = max(history, key=lambda row: float(row["val_accuracy"]))  # 找出验证集最佳记录。
    summary = {
        "experiment": preset.name,
        "description": preset.description,
        "best_epoch": int(best_epoch["epoch"]),
        "best_val_accuracy": float(best_epoch["val_accuracy"]),
        "best_val_loss": float(best_epoch["val_loss"]),
        "final_train_accuracy": float(history[-1]["train_accuracy"]),
        "final_val_accuracy": float(history[-1]["val_accuracy"]),
        "generalization_gap": float(history[-1]["train_accuracy"]) - float(history[-1]["val_accuracy"]),
        "total_training_seconds": sum(float(row["train_seconds"]) for row in history),
        "experiment_config": experiment_config,
        "split_config": split_config,
        "best_checkpoint": str(output_dirs["checkpoints"] / "best_model.pth"),
    }
    save_json(summary, output_dirs["results"] / "training_summary.json")  # 保存便于横向比较的摘要。

    print("=" * 80)
    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best checkpoint: {output_dirs['checkpoints'] / 'best_model.pth'}")
    print("注意：训练阶段不查看测试集；先比较验证集结果，再单独运行 evaluate.py。")


if __name__ == "__main__":
    main()  # 只有直接运行当前文件时才启动训练。
