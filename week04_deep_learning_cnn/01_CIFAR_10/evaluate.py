"""加载选定的最佳检查点，并在 CIFAR-10 官方测试集上做一次最终评估。"""

from __future__ import annotations

import argparse  # 用于读取命令行参数。
import csv  # 用于保存逐样本预测和每类准确率。
from pathlib import Path  # 用于管理文件路径。

import numpy as np  # 用于指标计算和 CSV 数据处理。
import torch  # 用于设备和模型加载。
from torch import nn  # 用于交叉熵损失。

from data_utils import CIFAR10_CLASSES, build_dataloaders  # 导入测试数据和类别名称。
from engine import collect_predictions, evaluate_one_epoch  # 导入测试函数和预测收集函数。
from models import create_model  # 根据检查点配置重建模型。
from utils import (
    load_checkpoint,
    plot_confusion_matrix,
    plot_sample_predictions,
    resolve_device,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """定义最终测试命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Evaluate one selected CIFAR-10 checkpoint on the untouched test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="best_model.pth 路径。")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="CIFAR-10 数据目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="测试结果目录；默认与检查点实验名对应。")
    parser.add_argument("--batch-size", type=int, default=256, help="测试批大小。")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 工作进程数。")
    parser.add_argument("--device", type=str, default="auto", help="auto、cpu、cuda 或 mps。")
    parser.add_argument("--no-download", action="store_true", help="不自动下载 CIFAR-10。")
    return parser.parse_args()


def write_per_class_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> list[dict[str, float | int | str]]:
    """计算并保存每个类别的准确率。"""

    rows: list[dict[str, float | int | str]] = []  # 保存每个类别一行结果。

    for class_id, class_name in enumerate(CIFAR10_CLASSES):  # 逐类统计。
        class_mask = labels == class_id  # 找到该类全部测试样本。
        total = int(class_mask.sum())  # 该类样本总数。
        correct = int((predictions[class_mask] == class_id).sum())  # 该类预测正确数。
        accuracy = correct / total if total > 0 else 0.0  # 防止除以 0。
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录。
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))  # 使用字典键作为表头。
        writer.writeheader()  # 写入表头。
        writer.writerows(rows)  # 写入每个类别结果。

    return rows  # 返回结构化结果，便于写入 JSON。


def write_predictions_csv(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """保存逐样本真实标签、预测标签和最高预测概率。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保结果目录存在。

    with output_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "sample_index",
            "true_label_id",
            "true_label_name",
            "predicted_label_id",
            "predicted_label_name",
            "confidence",
            "is_correct",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)  # 创建字典 CSV 写入器。
        writer.writeheader()  # 写入表头。

        for index in range(len(labels)):  # 逐样本保存预测结果。
            true_id = int(labels[index])  # 真实类别编号。
            predicted_id = int(predictions[index])  # 预测类别编号。
            confidence = float(probabilities[index, predicted_id])  # 预测类别对应概率。
            writer.writerow(
                {
                    "sample_index": index,
                    "true_label_id": true_id,
                    "true_label_name": CIFAR10_CLASSES[true_id],
                    "predicted_label_id": predicted_id,
                    "predicted_label_name": CIFAR10_CLASSES[predicted_id],
                    "confidence": confidence,
                    "is_correct": int(true_id == predicted_id),
                }
            )


def main() -> None:
    """执行独立测试集评估。"""

    args = parse_args()  # 读取参数。
    device = resolve_device(args.device)  # 选择计算设备。
    checkpoint = load_checkpoint(args.checkpoint, device)  # 读取检查点。
    experiment_config = checkpoint["experiment_config"]  # 提取模型与训练配置。
    split_config = checkpoint["split_config"]  # 提取训练时的数据划分配置。
    experiment_name = str(experiment_config["name"])  # 获取实验名称。
    seed = int(experiment_config["seed"])  # 使用训练时相同的随机种子。
    set_seed(seed)  # 固定随机性。

    if args.output_dir is None:  # 未指定时写入默认 outputs/results/<experiment>/final_test。
        output_dir = Path("outputs") / "results" / experiment_name / "final_test"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建最终测试目录。

    loaders, _ = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=float(split_config["val_ratio"]),
        seed=int(split_config["seed"]),
        use_augmentation=False,  # 测试阶段绝不使用随机数据增强。
        num_workers=args.num_workers,
        download=not args.no_download,
    )

    model = create_model(
        model_name=str(experiment_config["model_name"]),
        num_classes=10,
        use_batch_norm=bool(experiment_config["use_batch_norm"]),
        dropout=float(experiment_config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])  # 恢复最佳模型权重。
    model = model.to(device)  # 将模型移动到评估设备。

    criterion = nn.CrossEntropyLoss()  # 使用与训练一致的损失函数。
    test_metrics = evaluate_one_epoch(
        model=model,
        data_loader=loaders["test"],
        criterion=criterion,
        device=device,
        description="Final test",
    )

    prediction_data = collect_predictions(
        model=model,
        data_loader=loaders["test"],
        device=device,
        max_images_to_keep=32,
    )

    labels = prediction_data["labels"].numpy()  # 转换为 NumPy 数组。
    predictions = prediction_data["predictions"].numpy()  # 转换为 NumPy 数组。
    probabilities = prediction_data["probabilities"].numpy()  # 转换为 NumPy 数组。

    matrix = plot_confusion_matrix(
        labels=labels,
        predictions=predictions,
        output_path=output_dir / "confusion_matrix.png",
    )
    plot_sample_predictions(
        images=prediction_data["sample_images"],
        labels=prediction_data["labels"][: prediction_data["sample_images"].size(0)],
        predictions=prediction_data["predictions"][: prediction_data["sample_images"].size(0)],
        output_path=output_dir / "sample_predictions.png",
    )

    per_class_rows = write_per_class_accuracy(
        labels=labels,
        predictions=predictions,
        output_path=output_dir / "per_class_accuracy.csv",
    )
    write_predictions_csv(
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        output_path=output_dir / "test_predictions.csv",
    )

    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")  # 保存原始矩阵。

    final_metrics = {
        "experiment": experiment_name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_accuracy": float(checkpoint["best_val_accuracy"]),
        "test_loss": test_metrics.loss,
        "test_accuracy": test_metrics.accuracy,
        "test_correct": test_metrics.correct,
        "test_samples": test_metrics.samples,
        "test_seconds": test_metrics.seconds,
        "mean_per_class_accuracy": float(np.mean([row["accuracy"] for row in per_class_rows])),
        "per_class_accuracy": per_class_rows,
    }
    save_json(final_metrics, output_dir / "final_test_metrics.json")  # 保存最终测试摘要。

    print("=" * 80)
    print("Final CIFAR-10 Test Result")
    print("=" * 80)
    print(f"Experiment:              {experiment_name}")
    print(f"Checkpoint epoch:        {checkpoint['epoch']}")
    print(f"Best validation accuracy:{checkpoint['best_val_accuracy']:.4f}")
    print(f"Test loss:               {test_metrics.loss:.4f}")
    print(f"Test accuracy:           {test_metrics.accuracy:.4f}")
    print(f"Results directory:       {output_dir}")


if __name__ == "__main__":
    main()  # 直接运行文件时启动最终测试。
