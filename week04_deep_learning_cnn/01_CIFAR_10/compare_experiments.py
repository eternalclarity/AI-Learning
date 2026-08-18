"""汇总多个实验的验证集结果，并生成公平对比表和图。"""

from __future__ import annotations

import argparse  # 用于命令行参数。
import json  # 用于读取 training_summary.json。
from pathlib import Path  # 用于搜索结果文件。

import matplotlib.pyplot as plt  # 用于绘制实验对比图。
import pandas as pd  # 用于整理和保存表格。


def parse_args() -> argparse.Namespace:
    """读取汇总脚本参数。"""

    parser = argparse.ArgumentParser(
        description="Compare Week 4 validation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"), help="训练摘要根目录。")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/results/comparison"), help="汇总输出目录。")
    return parser.parse_args()


def main() -> None:
    """读取全部训练摘要并生成横向对比。"""

    args = parse_args()  # 读取参数。
    summary_paths = sorted(args.results_dir.glob("*/training_summary.json"))  # 搜索每个实验摘要。

    if not summary_paths:  # 没有完成训练时给出明确提示。
        raise FileNotFoundError(
            f"在 {args.results_dir} 下没有找到 training_summary.json；请先运行 train.py。"
        )

    rows: list[dict[str, object]] = []  # 保存每个实验一行结果。

    for summary_path in summary_paths:  # 逐个读取实验摘要。
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = summary["experiment_config"]  # 提取配置子字典。
        rows.append(
            {
                "experiment": summary["experiment"],
                "model": config["model_name"],
                "batch_norm": config["use_batch_norm"],
                "dropout": config["dropout"],
                "augmentation": config["use_augmentation"],
                "optimizer": config["optimizer_name"],
                "learning_rate": config["learning_rate"],
                "parameters": config["trainable_parameters"],
                "best_epoch": summary["best_epoch"],
                "best_val_accuracy": summary["best_val_accuracy"],
                "best_val_loss": summary["best_val_loss"],
                "final_train_accuracy": summary["final_train_accuracy"],
                "final_val_accuracy": summary["final_val_accuracy"],
                "generalization_gap": summary["generalization_gap"],
                "training_seconds": summary["total_training_seconds"],
            }
        )

    dataframe = pd.DataFrame(rows)  # 将记录转换为 DataFrame。
    dataframe = dataframe.sort_values("best_val_accuracy", ascending=False).reset_index(drop=True)  # 按最佳验证准确率排序。

    args.output_dir.mkdir(parents=True, exist_ok=True)  # 创建汇总目录。
    dataframe.to_csv(args.output_dir / "experiment_comparison.csv", index=False, encoding="utf-8-sig")

    figure, axis = plt.subplots(figsize=(11, 5.5))  # 创建横向柱状图。
    axis.bar(dataframe["experiment"], dataframe["best_val_accuracy"])  # 绘制最佳验证准确率。
    axis.set_xlabel("Experiment")  # 设置横轴。
    axis.set_ylabel("Best Validation Accuracy")  # 设置纵轴。
    axis.set_title("Week 4 Experiment Comparison")  # 设置标题。
    axis.set_ylim(0.0, 1.0)  # 准确率范围固定在 0～1。
    axis.tick_params(axis="x", rotation=30)  # 旋转实验名称避免重叠。
    axis.grid(axis="y", alpha=0.3)  # 添加横向参考线。

    for index, value in enumerate(dataframe["best_val_accuracy"]):  # 在柱顶标注准确率。
        axis.text(index, float(value) + 0.01, f"{float(value):.4f}", ha="center", fontsize=9)

    figure.tight_layout()  # 调整边距。
    figure.savefig(args.output_dir / "validation_accuracy_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(figure)  # 释放图像资源。

    print(dataframe.to_string(index=False))  # 在终端显示完整对比表。
    print(f"\nSaved comparison to: {args.output_dir}")
    print("请先根据验证集选择最终方案，再只对被选中的检查点运行 evaluate.py。")


if __name__ == "__main__":
    main()  # 直接运行文件时执行汇总。
