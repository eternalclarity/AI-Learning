"""
对比 三个模型在 训练，验证，测试集上的结果
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

# 文件夹名称、图例名称和绘图颜色保持一一对应。
MODEL_SETTINGS = {
    "MLP": {"label": "MLP", "color": "#4C78A8"},
    "B_CNN": {"label": "Basic CNN", "color": "#F58518"},
    "I_CNN": {"label": "Improved CNN", "color": "#54A24B"},
}


TEST_METRICS = {
    "MLP": {"test_loss": 0.3252, "test_accuracy": 88.61},
    "B_CNN": {"test_loss": 0.2460, "test_accuracy": 91.89},
    "I_CNN": {"test_loss": 0.2217, "test_accuracy": 92.37},
}

REQUIRED_HISTORY_KEYS = (
    "train_loss",
    "val_loss",
    "train_accuracy",
    "val_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot result comparisons for MLP, Basic CNN, and Improved CNN."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=BASE_DIR / "outputs",
        help="Directory containing the MLP, B_CNN, and I_CNN output folders.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=BASE_DIR / "outputs" / "comparison",
        help="Directory used to save comparison figures.",
    )
    return parser.parse_args()


def load_history(history_path: Path) -> dict[str, list[float]]:
    """读取并检查单个模型的训练历史。"""

    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    with history_path.open("r", encoding="utf-8") as file:
        history: dict[str, Any] = json.load(file)

    for key in REQUIRED_HISTORY_KEYS:
        values = history.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{history_path}: '{key}' must be a non-empty list.")
        if not all(isinstance(value, (int, float)) for value in values):
            raise ValueError(f"{history_path}: '{key}' contains non-numeric data.")

    return {key: [float(value) for value in history[key]] for key in REQUIRED_HISTORY_KEYS}


def load_all_histories(outputs_dir: Path) -> dict[str, dict[str, list[float]]]:
    return {
        model_name: load_history(outputs_dir / model_name / "history.json")
        for model_name in MODEL_SETTINGS
    }


def plot_metric_curves(
    axis: plt.Axes,
    histories: dict[str, dict[str, list[float]]],
    metric_key: str,
    title: str,
    y_label: str,
) -> None:
    """在同一坐标轴上绘制三个模型的一项指标。"""

    for model_name, settings in MODEL_SETTINGS.items():
        values = histories[model_name][metric_key]
        epochs = range(1, len(values) + 1)
        axis.plot(
            epochs,
            values,
            marker="o",
            linewidth=2,
            markersize=4,
            label=settings["label"],
            color=settings["color"],
        )

    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.25, linestyle="--")
    axis.legend()


def save_training_comparison(
    histories: dict[str, dict[str, list[float]]], save_path: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_metric_curves(axes[0], histories, "train_loss", "Training Loss", "Loss")
    plot_metric_curves(
        axes[1], histories, "train_accuracy", "Training Accuracy", "Accuracy (%)"
    )
    figure.suptitle("FashionMNIST Training Comparison", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_validation_comparison(
    histories: dict[str, dict[str, list[float]]], save_path: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_metric_curves(axes[0], histories, "val_loss", "Validation Loss", "Loss")
    plot_metric_curves(
        axes[1], histories, "val_accuracy", "Validation Accuracy", "Accuracy (%)"
    )
    figure.suptitle("FashionMNIST Validation Comparison", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def add_bar_labels(axis: plt.Axes, bars: Any, value_format: str) -> None:
    for bar in bars:
        value = bar.get_height()
        axis.annotate(
            value_format.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def save_test_comparison(save_path: Path) -> None:
    labels = [settings["label"] for settings in MODEL_SETTINGS.values()]
    colors = [settings["color"] for settings in MODEL_SETTINGS.values()]
    losses = [TEST_METRICS[name]["test_loss"] for name in MODEL_SETTINGS]
    accuracies = [TEST_METRICS[name]["test_accuracy"] for name in MODEL_SETTINGS]

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    loss_bars = axes[0].bar(labels, losses, color=colors, width=0.62)
    axes[0].set_title("Test Loss", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Loss")
    axes[0].set_ylim(0, max(losses) * 1.25)
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")
    add_bar_labels(axes[0], loss_bars, "{:.4f}")

    accuracy_bars = axes[1].bar(labels, accuracies, color=colors, width=0.62)
    axes[1].set_title("Test Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    add_bar_labels(axes[1], accuracy_bars, "{:.2f}%")

    figure.suptitle("FashionMNIST Test Comparison", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    histories = load_all_histories(args.outputs_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "training": args.save_dir / "training_comparison.png",
        "validation": args.save_dir / "validation_comparison.png",
        "test": args.save_dir / "test_comparison.png",
    }

    save_training_comparison(histories, output_paths["training"])
    save_validation_comparison(histories, output_paths["validation"])
    save_test_comparison(output_paths["test"])

    print("Comparison figures generated successfully:")
    for name, path in output_paths.items():
        print(f"  {name:<10}: {path}")

    print("\nTest metrics used:")
    for model_name, settings in MODEL_SETTINGS.items():
        metrics = TEST_METRICS[model_name]
        print(
            f"  {settings['label']:<12} | "
            f"loss={metrics['test_loss']:.4f}, "
            f"accuracy={metrics['test_accuracy']:.2f}%"
        )


if __name__ == "__main__":
    main()
