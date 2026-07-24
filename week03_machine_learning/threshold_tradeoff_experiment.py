"""在验证集上观察分类阈值对 Precision、Recall 和 F1 的影响。

重要规则：
1. 阈值只能使用训练集或验证集选择；
2. 测试集不能用于阈值搜索；
3. 本脚本只生成诊断结果，不会自动修改最终模型文件。
"""

# 允许在 Python 3.10 中使用现代类型注解。
from __future__ import annotations

# 导入 argparse，用于接收模型路径和随机种子。
import argparse

# 导入 Path，用于处理模型文件路径。
from pathlib import Path

# 导入 joblib，用于加载验证阶段保存的最佳模型。
import joblib

# 导入 Matplotlib，用于绘制阈值与指标曲线。
import matplotlib.pyplot as plt

# 导入 NumPy，用于生成阈值序列和执行数组判断。
import numpy as np

# 导入 Pandas，用于保存阈值实验表格。
import pandas as pd

# 导入分类指标函数。
from sklearn.metrics import f1_score, precision_score, recall_score

# 从公共工具模块导入数据、模型分数和输出路径。
from utils import (
    DEFAULT_RANDOM_STATE,
    MODEL_DIR,
    PLOT_DIR,
    RESULT_DIR,
    ensure_directories,
    get_positive_scores,
    load_dataset,
    save_json,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    # 创建参数解析器。
    parser = argparse.ArgumentParser(
        description="Sweep classification thresholds on the validation set."
    )

    # 添加最佳模型文件路径。
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_DIR / "best_model.joblib",
        help="Best-model payload produced by compare_models.py.",
    )

    # 添加随机种子，必须与 compare_models.py 保持一致。
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
    )

    # 添加可选最低 Precision 约束。
    parser.add_argument(
        "--minimum-precision",
        type=float,
        default=None,
        help=(
            "Optional precision constraint. Among thresholds satisfying it, "
            "the script reports the one with highest recall."
        ),
    )

    # 返回解析后的参数对象。
    return parser.parse_args()


def main() -> None:
    """执行验证集阈值扫描。"""

    # 解析参数。
    args = parse_args()

    # 创建输出目录。
    ensure_directories()

    # 检查模型文件是否存在。
    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model payload not found: {args.model_path}\n"
            "Run compare_models.py before this experiment."
        )

    # 加载模型载荷。
    payload = joblib.load(args.model_path)

    # 取出已经训练好的模型。
    model = payload["model"]

    # 取出模型名称。
    model_name = str(payload["model_name"])

    # 检查随机种子是否与模型训练阶段一致。
    saved_random_state = int(payload["random_state"])

    # 随机种子不一致会导致验证集变化，因此直接报错。
    if saved_random_state != args.random_state:
        raise ValueError(
            "random_state does not match model selection. "
            f"Saved={saved_random_state}, current={args.random_state}."
        )

    # 加载完整数据。
    X, y, _ = load_dataset()

    # 重建与训练阶段完全相同的数据划分。
    splits = split_dataset(X, y, random_state=args.random_state)

    # 只获取验证集的正类连续分数。
    validation_scores = get_positive_scores(model, splits["X_val"])

    # 生成 0.05 到 0.95 之间的阈值，共 91 个点。
    thresholds = np.linspace(0.05, 0.95, 91)

    # 创建结果列表。
    rows: list[dict[str, float]] = []

    # 依次使用每个阈值把连续分数转换为 0/1 类别。
    for threshold in thresholds:
        # 分数大于等于阈值时预测为恶性正类 1。
        predictions = (validation_scores >= threshold).astype(int)

        # 计算当前阈值下的 Precision。
        precision = precision_score(
            splits["y_val"],
            predictions,
            zero_division=0,
        )

        # 计算当前阈值下的 Recall。
        recall = recall_score(
            splits["y_val"],
            predictions,
            zero_division=0,
        )

        # 计算当前阈值下的 F1。
        f1 = f1_score(
            splits["y_val"],
            predictions,
            zero_division=0,
        )

        # 保存当前阈值的一行结果。
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "predicted_positive_count": int(predictions.sum()),
            }
        )

    # 把结果转换为 DataFrame。
    results = pd.DataFrame(rows)

    # 保存完整阈值扫描表。
    results.to_csv(
        RESULT_DIR / "validation_threshold_tradeoff.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 找到验证集 F1 最大的行。
    best_f1_row = results.loc[results["f1"].idxmax()]

    # 默认不生成 Precision 约束方案。
    constrained_summary: dict[str, float] | None = None

    # 如果用户提供最低 Precision，就寻找满足约束时 Recall 最大的阈值。
    if args.minimum_precision is not None:
        # 筛选满足 Precision 约束的候选行。
        candidates = results[
            results["precision"] >= args.minimum_precision
        ]

        # 只有存在候选时才生成方案。
        if not candidates.empty:
            # 按 Recall、F1 从高到低排序并选择第一行。
            constrained_row = candidates.sort_values(
                by=["recall", "f1"],
                ascending=[False, False],
            ).iloc[0]

            # 整理约束方案。
            constrained_summary = {
                "minimum_precision": float(args.minimum_precision),
                "threshold": float(constrained_row["threshold"]),
                "precision": float(constrained_row["precision"]),
                "recall": float(constrained_row["recall"]),
                "f1": float(constrained_row["f1"]),
            }

    # 保存阈值实验摘要。
    save_json(
        {
            "model_name": model_name,
            "dataset_used": "validation",
            "test_set_used": False,
            "default_threshold": 0.5,
            "best_validation_f1": {
                "threshold": float(best_f1_row["threshold"]),
                "precision": float(best_f1_row["precision"]),
                "recall": float(best_f1_row["recall"]),
                "f1": float(best_f1_row["f1"]),
            },
            "precision_constrained_solution": constrained_summary,
            "warning": (
                "This is a validation-set diagnostic. "
                "Do not repeatedly tune against the test set."
            ),
        },
        RESULT_DIR / "validation_threshold_summary.json",
    )

    # 创建阈值曲线画布。
    figure, axis = plt.subplots(figsize=(9, 6))

    # 绘制 Precision 曲线。
    axis.plot(
        results["threshold"],
        results["precision"],
        label="Precision",
    )

    # 绘制 Recall 曲线。
    axis.plot(
        results["threshold"],
        results["recall"],
        label="Recall",
    )

    # 绘制 F1 曲线。
    axis.plot(
        results["threshold"],
        results["f1"],
        label="F1",
    )

    # 绘制默认阈值 0.5 的竖线。
    axis.axvline(
        0.5,
        linestyle="--",
        label="Default threshold = 0.5",
    )

    # 设置横轴名称。
    axis.set_xlabel("Classification Threshold")

    # 设置纵轴名称。
    axis.set_ylabel("Validation Score")

    # 设置纵轴范围。
    axis.set_ylim(0.0, 1.05)

    # 设置图标题。
    axis.set_title(f"Precision-Recall-F1 Threshold Trade-off - {model_name}")

    # 显示网格。
    axis.grid(alpha=0.3)

    # 显示图例。
    axis.legend()

    # 调整布局。
    figure.tight_layout()

    # 保存阈值曲线。
    figure.savefig(
        PLOT_DIR / "validation_threshold_tradeoff.png",
        dpi=300,
        bbox_inches="tight",
    )

    # 关闭画布。
    plt.close(figure)

    # 打印结果摘要。
    print(f"Model: {model_name}")
    print("Dataset used: validation")
    print("Test set used: False")
    print(
        "Best validation F1 threshold: "
        f"{best_f1_row['threshold']:.2f} | "
        f"Precision={best_f1_row['precision']:.4f} | "
        f"Recall={best_f1_row['recall']:.4f} | "
        f"F1={best_f1_row['f1']:.4f}"
    )

    # 如果存在满足 Precision 约束的方案，则打印。
    if constrained_summary is not None:
        print("Precision-constrained solution:")
        print(constrained_summary)


# 只有直接运行当前文件时才执行主函数。
if __name__ == "__main__":
    # 启动阈值权衡实验。
    main()
