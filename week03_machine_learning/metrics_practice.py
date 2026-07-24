"""通过一组人工二分类结果，手动理解混淆矩阵和常见分类指标。

建议学习顺序：
1. 先不运行程序，自己在纸上统计 TP、TN、FP、FN；
2. 自己代入公式计算 Accuracy、Precision、Recall 和 F1；
3. 再运行本程序，与 scikit-learn 的结果对照。
"""

# 允许在 Python 3.10 中使用现代类型注解。
from __future__ import annotations

# 导入 NumPy，用于保存真实标签和预测标签。
import numpy as np

# 导入 scikit-learn 的标准评估指标，用于验证手算结果。
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[int, int, int, int]:
    """手动统计 TN、FP、FN 和 TP。"""

    # 统计真实为 0 且预测也为 0 的样本数量。
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))

    # 统计真实为 0 但预测为 1 的样本数量。
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))

    # 统计真实为 1 但预测为 0 的样本数量。
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))

    # 统计真实为 1 且预测也为 1 的样本数量。
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))

    # 返回四种情况。
    return true_negative, false_positive, false_negative, true_positive


def safe_divide(numerator: float, denominator: float) -> float:
    """安全执行除法，避免分母为 0 时程序报错。"""

    # 如果分母为 0，就返回 0.0。
    if denominator == 0:
        # 返回 0.0 表示当前指标无法由有效样本计算。
        return 0.0

    # 分母非 0 时正常计算商。
    return numerator / denominator


def calculate_metrics_manually(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
) -> dict[str, float]:
    """根据混淆矩阵中的四个数量手动计算分类指标。"""

    # 计算总样本数量。
    total = tn + fp + fn + tp

    # 计算准确率：所有预测正确样本占总样本的比例。
    accuracy = safe_divide(tp + tn, total)

    # 计算精确率：预测为正类的样本中，真正为正类的比例。
    precision = safe_divide(tp, tp + fp)

    # 计算召回率：所有真实正类中，被模型找出的比例。
    recall = safe_divide(tp, tp + fn)

    # 计算特异度：所有真实负类中，被正确识别为负类的比例。
    specificity = safe_divide(tn, tn + fp)

    # 计算 F1，它是 Precision 和 Recall 的调和平均数。
    f1 = safe_divide(2 * precision * recall, precision + recall)

    # 返回手动计算出的指标。
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def main() -> None:
    """运行完整的指标练习。"""

    # 构造真实标签：1 表示正类，0 表示负类。
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    # 构造模型预测标签，其中故意包含一次漏报和一次误报。
    y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 0])

    # 手动统计 TN、FP、FN 和 TP。
    tn, fp, fn, tp = calculate_confusion_counts(y_true, y_pred)

    # 根据四个数量手动计算指标。
    manual_metrics = calculate_metrics_manually(tn, fp, fn, tp)

    # 使用 scikit-learn 计算混淆矩阵。
    sklearn_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 使用 scikit-learn 计算标准指标。
    sklearn_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # 打印分隔线。
    print("=" * 60)

    # 打印标题。
    print("Binary Classification Metrics Practice")

    # 打印分隔线。
    print("=" * 60)

    # 打印真实标签。
    print(f"y_true: {y_true.tolist()}")

    # 打印预测标签。
    print(f"y_pred: {y_pred.tolist()}")

    # 打印空行。
    print()

    # 打印手动统计的四个混淆矩阵元素。
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # 打印标准混淆矩阵。
    print("\nscikit-learn confusion matrix:")

    # 打印矩阵内容。
    print(sklearn_matrix)

    # 提示矩阵布局。
    print("Layout: [[TN, FP], [FN, TP]]")

    # 打印手算指标标题。
    print("\nManual metrics:")

    # 逐项打印手算指标。
    for metric_name, metric_value in manual_metrics.items():
        # 按 4 位小数显示指标。
        print(f"{metric_name:<12}: {metric_value:.4f}")

    # 打印 scikit-learn 指标标题。
    print("\nscikit-learn metrics:")

    # 逐项打印 scikit-learn 指标。
    for metric_name, metric_value in sklearn_metrics.items():
        # 按 4 位小数显示指标。
        print(f"{metric_name:<12}: {metric_value:.4f}")

    # 使用断言检查 Accuracy 是否一致。
    assert np.isclose(manual_metrics["accuracy"], sklearn_metrics["accuracy"])

    # 使用断言检查 Precision 是否一致。
    assert np.isclose(manual_metrics["precision"], sklearn_metrics["precision"])

    # 使用断言检查 Recall 是否一致。
    assert np.isclose(manual_metrics["recall"], sklearn_metrics["recall"])

    # 使用断言检查 F1 是否一致。
    assert np.isclose(manual_metrics["f1"], sklearn_metrics["f1"])

    # 打印验证成功提示。
    print("\nAll manual calculations match scikit-learn.")


# 只有直接运行当前文件时，才执行 main 函数。
if __name__ == "__main__":
    # 调用程序入口。
    main()
