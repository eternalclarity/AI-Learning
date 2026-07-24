"""在统一数据划分下比较多种二分类模型。

默认 ``--model-set core`` 完成原第三周计划要求的四模型实验：
Logistic Regression、SVM、Random Forest、MLP。

使用 ``--model-set extended`` 时，会加入新版课程重点模型：
Decision Tree 和 Gradient Boosting。

本脚本只使用训练集拟合模型、使用验证集比较模型。
测试集完全不参与模型选择，最终测试由 evaluate_best_model.py 完成。
"""

# 允许在 Python 3.10 中使用现代类型注解。
from __future__ import annotations

# 导入 argparse，用于从命令行接收随机种子等参数。
import argparse

# 导入 time，用于统计每个模型的训练和预测时间。
import time

# 导入 Path，用于处理文件路径参数。
from pathlib import Path

# 导入 joblib，用于保存已经训练好的 scikit-learn 模型。
import joblib

# 导入 Pandas，用于整理模型对比结果表。
import pandas as pd

# 从公共工具模块导入本实验需要的函数和路径常量。
from utils import (
    DEFAULT_RANDOM_STATE,
    MODEL_DIR,
    PLOT_DIR,
    RESULT_DIR,
    create_models,
    ensure_directories,
    evaluate_model,
    load_dataset,
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_roc_curves,
    save_json,
    save_split_summary,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    # 创建命令行参数解析器。
    parser = argparse.ArgumentParser(
        description="Compare classification models with a validation-only selection workflow."
    )

    # 添加模型集合参数。
    parser.add_argument(
        "--model-set",
        type=str,
        choices=["core", "new_course", "extended"],
        default="core",
        help=(
            "core=原计划四模型；"
            "new_course=决策树/随机森林/梯度提升；"
            "extended=全部模型。"
        ),
    )

    # 添加随机种子参数。
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for data splitting and model initialization.",
    )

    # 返回解析后的参数对象。
    return parser.parse_args()


def add_prefix(
    metrics: dict[str, float],
    prefix: str,
) -> dict[str, float]:
    """给指标名称添加 train_ 或 val_ 前缀。"""

    # 使用字典推导式给每个键添加前缀。
    return {
        f"{prefix}_{metric_name}": metric_value
        for metric_name, metric_value in metrics.items()
    }


def model_name_to_filename(model_name: str) -> str:
    """把模型显示名称转换为适合作为文件名的字符串。"""

    # 转成小写，并把空格替换成下划线。
    return model_name.lower().replace(" ", "_")


def main() -> None:
    """执行候选模型的训练、验证、比较和保存。"""

    # 解析命令行参数。
    args = parse_args()

    # 创建模型、图片和结果目录。
    ensure_directories()

    # 加载输入特征、标签和数据集元数据。
    X, y, metadata = load_dataset()

    # 按照 70% / 15% / 15% 划分数据。
    splits = split_dataset(X, y, random_state=args.random_state)

    # 保存数据集基本信息。
    save_json(metadata, RESULT_DIR / "dataset_metadata.json")

    # 保存数据划分摘要。
    save_split_summary(splits, RESULT_DIR / "data_split_summary.csv")

    # 根据命令行选择创建核心模型、新版树模型或全部模型。
    models = create_models(
        random_state=args.random_state,
        model_set=args.model_set,
    )

    # 创建一个空列表，用于保存每个模型的训练和验证指标。
    result_rows: list[dict[str, float | str]] = []

    # 创建一个空字典，用于保存已经训练好的模型对象。
    fitted_models: dict[str, object] = {}

    # 创建一个空字典，用于保存每个模型在验证集上的预测结果。
    validation_predictions: dict[str, object] = {}

    # 打印实验标题。
    print("=" * 100)

    # 打印实验名称。
    print("Machine Learning Model Comparison")

    # 打印本次使用的模型集合。
    print(f"Model set: {args.model_set}")

    # 打印实际参与比较的模型。
    print(f"Models: {', '.join(models.keys())}")

    # 打印说明。
    print("Positive class: malignant = 1")

    # 打印训练集规模。
    print(f"Training samples:   {len(splits['y_train'])}")

    # 打印验证集规模。
    print(f"Validation samples: {len(splits['y_val'])}")

    # 打印测试集规模，并明确指出当前脚本不会使用测试集。
    print(f"Test samples:       {len(splits['y_test'])} (not used for selection)")

    # 打印分隔线。
    print("=" * 100)

    # 依次训练和验证当前集合中的所有模型。
    for model_name, model in models.items():
        # 打印当前模型名称。
        print(f"\nTraining: {model_name}")

        # 记录训练开始时间。
        training_start = time.perf_counter()

        # 只使用训练集拟合模型。
        model.fit(splits["X_train"], splits["y_train"])

        # 计算训练所用时间。
        training_time = time.perf_counter() - training_start

        # 在训练集上评估模型，用于观察是否可能过拟合。
        train_metrics, _, _ = evaluate_model(
            model,
            splits["X_train"],
            splits["y_train"],
        )

        # 记录验证预测开始时间。
        prediction_start = time.perf_counter()

        # 在验证集上评估模型，用于模型比较和选择。
        val_metrics, val_predictions, _ = evaluate_model(
            model,
            splits["X_val"],
            splits["y_val"],
        )

        # 计算验证集预测时间。
        prediction_time = time.perf_counter() - prediction_start

        # 创建当前模型的完整结果行。
        result_row = {
            "model": model_name,
            "training_time_seconds": training_time,
            "validation_prediction_time_seconds": prediction_time,
            **add_prefix(train_metrics, "train"),
            **add_prefix(val_metrics, "val"),
        }

        # 把当前模型结果加入总结果列表。
        result_rows.append(result_row)

        # 保存当前已经训练好的模型对象。
        fitted_models[model_name] = model

        # 保存当前模型在验证集上的预测标签。
        validation_predictions[model_name] = val_predictions

        # 根据模型名称生成安全文件名。
        model_filename = model_name_to_filename(model_name)

        # 保存当前模型，方便后续单独加载分析。
        joblib.dump(model, MODEL_DIR / f"{model_filename}.joblib")

        # 打印训练集 F1。
        print(f"  Train F1:      {train_metrics['f1']:.4f}")

        # 打印验证集 F1。
        print(f"  Validation F1: {val_metrics['f1']:.4f}")

        # 打印验证集 Recall。
        print(f"  Validation Recall: {val_metrics['recall']:.4f}")

        # 打印验证集 ROC-AUC。
        print(f"  Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")

    # 把所有模型结果转换为 DataFrame。
    results = pd.DataFrame(result_rows)

    # 按验证集 F1、ROC-AUC、Recall 依次从高到低排序。
    results = results.sort_values(
        by=["val_f1", "val_roc_auc", "val_recall"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # 保存完整模型对比结果。
    results.to_csv(
        RESULT_DIR / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 读取排序后第一行的模型名称，作为最佳模型。
    best_model_name = str(results.loc[0, "model"])

    # 从已训练模型字典中取出最佳模型。
    best_model = fitted_models[best_model_name]

    # 取出最佳模型在验证集上的预测标签。
    best_val_predictions = validation_predictions[best_model_name]

    # 建立最佳模型保存载荷，附带模型名称、随机种子和选择规则。
    best_model_payload = {
        "model": best_model,
        "model_name": best_model_name,
        "random_state": args.random_state,
        "positive_label": 1,
        "positive_class": "malignant",
        "selection_rule": "Highest validation F1; ROC-AUC and Recall used as tie-breakers.",
        "model_set": args.model_set,
    }

    # 保存最佳模型及其元数据。
    joblib.dump(best_model_payload, MODEL_DIR / "best_model.joblib")

    # 整理最佳模型的验证集指标。
    best_validation_metrics = {
        key.removeprefix("val_"): float(results.loc[0, key])
        for key in results.columns
        if key.startswith("val_")
    }

    # 保存模型选择摘要。
    save_json(
        {
            "best_model": best_model_name,
            "random_state": args.random_state,
            "selection_metric": "validation_f1",
            "model_set": args.model_set,
            "candidate_models": list(models.keys()),
            "validation_metrics": best_validation_metrics,
            "test_set_used": False,
        },
        RESULT_DIR / "model_selection_summary.json",
    )

    # 绘制候选模型的验证指标对比图。
    plot_metric_comparison(
        results,
        PLOT_DIR / "model_metrics_comparison.png",
    )

    # 绘制候选模型在验证集上的 ROC 曲线。
    plot_roc_curves(
        fitted_models,
        splits["X_val"],
        splits["y_val"],
        PLOT_DIR / "validation_roc_curves.png",
    )

    # 绘制最佳模型在验证集上的混淆矩阵。
    plot_confusion_matrix(
        splits["y_val"],
        best_val_predictions,
        f"Validation Confusion Matrix - {best_model_name}",
        PLOT_DIR / "best_model_validation_confusion_matrix.png",
    )

    # 打印完整对比表中的主要列。
    print("\n" + "=" * 100)

    # 打印结果标题。
    print("Validation comparison")

    # 打印主要指标，保留四位小数。
    print(
        results[
            [
                "model",
                "train_f1",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_f1",
                "val_roc_auc",
                "training_time_seconds",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    # 打印最佳模型名称。
    print(f"\nBest model selected on validation set: {best_model_name}")

    # 打印模型文件路径。
    print(f"Saved best model: {MODEL_DIR / 'best_model.joblib'}")

    # 明确说明测试集尚未使用。
    print("The test set has not been evaluated yet.")

    # 提示运行最终测试脚本。
    print("Run evaluate_best_model.py for the final independent test.")


# 只有直接运行当前脚本时才启动实验。
if __name__ == "__main__":
    # 调用主函数。
    main()
