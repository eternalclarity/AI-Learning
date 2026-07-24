"""新版吴恩达课程 87～99 集的树模型扩展实验。

本脚本完成：
1. 比较 Decision Tree、Random Forest 和 Gradient Boosting；
2. 扫描不同决策树深度，观察训练/验证 F1 与高方差；
3. 保存随机森林特征重要性；
4. 使用验证集选择最佳树模型；
5. 测试集仍保持封存，不在本脚本中使用。
"""

# 允许在 Python 3.10 中使用现代类型注解。
from __future__ import annotations

# 导入 argparse，用于接收随机种子等命令行参数。
import argparse

# 导入 time，用于统计模型训练时间。
import time

# 导入 joblib，用于保存已经训练好的模型。
import joblib

# 导入 Matplotlib，用于绘制深度曲线和特征重要性。
import matplotlib.pyplot as plt

# 导入 Pandas，用于整理实验结果表。
import pandas as pd

# 导入决策树分类器，用于扫描不同最大深度。
from sklearn.tree import DecisionTreeClassifier

# 从公共工具模块导入数据、模型、指标和输出函数。
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
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    # 创建参数解析器。
    parser = argparse.ArgumentParser(
        description="Run the new-course decision-tree and ensemble experiment."
    )

    # 添加统一随机种子参数。
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for data splitting and model initialization.",
    )

    # 返回解析后的参数。
    return parser.parse_args()


def safe_filename(model_name: str) -> str:
    """把模型显示名称转换成适合文件系统的名称。"""

    # 转换成小写，并用下划线替换空格。
    return model_name.lower().replace(" ", "_")


def run_model_comparison(
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, object], str]:
    """训练并比较新版课程中的树模型。"""

    # 加载数据集。
    X, y, _ = load_dataset()

    # 使用统一随机种子划分训练、验证和测试集。
    splits = split_dataset(X, y, random_state=random_state)

    # 创建新版课程树模型集合。
    models = create_models(
        random_state=random_state,
        model_set="new_course",
    )

    # 创建结果列表，用于保存每个模型的一行指标。
    result_rows: list[dict[str, float | str]] = []

    # 创建已训练模型字典。
    fitted_models: dict[str, object] = {}

    # 创建验证集预测字典。
    validation_predictions: dict[str, object] = {}

    # 依次训练每个树模型。
    for model_name, model in models.items():
        # 打印当前模型名称。
        print(f"Training: {model_name}")

        # 记录训练开始时间。
        start_time = time.perf_counter()

        # 只使用训练集拟合模型。
        model.fit(splits["X_train"], splits["y_train"])

        # 计算训练时长。
        training_time = time.perf_counter() - start_time

        # 在训练集上计算指标，用于观察是否高方差。
        train_metrics, _, _ = evaluate_model(
            model,
            splits["X_train"],
            splits["y_train"],
        )

        # 在验证集上计算指标，用于模型比较。
        val_metrics, val_predictions, _ = evaluate_model(
            model,
            splits["X_val"],
            splits["y_val"],
        )

        # 整理当前模型的一行结果。
        result_rows.append(
            {
                "model": model_name,
                "training_time_seconds": training_time,
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "train_roc_auc": train_metrics["roc_auc"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "generalization_gap_f1": train_metrics["f1"] - val_metrics["f1"],
            }
        )

        # 保存已经训练好的模型对象。
        fitted_models[model_name] = model

        # 保存验证集预测标签。
        validation_predictions[model_name] = val_predictions

        # 把当前模型单独保存到模型目录。
        joblib.dump(
            model,
            MODEL_DIR / f"tree_{safe_filename(model_name)}.joblib",
        )

    # 把结果列表转换为 DataFrame。
    results = pd.DataFrame(result_rows)

    # 按验证 F1、ROC-AUC、Recall 排序。
    results = results.sort_values(
        by=["val_f1", "val_roc_auc", "val_recall"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # 保存树模型比较结果。
    results.to_csv(
        RESULT_DIR / "tree_ensemble_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 取出验证集上排名第一的模型名称。
    best_model_name = str(results.loc[0, "model"])

    # 取出最佳树模型。
    best_model = fitted_models[best_model_name]

    # 保存最佳树模型及元数据。
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "random_state": random_state,
            "positive_label": 1,
            "positive_class": "malignant",
            "selection_rule": "Highest validation F1; ROC-AUC and Recall as tie-breakers.",
            "test_set_used": False,
        },
        MODEL_DIR / "best_tree_ensemble_model.joblib",
    )

    # 保存结构化模型选择摘要。
    save_json(
        {
            "best_tree_model": best_model_name,
            "candidate_models": list(models.keys()),
            "test_set_used": False,
            "selection_metric": "validation_f1",
        },
        RESULT_DIR / "tree_ensemble_selection_summary.json",
    )

    # 绘制树模型验证指标对比图。
    plot_metric_comparison(
        results,
        PLOT_DIR / "tree_ensemble_metrics_comparison.png",
    )

    # 绘制树模型验证集 ROC 曲线。
    plot_roc_curves(
        fitted_models,
        splits["X_val"],
        splits["y_val"],
        PLOT_DIR / "tree_ensemble_validation_roc.png",
    )

    # 绘制最佳树模型验证集混淆矩阵。
    plot_confusion_matrix(
        splits["y_val"],
        validation_predictions[best_model_name],
        f"Validation Confusion Matrix - {best_model_name}",
        PLOT_DIR / "best_tree_model_validation_confusion_matrix.png",
    )

    # 返回结果、已训练模型和最佳模型名称。
    return results, fitted_models, best_model_name


def run_depth_sweep(random_state: int) -> pd.DataFrame:
    """扫描决策树深度，观察模型复杂度与泛化差距。"""

    # 加载数据集。
    X, y, _ = load_dataset()

    # 构造相同的数据划分。
    splits = split_dataset(X, y, random_state=random_state)

    # 设置需要比较的最大深度；None 表示不限制深度。
    max_depth_values: list[int | None] = [1, 2, 3, 4, 5, 6, 8, 10, None]

    # 创建结果列表。
    rows: list[dict[str, float | int | str]] = []

    # 依次训练不同深度的决策树。
    for max_depth in max_depth_values:
        # 创建当前深度的决策树。
        model = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=max_depth,
            random_state=random_state,
        )

        # 使用训练集拟合模型。
        model.fit(splits["X_train"], splits["y_train"])

        # 计算训练集指标。
        train_metrics, _, _ = evaluate_model(
            model,
            splits["X_train"],
            splits["y_train"],
        )

        # 计算验证集指标。
        val_metrics, _, _ = evaluate_model(
            model,
            splits["X_val"],
            splits["y_val"],
        )

        # 保存当前深度的结果。
        rows.append(
            {
                "max_depth": "None" if max_depth is None else max_depth,
                "actual_tree_depth": int(model.get_depth()),
                "leaf_count": int(model.get_n_leaves()),
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
                "generalization_gap_f1": train_metrics["f1"] - val_metrics["f1"],
            }
        )

    # 转换为 DataFrame。
    depth_results = pd.DataFrame(rows)

    # 保存原始数值。
    depth_results.to_csv(
        RESULT_DIR / "decision_tree_depth_sweep.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 使用顺序编号作为横轴，便于同时表示有限深度和 None。
    x_positions = list(range(len(depth_results)))

    # 创建画布。
    figure, axis = plt.subplots(figsize=(9, 6))

    # 绘制训练 F1。
    axis.plot(
        x_positions,
        depth_results["train_f1"],
        marker="o",
        label="Training F1",
    )

    # 绘制验证 F1。
    axis.plot(
        x_positions,
        depth_results["val_f1"],
        marker="s",
        label="Validation F1",
    )

    # 设置横轴刻度。
    axis.set_xticks(x_positions)

    # 设置横轴刻度文字。
    axis.set_xticklabels(depth_results["max_depth"].astype(str))

    # 设置横轴标题。
    axis.set_xlabel("Decision Tree max_depth")

    # 设置纵轴标题。
    axis.set_ylabel("F1 Score")

    # 设置纵轴范围。
    axis.set_ylim(0.0, 1.05)

    # 设置图标题。
    axis.set_title("Decision Tree Complexity: Training vs Validation F1")

    # 显示网格。
    axis.grid(alpha=0.3)

    # 显示图例。
    axis.legend()

    # 调整布局。
    figure.tight_layout()

    # 保存图片。
    figure.savefig(
        PLOT_DIR / "decision_tree_depth_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    # 关闭画布。
    plt.close(figure)

    # 返回深度扫描结果。
    return depth_results


def save_random_forest_feature_importance(
    fitted_models: dict[str, object],
) -> pd.DataFrame:
    """保存随机森林的前 15 个内置特征重要性。"""

    # 重新加载特征名称。
    X, _, _ = load_dataset()

    # 取出已经训练好的随机森林。
    random_forest = fitted_models["Random Forest"]

    # 把特征名称与内置重要性组合成 DataFrame。
    importance_frame = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": random_forest.feature_importances_,
        }
    )

    # 按重要性从高到低排序。
    importance_frame = importance_frame.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)

    # 保存全部特征重要性。
    importance_frame.to_csv(
        RESULT_DIR / "random_forest_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 选择前 15 个特征并反转顺序，便于水平柱状图从下到上显示。
    top_features = importance_frame.head(15).iloc[::-1]

    # 创建画布。
    figure, axis = plt.subplots(figsize=(9, 7))

    # 绘制水平柱状图。
    axis.barh(top_features["feature"], top_features["importance"])

    # 设置横轴标题。
    axis.set_xlabel("Impurity-based Feature Importance")

    # 设置图标题。
    axis.set_title("Random Forest Feature Importance")

    # 调整布局。
    figure.tight_layout()

    # 保存图片。
    figure.savefig(
        PLOT_DIR / "random_forest_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    # 关闭画布。
    plt.close(figure)

    # 返回完整重要性表。
    return importance_frame


def main() -> None:
    """运行树模型扩展实验。"""

    # 解析命令行参数。
    args = parse_args()

    # 创建输出目录。
    ensure_directories()

    # 运行树模型对比。
    results, fitted_models, best_model_name = run_model_comparison(
        random_state=args.random_state,
    )

    # 运行决策树深度扫描。
    depth_results = run_depth_sweep(
        random_state=args.random_state,
    )

    # 保存随机森林特征重要性。
    feature_importance = save_random_forest_feature_importance(
        fitted_models=fitted_models,
    )

    # 打印树模型对比结果。
    print("\nTree ensemble validation comparison:")
    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # 打印深度扫描结果。
    print("\nDecision tree depth sweep:")
    print(
        depth_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # 打印前 10 个随机森林重要特征。
    print("\nTop random-forest features:")
    print(
        feature_importance.head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    # 打印最佳树模型名称。
    print(f"\nBest tree model selected on validation set: {best_model_name}")

    # 明确说明测试集没有被使用。
    print("Test set used: False")


# 只有直接运行当前文件时才执行主函数。
if __name__ == "__main__":
    # 启动树模型扩展实验。
    main()
