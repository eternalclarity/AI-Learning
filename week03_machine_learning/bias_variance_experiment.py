"""使用学习曲线观察模型的偏差与方差表现。

学习曲线会随着训练样本数量增加，分别记录：
1. 训练得分；
2. 交叉验证得分。

典型判断：
- 训练分数和验证分数都低且接近：可能高偏差 / 欠拟合；
- 训练分数高、验证分数明显更低：可能高方差 / 过拟合；
- 两条曲线都较高且逐渐接近：泛化表现相对合理。
"""

# 允许在 Python 3.10 中使用现代类型注解。
from __future__ import annotations

# 导入 argparse，用于选择要分析的模型和交叉验证参数。
import argparse

# 导入 Path，用于处理保存路径。
from pathlib import Path

# 导入 matplotlib.pyplot，用于绘制学习曲线。
import matplotlib.pyplot as plt

# 导入 NumPy，用于生成训练集比例和计算均值、标准差。
import numpy as np

# 导入 Pandas，用于保存学习曲线数值表。
import pandas as pd

# 导入 StratifiedKFold，用于保持每折类别比例近似一致。
from sklearn.model_selection import StratifiedKFold, learning_curve

# 从公共工具模块导入模型、数据和路径函数。
from utils import (
    DEFAULT_RANDOM_STATE,
    PLOT_DIR,
    RESULT_DIR,
    create_models,
    ensure_directories,
    get_available_model_names,
    load_dataset,
    save_json,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    # 创建参数解析器。
    parser = argparse.ArgumentParser(
        description="Generate a learning curve for bias-variance diagnosis."
    )

    # 添加模型选择参数。
    parser.add_argument(
        "--model",
        type=str,
        default="Logistic Regression",
        choices=get_available_model_names(),
        help="Model used to generate the learning curve.",
    )

    # 添加可选基准 F1。
    # 基准可以来自旧系统、人工水平、简单规则或业务目标。
    parser.add_argument(
        "--benchmark-f1",
        type=float,
        default=None,
        help=(
            "Optional reference F1 used to discuss avoidable bias. "
            "Do not invent a benchmark only to force a diagnosis."
        ),
    )

    # 添加交叉验证折数参数。
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds.",
    )

    # 添加并行任务数参数；Windows 初学环境下默认 1 最稳定。
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs used by learning_curve.",
    )

    # 添加随机种子参数。
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
    )

    # 返回参数对象。
    return parser.parse_args()


def diagnose_curve(
    final_train_score: float,
    final_validation_score: float,
    benchmark_score: float | None,
) -> dict[str, float | str | None]:
    """整理学习曲线证据，而不是用固定阈值武断下结论。

    新版课程强调：
    1. 训练表现应当与合理基准比较，才能讨论可避免偏差；
    2. 训练与验证之间的差距用于观察方差；
    3. 结论还要结合完整学习曲线、交叉验证波动和错误分析。
    """

    # 计算训练分数与验证分数之间的泛化差距。
    generalization_gap = final_train_score - final_validation_score

    # 如果提供了基准，就计算训练表现距离基准还有多远。
    avoidable_bias_gap = (
        benchmark_score - final_train_score
        if benchmark_score is not None
        else None
    )

    # 创建解释性提示列表。
    messages: list[str] = []

    # 对泛化差距进行描述，但不把某个数值当成跨任务通用真理。
    if generalization_gap > 0.08:
        messages.append(
            "The final training F1 is noticeably higher than the "
            "cross-validation F1; this is evidence consistent with high variance."
        )
    elif generalization_gap > 0.03:
        messages.append(
            "A moderate train-validation gap remains; inspect the curve and fold variance."
        )
    else:
        messages.append(
            "The final train-validation gap is small; this alone does not prove low bias."
        )

    # 只有用户提供了有依据的基准，才讨论可避免偏差。
    if benchmark_score is None:
        messages.append(
            "No benchmark F1 was supplied, so avoidable bias cannot be judged reliably."
        )
    elif avoidable_bias_gap is not None and avoidable_bias_gap > 0.05:
        messages.append(
            "Training F1 is materially below the supplied benchmark; "
            "this is evidence consistent with avoidable bias."
        )
    else:
        messages.append(
            "Training F1 is close to the supplied benchmark; "
            "focus more on the generalization gap and concrete errors."
        )

    # 强调结论需要综合证据。
    messages.append(
        "Treat these as diagnostic clues, not automatic labels; "
        "review the full curve, standard deviations, baseline quality and error cases."
    )

    # 返回结构化诊断，方便写入 JSON。
    return {
        "final_train_f1": final_train_score,
        "final_cross_validation_f1": final_validation_score,
        "generalization_gap": generalization_gap,
        "benchmark_f1": benchmark_score,
        "avoidable_bias_gap": avoidable_bias_gap,
        "interpretation": " ".join(messages),
    }


def main() -> None:
    """生成指定模型的学习曲线。"""

    # 解析命令行参数。
    args = parse_args()

    # 创建输出目录。
    ensure_directories()

    # 加载完整数据集。
    X, y, _ = load_dataset()

    # 按统一规则划分数据；独立测试集仍然不参与学习曲线。
    splits = split_dataset(X, y, random_state=args.random_state)

    # 合并训练集和验证集作为“开发数据”，测试集仍保持封存。
    X_development = pd.concat(
        [splits["X_train"], splits["X_val"]],
        axis=0,
    )

    # 合并训练标签和验证标签。
    y_development = pd.concat(
        [splits["y_train"], splits["y_val"]],
        axis=0,
    )

    # 创建扩展模型集合，使逻辑回归、树模型和 MLP 都可分析。
    models = create_models(
        random_state=args.random_state,
        model_set="extended",
    )

    # 选择用户指定的模型。
    model = models[args.model]

    # 创建分层 K 折交叉验证器。
    cross_validator = StratifiedKFold(
        n_splits=args.cv,
        shuffle=True,
        random_state=args.random_state,
    )

    # 设置五个训练数据比例，从 20% 逐渐增加到 100%。
    train_size_ratios = np.linspace(0.20, 1.00, 5)

    # 计算学习曲线。
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=model,
        X=X_development,
        y=y_development,
        train_sizes=train_size_ratios,
        cv=cross_validator,
        scoring="f1",
        shuffle=True,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    # 计算每个训练规模下的平均训练 F1。
    train_mean = train_scores.mean(axis=1)

    # 计算每个训练规模下训练 F1 的标准差。
    train_std = train_scores.std(axis=1)

    # 计算每个训练规模下的平均验证 F1。
    validation_mean = validation_scores.mean(axis=1)

    # 计算每个训练规模下验证 F1 的标准差。
    validation_std = validation_scores.std(axis=1)

    # 整理学习曲线数据表。
    curve_frame = pd.DataFrame(
        {
            "train_size": train_sizes,
            "train_f1_mean": train_mean,
            "train_f1_std": train_std,
            "validation_f1_mean": validation_mean,
            "validation_f1_std": validation_std,
            "generalization_gap": train_mean - validation_mean,
        }
    )

    # 把模型名称转换成安全文件名。
    file_stem = args.model.lower().replace(" ", "_")

    # 保存学习曲线原始数值。
    curve_frame.to_csv(
        RESULT_DIR / f"learning_curve_{file_stem}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 对最后一个点进行简单诊断。
    diagnosis = diagnose_curve(
        final_train_score=float(train_mean[-1]),
        final_validation_score=float(validation_mean[-1]),
        benchmark_score=args.benchmark_f1,
    )

    # 保存学习曲线摘要。
    save_json(
        {
            "model": args.model,
            "cv_folds": args.cv,
            "scoring": "f1",
            "test_set_used": False,
            "final_train_f1": float(train_mean[-1]),
            "final_validation_f1": float(validation_mean[-1]),
            "final_generalization_gap": float(
                train_mean[-1] - validation_mean[-1]
            ),
            "diagnosis": diagnosis,
        },
        RESULT_DIR / f"learning_curve_{file_stem}_summary.json",
    )

    # 创建学习曲线画布。
    figure, axis = plt.subplots(figsize=(8, 6))

    # 绘制平均训练 F1 曲线。
    axis.plot(
        train_sizes,
        train_mean,
        marker="o",
        label="Training F1",
    )

    # 绘制训练 F1 的标准差阴影。
    axis.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
    )

    # 绘制平均交叉验证 F1 曲线。
    axis.plot(
        train_sizes,
        validation_mean,
        marker="s",
        label="Cross-validation F1",
    )

    # 绘制交叉验证 F1 的标准差阴影。
    axis.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.15,
    )

    # 设置横轴名称。
    axis.set_xlabel("Training Samples")

    # 设置纵轴名称。
    axis.set_ylabel("F1 Score")

    # 设置纵轴范围。
    axis.set_ylim(0.0, 1.05)

    # 设置标题。
    axis.set_title(f"Learning Curve - {args.model}")

    # 显示图例。
    axis.legend()

    # 显示网格。
    axis.grid(alpha=0.3)

    # 调整布局。
    figure.tight_layout()

    # 保存学习曲线图片。
    figure.savefig(
        PLOT_DIR / f"learning_curve_{file_stem}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # 关闭图片。
    plt.close(figure)

    # 打印结果表。
    print(curve_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    # 打印诊断结果。
    print("\nDiagnosis evidence:")
    print(diagnosis["interpretation"])

    # 明确说明测试集没有被使用。
    print("Test set used: False")


# 只有直接运行本文件时才执行 main。
if __name__ == "__main__":
    # 启动学习曲线实验。
    main()
