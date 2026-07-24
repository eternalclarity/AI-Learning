"""第三周机器学习对比实验的公共工具函数。

本文件集中负责：
1. 加载并整理数据集；
2. 划分训练集、验证集和测试集；
3. 根据实验集合创建待比较模型；
4. 计算分类指标；
5. 保存 JSON、CSV 和图片。

把公共逻辑集中在 utils.py 中，可以避免 compare_models.py、
evaluate_best_model.py 和 bias_variance_experiment.py 重复编写代码。
"""

# 允许在 Python 3.10 中更自然地使用较新的类型注解写法。
from __future__ import annotations

# 导入 json，用于把实验配置和指标保存成 JSON 文件。
import json

# 导入 Path，用面向对象的方式处理跨平台文件路径。
from pathlib import Path

# 导入 Any，用于表示函数参数可以接受多种数据类型。
from typing import Any

# 导入 matplotlib.pyplot，用于绘制模型对比图、ROC 曲线和混淆矩阵。
import matplotlib.pyplot as plt

# 导入 NumPy，用于数值运算和数组转换。
import numpy as np

# 导入 Pandas，用于保存表格、处理 DataFrame 和拼接数据。
import pandas as pd

# 导入 Wisconsin Breast Cancer 二分类数据集。
from sklearn.datasets import load_breast_cancer

# 导入随机森林和直方图梯度提升分类器。
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# 导入逻辑回归分类器。
from sklearn.linear_model import LogisticRegression

# 导入常用分类评估指标。
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# 导入数据集划分函数。
from sklearn.model_selection import train_test_split

# 导入 MLP 多层感知机分类器。
from sklearn.neural_network import MLPClassifier

# 导入 Pipeline，把标准化与模型训练串成一个完整流程。
from sklearn.pipeline import Pipeline

# 导入 StandardScaler，用于把特征标准化到均值约为 0、标准差约为 1。
from sklearn.preprocessing import StandardScaler

# 导入支持向量机分类器。
from sklearn.svm import SVC

# 导入单棵决策树分类器。
from sklearn.tree import DecisionTreeClassifier


# 设置项目根目录，也就是当前 utils.py 所在目录。
BASE_DIR = Path(__file__).resolve().parent

# 设置输出目录，用于统一保存模型、图片和结果文件。
OUTPUT_DIR = BASE_DIR / "outputs"

# 设置模型保存目录。
MODEL_DIR = OUTPUT_DIR / "models"

# 设置图片保存目录。
PLOT_DIR = OUTPUT_DIR / "plots"

# 设置实验结果保存目录。
RESULT_DIR = OUTPUT_DIR / "results"

# 设置全项目统一使用的随机种子，保证数据划分和模型初始化尽量可复现。
DEFAULT_RANDOM_STATE = 42

# 明确本项目的二分类标签含义。
# 0 表示良性，1 表示恶性。
CLASS_NAMES = {0: "benign", 1: "malignant"}


def ensure_directories() -> None:
    """创建项目运行时需要的全部输出目录。"""

    # 创建模型目录；如果上级目录不存在，也一并创建。
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 创建图片目录；如果目录已经存在，不报错。
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 创建结果目录；如果目录已经存在，不报错。
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """加载并整理 Wisconsin Breast Cancer 数据集。

    返回：
        X:
            输入特征 DataFrame，形状为 [样本数, 特征数]。

        y:
            二分类标签 Series。
            本项目人为规定：1=malignant，0=benign。

        metadata:
            数据集名称、样本数、特征数和类别含义等元数据。
    """

    # 以 Pandas DataFrame 形式加载数据，方便观察列名和统计信息。
    dataset = load_breast_cancer(as_frame=True)

    # 复制输入特征，避免后续修改影响原始数据对象。
    X = dataset.data.copy()

    # scikit-learn 原始标签是：0=malignant，1=benign。
    # 为了让“恶性”成为我们重点关注的正类，把原标签 0 映射为新标签 1。
    y = (dataset.target == 0).astype(int)

    # 给标签列设置一个清晰的名称。
    y.name = "is_malignant"

    # 整理数据集元数据，后续可写入 JSON 文件或 README。
    metadata = {
        "dataset_name": "Wisconsin Breast Cancer Diagnostic",
        "sample_count": int(X.shape[0]),
        "feature_count": int(X.shape[1]),
        "negative_class": "benign",
        "positive_class": "malignant",
        "positive_label": 1,
        "negative_label": 0,
        "feature_names": list(X.columns),
    }

    # 返回输入特征、标签和元数据。
    return X, y, metadata


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, pd.DataFrame | pd.Series]:
    """按照 70% / 15% / 15% 划分训练、验证和测试数据。

    划分原则：
    1. 使用 stratify 保持三个集合中的类别比例近似一致；
    2. 使用固定 random_state 保证每次运行得到相同划分；
    3. 测试集在模型选择阶段完全不参与训练和比较。
    """

    # 第一次划分：保留 70% 作为训练集，剩余 30% 作为临时集合。
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    # 第二次划分：把临时集合平均拆成验证集和测试集，各占原始数据的约 15%。
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    # 把所有划分结果放入字典，方便其他脚本统一访问。
    splits = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    # 返回完整的数据划分结果。
    return splits


def create_models(
    random_state: int = DEFAULT_RANDOM_STATE,
    model_set: str = "core",
) -> dict[str, Any]:
    """根据实验目的创建一组分类模型。

    参数：
        random_state:
            数据和模型初始化统一使用的随机种子。

        model_set:
            ``core``：原学习计划要求的四模型；
            ``new_course``：新版课程树模型扩展；
            ``extended``：把核心模型与树模型放在一起比较。

    返回：
        以模型显示名称为键、scikit-learn 估计器为值的字典。

    说明：
        Logistic Regression、SVM 和 MLP 对特征尺度较敏感，
        因此使用 Pipeline 把 StandardScaler 与模型绑定。
        决策树、随机森林和梯度提升树按特征阈值分裂，
        通常不要求先做 StandardScaler。
    """

    # 创建“标准化 + 逻辑回归”流水线。
    logistic_regression = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # 创建“标准化 + RBF 核 SVM”流水线。
    svm = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                SVC(
                    C=1.0,
                    kernel="rbf",
                    gamma="scale",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # 创建单棵决策树。
    # criterion="entropy" 与新版课程中的熵和信息增益讲解对应。
    decision_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
    )

    # 创建随机森林分类器。
    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )

    # 创建经典梯度提升分类器，用于实践新版 Boosting 主线。
    # 它属于梯度提升树，但不是 XGBoost 的同义词。
    gradient_boosting = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=3,
        random_state=random_state,
    )

    # 创建“标准化 + MLP”流水线。
    mlp = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    learning_rate_init=0.001,
                    max_iter=2000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=50,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # 所有可用模型。
    all_models = {
        "Logistic Regression": logistic_regression,
        "SVM": svm,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
        "Gradient Boosting": gradient_boosting,
        "MLP": mlp,
    }

    # 不同实验集合对应不同模型名称。
    model_sets = {
        "core": [
            "Logistic Regression",
            "SVM",
            "Random Forest",
            "MLP",
        ],
        "new_course": [
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
        ],
        "extended": [
            "Logistic Regression",
            "SVM",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "MLP",
        ],
    }

    # 检查集合名称是否合法。
    if model_set not in model_sets:
        raise ValueError(
            f"Unknown model_set={model_set!r}. "
            f"Choose from {sorted(model_sets)}."
        )

    # 只返回当前实验需要的模型，并保持预先定义的顺序。
    return {
        model_name: all_models[model_name]
        for model_name in model_sets[model_set]
    }


def get_available_model_names() -> list[str]:
    """返回全部可用于学习曲线分析的模型名称。"""

    # 创建扩展模型集合并返回其中所有键。
    return list(create_models(model_set="extended").keys())


def get_positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """获得每个样本属于正类（malignant=1）的连续分数。

    ROC-AUC 和 ROC 曲线需要使用连续分数，而不能只使用 0/1 预测标签。
    优先使用 predict_proba；如果模型没有该方法，再使用 decision_function。
    """

    # 判断模型是否支持 predict_proba。
    if hasattr(model, "predict_proba"):
        # 计算每个类别的预测概率。
        probabilities = model.predict_proba(X)

        # 获取模型内部记录的类别顺序。
        classes = np.asarray(model.classes_)

        # 找到正类标签 1 在类别数组中的列索引。
        positive_index = int(np.where(classes == 1)[0][0])

        # 返回正类对应的预测概率。
        return probabilities[:, positive_index]

    # 如果模型没有 predict_proba，但支持 decision_function，就使用决策分数。
    if hasattr(model, "decision_function"):
        # 返回模型的连续决策分数。
        return np.asarray(model.decision_function(X))

    # 如果两种连续分数接口都不存在，就抛出明确错误。
    raise AttributeError(
        "The estimator must provide predict_proba or decision_function."
    )


def calculate_binary_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """计算本项目使用的全部二分类指标。"""

    # 根据真实标签和预测标签计算混淆矩阵。
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 按照 [[TN, FP], [FN, TP]] 的顺序拆出四个值。
    tn, fp, fn, tp = matrix.ravel()

    # 计算特异度，也就是负类中被正确识别为负类的比例。
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 计算并整理所有指标。
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    # 返回指标字典。
    return metrics


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """对一个已经训练好的模型进行预测并计算指标。"""

    # 预测离散类别标签 0 或 1。
    y_pred = model.predict(X)

    # 获得属于正类的连续概率或决策分数。
    y_score = get_positive_scores(model, X)

    # 计算完整指标。
    metrics = calculate_binary_metrics(y, y_pred, y_score)

    # 同时返回指标、离散预测和连续分数。
    return metrics, y_pred, y_score


def save_json(data: Any, save_path: Path) -> None:
    """把 Python 对象保存为可读的 UTF-8 JSON 文件。"""

    # 确保目标文件的上级目录存在。
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 以写入模式打开文件，并明确使用 UTF-8 编码。
    with save_path.open("w", encoding="utf-8") as file:
        # 把对象序列化为 JSON；ensure_ascii=False 可以正常保存中文。
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_split_summary(
    splits: dict[str, pd.DataFrame | pd.Series],
    save_path: Path,
) -> None:
    """保存训练集、验证集和测试集的规模与类别比例。"""

    # 创建一个空列表，用于逐行记录三个数据子集的信息。
    rows: list[dict[str, Any]] = []

    # 依次处理训练集、验证集和测试集。
    for split_name in ("train", "val", "test"):
        # 根据名称取出当前集合的标签。
        y_split = splits[f"y_{split_name}"]

        # 统计当前集合中的样本数和类别比例。
        row = {
            "split": split_name,
            "sample_count": int(len(y_split)),
            "benign_count": int((y_split == 0).sum()),
            "malignant_count": int((y_split == 1).sum()),
            "malignant_ratio": float((y_split == 1).mean()),
        }

        # 把当前行加入列表。
        rows.append(row)

    # 把列表转换为 DataFrame。
    summary = pd.DataFrame(rows)

    # 把数据划分摘要保存为 CSV。
    summary.to_csv(save_path, index=False, encoding="utf-8-sig")


def plot_metric_comparison(
    results: pd.DataFrame,
    save_path: Path,
) -> None:
    """绘制多个模型在验证集上的主要指标对比图。"""

    # 指定需要展示的主要指标列。
    metric_columns = [
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_roc_auc",
    ]

    # 复制模型名称和指标，避免修改原始结果表。
    plot_data = results[["model", *metric_columns]].copy()

    # 将模型名称设置为横轴索引。
    plot_data = plot_data.set_index("model")

    # 把列名改成更适合图例显示的名称。
    plot_data.columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]

    # 创建画布。
    figure, axis = plt.subplots(figsize=(11, 6))

    # 绘制分组柱状图。
    plot_data.plot(kind="bar", ax=axis)

    # 设置纵轴范围为 0 到 1，方便直观比较分类指标。
    axis.set_ylim(0.0, 1.05)

    # 设置横轴名称。
    axis.set_xlabel("Model")

    # 设置纵轴名称。
    axis.set_ylabel("Score")

    # 设置图标题。
    axis.set_title("Validation Metrics Comparison")

    # 添加水平网格线，帮助读取数值。
    axis.grid(axis="y", alpha=0.3)

    # 旋转横轴标签，避免模型名称重叠。
    axis.tick_params(axis="x", rotation=15)

    # 调整布局，避免文字被裁剪。
    figure.tight_layout()

    # 保存图片。
    figure.savefig(save_path, dpi=300, bbox_inches="tight")

    # 关闭画布，释放内存。
    plt.close(figure)


def plot_roc_curves(
    fitted_models: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    save_path: Path,
) -> None:
    """绘制多个模型在同一数据集上的 ROC 曲线。"""

    # 创建画布和坐标轴。
    figure, axis = plt.subplots(figsize=(8, 6))

    # 依次处理每个已经训练好的模型。
    for model_name, model in fitted_models.items():
        # 获取当前模型的正类连续分数。
        y_score = get_positive_scores(model, X)

        # 计算 ROC 曲线上的假正例率、真正例率和阈值。
        false_positive_rate, true_positive_rate, _ = roc_curve(y, y_score)

        # 计算 ROC-AUC。
        auc_value = roc_auc_score(y, y_score)

        # 绘制当前模型的 ROC 曲线，并在图例中显示 AUC。
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{model_name} (AUC={auc_value:.3f})",
        )

    # 绘制随机猜测对应的对角虚线。
    axis.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

    # 设置横轴名称。
    axis.set_xlabel("False Positive Rate")

    # 设置纵轴名称。
    axis.set_ylabel("True Positive Rate")

    # 设置标题。
    axis.set_title("Validation ROC Curves")

    # 设置坐标范围。
    axis.set_xlim(0.0, 1.0)

    # 设置坐标范围。
    axis.set_ylim(0.0, 1.05)

    # 显示图例。
    axis.legend(loc="lower right")

    # 显示网格。
    axis.grid(alpha=0.3)

    # 调整布局。
    figure.tight_layout()

    # 保存图片。
    figure.savefig(save_path, dpi=300, bbox_inches="tight")

    # 关闭图片。
    plt.close(figure)


def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """绘制二分类混淆矩阵。"""

    # 计算混淆矩阵，并固定标签顺序为 0、1。
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 创建画布和坐标轴。
    figure, axis = plt.subplots(figsize=(6, 5))

    # 把二维矩阵显示成图像。
    image = axis.imshow(matrix)

    # 添加颜色条，表示不同颜色对应的数量大小。
    figure.colorbar(image, ax=axis)

    # 设置横轴类别刻度位置。
    axis.set_xticks([0, 1])

    # 设置横轴类别名称。
    axis.set_xticklabels(["Benign", "Malignant"])

    # 设置纵轴类别刻度位置。
    axis.set_yticks([0, 1])

    # 设置纵轴类别名称。
    axis.set_yticklabels(["Benign", "Malignant"])

    # 设置横轴说明。
    axis.set_xlabel("Predicted Label")

    # 设置纵轴说明。
    axis.set_ylabel("True Label")

    # 设置图标题。
    axis.set_title(title)

    # 遍历矩阵的每个位置，在格子中写入具体数量。
    for row_index in range(matrix.shape[0]):
        # 遍历矩阵的每一列。
        for column_index in range(matrix.shape[1]):
            # 在当前格子中心显示数值。
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
            )

    # 调整布局。
    figure.tight_layout()

    # 保存图片。
    figure.savefig(save_path, dpi=300, bbox_inches="tight")

    # 关闭画布。
    plt.close(figure)
