"""第四周 CIFAR-10 项目的实验配置。

这个文件集中管理所有实验预设，目的是保证不同实验之间只改变我们真正想研究的变量。
例如：
- Exp 1 与 Exp 2 只比较是否使用 BatchNorm；
- Exp 2 与 Exp 3 只比较是否使用 Dropout；
- Exp 3 与 Exp 4 只比较是否使用数据增强。
"""

from __future__ import annotations

from dataclasses import dataclass  # dataclass 可以简化“只保存配置数据”的类。
from typing import Final  # Final 用于标记不希望被随意修改的常量。


@dataclass(frozen=True)
class ExperimentPreset:
    """保存一组完整、可复现的实验配置。"""

    name: str  # 实验名称，也会用作输出目录名。
    description: str  # 对实验目的的简短说明。
    model_name: str  # 交给模型工厂创建的模型名称。
    use_batch_norm: bool  # BasicCNN 是否加入 BatchNorm。
    dropout: float  # Dropout 概率；0.0 表示不使用 Dropout。
    use_augmentation: bool  # 训练集是否使用随机裁剪和随机翻转。
    optimizer_name: str = "adam"  # 优化器名称。
    learning_rate: float = 1e-3  # 初始学习率。
    weight_decay: float = 0.0  # 权重衰减系数。
    scheduler_name: str = "none"  # 学习率调度器名称。


# 核心四组实验保持相同模型骨架、优化器和学习率，只逐步增加训练技术。
EXPERIMENT_PRESETS: Final[dict[str, ExperimentPreset]] = {
    "exp0_mlp": ExperimentPreset(
        name="exp0_mlp",
        description="可选对照：把 CIFAR-10 图片展平后交给 MLP。",
        model_name="mlp",
        use_batch_norm=False,
        dropout=0.2,
        use_augmentation=False,
    ),
    "exp1_basic_cnn": ExperimentPreset(
        name="exp1_basic_cnn",
        description="CNN 基线：卷积、ReLU、池化，不使用 BN、Dropout 和数据增强。",
        model_name="basic_cnn",
        use_batch_norm=False,
        dropout=0.0,
        use_augmentation=False,
    ),
    "exp2_cnn_bn": ExperimentPreset(
        name="exp2_cnn_bn",
        description="在同一 BasicCNN 上加入 BatchNorm。",
        model_name="basic_cnn",
        use_batch_norm=True,
        dropout=0.0,
        use_augmentation=False,
    ),
    "exp3_cnn_bn_dropout": ExperimentPreset(
        name="exp3_cnn_bn_dropout",
        description="在 CNN + BatchNorm 的基础上加入 Dropout。",
        model_name="basic_cnn",
        use_batch_norm=True,
        dropout=0.5,
        use_augmentation=False,
    ),
    "exp4_cnn_bn_dropout_aug": ExperimentPreset(
        name="exp4_cnn_bn_dropout_aug",
        description="在 Exp 3 基础上只增加训练集数据增强。",
        model_name="basic_cnn",
        use_batch_norm=True,
        dropout=0.5,
        use_augmentation=True,
    ),
    "exp5_lenet": ExperimentPreset(
        name="exp5_lenet",
        description="经典 LeNet 的 CIFAR-10 适配版本。",
        model_name="lenet",
        use_batch_norm=False,
        dropout=0.0,
        use_augmentation=False,
    ),
    "exp6_vgg_small": ExperimentPreset(
        name="exp6_vgg_small",
        description="使用可复用 VGG Block 构建的小型 VGG。",
        model_name="vgg_small",
        use_batch_norm=True,
        dropout=0.5,
        use_augmentation=True,
    ),
    "exp7_small_resnet": ExperimentPreset(
        name="exp7_small_resnet",
        description="面向 CIFAR-10 的小型残差网络。",
        model_name="small_resnet",
        use_batch_norm=True,
        dropout=0.0,
        use_augmentation=True,
        scheduler_name="cosine",
    ),
}


CORE_EXPERIMENT_NAMES: Final[tuple[str, ...]] = (
    "exp1_basic_cnn",
    "exp2_cnn_bn",
    "exp3_cnn_bn_dropout",
    "exp4_cnn_bn_dropout_aug",
)


OPTIONAL_EXPERIMENT_NAMES: Final[tuple[str, ...]] = (
    "exp0_mlp",
    "exp5_lenet",
    "exp6_vgg_small",
    "exp7_small_resnet",
)


def get_experiment_preset(name: str) -> ExperimentPreset:
    """根据名称返回实验预设，并在名称错误时给出清晰提示。"""

    if name not in EXPERIMENT_PRESETS:  # 先检查用户输入的名称是否合法。
        available = ", ".join(EXPERIMENT_PRESETS)  # 将全部合法名称拼接成提示文本。
        raise ValueError(f"未知实验：{name!r}。可选值：{available}")

    return EXPERIMENT_PRESETS[name]  # 返回不可变的实验配置对象。
