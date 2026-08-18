"""第四周项目的模型注册表与统一创建函数。"""

from __future__ import annotations

from torch import nn  # 用于返回类型标注。

from .basic_cnn import BasicCNN  # 核心实验模型。
from .lenet import LeNetCIFAR10  # 经典 LeNet。
from .mlp import CIFAR10MLP  # MLP 对照模型。
from .small_resnet import SmallResNet  # 小型残差网络。
from .vgg_small import SmallVGG  # 小型 VGG。


AVAILABLE_MODELS: tuple[str, ...] = (
    "mlp",
    "basic_cnn",
    "lenet",
    "vgg_small",
    "small_resnet",
)


def create_model(
    model_name: str,
    num_classes: int = 10,
    use_batch_norm: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """根据字符串名称创建模型。"""

    normalized_name = model_name.strip().lower()  # 去除多余空格并统一成小写。

    if normalized_name == "mlp":
        return CIFAR10MLP(num_classes=num_classes, dropout=dropout)

    if normalized_name == "basic_cnn":
        return BasicCNN(
            num_classes=num_classes,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

    if normalized_name == "lenet":
        return LeNetCIFAR10(num_classes=num_classes)

    if normalized_name == "vgg_small":
        return SmallVGG(
            num_classes=num_classes,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

    if normalized_name == "small_resnet":
        return SmallResNet(num_classes=num_classes)

    available = ", ".join(AVAILABLE_MODELS)  # 为异常信息列出所有合法名称。
    raise ValueError(f"未知模型：{model_name!r}。可选值：{available}")


__all__ = [
    "AVAILABLE_MODELS",
    "BasicCNN",
    "CIFAR10MLP",
    "LeNetCIFAR10",
    "SmallResNet",
    "SmallVGG",
    "create_model",
]
