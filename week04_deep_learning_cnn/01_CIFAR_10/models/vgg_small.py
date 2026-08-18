"""使用 VGG Block 思想构建的小型 CIFAR-10 网络。"""

from __future__ import annotations

import torch  # 用于张量类型标注。
from torch import nn  # nn 提供卷积层、池化层和全连接层。


def vgg_block(
    num_convs: int,
    in_channels: int,
    out_channels: int,
    use_batch_norm: bool = True,
) -> nn.Sequential:
    """创建“若干个 3×3 卷积 + 一次池化”的 VGG Block。"""

    if num_convs <= 0:  # 每个块至少包含一个卷积层。
        raise ValueError("num_convs 必须大于 0。")

    layers: list[nn.Module] = []  # 保存块中的各层。
    current_in_channels = in_channels  # 第一层卷积使用外部传入的输入通道数。

    for _ in range(num_convs):  # 按 num_convs 重复堆叠卷积层。
        layers.append(
            nn.Conv2d(
                current_in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batch_norm,
            )
        )
        if use_batch_norm:  # 可选地在每个卷积后加入 BatchNorm。
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))  # 每个卷积后都使用 ReLU。
        current_in_channels = out_channels  # 后续卷积的输入通道等于前一层输出通道。

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 每个块末尾统一做一次下采样。
    return nn.Sequential(*layers)  # 返回可复用的 VGG Block。


class SmallVGG(nn.Module):
    """适合 CIFAR-10 训练规模的小型 VGG。"""

    def __init__(
        self,
        num_classes: int = 10,
        use_batch_norm: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()  # 初始化父类。

        self.features = nn.Sequential(
            vgg_block(1, 3, 32, use_batch_norm),  # 32×32 → 16×16。
            vgg_block(1, 32, 64, use_batch_norm),  # 16×16 → 8×8。
            vgg_block(2, 64, 128, use_batch_norm),  # 8×8 → 4×4。
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # [N,128,4,4] → [N,2048]。
            nn.Linear(128 * 4 * 4, 256),  # 2048 → 256。
            nn.ReLU(inplace=True),  # 非线性激活。
            nn.Dropout(dropout),  # 缓解分类器过拟合。
            nn.Linear(256, num_classes),  # 256 → 10。
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行一次前向传播。"""

        x = self.features(x)  # 使用重复卷积块提取特征。
        logits = self.classifier(x)  # 输出类别分数。
        return logits
