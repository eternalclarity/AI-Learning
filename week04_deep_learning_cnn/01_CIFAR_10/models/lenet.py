"""LeNet 的 CIFAR-10 适配实现。"""

from __future__ import annotations

import torch  # 用于张量类型标注。
from torch import nn  # nn 提供卷积层和全连接层。


class LeNetCIFAR10(nn.Module):
    """将经典 LeNet 从单通道图片适配到 3×32×32 的 CIFAR-10。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()  # 初始化父类。

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),  # [N,3,32,32] → [N,6,28,28]。
            nn.Sigmoid(),  # 保留经典 LeNet 常用的 Sigmoid 风格。
            nn.AvgPool2d(kernel_size=2, stride=2),  # [N,6,28,28] → [N,6,14,14]。
            nn.Conv2d(6, 16, kernel_size=5),  # [N,6,14,14] → [N,16,10,10]。
            nn.Sigmoid(),  # 第二个激活函数。
            nn.AvgPool2d(kernel_size=2, stride=2),  # [N,16,10,10] → [N,16,5,5]。
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # [N,16,5,5] → [N,400]。
            nn.Linear(16 * 5 * 5, 120),  # 400 → 120。
            nn.Sigmoid(),  # 保留经典网络风格。
            nn.Linear(120, 84),  # 120 → 84。
            nn.Sigmoid(),  # 再次加入非线性。
            nn.Linear(84, num_classes),  # 84 → 10。
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行一次前向传播。"""

        x = self.features(x)  # 先经过卷积编码器。
        logits = self.classifier(x)  # 再经过全连接分类器。
        return logits  # 返回每个类别的 logits。
