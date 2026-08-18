"""CIFAR-10 的多层感知机对照模型。"""

from __future__ import annotations

import torch  # 用于张量类型标注。
from torch import nn  # nn 提供神经网络层。


class CIFAR10MLP(nn.Module):
    """先展平彩色图片，再使用三层全连接网络分类。"""

    def __init__(
        self,
        input_size: int = 3 * 32 * 32,
        hidden_sizes: tuple[int, int] = (512, 256),
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()  # 初始化 nn.Module 内部状态。

        hidden_size1, hidden_size2 = hidden_sizes  # 解包两个隐藏层宽度。

        self.flatten = nn.Flatten()  # [N,3,32,32] → [N,3072]。
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),  # 第一层全连接：3072 → 512。
            nn.ReLU(),  # 加入非线性表达能力。
            nn.Dropout(dropout),  # 训练时随机丢弃部分隐藏层输出。
            nn.Linear(hidden_size1, hidden_size2),  # 第二层全连接：512 → 256。
            nn.ReLU(),  # 第二个非线性激活。
            nn.Dropout(dropout),  # 第二个 Dropout。
            nn.Linear(hidden_size2, num_classes),  # 输出 10 个类别的 logits。
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行一次前向传播。"""

        x = self.flatten(x)  # 丢失二维空间布局，只保留所有像素值。
        logits = self.network(x)  # 通过多层全连接网络得到分类分数。
        return logits  # CrossEntropyLoss 需要未经 Softmax 的 logits。
