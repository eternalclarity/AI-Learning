"""ResNet 的基本残差块。"""

from __future__ import annotations

import torch  # 用于张量类型标注。
from torch import nn  # nn 提供卷积、BN 和激活函数。


class BasicResidualBlock(nn.Module):
    """实现 y = ReLU(F(x) + shortcut(x))。"""

    expansion: int = 1  # BasicBlock 的输出通道不做额外倍增。

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()  # 初始化父类。

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)  # 规范化第一层卷积输出。
        self.relu = nn.ReLU(inplace=True)  # 两处复用同一个无参数 ReLU 模块。
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)  # 第二层卷积后先不激活，等待残差相加。

        needs_projection = stride != 1 or in_channels != out_channels  # 判断输入与输出形状是否一致。

        if needs_projection:  # 形状不一致时，用 1×1 卷积调整捷径分支。
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:  # 形状一致时，捷径分支直接返回输入。
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行残差块前向传播。"""

        identity = self.shortcut(x)  # 得到可与主分支相加的捷径输出。
        out = self.conv1(x)  # 主分支第一个 3×3 卷积。
        out = self.bn1(out)  # 批量归一化。
        out = self.relu(out)  # 第一次激活。
        out = self.conv2(out)  # 主分支第二个 3×3 卷积。
        out = self.bn2(out)  # 第二次批量归一化。
        out = out + identity  # 核心残差连接：F(x) + x 或 F(x) + projection(x)。
        out = self.relu(out)  # 相加后再激活。
        return out  # 返回残差块输出。
