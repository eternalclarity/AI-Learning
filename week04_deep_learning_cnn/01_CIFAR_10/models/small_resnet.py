"""面向 CIFAR-10 的小型 ResNet。"""

from __future__ import annotations

import torch  # 用于张量类型标注。
from torch import nn  # nn 提供网络层。

from .residual_block import BasicResidualBlock  # 导入本项目的残差块。


class SmallResNet(nn.Module):
    """使用三个阶段、每阶段两个残差块的小型网络。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()  # 初始化父类。

        self.in_channels = 32  # 记录当前阶段输入通道，供 _make_stage 动态更新。

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 保持 32×32。
            nn.BatchNorm2d(32),  # 稳定第一层特征。
            nn.ReLU(inplace=True),  # 加入非线性。
        )

        self.stage1 = self._make_stage(out_channels=32, num_blocks=2, first_stride=1)  # 32×32。
        self.stage2 = self._make_stage(out_channels=64, num_blocks=2, first_stride=2)  # 16×16。
        self.stage3 = self._make_stage(out_channels=128, num_blocks=2, first_stride=2)  # 8×8。

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 将任意空间尺寸聚合为每通道一个数。
        self.classifier = nn.Linear(128, num_classes)  # 128 维全局特征 → 10 类。

        self._initialize_weights()  # 初始化全部可学习参数。

    def _make_stage(
        self,
        out_channels: int,
        num_blocks: int,
        first_stride: int,
    ) -> nn.Sequential:
        """创建由多个残差块组成的一个阶段。"""

        blocks: list[nn.Module] = []  # 保存该阶段的残差块。
        blocks.append(
            BasicResidualBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=first_stride,
            )
        )
        self.in_channels = out_channels  # 第一个块之后，后续输入通道已经改变。

        for _ in range(1, num_blocks):  # 添加不改变形状的剩余残差块。
            blocks.append(
                BasicResidualBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1,
                )
            )

        return nn.Sequential(*blocks)  # 返回一个完整阶段。

    def _initialize_weights(self) -> None:
        """使用常见方式初始化卷积、BN 和全连接层。"""

        for module in self.modules():  # 遍历所有子模块。
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行一次前向传播。"""

        x = self.stem(x)  # 初始卷积提取低层特征。
        x = self.stage1(x)  # 第一阶段保持分辨率。
        x = self.stage2(x)  # 第二阶段降低分辨率并增加通道。
        x = self.stage3(x)  # 第三阶段继续降低分辨率并增加通道。
        x = self.pool(x)  # [N,128,8,8] → [N,128,1,1]。
        x = torch.flatten(x, start_dim=1)  # [N,128,1,1] → [N,128]。
        logits = self.classifier(x)  # 输出 10 类 logits。
        return logits
