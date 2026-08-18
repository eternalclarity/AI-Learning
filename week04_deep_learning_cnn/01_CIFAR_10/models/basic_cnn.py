"""用于四组核心对比实验的 BasicCNN。"""

from __future__ import annotations

import torch  # 用于张量类型标注。
from torch import nn  # nn 提供卷积、归一化、池化等网络层。


def make_conv_block(
    in_channels: int,
    out_channels: int,
    use_batch_norm: bool,
) -> nn.Sequential:
    """创建一个 Conv → (BN) → ReLU → MaxPool 卷积块。"""

    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not use_batch_norm,  # 使用 BN 时卷积偏置通常可以省略。
        )
    ]

    if use_batch_norm:  # 根据实验配置决定是否加入批量归一化。
        layers.append(nn.BatchNorm2d(out_channels))  # 每个输出通道拥有独立的缩放和平移参数。

    layers.extend(
        [
            nn.ReLU(inplace=True),  # 原地 ReLU 可少占用一部分中间内存。
            nn.MaxPool2d(kernel_size=2, stride=2),  # 高和宽各缩小一半。
        ]
    )

    return nn.Sequential(*layers)  # 按列表顺序组成一个可调用模块。


class BasicCNN(nn.Module):
    """面向 32×32 彩色图片的小型卷积神经网络。"""

    def __init__(
        self,
        num_classes: int = 10,
        use_batch_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()  # 初始化父类。

        if not 0.0 <= dropout < 1.0:  # 检查 Dropout 概率是否合法。
            raise ValueError("dropout 必须位于 [0, 1) 区间。")

        self.features = nn.Sequential(
            make_conv_block(3, 32, use_batch_norm),  # [N,3,32,32] → [N,32,16,16]。
            make_conv_block(32, 64, use_batch_norm),  # [N,32,16,16] → [N,64,8,8]。
            make_conv_block(64, 128, use_batch_norm),  # [N,64,8,8] → [N,128,4,4]。
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # [N,128,4,4] → [N,2048]。
            nn.Linear(128 * 4 * 4, 256),  # 将卷积特征映射到 256 维隐藏表示。
            nn.ReLU(inplace=True),  # 添加非线性。
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),  # 仅在需要时启用 Dropout。
            nn.Linear(256, num_classes),  # 输出 10 个类别 logits。
        )

        self._initialize_weights()  # 使用适合 ReLU 的方式初始化卷积和全连接权重。

    def _initialize_weights(self) -> None:
        """初始化模型参数。"""

        for module in self.modules():  # 遍历当前模型中的所有子模块。
            if isinstance(module, nn.Conv2d):  # 卷积层使用 Kaiming 初始化。
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:  # 只有存在偏置时才初始化偏置。
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):  # BN 初始时保持恒等缩放。
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):  # 全连接层使用 Kaiming 初始化。
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行一次前向传播。"""

        features = self.features(x)  # 卷积块逐步提取空间特征。
        logits = self.classifier(features)  # 将特征映射为类别分数。
        return logits  # 返回 logits，不手动调用 Softmax。
