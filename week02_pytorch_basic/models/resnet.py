"""
改进三：设计的小型残差网络 ResNet-14
"""

import torch
from torch import nn


class BasicResidualBlock(nn.Module):
    """
    定义残差块，学习主分支 F(x)，再与捷径分支 x 相加，输出 ReLU(F(x)+x)。
    主分支：
        输入 x -> 第一个 3×3 卷积 -> BatchNorm -> ReLU -> 第二个 3×3 卷积 -> BatchNorm -> 得到 F(x)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        # 创建 残差主分支 中的 第一层 3×3 卷积 ,提取第一轮局部特征, 必要时通过 stride=2 缩小宽高,并改变通道数
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False,)   # stride=2 时会把特征图长宽缩小一半。padding=1 让 stride=1 时空间尺寸保持不变。后面紧跟 BatchNorm，因此卷积层不需要单独偏置。
        # 对第一层卷积的输出做批归一化, 稳定输出特征分布
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        # 创建 残差主分支 中的 第二层 3×3 卷积 ,第二层输出通道保持不变,不再缩小空间尺寸, 进一步组合邻域和通道信息
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 对第二层卷积的输出做批归一化
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        # 创建能够反复使用的 ReLU，并允许原地修改以节省内存
        self.relu = nn.ReLU(inplace=True)

        # 捷径分支shortcut: 判断输入 x 能不能直接与主分支输出 F(x) 相加 -> F(x)是否高宽减半，通道数加倍
        if stride == 1 and in_channels == out_channels:
            # x 与 F(x) 形状相同时，捷径分支直接原样传递输入
            self.shortcut = nn.Identity()   # nn.Identity() 是恒等映射, 它接收到什么，就原样输出什么
        else:
            # x 与 F(x) 形状不同时，用 1×1 卷积层和 BatchNorm 调整捷径分支的 x 匹配 F(x) 的形状(通道/宽高)
            self.shortcut = nn.Sequential(
                # 1×1 卷积负责改变通道数，并按照 stride 调整空间尺寸
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                # 对变换后的捷径分支进行批归一化。
                nn.BatchNorm2d(num_features=out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 让原始输入经过捷径分支shortcut
        identity = self.shortcut(x)

        # 让输入经过主分支第一层卷积
        out = self.conv1(x)
        # 对第一层卷积结果做批归一化
        out = self.bn1(out)
        # 对归一化结果使用 ReLU 激活
        out = self.relu(out)

        # 让特征继续经过主分支第二层卷积
        out = self.conv2(out)
        # 对第二层卷积结果做批归一化，但暂时不激活
        out = self.bn2(out)

        # 把主分支学到的残差 F(x) 与捷径分支 identity 相加
        out = out + identity

        # 对相加结果使用 ReLU，得到该残差块的最终输出
        out = self.relu(out)
        # 返回残差块输出。
        return out


class SmallResNet(nn.Module):
    """定义由一个 卷积入口 和 三个残差阶段 组成的小型 ResNet。"""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.dropout = dropout

        # 创建卷积网络入口，把一通道灰度图转换成 32 通道特征图, 宽高不变 [batch, 1, 28, 28] -> [batch, 32, 28, 28]
        self.stem = nn.Sequential(
            # 使用 3×3 卷积，保持 FashionMNIST 的 28×28 尺寸
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            # 对入口卷积产生的 32 个通道进行批归一化
            nn.BatchNorm2d(num_features=32),
            # 使用 ReLU 激活入口特征
            nn.ReLU(inplace=True),
            # Dropout丢弃层
            nn.Dropout2d(p=dropout),
        )

        # 第一阶段块包含两个不改变通道, 宽高的残差块 [batch, 32, 28, 28] -> [batch, 32, 28, 28]
        self.stage1 = nn.Sequential(
            # 第一个残差块输入输出形状相同，捷径分支使用 Identity
            BasicResidualBlock(in_channels=32, out_channels=32, stride=1),
            # 第二个残差块继续提取 32 通道特征
            BasicResidualBlock(in_channels=32, out_channels=32, stride=1),
            # Dropout丢弃层
            nn.Dropout2d(p=dropout),
        )

        # 第二阶段把通道数增加到 64，并把宽高缩小到 14×14 [batch, 32, 28, 28] -> [batch, 64, 14, 14]
        self.stage2 = nn.Sequential(
            # stride=2 同时完成通道增加和下采样，捷径分支会用 1×1 卷积匹配
            BasicResidualBlock(in_channels=32, out_channels=64, stride=2),
            # 第二个块保持 64 通道和 14×14 空间尺寸
            BasicResidualBlock(in_channels=64, out_channels=64, stride=1),
            # Dropout丢弃层
            nn.Dropout2d(p=dropout),
        )

        # 第三阶段把通道数增加到 128，并把宽高缩小到 7×7 [batch, 64, 14, 14] -> [batch, 128, 7, 7]
        self.stage3 = nn.Sequential(
            # stride=2 把 [64, 14, 14] 变换成 [128, 7, 7]。
            BasicResidualBlock(in_channels=64, out_channels=128, stride=2),
            # 第二个块继续在 128 个通道上提取更高级的特征
            BasicResidualBlock(in_channels=128, out_channels=128, stride=1),
            # Dropout丢弃层
            nn.Dropout2d(p=dropout),
        )

        # 自适应平均池化层 把 每个通道 的 任意宽高 压缩成 1×1    [batch, 128, 7, 7] -> [batch, 128, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # 最终全连接层把 128 个通道的全局特征映射成 10 个类别分数  [batch, 128] -> [batch, 10]
        self.classifier = nn.Linear(in_features=128, out_features=num_classes)

    # 定义整个 SmallResNet 的前向传播过程。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)    # [batch, 1, 28, 28] -> [batch, 32, 28, 28]

        x = self.stage1(x)  # [batch, 32, 28, 28] -> [batch, 32, 28, 28]

        x = self.stage2(x)  # [batch, 32, 28, 28] -> [batch, 64, 14, 14]

        x = self.stage3(x)  # [batch, 64, 14, 14] -> [batch, 128, 7, 7]

        x = self.global_pool(x) # [batch, 128, 7, 7] -> [batch, 128, 1, 1]

        x = torch.flatten(x, start_dim=1)   # [batch, 128, 1, 1] -> [batch, 128]

        x = self.classifier(x)  # [batch, 128] -> [batch, 10] , 返回 logits，交给 CrossEntropyLoss 或 argmax 使用

        return x

    def get_config(self) -> dict:
        """返回模型构造参数，用于从检查点重新创建模型."""
        return {
            "num_classes": self.num_classes,
            "dropout": self.dropout,
        }
