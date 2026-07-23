"""
改进二：在基础 CNN 中加入 BatchNorm 和 Dropout。

归一化层(nn.BatchNorm2d - 二维批归一化层) - 减小数值尺度差异,稳定特征数值分布
    - 归一化: 归一化会先让它们围绕 0 分布,第一步，减去平均值,然后再除以数据的波动程度，使这些数不要过于分散, 让梯度下降更容易
    - 卷积层中的参数会不断更新，因此卷积层每一次输出的数据范围都可能发生变化,后面的网络层刚刚适应了前一种数据范围，输入突然又变得很大，学习就会比较困难
    - BatchNorm: 根据当前这一批数据计算均值和方差，再进行归一化,假设一个 batch 有 64 张图片,BatchNorm 会观察这 64 张图片对应的特征数据，然后计算统计值
    - BatchNorm2d 是每个通道单独计算均值和方差，但统计范围包括 batch、高度和宽度
    - 参数数量: 对于每个通道都有一个 γ一个 β: num_features * 2
    - 测试模式下使用运行均值和方差

丢弃层(nn.Dropout) - 随机关闭部分输出, 缓解过拟合
    - Dropout 在训练过程中，会随机把一部分神经元的输出暂时变成 0，迫使模型不能过度依赖某几个神经元，从而提高模型对新数据的泛化能力
    - 每执行一次 Dropout 的前向传播，都会重新随机生成一个丢弃掩码
    - 测试模式不生效
"""

import torch
from torch import nn


class ImprovedCNN(nn.Module):
    """在与基础 CNN 相近的结构上加入批归一化和随机失活。"""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.dropout = dropout

        # 创建卷积特征提取部分。
        self.features = nn.Sequential(
            # 第一层卷积不使用偏置，因为后面的 BatchNorm 已经带有可学习平移参数。
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, bias=False),
            # 对 32 个通道分别做批归一化，使中间特征分布更加稳定, 不改变张量形状
            nn.BatchNorm2d(num_features=32),    # 二维批归一化层, 表示输入有 32 个通道, 它必须和前面卷积层的输出通道数对应
            # 使用 ReLU 加入非线性。
            nn.ReLU(),
            # 最大池化把空间尺寸从 28×28 缩小到 14×14。
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 在训练时随机把部分特征图位置置零，降低模型对局部特征的过度依赖。
            nn.Dropout2d(p=dropout),

            # 第二层卷积把通道数从 32 增加到 64。
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            # 对第二层卷积产生的 64 个通道进行批归一化。
            nn.BatchNorm2d(num_features=64),
            # 再次使用 ReLU 加入非线性。
            nn.ReLU(),
            # 第二次池化把空间尺寸从 14×14 缩小到 7×7。
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 再次随机屏蔽部分特征，增强模型的泛化能力。
            nn.Dropout2d(p=dropout),
        )

        # 创建全连接分类部分。
        self.classifier = nn.Sequential(
            # 把 [batch, 64, 7, 7] 展平成 [batch, 3136]。
            nn.Flatten(),
            # 把 3136 维特征映射成 128 维综合表示。
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            # 对 128 维全连接特征进行批归一化。
            nn.BatchNorm1d(num_features=128),
            # 使用 ReLU 加入非线性。
            nn.ReLU(),
            # 训练时随机丢弃 30% 的神经元输出，缓解过拟合。
            nn.Dropout(p=dropout),
            # 把 128 维特征映射成 10 个类别分数。
            nn.Linear(in_features=128, out_features=num_classes),
        )

    # 定义模型的前向传播过程。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先用卷积、BatchNorm、ReLU、池化和 Dropout 提取特征。
        x = self.features(x)
        # 再用全连接分类器得到 10 个类别的 logits。
        x = self.classifier(x)
        # 返回未经 softmax 的分类分数。
        return x

    def get_config(self) -> dict:
        """返回模型构造参数，用于从检查点重新创建模型."""
        return {
            "num_classes": self.num_classes,
            "dropout": self.dropout,
        }
