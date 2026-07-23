"""
改进一：使用卷积(conv2d)、ReLU、最大池化和全连接层的基础 CNN(Convolutional Neural Network - 卷积神经网络)
CNN： 是由卷积层等多种层组成的一类神经网络

展平层(nn.Flatten) - 把多维特征展开成一维

全连接层(nn.Linear) - 根据特征放缩或完成最终分类

激活层(nn.ReLU) - 把负参数变0，增加非线性能力

卷积层(nn.conv2d - 二维卷积层) - 保持图片的二维空间关系,提取局部特征:
    - 卷积层就是拿着很多个可训练的“小窗口”(卷积核)，在图片上从左到右、从上到下地扫描，寻找边缘、纹理、形状等局部特征，并把这些特征保存成多张特征图，交给后面的网络继续处理。
    - 卷积核实际上就是模型中的一组可训练参数w, 移动到一个位置x, w*x+b计算出特征图中的一个数字, 最终会生成一张新的数字矩阵,即 特征图(Feature Map)
    - 特征图 -> 这张图片在某种特征上的检测结果: 一个卷积核可能特别擅长检测竖直边缘。如果图片某个位置存在明显的竖直边缘，输出数字可能较大，说明那里很可能存在卷积核关注的特征
    - 通常会同时使用很多个卷积核 -> 卷积核1：检测竖直边缘,卷积核2：检测水平边缘,卷积核3：检测斜线, 每个卷积核都会生成一张特征图 -> 32 组不同的特征信息
    - 在图片周围补0: 卷积核不能以最边缘的像素为中心进行计算，边缘像素参与计算的次数也更少, 3*3->padding=1, 5*5->padding=2. 补0->不会改变图片的高和宽
    - 多个卷积层串联: 像素 -> 边缘 -> 局部形状 -> 完整物体特征 -> 分类结果
    - 参数数量: [in_channels * kernel_sizes * kernel_sizes + 1(bias)] * out_channels

池化层(nn.MaxPool2d - 二维最大池化层) - 保留重要特征, 缩小特征图
    - 最大池化层 把特征图划分成一个个"小区域"(池化核，默认不滑动，无参数)，每个区域只保留最大的那个数，从而缩小特征图，同时保留最明显的特征。 - "下采样层"
    - 卷积层产生的特征图中，数值较大的位置通常表示：这个位置出现了卷积核正在寻找的特征，而且这个特征比较明显
    - 平均池化:nn.AvgPool2d(): 保留区域整体信息
"""

import torch
from torch import nn


class BasicCNN(nn.Module):
    """基础 CNN：重点观察卷积结构相对于 MLP 的改进。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.num_classes = num_classes

        # 创建负责提取图像空间特征的卷积部分。
        self.features = nn.Sequential(
            # 第一层二维卷积层：把 1 个灰度通道变成 32 张特征图. [batch, 1, 28, 28] -> [batch, 32, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # 使用 32 个 3*3 卷积核,在图片周围补 一 圈0,依次扫描 单 通道灰度图,生成32张特征图,用于提取32组不同的局部特征
            # ReLU 把负数变成 0，为网络加入非线性表达能力。
            nn.ReLU(),
            # 2×2 最大池化把每张特征图从 28×28 缩小到 14×14. [batch, 32, 28, 28] -> [batch, 32, 14, 14]
            nn.MaxPool2d(kernel_size=2, stride=2), # 把特征图划分成 2×2 的区域,每次移动两格，取其中的最大值 -> 缩小特征图

            # 第二层卷积层：把 32 个通道组合成 64 个更高级的特征通道. [batch, 32, 14, 14] -> [batch, 64, 14, 14]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # 再次使用 ReLU，让第二层卷积能够学习非线性组合。
            nn.ReLU(),
            # 第二次池化把特征图从 14×14 缩小到 7×7。[batch, 64, 14, 14] -> [batch, 64, 7, 7]
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 创建负责根据提取到的特征完成最终分类的全连接部分。
        self.classifier = nn.Sequential(
            # 把 [batch, 64, 7, 7] 展平成 [batch, 3136]。
            nn.Flatten(),
            # 把 3136 个特征压缩成 128 个综合特征。
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            # 使用 ReLU 为全连接分类部分加入非线性。
            nn.ReLU(),
            # 输出 10 个原始分类分数，分别对应 10 类服饰。
            nn.Linear(in_features=128, out_features=num_classes),
        )

    # 定义前向传播，规定一批图片如何依次经过模型各层。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状为 [batch, 1, 28, 28]，卷积后变成 [batch, 64, 7, 7]。
        x = self.features(x)
        # 展平并经过全连接层，最终得到 [batch, 10] 的 logits。
        x = self.classifier(x)
        # 返回 10 类的原始分数；CrossEntropyLoss 会在内部处理 softmax。
        return x

    def get_config(self) -> dict:
        """返回模型构造参数，用于从检查点重新创建模型."""
        return {
            "num_classes": self.num_classes,
        }