"""
定义一个用于 FashionMNIST 数据集的简单多层感知机模型。
 (MLP 是 Multilayer Perceptron 的缩写， 中文通常称为多层感知机)

 FashionMNIST 中的每张图片是：
    28 × 28 的 灰度 图片 -> 28 * 28 * 1

 模型任务：
    输入一张服装图片
    输出它属于 10 个类别中每个类别的预测分数
"""

from __future__ import annotations

from typing import Sequence # Sequence 表示有顺序的序列, 可以是多种序列,比如 List, tuple, range

import torch
from torch import nn


class MLP(nn.Module):
    """
    用于 28 × 28 灰度 图片分类的多层感知机。
    - 输入形状： [batch_size, 1, 28, 28]
        各维度含义：
        batch_size： 一个批次中的图片数量。 -> batch_size批次来源于 dataloader
        1： 灰度图片只有一个颜色通道。
        28, 28： 图片的高度和宽度。

    - 输出形状： [batch_size, num_classes]
        例如 batch_size=32，num_classes=10： [32, 10] 表示： 一共有 32 张图片； 每张图片输出 10 个类别分数。

    -  nn.Linear(in_features, out_features) 永远只处理输入张量的最后一维
        images: tensor[batch_size, 1, 28, 28] -> nn.Flatten() -> tensor[batch_size, 784] -> in_features = 784

    nn.Linear参数量 = in_features × out_features + out_features
    """

    def __init__(
        self,
        input_size: int = 28 * 28 * 1,              # 每张图片展开后的输入特征数量
        hidden_sizes: Sequence[int] = (256, 128),   # 两个隐藏层的神经元数量,Sequence[int]表示由整数构成的序列
        num_classes: int = 10,                      # 分类类别数量，表示 FashionMNIST 有 10 个类别
        dropout: float = 0.2,                       # Dropout 的丢弃概率， 训练时随机将 20% 的神经元输出置为 0
    ) -> None:
        super().__init__()

        # 检查 hidden_sizes 是否正好包含两个元素
        if len(hidden_sizes) != 2:
            raise ValueError("This beginner MLP expects exactly two hidden layer sizes.")
        # 检查 dropout 是否在合法范围内
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        hidden_size1, hidden_size2 = hidden_sizes

        self.input_size = input_size    # 保存模型的输入特征数量
        self.hidden_sizes = (int(hidden_size1), int(hidden_size2))  # 保存两个隐藏层大小
        self.num_classes = num_classes  # 保存分类类别数量
        self.dropout = dropout  # 保存 Dropout 概率

        self.flatten = nn.Flatten()     # 创建 Flatten 层，将每张多维图片展开成一维特征：[batch_size, 1, 28, 28] -> [batch_size, 784]
        self.network = nn.Sequential(   # 使用 nn.Sequential 按顺序组合多个网络层
            # nn.Linear(in_features, out_features) 永远只处理输入张量的最后一维
            nn.Linear(input_size, hidden_size1),    # 第一层全连接层 [batch_size, 784] -> [batch_size, 256] ;
            nn.ReLU(),                              # ReLU 激活函数 ReLU(x) = max(0, x), 激活函数可以让模型拥有非线性表达能力
            nn.Dropout(dropout),                    # 第一层 Dropout, 表示是每个 batch 都随机丢弃 20% 的输出, 主要作用是减少过拟合

            nn.Linear(hidden_size1, hidden_size2),  # 第二层全连接层 [batch_size, 256] -> [batch_size, 128]
            nn.ReLU(),                              # 第二个 ReLU 激活函数
            nn.Dropout(dropout),                    # 第二个 Dropout 层

            nn.Linear(hidden_size2, num_classes),   # 最后一层全连接层，也叫输出层 [batch_size, 128] -> [batch_size, 10]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 执行一次前向传播，并返回每个类别的 logits"""
        x = self.flatten(x)             # 将图片展开
        logits = self.network(x)        # 将展开后的数据传入整个神经网络, logits（逻辑值 / 未归一化分数）：模型最后一层直接输出的原始分数，还没有转换成概率
        return logits                   # [batch_size, num_classes]

    def get_config(self) -> dict:
        """返回模型构造参数，用于从检查点重新创建模型."""
        return {
            "input_size": self.input_size,
            "hidden_sizes": list(self.hidden_sizes),
            "num_classes": self.num_classes,
            "dropout": self.dropout,
        }


if __name__ == "__main__":
    model = MLP()
    dummy_input = torch.randn(32, 1, 28, 28)
    output = model(dummy_input)

    print(model)
    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
