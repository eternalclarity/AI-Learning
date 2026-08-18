"""按照 D2L TinySSD 思路实现的多尺度单发多框检测器。"""

from __future__ import annotations

import torch
from torch import nn

from box_ops import multibox_prior


def cls_predictor(in_channels: int, num_anchors: int, num_classes: int) -> nn.Conv2d:
    """每个锚框预测 background + num_classes 个类别分数。"""
    return nn.Conv2d(
        in_channels,
        num_anchors * (num_classes + 1),
        kernel_size=3,
        padding=1,
    )


def bbox_predictor(in_channels: int, num_anchors: int) -> nn.Conv2d:
    """每个锚框预测 4 个边界框偏移量。"""
    return nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)


def down_sample_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """两个卷积后用最大池化把特征图宽高减半。"""
    layers: list[nn.Module] = []
    for _ in range(2):
        layers.extend(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2))
    return nn.Sequential(*layers)


def base_network() -> nn.Sequential:
    """TinySSD 的基础特征提取网络。"""
    channels = [3, 16, 32, 64]
    blocks = [down_sample_block(channels[i], channels[i + 1]) for i in range(3)]
    return nn.Sequential(*blocks)


def flatten_prediction(prediction: torch.Tensor) -> torch.Tensor:
    """把 [B,C,H,W] 变成按位置排列的一维预测。"""
    return prediction.permute(0, 2, 3, 1).reshape(prediction.shape[0], -1)


def concatenate_predictions(predictions: list[torch.Tensor]) -> torch.Tensor:
    """拼接多个尺度的预测结果。"""
    return torch.cat([flatten_prediction(pred) for pred in predictions], dim=1)


class TinySSD(nn.Module):
    """一个适合教学和香蕉数据集的 TinySSD。"""

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.sizes = [
            [0.20, 0.272],
            [0.37, 0.447],
            [0.54, 0.619],
            [0.71, 0.790],
            [0.88, 0.961],
        ]
        self.ratios = [[1.0, 2.0, 0.5] for _ in range(5)]
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1

        self.blocks = nn.ModuleList(
            [
                base_network(),
                down_sample_block(64, 128),
                down_sample_block(128, 128),
                down_sample_block(128, 128),
                nn.AdaptiveMaxPool2d((1, 1)),
            ]
        )
        predictor_channels = [64, 128, 128, 128, 128]
        self.class_predictors = nn.ModuleList(
            [
                cls_predictor(ch, self.num_anchors, num_classes)
                for ch in predictor_channels
            ]
        )
        self.bbox_predictors = nn.ModuleList(
            [bbox_predictor(ch, self.num_anchors) for ch in predictor_channels]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors_all: list[torch.Tensor] = []
        class_all: list[torch.Tensor] = []
        bbox_all: list[torch.Tensor] = []

        feature = x
        for level in range(5):
            feature = self.blocks[level](feature)
            anchors = multibox_prior(feature, self.sizes[level], self.ratios[level])
            class_prediction = self.class_predictors[level](feature)
            bbox_prediction = self.bbox_predictors[level](feature)
            anchors_all.append(anchors)
            class_all.append(class_prediction)
            bbox_all.append(bbox_prediction)

        anchors = torch.cat(anchors_all, dim=1)
        class_predictions = concatenate_predictions(class_all).reshape(
            x.shape[0], -1, self.num_classes + 1
        )
        bbox_predictions = concatenate_predictions(bbox_all)
        return anchors, class_predictions, bbox_predictions
