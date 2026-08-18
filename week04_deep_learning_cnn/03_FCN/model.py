"""D2L 风格的 ResNet18-FCN 语义分割模型。"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def bilinear_kernel(in_channels: int, out_channels: int, kernel_size: int) -> torch.Tensor:
    """生成用于初始化转置卷积的双线性插值核。"""
    factor = (kernel_size + 1) // 2
    center = factor - 1 if kernel_size % 2 == 1 else factor - 0.5
    og_y = torch.arange(kernel_size).reshape(-1, 1)
    og_x = torch.arange(kernel_size).reshape(1, -1)
    filt = (1 - torch.abs(og_y - center) / factor) * (
        1 - torch.abs(og_x - center) / factor
    )
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    shared = min(in_channels, out_channels)
    weight[torch.arange(shared), torch.arange(shared)] = filt
    return weight


class FCNResNet18(nn.Module):
    """ResNet18 提取特征 + 1×1 分类 + 转置卷积上采样。"""

    def __init__(
        self,
        num_classes: int = 21,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone_model = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-2])
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        self.upsample = nn.ConvTranspose2d(
            num_classes,
            num_classes,
            kernel_size=64,
            stride=32,
            padding=16,
            bias=False,
        )

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        with torch.no_grad():
            self.upsample.weight.copy_(bilinear_kernel(num_classes, num_classes, 64))

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits_low_resolution = self.classifier(features)
        logits = self.upsample(logits_low_resolution)
        return logits
