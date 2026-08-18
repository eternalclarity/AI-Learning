"""TinySSD 损失函数。"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def ssd_loss(
    class_predictions: torch.Tensor,
    class_labels: torch.Tensor,
    bbox_predictions: torch.Tensor,
    bbox_labels: torch.Tensor,
    bbox_masks: torch.Tensor,
    bbox_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算分类损失与边界框回归损失。

    分类使用交叉熵；边界框使用 Smooth L1。
    D2L 主示例使用 L1，教材练习明确建议尝试 Smooth L1；这里采用后者。
    """
    batch_size = class_predictions.shape[0]
    class_loss = F.cross_entropy(
        class_predictions.reshape(-1, class_predictions.shape[-1]),
        class_labels.reshape(-1),
        reduction="none",
    ).reshape(batch_size, -1).mean(dim=1)

    bbox_loss = F.smooth_l1_loss(
        bbox_predictions * bbox_masks,
        bbox_labels * bbox_masks,
        reduction="none",
    ).sum(dim=1) / bbox_masks.sum(dim=1).clamp(min=1.0)

    total_loss = class_loss + bbox_weight * bbox_loss
    return total_loss, class_loss, bbox_loss
