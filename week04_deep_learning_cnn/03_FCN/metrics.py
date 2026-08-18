"""语义分割指标：像素准确率、每类 IoU 与 mIoU。"""

from __future__ import annotations

import torch


class SegmentationConfusionMatrix:
    """累计整个数据集的像素级混淆矩阵。"""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, logits_or_pred: torch.Tensor, target: torch.Tensor) -> None:
        if logits_or_pred.ndim == 4:
            prediction = logits_or_pred.argmax(dim=1)
        else:
            prediction = logits_or_pred

        prediction = prediction.detach().cpu().long().reshape(-1)
        target = target.detach().cpu().long().reshape(-1)
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        prediction = prediction[valid]
        target = target[valid]
        indices = target * self.num_classes + prediction
        bincount = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.matrix += bincount.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, object]:
        matrix = self.matrix.float()
        diagonal = torch.diag(matrix)
        total = matrix.sum()
        pixel_accuracy = diagonal.sum() / total.clamp(min=1)

        union = matrix.sum(dim=1) + matrix.sum(dim=0) - diagonal
        class_iou = diagonal / union.clamp(min=1)
        valid_classes = union > 0
        mean_iou = class_iou[valid_classes].mean() if valid_classes.any() else torch.tensor(0.0)

        return {
            "pixel_accuracy": float(pixel_accuracy.item()),
            "mean_iou": float(mean_iou.item()),
            "class_iou": [float(x) for x in class_iou.tolist()],
            "confusion_matrix": self.matrix.tolist(),
        }
