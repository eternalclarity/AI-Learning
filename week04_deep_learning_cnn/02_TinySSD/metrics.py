"""目标检测评价指标：锚框分类准确率、边界框 MAE 与 AP@0.5。"""

from __future__ import annotations

import numpy as np
import torch

from box_ops import box_iou


def anchor_class_accuracy(class_predictions: torch.Tensor, class_labels: torch.Tensor) -> float:
    """训练诊断指标；注意背景锚框很多，因此不能当最终检测指标。"""
    predicted = class_predictions.argmax(dim=-1)
    return float((predicted == class_labels).float().mean().item())


def positive_bbox_mae(
    bbox_predictions: torch.Tensor,
    bbox_labels: torch.Tensor,
    bbox_masks: torch.Tensor,
) -> float:
    """只在正锚框对应的回归维度上计算平均绝对误差。"""
    error = torch.abs((bbox_predictions - bbox_labels) * bbox_masks).sum()
    denominator = bbox_masks.sum().clamp(min=1.0)
    return float((error / denominator).item())


def average_precision(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """计算单类别香蕉检测的 AP@IoU。

    predictions 每张图为 [N,6]：class, score, x1, y1, x2, y2。
    targets 每张图为 [M,5]：class, x1, y1, x2, y2。
    """
    records: list[tuple[int, float, torch.Tensor]] = []
    total_gt = 0

    for image_index, (pred, target) in enumerate(zip(predictions, targets)):
        total_gt += int((target[:, 0] >= 0).sum().item())
        for row in pred.cpu():
            records.append((image_index, float(row[1]), row))

    if total_gt == 0:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0, "mean_matched_iou": 0.0}

    records.sort(key=lambda item: item[1], reverse=True)
    matched: list[set[int]] = [set() for _ in targets]
    tp = np.zeros(len(records), dtype=np.float64)
    fp = np.zeros(len(records), dtype=np.float64)
    matched_ious: list[float] = []

    for rank, (image_index, _, pred_row) in enumerate(records):
        target = targets[image_index].cpu()
        valid = target[:, 0] >= 0
        target = target[valid]
        if target.numel() == 0:
            fp[rank] = 1
            continue

        same_class = target[:, 0].long() == int(pred_row[0].item())
        candidate_indices = torch.where(same_class)[0]
        if candidate_indices.numel() == 0:
            fp[rank] = 1
            continue

        ious = box_iou(pred_row[2:6].unsqueeze(0), target[candidate_indices, 1:5]).squeeze(0)
        best_local = int(torch.argmax(ious).item())
        best_iou = float(ious[best_local].item())
        gt_index = int(candidate_indices[best_local].item())

        if best_iou >= iou_threshold and gt_index not in matched[image_index]:
            tp[rank] = 1
            matched[image_index].add(gt_index)
            matched_ious.append(best_iou)
        else:
            fp[rank] = 1

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recall = cumulative_tp / max(total_gt, 1)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))

    final_precision = float(precision[-1]) if len(precision) else 0.0
    final_recall = float(recall[-1]) if len(recall) else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return {
        "ap": ap,
        "precision": final_precision,
        "recall": final_recall,
        "mean_matched_iou": mean_iou,
    }
