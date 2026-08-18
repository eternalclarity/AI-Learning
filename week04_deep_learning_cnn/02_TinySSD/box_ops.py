"""SSD 中最关键的边界框、锚框、目标分配与 NMS 操作。

这些函数刻意不依赖 d2l 包，便于真正理解 SSD 的数据流。
"""

from __future__ import annotations

import math

import torch


def box_corner_to_center(boxes: torch.Tensor) -> torch.Tensor:
    """[xmin, ymin, xmax, ymax] -> [cx, cy, w, h]。"""
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)


def box_center_to_corner(boxes: torch.Tensor) -> torch.Tensor:
    """[cx, cy, w, h] -> [xmin, ymin, xmax, ymax]。"""
    cx, cy, w, h = boxes.unbind(dim=-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """计算两组边界框两两之间的 IoU。"""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    upper_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    lower_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection_wh = (lower_right - upper_left).clamp(min=0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)

    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1e-12)


def multibox_prior(
    feature_map: torch.Tensor,
    sizes: list[float],
    ratios: list[float],
) -> torch.Tensor:
    """在特征图每个位置生成一组归一化锚框。

    返回形状：[1, H * W * boxes_per_pixel, 4]。
    """
    device = feature_map.device
    height, width = feature_map.shape[-2:]
    num_sizes = len(sizes)
    num_ratios = len(ratios)
    boxes_per_pixel = num_sizes + num_ratios - 1

    size_tensor = torch.tensor(sizes, dtype=torch.float32, device=device)
    ratio_tensor = torch.tensor(ratios, dtype=torch.float32, device=device)

    center_y = (torch.arange(height, device=device) + 0.5) / height
    center_x = (torch.arange(width, device=device) + 0.5) / width
    shift_y, shift_x = torch.meshgrid(center_y, center_x, indexing="ij")
    shift_y = shift_y.reshape(-1)
    shift_x = shift_x.reshape(-1)

    widths = torch.cat(
        (
            size_tensor * torch.sqrt(ratio_tensor[0]),
            size_tensor[0] * torch.sqrt(ratio_tensor[1:]),
        )
    ) * height / width
    heights = torch.cat(
        (
            size_tensor / torch.sqrt(ratio_tensor[0]),
            size_tensor[0] / torch.sqrt(ratio_tensor[1:]),
        )
    )

    anchor_manipulations = torch.stack(
        (-widths, -heights, widths, heights), dim=1
    ).repeat(height * width, 1) / 2

    centers = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
    centers = centers.repeat_interleave(boxes_per_pixel, dim=0)
    anchors = centers + anchor_manipulations
    return anchors.unsqueeze(0)


def assign_anchor_to_bbox(
    ground_truth: torch.Tensor,
    anchors: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """为每个锚框分配一个真实框索引；未分配返回 -1。"""
    num_anchors = anchors.shape[0]
    num_gt = ground_truth.shape[0]
    mapping = torch.full((num_anchors,), -1, dtype=torch.long, device=anchors.device)

    if num_gt == 0:
        return mapping

    ious = box_iou(anchors, ground_truth)
    max_ious, indices = ious.max(dim=1)
    positive = max_ious >= iou_threshold
    mapping[positive] = indices[positive]

    ious_for_matching = ious.clone()
    for _ in range(num_gt):
        flat_index = torch.argmax(ious_for_matching)
        anchor_index = flat_index // num_gt
        gt_index = flat_index % num_gt
        mapping[anchor_index] = gt_index
        ious_for_matching[anchor_index, :] = -1
        ious_for_matching[:, gt_index] = -1

    return mapping


def offset_boxes(anchors: torch.Tensor, assigned_boxes: torch.Tensor) -> torch.Tensor:
    """把真实框编码成 SSD 需要学习的中心偏移与尺度偏移。"""
    anchor_center = box_corner_to_center(anchors)
    assigned_center = box_corner_to_center(assigned_boxes)
    offset_xy = 10 * (assigned_center[:, :2] - anchor_center[:, :2]) / anchor_center[:, 2:]
    offset_wh = 5 * torch.log(assigned_center[:, 2:] / anchor_center[:, 2:].clamp(min=1e-12))
    return torch.cat((offset_xy, offset_wh), dim=1)


def offset_inverse(anchors: torch.Tensor, offset_preds: torch.Tensor) -> torch.Tensor:
    """把网络预测的偏移量解码回边界框坐标。"""
    anchor_center = box_corner_to_center(anchors)
    pred_xy = offset_preds[:, :2] * anchor_center[:, 2:] / 10 + anchor_center[:, :2]
    pred_wh = torch.exp(offset_preds[:, 2:] / 5) * anchor_center[:, 2:]
    return box_center_to_corner(torch.cat((pred_xy, pred_wh), dim=1))


def multibox_target(
    anchors: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把真实标签转换成所有锚框的分类标签与回归标签。"""
    anchors_2d = anchors.squeeze(0)
    batch_size = labels.shape[0]
    bbox_offsets, bbox_masks, class_labels = [], [], []

    for batch_index in range(batch_size):
        label = labels[batch_index]
        valid = label[:, 0] >= 0
        label = label[valid]
        gt_boxes = label[:, 1:5]

        mapping = assign_anchor_to_bbox(gt_boxes, anchors_2d, iou_threshold)
        positive_mask = mapping >= 0
        bbox_mask = positive_mask.float().unsqueeze(-1).repeat(1, 4)

        assigned_boxes = anchors_2d.clone()
        cls = torch.zeros(anchors_2d.shape[0], dtype=torch.long, device=anchors.device)
        if positive_mask.any():
            gt_indices = mapping[positive_mask]
            assigned_boxes[positive_mask] = gt_boxes[gt_indices]
            cls[positive_mask] = label[gt_indices, 0].long() + 1

        offsets = offset_boxes(anchors_2d, assigned_boxes) * bbox_mask
        bbox_offsets.append(offsets.reshape(-1))
        bbox_masks.append(bbox_mask.reshape(-1))
        class_labels.append(cls)

    return (
        torch.stack(bbox_offsets),
        torch.stack(bbox_masks),
        torch.stack(class_labels),
    )


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """手写非极大值抑制（NMS），返回保留框的索引。"""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    order = torch.argsort(scores, descending=True)
    keep: list[int] = []

    while order.numel() > 0:
        current = int(order[0])
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = box_iou(boxes[current].unsqueeze(0), boxes[remaining]).squeeze(0)
        order = remaining[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def multibox_detection(
    class_probs: torch.Tensor,
    offset_preds: torch.Tensor,
    anchors: torch.Tensor,
    score_threshold: float = 0.05,
    nms_threshold: float = 0.5,
) -> list[torch.Tensor]:
    """把 SSD 网络输出解码成最终检测结果。

    class_probs: [B, num_classes + 1, num_anchors]
    offset_preds: [B, num_anchors * 4]
    每个输出行：[class_id, score, xmin, ymin, xmax, ymax]。
    """
    anchors_2d = anchors.squeeze(0)
    batch_size = class_probs.shape[0]
    num_classes = class_probs.shape[1] - 1
    outputs: list[torch.Tensor] = []

    for batch_index in range(batch_size):
        decoded = offset_inverse(
            anchors_2d,
            offset_preds[batch_index].reshape(-1, 4),
        ).clamp(0, 1)
        image_detections: list[torch.Tensor] = []

        for class_index in range(1, num_classes + 1):
            scores = class_probs[batch_index, class_index]
            candidate_mask = scores >= score_threshold
            candidate_indices = torch.where(candidate_mask)[0]
            if candidate_indices.numel() == 0:
                continue

            candidate_boxes = decoded[candidate_indices]
            candidate_scores = scores[candidate_indices]
            kept_local = nms(candidate_boxes, candidate_scores, nms_threshold)
            kept_indices = candidate_indices[kept_local]

            rows = torch.cat(
                (
                    torch.full(
                        (kept_indices.numel(), 1),
                        class_index - 1,
                        device=decoded.device,
                        dtype=decoded.dtype,
                    ),
                    scores[kept_indices].unsqueeze(1),
                    decoded[kept_indices],
                ),
                dim=1,
            )
            image_detections.append(rows)

        if image_detections:
            output = torch.cat(image_detections, dim=0)
            output = output[torch.argsort(output[:, 1], descending=True)]
        else:
            output = torch.empty((0, 6), device=anchors.device)
        outputs.append(output)

    return outputs
