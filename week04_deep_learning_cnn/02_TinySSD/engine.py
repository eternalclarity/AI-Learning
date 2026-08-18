"""TinySSD 训练与验证循环。"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.nn import functional as F
from tqdm import tqdm

from box_ops import multibox_detection, multibox_target
from losses import ssd_loss
from metrics import anchor_class_accuracy, average_precision, positive_bbox_mae


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    iou_threshold: float = 0.5,
    amp: bool = False,
) -> dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    totals = {"loss": 0.0, "class_loss": 0.0, "bbox_loss": 0.0, "class_acc": 0.0, "bbox_mae": 0.0}
    batches = 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, amp):
            anchors, class_predictions, bbox_predictions = model(images)
            bbox_labels, bbox_masks, class_labels = multibox_target(
                anchors,
                targets,
                iou_threshold=iou_threshold,
            )
            total_loss, class_loss, bbox_loss = ssd_loss(
                class_predictions,
                class_labels,
                bbox_predictions,
                bbox_labels,
                bbox_masks,
            )
            loss = total_loss.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totals["loss"] += float(loss.item())
        totals["class_loss"] += float(class_loss.mean().item())
        totals["bbox_loss"] += float(bbox_loss.mean().item())
        totals["class_acc"] += anchor_class_accuracy(class_predictions.detach(), class_labels)
        totals["bbox_mae"] += positive_bbox_mae(
            bbox_predictions.detach(), bbox_labels, bbox_masks
        )
        batches += 1

    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    amp: bool = False,
) -> tuple[dict[str, float], list[torch.Tensor], list[torch.Tensor]]:
    model.eval()
    totals = {"loss": 0.0, "class_acc": 0.0, "bbox_mae": 0.0}
    batches = 0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for images, targets in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with _autocast_context(device, amp):
            anchors, class_predictions, bbox_predictions = model(images)
            bbox_labels, bbox_masks, class_labels = multibox_target(
                anchors,
                targets,
                iou_threshold=iou_threshold,
            )
            total_loss, _, _ = ssd_loss(
                class_predictions,
                class_labels,
                bbox_predictions,
                bbox_labels,
                bbox_masks,
            )

        probabilities = F.softmax(class_predictions.float(), dim=-1).permute(0, 2, 1)
        predictions = multibox_detection(
            probabilities,
            bbox_predictions.float(),
            anchors.float(),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )

        totals["loss"] += float(total_loss.mean().item())
        totals["class_acc"] += anchor_class_accuracy(class_predictions, class_labels)
        totals["bbox_mae"] += positive_bbox_mae(bbox_predictions, bbox_labels, bbox_masks)
        batches += 1

        all_predictions.extend([pred.detach().cpu() for pred in predictions])
        all_targets.extend([target.detach().cpu() for target in targets])

    ap = average_precision(all_predictions, all_targets, iou_threshold=iou_threshold)
    metrics = {key: value / max(batches, 1) for key, value in totals.items()}
    metrics.update(
        {
            "ap50": ap["ap"],
            "precision": ap["precision"],
            "recall": ap["recall"],
            "mean_matched_iou": ap["mean_matched_iou"],
        }
    )
    return metrics, all_predictions, all_targets
