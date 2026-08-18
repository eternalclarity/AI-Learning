"""FCN 语义分割训练与验证循环。"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.nn import functional as F
from tqdm import tqdm

from metrics import SegmentationConfusionMatrix


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int = 21,
    ignore_index: int = 255,
    amp: bool = False,
) -> dict[str, object]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    confusion = SegmentationConfusionMatrix(num_classes, ignore_index)
    total_loss = 0.0
    batches = 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, amp):
            logits = model(images)
            loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        batches += 1
        confusion.update(logits.detach(), targets)

    metrics = confusion.compute()
    metrics["loss"] = total_loss / max(batches, 1)
    return metrics


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int = 21,
    ignore_index: int = 255,
    amp: bool = False,
) -> dict[str, object]:
    model.eval()
    confusion = SegmentationConfusionMatrix(num_classes, ignore_index)
    total_loss = 0.0
    batches = 0

    for images, targets in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with _autocast_context(device, amp):
            logits = model(images)
            loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)

        total_loss += float(loss.item())
        batches += 1
        confusion.update(logits, targets)

    metrics = confusion.compute()
    metrics["loss"] = total_loss / max(batches, 1)
    return metrics
