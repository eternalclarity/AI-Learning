"""BERT 训练/验证核心；支持 AMP、Gradient Accumulation、Gradient Clipping。"""

from __future__ import annotations

import math

import torch
from tqdm import tqdm

from metrics import compute_metrics


def move_batch(batch: dict, device: torch.device):
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def evaluate_epoch(model, loader, device: torch.device, amp: bool = False):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels_all = []
    preds_all = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="validation", leave=False):
            batch = move_batch(batch, device)
            paths = batch.pop("relative_paths", None)
            del paths
            labels = batch["labels"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                outputs = model(**batch)
                loss = outputs.loss
            preds = outputs.logits.argmax(dim=-1)
            total_loss += float(loss.item()) * labels.size(0)
            total_examples += labels.size(0)
            labels_all.append(labels.detach().cpu())
            preds_all.append(preds.detach().cpu())

    labels = torch.cat(labels_all)
    preds = torch.cat(preds_all)
    result = compute_metrics(labels, preds)
    result["loss"] = total_loss / max(total_examples, 1)
    return result


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    grad_accum_steps: int,
    grad_clip: float,
    amp: bool,
    scaler,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_examples = 0
    labels_all = []
    preds_all = []

    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        batch = move_batch(batch, device)
        batch.pop("relative_paths", None)
        labels = batch["labels"]

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            outputs = model(**batch)
            raw_loss = outputs.loss
            loss = raw_loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = step % grad_accum_steps == 0 or step == len(loader)
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    grad_clip,
                )
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        preds = outputs.logits.argmax(dim=-1)
        total_loss += float(raw_loss.item()) * labels.size(0)
        total_examples += labels.size(0)
        labels_all.append(labels.detach().cpu())
        preds_all.append(preds.detach().cpu())

    labels = torch.cat(labels_all)
    preds = torch.cat(preds_all)
    result = compute_metrics(labels, preds)
    result["loss"] = total_loss / max(total_examples, 1)
    return result


def optimizer_steps_per_epoch(loader_length: int, grad_accum_steps: int) -> int:
    return math.ceil(loader_length / grad_accum_steps)
