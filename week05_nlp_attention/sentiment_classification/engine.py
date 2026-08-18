"""训练与评估一个 epoch 的通用逻辑。"""

from __future__ import annotations

import torch
from torch import nn
from tqdm import tqdm

from .metrics import compute_classification_metrics


def move_batch_to_device(
    batch: dict,
    device: torch.device,
) -> dict:
    return {
        key: (
            value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def run_epoch(
    model: nn.Module,
    data_loader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = 1.0,
    description: str = "",
) -> dict:
    """optimizer 非空表示训练；为空表示验证/测试。"""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0

    all_labels: list[int] = []
    all_predictions: list[int] = []

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch in tqdm(data_loader, desc=description, leave=False):
            batch = move_batch_to_device(batch, device)
            labels = batch["labels"]

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(
                input_ids=batch["input_ids"],
                lengths=batch["lengths"],
                attention_mask=batch["attention_mask"],
            )

            loss = loss_fn(logits, labels)

            if is_training:
                loss.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip,
                    )

                optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)

            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(
                predictions.detach().cpu().tolist()
            )

    metrics = compute_classification_metrics(
        all_labels,
        all_predictions,
    )

    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics
