"""纯 PyTorch 二分类指标；不依赖 sklearn。"""

from __future__ import annotations

import torch


def confusion_counts(labels: torch.Tensor, preds: torch.Tensor):
    labels = labels.to(torch.long).view(-1)
    preds = preds.to(torch.long).view(-1)
    tn = ((labels == 0) & (preds == 0)).sum()
    fp = ((labels == 0) & (preds == 1)).sum()
    fn = ((labels == 1) & (preds == 0)).sum()
    tp = ((labels == 1) & (preds == 1)).sum()
    return tn, fp, fn, tp


def _safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(b > 0, a.float() / b.float(), torch.zeros((), device=a.device))


def compute_metrics(labels: torch.Tensor, preds: torch.Tensor) -> dict:
    tn, fp, fn, tp = confusion_counts(labels, preds)
    total = tn + fp + fn + tp
    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "accuracy": float(accuracy.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "specificity": float(specificity.item()),
        "f1": float(f1.item()),
        "tn": int(tn.item()),
        "fp": int(fp.item()),
        "fn": int(fn.item()),
        "tp": int(tp.item()),
    }


def confusion_matrix_tensor(labels: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    tn, fp, fn, tp = confusion_counts(labels, preds)
    return torch.tensor([[tn, fp], [fn, tp]], dtype=torch.long)
