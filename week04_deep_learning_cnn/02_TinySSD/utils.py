"""目标检测项目通用工具。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_checkpoint(state: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def plot_history(history: list[dict[str, float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("SSD Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_ap50"] for row in history], label="AP@0.5")
    axes[1].set_title("Validation AP@0.5")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_detections(
    image: torch.Tensor,
    target: torch.Tensor | None,
    prediction: torch.Tensor | None,
    path: str | Path,
    score_threshold: float = 0.5,
) -> None:
    """绘制真实框和预测框。绿色虚线=GT，红色实线=预测。"""
    image_np = image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    height, width = image_np.shape[:2]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_np)
    ax.axis("off")

    if target is not None:
        for row in target.detach().cpu():
            if row[0] < 0:
                continue
            x1, y1, x2, y2 = row[1:5].tolist()
            rect = Rectangle(
                (x1 * width, y1 * height),
                (x2 - x1) * width,
                (y2 - y1) * height,
                fill=False,
                linewidth=2,
                linestyle="--",
                edgecolor="lime",
            )
            ax.add_patch(rect)
            ax.text(x1 * width, y1 * height, "GT", color="lime", fontsize=9)

    if prediction is not None:
        for row in prediction.detach().cpu():
            score = float(row[1])
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = row[2:6].tolist()
            rect = Rectangle(
                (x1 * width, y1 * height),
                (x2 - x1) * width,
                (y2 - y1) * height,
                fill=False,
                linewidth=2,
                edgecolor="red",
            )
            ax.add_patch(rect)
            ax.text(x1 * width, y1 * height, f"banana {score:.2f}", color="red", fontsize=9)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
