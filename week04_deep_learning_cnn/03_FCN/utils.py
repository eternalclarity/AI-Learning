"""语义分割项目通用工具。"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import IMAGENET_MEAN, IMAGENET_STD, VOC_COLORMAP, VOC_CLASSES


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


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)


def class_map_to_rgb(class_map: torch.Tensor) -> np.ndarray:
    class_map = class_map.detach().cpu().numpy()
    rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for class_index, color in enumerate(VOC_COLORMAP):
        rgb[class_map == class_index] = color
    rgb[class_map == 255] = [224, 224, 192]
    return rgb


def save_prediction_panel(
    image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    path: str | Path,
) -> None:
    image_np = denormalize(image).cpu().permute(1, 2, 0).numpy()
    target_rgb = class_map_to_rgb(target)
    pred_rgb = class_map_to_rgb(prediction)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[1].imshow(target_rgb)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_overlay(
    image_rgb: np.ndarray,
    prediction: torch.Tensor,
    path: str | Path,
    alpha: float = 0.5,
) -> None:
    pred_rgb = class_map_to_rgb(prediction).astype(np.float32) / 255.0
    image_float = image_rgb.astype(np.float32) / 255.0
    overlay = (1 - alpha) * image_float + alpha * pred_rgb
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(np.clip(overlay, 0, 1))
    ax.axis("off")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: list[dict[str, float]], path: str | Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Cross Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_miou"] for row in history], label="mIoU")
    axes[1].plot(epochs, [row["val_pixel_acc"] for row in history], label="Pixel Acc")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_per_class_iou(class_iou: list[float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["class_index", "class_name", "iou"])
        for index, value in enumerate(class_iou):
            writer.writerow([index, VOC_CLASSES[index], value])
