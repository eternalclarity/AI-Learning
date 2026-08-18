"""项目通用工具。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from .models import (
    BiLSTMAttentionClassifier,
    BiLSTMClassifier,
    MeanPoolingClassifier,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "aclImdb"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "你指定了 CUDA，但当前 PyTorch 检测不到 CUDA。"
        )

    return torch.device(device_name)


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def create_model(
    model_name: str,
    vocab_size: int,
    embedding_dim: int,
    hidden_size: int,
    attention_dim: int,
    num_layers: int,
    dropout: float,
    pad_id: int,
) -> nn.Module:
    if model_name == "mean_pooling":
        return MeanPoolingClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pad_id=pad_id,
            dropout=dropout,
        )

    if model_name == "bilstm":
        return BiLSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            pad_id=pad_id,
            dropout=dropout,
        )

    if model_name == "bilstm_attention":
        return BiLSTMAttentionClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            attention_dim=attention_dim,
            num_layers=num_layers,
            pad_id=pad_id,
            dropout=dropout,
        )

    raise ValueError(f"未知 model_name：{model_name}")


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_f1: float,
    model_name: str,
    model_config: dict,
    train_config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "best_val_f1": best_val_f1,
            "model_name": model_name,
            "model_config": model_config,
            "train_config": train_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def plot_training_history(
    history: list[dict],
    output_stem: Path,
) -> None:
    if not history:
        return

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_stem.with_name(output_stem.name + "_loss.png"),
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        df["epoch"],
        df["train_accuracy"],
        label="Train Accuracy",
    )
    plt.plot(
        df["epoch"],
        df["val_accuracy"],
        label="Validation Accuracy",
    )
    plt.plot(
        df["epoch"],
        df["val_f1"],
        label="Validation F1",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Classification Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_stem.with_name(output_stem.name + "_metrics.png"),
        dpi=160,
    )
    plt.close()
