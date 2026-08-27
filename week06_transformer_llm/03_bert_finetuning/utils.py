from __future__ import annotations

import json
import random
from pathlib import Path

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前不可用")
    return torch.device(name)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def linear_warmup_decay_lambda(current_step: int, warmup_steps: int, total_steps: int) -> float:
    """LambdaLR factor：warmup 后线性衰减。

    LambdaLR 在初始化时会查询 step=0，因此 warmup 时返回 1/warmup_steps，
    避免第一个真正 optimizer.step() 使用完全为 0 的学习率。
    """
    if warmup_steps > 0 and current_step < warmup_steps:
        return (current_step + 1) / warmup_steps
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 1.0 - progress)
