"""Transformer 中最容易混淆的 Mask，统一约定 True=允许关注。"""

from __future__ import annotations

import torch


def make_valid_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """返回 [B,T]，真实 token 为 True，padding 为 False。"""
    return tokens.ne(pad_id)


def make_causal_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
    """返回 [T,T] 下三角布尔矩阵，True 表示该 Key 可以被看到。"""
    return torch.ones(length, length, dtype=torch.bool, device=device).tril()
