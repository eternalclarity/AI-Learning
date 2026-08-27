"""可选拓展：LoRA Linear。默认不接入主训练，先理解低秩增量本质。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank 必须 > 0")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_rank = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        return self.base(x) + self.scale * low_rank
