"""Bahdanau 风格加性注意力池化。"""

from __future__ import annotations

import torch
from torch import nn


class AdditiveAttentionPooling(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        attention_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.key_projection = nn.Linear(
            feature_dim,
            attention_dim,
            bias=False,
        )

        self.query = nn.Parameter(torch.empty(attention_dim))

        self.score_projection = nn.Linear(
            attention_dim,
            1,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self.key_projection(sequence)                # [B,L,D] -> [B,L,A]
        query = self.query.view(1, 1, -1)                  # [1,1,A]
        features = torch.tanh(keys + query)                # [B,L,A]
        scores = self.score_projection(features).squeeze(-1)  # [B,L]

        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )

        weights = torch.softmax(scores, dim=-1)            # [B,L]
        dropped_weights = self.dropout(weights)

        context = (
            dropped_weights.unsqueeze(-1) * sequence
        ).sum(dim=1)                                       # [B,D]

        return context, weights
