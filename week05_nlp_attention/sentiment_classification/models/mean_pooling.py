"""Embedding + Masked Mean Pooling 文本分类器。"""

from __future__ import annotations

import torch
from torch import nn


class MeanPoolingClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int = 2,
        pad_id: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)              # [B,L] -> [B,L,E]
        mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        summed = (embeddings * mask).sum(dim=1)            # [B,E]
        denominator = lengths.clamp_min(1).unsqueeze(1).to(embeddings.dtype)
        pooled = summed / denominator
        return self.classifier(self.dropout(pooled))       # [B,2]
