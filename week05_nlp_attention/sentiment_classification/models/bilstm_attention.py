"""Embedding + BiLSTM + Additive Attention 分类器。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .attention import AdditiveAttentionPooling


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        attention_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 2,
        pad_id: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = AdditiveAttentionPooling(
            feature_dim=hidden_size * 2,
            attention_dim=attention_dim,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def encode(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.lstm(packed)

        sequence_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.size(1),
        )

        return sequence_output

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        sequence_output = self.encode(input_ids, lengths)   # [B,L,2H]

        context, weights = self.attention(
            sequence_output,
            attention_mask,
        )                                                   # [B,2H], [B,L]

        logits = self.classifier(self.dropout(context))

        if return_attention:
            return logits, weights

        return logits
