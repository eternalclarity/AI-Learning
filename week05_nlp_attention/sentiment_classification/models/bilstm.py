"""Embedding + BiLSTM 文本分类器。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
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

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del attention_mask

        embeddings = self.embedding(input_ids)             # [B,L,E]

        packed = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (h_n, _) = self.lstm(packed)                    # [2*num_layers,B,H]

        forward_hidden = h_n[-2]                           # [B,H]
        backward_hidden = h_n[-1]                          # [B,H]

        representation = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1,
        )                                                  # [B,2H]

        return self.classifier(self.dropout(representation))
