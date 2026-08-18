"""不依赖真实 IMDB 数据的模型前反向传播冒烟测试。"""

from __future__ import annotations

import torch
from torch import nn

from .models import (
    BiLSTMAttentionClassifier,
    BiLSTMClassifier,
    MeanPoolingClassifier,
)


def main() -> None:
    torch.manual_seed(42)

    input_ids = torch.tensor(
        [
            [2, 3, 4, 5, 0, 0],
            [6, 7, 8, 9, 10, 11],
            [12, 13, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    attention_mask = input_ids.ne(0)
    lengths = attention_mask.sum(dim=1)
    labels = torch.tensor([1, 0, 1], dtype=torch.long)

    models = {
        "mean_pooling": MeanPoolingClassifier(
            vocab_size=100,
            embedding_dim=16,
        ),
        "bilstm": BiLSTMClassifier(
            vocab_size=100,
            embedding_dim=16,
            hidden_size=12,
        ),
        "bilstm_attention": BiLSTMAttentionClassifier(
            vocab_size=100,
            embedding_dim=16,
            hidden_size=12,
            attention_dim=10,
        ),
    }

    loss_fn = nn.CrossEntropyLoss()

    for name, model in models.items():
        logits = model(
            input_ids=input_ids,
            lengths=lengths,
            attention_mask=attention_mask,
        )

        assert logits.shape == (3, 2)

        loss = loss_fn(logits, labels)
        loss.backward()

        print(
            f"{name:<20} "
            f"logits={tuple(logits.shape)} "
            f"loss={loss.item():.4f}"
        )

    attention_model = models["bilstm_attention"]
    attention_model.eval()

    with torch.no_grad():
        _, weights = attention_model(
            input_ids=input_ids,
            lengths=lengths,
            attention_mask=attention_mask,
            return_attention=True,
        )

    assert weights.shape == input_ids.shape
    assert torch.all(weights[~attention_mask] < 1e-6)
    assert torch.allclose(
        weights.sum(dim=1),
        torch.ones(3),
        atol=1e-5,
    )

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
