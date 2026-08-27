from __future__ import annotations

import torch
from torch import nn

from masks import make_valid_mask
from model import Transformer, TransformerConfig


def main() -> None:
    torch.manual_seed(42)
    config = TransformerConfig(
        src_vocab_size=30,
        tgt_vocab_size=40,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        dropout=0.0,
        max_len=32,
    )
    model = Transformer(config)
    src = torch.tensor([[4, 5, 2, 0], [6, 7, 8, 2]])
    tgt = torch.tensor([[1, 9, 10, 0], [1, 11, 12, 13]])
    src_valid = make_valid_mask(src, 0)
    tgt_valid = make_valid_mask(tgt, 0)
    logits = model(src, tgt, src_valid, tgt_valid)
    assert logits.shape == (2, 4, 40)
    labels = torch.tensor([[9, 10, 2, 0], [11, 12, 13, 2]])
    loss = nn.CrossEntropyLoss(ignore_index=0)(logits.reshape(-1, 40), labels.reshape(-1))
    loss.backward()
    print("logits:", tuple(logits.shape), "loss:", float(loss.detach()))
    print("cross attention:", tuple(model.last_cross_attention.shape))
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
