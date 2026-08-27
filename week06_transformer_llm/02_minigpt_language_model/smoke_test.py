from __future__ import annotations

import torch

from generation import generate_cached, generate_naive
from model import GPT, GPTConfig


def main() -> None:
    torch.manual_seed(42)
    config = GPTConfig(vocab_size=30, block_size=32, d_model=32, num_heads=4, num_layers=2, d_ff=64, dropout=0.0, attention_impl="manual")
    model = GPT(config)
    ids = torch.randint(0, 30, (2, 12))
    targets = torch.randint(0, 30, (2, 12))
    logits, loss, _ = model(ids, targets)
    assert logits.shape == (2, 12, 30)
    loss.backward()

    # Greedy 下 naive/cache 应产生相同 token。
    model.eval()
    prompt = ids[:1, :5]
    a = generate_naive(model, prompt, 8, greedy=True)
    b = generate_cached(model, prompt, 8, greedy=True)
    assert torch.equal(a, b)
    print("logits:", tuple(logits.shape), "loss:", float(loss.detach()))
    print("naive/cache identical:", torch.equal(a, b))
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
