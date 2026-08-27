"""Manual Attention vs PyTorch SDPA：先验证数值，再测速度。"""

from __future__ import annotations

import argparse
import time

import torch

from attention import CausalSelfAttention
from utils import resolve_device, synchronize


def benchmark(module, x, iterations, device):
    module.eval()
    with torch.no_grad():
        for _ in range(10):
            module(x)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            module(x)
        synchronize(device)
    return (time.perf_counter() - start) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    manual = CausalSelfAttention(args.d_model, args.num_heads, dropout=0.0, impl="manual").to(device).eval()
    sdpa = CausalSelfAttention(args.d_model, args.num_heads, dropout=0.0, impl="sdpa").to(device).eval()
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randn(args.batch_size, args.seq_len, args.d_model, device=device)

    with torch.no_grad():
        y_manual, _ = manual(x)
        y_sdpa, _ = sdpa(x)
    max_error = float((y_manual - y_sdpa).abs().max().item())

    manual_time = benchmark(manual, x, args.iterations, device)
    sdpa_time = benchmark(sdpa, x, args.iterations, device)
    print({
        "max_abs_error": max_error,
        "manual_ms": manual_time * 1000,
        "sdpa_ms": sdpa_time * 1000,
        "speedup": manual_time / sdpa_time,
    })


if __name__ == "__main__":
    main()
