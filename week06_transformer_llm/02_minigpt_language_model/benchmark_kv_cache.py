"""比较 Naive vs KV Cache；使用 greedy 保证两种方法生成同一序列。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from generation import generate_cached, generate_naive
from model import GPT, GPTConfig
from tokenizer import CharTokenizer
from utils import resolve_device, save_json, synchronize

ROOT = Path(__file__).resolve().parent


def timed(fn, model, ids, new_tokens, device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    out = fn(model, ids, max_new_tokens=new_tokens, greedy=True)
    synchronize(device)
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return out, elapsed, peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "checkpoints" / "best.pt")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    tok = CharTokenizer.load(args.artifact_dir / "tokenizer.json")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)

    # warm-up
    _ = generate_cached(model, ids, max_new_tokens=3, greedy=True)
    _ = generate_naive(model, ids, max_new_tokens=3, greedy=True)

    naive_out, naive_time, naive_peak = timed(generate_naive, model, ids, args.max_new_tokens, device)
    cache_out, cache_time, cache_peak = timed(generate_cached, model, ids, args.max_new_tokens, device)
    same = torch.equal(naive_out, cache_out)

    result = {
        "new_tokens": args.max_new_tokens,
        "outputs_identical_greedy": bool(same),
        "naive_seconds": naive_time,
        "cached_seconds": cache_time,
        "naive_tokens_per_second": args.max_new_tokens / naive_time,
        "cached_tokens_per_second": args.max_new_tokens / cache_time,
        "speedup": naive_time / cache_time,
        "naive_peak_memory_bytes": naive_peak,
        "cached_peak_memory_bytes": cache_peak,
    }
    save_json(result, ROOT / "outputs" / "results" / "kv_cache_benchmark.json")
    print(result)


if __name__ == "__main__":
    main()
