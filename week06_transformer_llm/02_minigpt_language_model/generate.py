from __future__ import annotations

import argparse
from pathlib import Path

import torch

from generation import generate_cached, generate_naive
from model import GPT, GPTConfig
from tokenizer import CharTokenizer
from utils import resolve_device

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "checkpoints" / "best.pt")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    tokenizer = CharTokenizer.load(args.artifact_dir / "tokenizer.json")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    fn = generate_cached if args.use_cache else generate_naive
    output = fn(
        model,
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
