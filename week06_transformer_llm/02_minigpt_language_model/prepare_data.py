"""连续文本按 90/10 切分；Tokenizer 只由 train text 建立。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-file", type=Path, default=ROOT / "data" / "raw" / "input.txt")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()

    text = args.raw_file.read_text(encoding="utf-8")
    split = int(len(text) * args.train_ratio)
    train_text = text[:split]
    val_text = text[split:]

    tokenizer = CharTokenizer.from_training_text(train_text)
    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.artifact_dir / "tokenizer.json")
    torch.save(train_ids, args.artifact_dir / "train.pt")
    torch.save(val_ids, args.artifact_dir / "val.pt")
    (args.artifact_dir / "summary.json").write_text(
        json.dumps({
            "train_chars": len(train_text),
            "val_chars": len(val_text),
            "vocab_size": tokenizer.vocab_size,
            "train_ratio": args.train_ratio,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"train tokens={train_ids.numel()} val tokens={val_ids.numel()} vocab={tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
