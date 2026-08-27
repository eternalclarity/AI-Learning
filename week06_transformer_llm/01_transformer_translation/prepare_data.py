"""划分 train/val/test；词表只使用 train 构建。"""

from __future__ import annotations

import argparse
from pathlib import Path

from data import build_vocab, read_raw_pairs, save_artifacts, split_pairs

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-file", type=Path, default=ROOT / "data" / "raw" / "fra.txt")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--max-examples", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-vocab-size", type=int, default=10000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = read_raw_pairs(args.raw_file, max_examples=args.max_examples)
    train, val, test = split_pairs(
        pairs,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # 关键：Vocabulary 只从 train 文本统计。
    src_vocab = build_vocab(
        [p.source for p in train],
        min_freq=args.min_freq,
        max_size=args.max_vocab_size,
    )
    tgt_vocab = build_vocab(
        [p.target for p in train],
        min_freq=args.min_freq,
        max_size=args.max_vocab_size,
    )

    config = {
        "seed": args.seed,
        "max_examples": args.max_examples,
        "min_freq": args.min_freq,
        "max_vocab_size": args.max_vocab_size,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }
    save_artifacts(args.artifact_dir, train, val, test, src_vocab, tgt_vocab, config)

    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print(f"src_vocab={len(src_vocab)} tgt_vocab={len(tgt_vocab)}")


if __name__ == "__main__":
    main()
