"""生成可复现划分，并只从训练子集建立 vocabulary。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from sentiment_classification.data import (
    build_vocab,
    discover_imdb_samples,
    save_split_manifest,
    stratified_train_val_split,
)
from sentiment_classification.utils import save_json


PROJECT_ROOT = Path(__file__).resolve().parent


def label_counts(records) -> dict[str, int]:
    counter = Counter(record.label for record in records)

    return {
        "negative": int(counter.get(0, 0)),
        "positive": int(counter.get(1, 0)),
        "total": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "aclImdb",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-vocab-size", type=int, default=20_000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    official_train = discover_imdb_samples(args.data_dir, "train")
    official_test = discover_imdb_samples(args.data_dir, "test")

    train_records, val_records = stratified_train_val_split(
        official_train,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print("Building vocabulary from TRAIN split only...")

    vocab = build_vocab(
        data_dir=args.data_dir,
        train_records=train_records,
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
    )

    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    vocab.save(args.artifact_dir / "vocab.json")

    save_split_manifest(
        path=args.artifact_dir / "splits.json",
        train_records=train_records,
        val_records=val_records,
        test_records=official_test,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )

    summary = {
        "train": label_counts(train_records),
        "validation": label_counts(val_records),
        "test": label_counts(official_test),
        "vocab_size": len(vocab),
        "max_vocab_size": args.max_vocab_size,
        "min_freq": args.min_freq,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
    }

    save_json(
        summary,
        args.artifact_dir / "data_summary.json",
    )

    print("\nData preparation finished.")
    print(f"Train: {len(train_records)}")
    print(f"Validation: {len(val_records)}")
    print(f"Test: {len(official_test)}")
    print(f"Vocabulary: {len(vocab)}")


if __name__ == "__main__":
    main()
