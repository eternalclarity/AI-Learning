"""官方 train 再分 train/validation；官方 test 封存到最终评估。"""

from __future__ import annotations

import argparse
from pathlib import Path

from data import discover, save_splits, stratified_split

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "aclImdb")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    official_train = discover(args.data_dir, "train")
    official_test = discover(args.data_dir, "test")
    train, val = stratified_split(official_train, args.val_ratio, args.seed)
    save_splits(
        args.artifact_dir / "splits.json",
        train,
        val,
        official_test,
        {"val_ratio": args.val_ratio, "seed": args.seed},
    )
    print(f"train={len(train)} val={len(val)} test={len(official_test)}")


if __name__ == "__main__":
    main()
