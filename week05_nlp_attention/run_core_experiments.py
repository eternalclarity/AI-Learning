"""顺序训练三个核心实验，并生成验证集比较表。"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-length", type=int, default=256)

    args = parser.parse_args()

    for model_name in [
        "mean_pooling",
        "bilstm",
        "bilstm_attention",
    ]:
        command = [
            sys.executable,
            "-m",
            "sentiment_classification.train",
            "--model",
            model_name,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--max-length",
            str(args.max_length),
        ]

        print("\n" + "=" * 72)
        print(f"Running: {model_name}")
        print("=" * 72)

        subprocess.run(command, check=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "sentiment_classification.compare_models",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
