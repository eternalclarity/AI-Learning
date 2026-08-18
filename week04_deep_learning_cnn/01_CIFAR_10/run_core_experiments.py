"""按顺序运行四组核心公平对比实验。"""

from __future__ import annotations

import argparse  # 用于读取公共训练参数。
import subprocess  # 用于按顺序调用 train.py。
import sys  # 用于获取当前 Python 解释器路径。
from pathlib import Path  # 用于传递目录参数。

from config import CORE_EXPERIMENT_NAMES  # 导入四组核心实验名称。


def parse_args() -> argparse.Namespace:
    """读取批量实验参数。"""

    parser = argparse.ArgumentParser(
        description="Run all four core Week 4 experiments sequentially.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    """逐个执行核心实验，任一实验失败时立即停止。"""

    args = parse_args()  # 读取公共参数。

    for experiment_name in CORE_EXPERIMENT_NAMES:  # 按既定顺序运行四组实验。
        command = [
            sys.executable,
            "train.py",
            "--experiment",
            experiment_name,
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(args.output_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
        ]

        if args.amp:  # 用户启用 AMP 时将开关传给 train.py。
            command.append("--amp")

        print("=" * 80)
        print(f"Running {experiment_name}")
        print("Command:", " ".join(command))
        subprocess.run(command, check=True)  # check=True 会在训练失败时抛出异常并停止。

    comparison_command = [
        sys.executable,
        "compare_experiments.py",
        "--results-dir",
        str(args.output_dir / "results"),
        "--output-dir",
        str(args.output_dir / "results" / "comparison"),
    ]
    subprocess.run(comparison_command, check=True)  # 四组训练完成后自动生成验证集对比表。


if __name__ == "__main__":
    main()  # 直接运行文件时开始批量实验。
