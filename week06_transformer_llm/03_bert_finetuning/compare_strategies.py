"""比较 Head-only / Last-N / Full；不依赖 pandas/sklearn。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
STRATEGIES = ["head_only", "last_n", "full"]


def main() -> None:
    result_dir = ROOT / "outputs" / "results"
    rows = []
    for strategy in STRATEGIES:
        path = result_dir / f"{strategy}_summary.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(payload)
        else:
            print(f"Skip {strategy}: missing {path.name}")
    if not rows:
        raise RuntimeError("还没有任何训练 summary")

    rows.sort(key=lambda x: x["best_val_f1"], reverse=True)
    fields = [
        "strategy", "total_parameters", "trainable_parameters", "trainable_ratio",
        "best_val_accuracy", "best_val_precision", "best_val_recall", "best_val_f1",
        "training_seconds", "peak_memory_bytes", "effective_batch_size",
    ]
    with (result_dir / "strategy_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    print("\nStrategy ranking by validation F1:")
    for row in rows:
        print(
            f"{row['strategy']:<10} F1={row['best_val_f1']:.4f} "
            f"trainable={row['trainable_parameters']:,} "
            f"time={row['training_seconds']:.1f}s"
        )

    plot_dir = ROOT / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar([r["strategy"] for r in rows], [r["best_val_f1"] for r in rows])
    plt.ylabel("Validation F1")
    plt.ylim(0, 1)
    plt.title("BERT Fine-tuning Strategy Comparison")
    plt.tight_layout()
    plt.savefig(plot_dir / "strategy_comparison.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
