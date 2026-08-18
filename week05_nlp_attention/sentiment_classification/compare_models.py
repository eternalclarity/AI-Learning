"""读取三个验证集摘要并生成公平对比表。"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from .utils import DEFAULT_OUTPUT_DIR


MODEL_NAMES = ["mean_pooling", "bilstm", "bilstm_attention"]


def main() -> None:
    result_dir = DEFAULT_OUTPUT_DIR / "results"
    rows: list[dict] = []

    for model_name in MODEL_NAMES:
        path = result_dir / f"{model_name}_summary.json"

        if not path.exists():
            print(f"Skip {model_name}: missing {path.name}")
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))

        rows.append(
            {
                "Model": payload["model"],
                "Parameters": payload["parameters"],
                "Best Epoch": payload["best_epoch"],
                "Validation Accuracy": payload["best_val_accuracy"],
                "Validation Precision": payload["best_val_precision"],
                "Validation Recall": payload["best_val_recall"],
                "Validation F1": payload["best_val_f1"],
                "Training Seconds": payload["training_seconds"],
            }
        )

    if not rows:
        raise RuntimeError("没有找到任何模型训练摘要。")

    df = pd.DataFrame(rows).sort_values(
        "Validation F1",
        ascending=False,
    ).reset_index(drop=True)

    df.to_csv(
        result_dir / "model_comparison.csv",
        index=False,
    )

    print(df.to_string(index=False))

    plot_dir = DEFAULT_OUTPUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    x = range(len(df))
    plt.bar(x, df["Validation F1"])
    plt.xticks(x, df["Model"], rotation=15)
    plt.ylabel("Validation F1")
    plt.title("Model Comparison")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(plot_dir / "model_comparison.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
