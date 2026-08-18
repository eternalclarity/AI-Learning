"""只在最终选模后使用官方 IMDB test split。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from .data import (
    IMDBDataset,
    Vocab,
    create_dataloader,
    load_split_manifest,
)
from .metrics import compute_classification_metrics
from .utils import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    create_model,
    resolve_device,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["mean_pooling", "bilstm", "bilstm_attention"],
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)

    checkpoint_path = (
        args.output_dir
        / "checkpoints"
        / args.model
        / "best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到最佳模型：{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    vocab = Vocab.load(args.artifact_dir / "vocab.json")
    split_manifest = load_split_manifest(
        args.artifact_dir / "splits.json"
    )

    max_length = int(checkpoint["train_config"]["max_length"])

    test_dataset = IMDBDataset(
        data_dir=args.data_dir,
        records=split_manifest["test"],
        vocab=vocab,
        max_length=max_length,
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = create_model(
        model_name=checkpoint["model_name"],
        **checkpoint["model_config"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_positive_probabilities: list[float] = []
    all_paths: list[str] = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                lengths=lengths,
                attention_mask=attention_mask,
            )

            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_positive_probabilities.extend(
                probabilities[:, 1].cpu().tolist()
            )
            all_paths.extend(batch["relative_paths"])

    metrics = compute_classification_metrics(
        all_labels,
        all_predictions,
    )

    result_dir = args.output_dir / "results"
    plot_dir = args.output_dir / "plots"
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        metrics,
        result_dir / f"{args.model}_test_metrics.json",
    )

    pd.DataFrame(
        {
            "relative_path": all_paths,
            "label": all_labels,
            "prediction": all_predictions,
            "positive_probability": all_positive_probabilities,
        }
    ).to_csv(
        result_dir / f"{args.model}_test_predictions.csv",
        index=False,
    )

    ConfusionMatrixDisplay.from_predictions(
        all_labels,
        all_predictions,
        display_labels=["negative", "positive"],
        values_format="d",
    )
    plt.title(f"{args.model} - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        plot_dir / f"{args.model}_confusion_matrix.png",
        dpi=160,
    )
    plt.close()

    print("Final test metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
