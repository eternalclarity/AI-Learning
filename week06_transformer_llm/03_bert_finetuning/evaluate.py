"""最终 test split 评估；纯 PyTorch 指标与 Confusion Matrix。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import IMDBTextDataset, create_loader, load_splits
from metrics import compute_metrics, confusion_matrix_tensor
from strategies import apply_strategy
from utils import resolve_device, save_json

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["head_only", "last_n", "full"], required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "aclImdb")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = args.output_dir / "checkpoints" / args.strategy / "best.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(payload["model_name"], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(payload["model_name"], num_labels=2)
    apply_strategy(model, payload["strategy"], payload["unfreeze_last_n"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()

    splits, _ = load_splits(args.artifact_dir / "splits.json")
    dataset = IMDBTextDataset(args.data_dir, splits["test"])
    loader = create_loader(
        dataset,
        tokenizer,
        args.batch_size,
        False,
        payload["max_length"],
        args.num_workers,
        device.type == "cuda",
        8 if device.type == "cuda" else None,
    )

    labels_all, preds_all, probs_all, paths_all = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            paths = batch.pop("relative_paths")
            labels = batch["labels"].to(device)
            model_inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**model_inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1)
            labels_all.append(labels.cpu())
            preds_all.append(preds.cpu())
            probs_all.extend(probs[:, 1].cpu().tolist())
            paths_all.extend(paths)

    labels = torch.cat(labels_all)
    preds = torch.cat(preds_all)
    metrics = compute_metrics(labels, preds)
    result_dir = args.output_dir / "results"
    plot_dir = args.output_dir / "plots"
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, result_dir / f"{args.strategy}_test_metrics.json")

    with (result_dir / f"{args.strategy}_test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "label", "prediction", "positive_probability"])
        writer.writeheader()
        for path, label, pred, prob in zip(paths_all, labels.tolist(), preds.tolist(), probs_all):
            writer.writerow({
                "relative_path": path,
                "label": label,
                "prediction": pred,
                "positive_probability": prob,
            })

    cm = confusion_matrix_tensor(labels, preds)
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm)
    plt.xticks([0, 1], ["negative", "positive"])
    plt.yticks([0, 1], ["negative", "positive"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"BERT Test Confusion Matrix - {args.strategy}")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(plot_dir / f"{args.strategy}_confusion_matrix.png", dpi=160)
    plt.close()
    print(metrics)


if __name__ == "__main__":
    main()
