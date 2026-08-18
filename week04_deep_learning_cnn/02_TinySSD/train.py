"""训练 TinySSD 香蕉目标检测模型。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from config import DEFAULT_CONFIG
from dataset import create_dataloaders
from download_data import download_and_extract
from engine import train_one_epoch, validate
from model import TinySSD
from utils import plot_history, resolve_device, save_checkpoint, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinySSD on banana detection dataset")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_CONFIG.dataset_dir)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.num_epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG.weight_decay)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download:
        args.data_dir = download_and_extract()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device = {device}")

    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = TinySSD(num_classes=1).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = DEFAULT_CONFIG.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    result_dir = output_dir / "results"
    plot_dir = output_dir / "plots"
    for directory in (checkpoint_dir, result_dir, plot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    best_ap50 = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            iou_threshold=DEFAULT_CONFIG.iou_threshold,
            amp=args.amp,
        )
        val_metrics, _, _ = validate(
            model,
            val_loader,
            device,
            iou_threshold=DEFAULT_CONFIG.iou_threshold,
            score_threshold=DEFAULT_CONFIG.score_threshold,
            nms_threshold=DEFAULT_CONFIG.nms_threshold,
            amp=args.amp,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_class_acc": train_metrics["class_acc"],
            "train_bbox_mae": train_metrics["bbox_mae"],
            "val_loss": val_metrics["loss"],
            "val_class_acc": val_metrics["class_acc"],
            "val_bbox_mae": val_metrics["bbox_mae"],
            "val_ap50": val_metrics["ap50"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={row['train_loss']:.4f} | "
            f"val loss={row['val_loss']:.4f} | "
            f"AP50={row['val_ap50']:.4f} | "
            f"P={row['val_precision']:.4f} R={row['val_recall']:.4f}"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_ap50": max(best_ap50, row["val_ap50"]),
            "history": history,
            "num_classes": 1,
        }
        save_checkpoint(state, checkpoint_dir / "last.pth")
        if row["val_ap50"] > best_ap50:
            best_ap50 = row["val_ap50"]
            save_checkpoint(state, checkpoint_dir / "best.pth")

    csv_path = result_dir / "training_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    save_json({"best_val_ap50": best_ap50}, result_dir / "summary.json")
    plot_history(history, plot_dir / "training_curves.png")
    print(f"best validation AP50 = {best_ap50:.4f}")


if __name__ == "__main__":
    main()
