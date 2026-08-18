"""训练 ResNet18-FCN 进行 Pascal VOC2012 语义分割。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from config import DEFAULT_CONFIG
from dataset import create_dataloaders
from download_voc import download_and_extract
from engine import train_one_epoch, validate
from model import FCNResNet18
from utils import plot_history, resolve_device, save_checkpoint, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FCN-ResNet18 on Pascal VOC2012")
    parser.add_argument("--voc-dir", type=Path, default=DEFAULT_CONFIG.voc_dir)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.num_epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--crop-height", type=int, default=DEFAULT_CONFIG.crop_height)
    parser.add_argument("--crop-width", type=int, default=DEFAULT_CONFIG.crop_width)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG.weight_decay)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download:
        args.voc_dir = download_and_extract()

    if args.crop_height % 32 != 0 or args.crop_width % 32 != 0:
        raise ValueError("FCN 本项目的 crop height/width 必须能被 32 整除。")

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device = {device}")

    train_loader, val_loader = create_dataloaders(
        args.voc_dir,
        batch_size=args.batch_size,
        crop_size=(args.crop_height, args.crop_width),
        num_workers=args.num_workers,
    )
    model = FCNResNet18(
        num_classes=DEFAULT_CONFIG.num_classes,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = DEFAULT_CONFIG.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    result_dir = output_dir / "results"
    plot_dir = output_dir / "plots"
    for directory in (checkpoint_dir, result_dir, plot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    best_miou = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            num_classes=DEFAULT_CONFIG.num_classes,
            ignore_index=DEFAULT_CONFIG.ignore_index,
            amp=args.amp,
        )
        val_metrics = validate(
            model,
            val_loader,
            device,
            num_classes=DEFAULT_CONFIG.num_classes,
            ignore_index=DEFAULT_CONFIG.ignore_index,
            amp=args.amp,
        )

        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_pixel_acc": float(train_metrics["pixel_accuracy"]),
            "train_miou": float(train_metrics["mean_iou"]),
            "val_loss": float(val_metrics["loss"]),
            "val_pixel_acc": float(val_metrics["pixel_accuracy"]),
            "val_miou": float(val_metrics["mean_iou"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={row['train_loss']:.4f} mIoU={row['train_miou']:.4f} | "
            f"val loss={row['val_loss']:.4f} mIoU={row['val_miou']:.4f} "
            f"pixel_acc={row['val_pixel_acc']:.4f}"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou": max(best_miou, row["val_miou"]),
            "history": history,
            "num_classes": DEFAULT_CONFIG.num_classes,
            "pretrained": not args.no_pretrained,
            "crop_size": [args.crop_height, args.crop_width],
        }
        save_checkpoint(state, checkpoint_dir / "last.pth")
        if row["val_miou"] > best_miou:
            best_miou = row["val_miou"]
            save_checkpoint(state, checkpoint_dir / "best.pth")

    csv_path = result_dir / "training_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    save_json({"best_val_miou": best_miou}, result_dir / "summary.json")
    plot_history(history, plot_dir / "training_curves.png")
    print(f"best validation mIoU = {best_miou:.4f}")


if __name__ == "__main__":
    main()
