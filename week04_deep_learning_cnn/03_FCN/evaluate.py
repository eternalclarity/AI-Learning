"""在 VOC2012 验证集上评估 FCN，并输出 mIoU 与可视化。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import DEFAULT_CONFIG
from dataset import create_dataloaders
from engine import validate
from model import FCNResNet18
from utils import resolve_device, save_json, save_per_class_iou, save_prediction_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CONFIG.output_dir / "checkpoints" / "best.pth")
    parser.add_argument("--voc-dir", type=Path, default=DEFAULT_CONFIG.voc_dir)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    crop_h, crop_w = checkpoint.get("crop_size", [320, 480])
    _, val_loader = create_dataloaders(
        args.voc_dir,
        batch_size=args.batch_size,
        crop_size=(crop_h, crop_w),
        num_workers=args.num_workers,
    )

    model = FCNResNet18(num_classes=int(checkpoint.get("num_classes", 21)), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = validate(model, val_loader, device)

    save_json(metrics, DEFAULT_CONFIG.output_dir / "results" / "evaluation.json")
    save_per_class_iou(metrics["class_iou"], DEFAULT_CONFIG.output_dir / "results" / "per_class_iou.csv")
    print({k: v for k, v in metrics.items() if k not in {"class_iou", "confusion_matrix"}})

    model.eval()
    dataset = val_loader.dataset
    with torch.no_grad():
        for index in range(min(8, len(dataset))):
            image, target = dataset[index]
            logits = model(image.unsqueeze(0).to(device))
            prediction = logits.argmax(dim=1).squeeze(0).cpu()
            save_prediction_panel(
                image,
                target,
                prediction,
                DEFAULT_CONFIG.output_dir / "plots" / f"prediction_{index:02d}.png",
            )


if __name__ == "__main__":
    main()
