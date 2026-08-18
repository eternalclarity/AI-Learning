"""在香蕉验证集上评估保存好的 TinySSD。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import DEFAULT_CONFIG
from dataset import create_dataloaders
from engine import validate
from model import TinySSD
from utils import draw_detections, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CONFIG.output_dir / "checkpoints" / "best.pth")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_CONFIG.dataset_dir)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--visual-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    _, val_loader = create_dataloaders(args.data_dir, batch_size=args.batch_size)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TinySSD(num_classes=int(checkpoint.get("num_classes", 1))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics, predictions, targets = validate(
        model,
        val_loader,
        device,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )
    save_json(metrics, DEFAULT_CONFIG.output_dir / "results" / "evaluation.json")
    print(metrics)

    dataset = val_loader.dataset
    for index in range(min(8, len(dataset))):
        image, target = dataset[index]
        draw_detections(
            image,
            target,
            predictions[index],
            DEFAULT_CONFIG.output_dir / "plots" / f"prediction_{index:02d}.png",
            score_threshold=args.visual_threshold,
        )


if __name__ == "__main__":
    main()
