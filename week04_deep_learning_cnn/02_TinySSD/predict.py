"""使用 TinySSD 对单张图片进行预测。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from box_ops import multibox_detection
from config import DEFAULT_CONFIG
from model import TinySSD
from utils import draw_detections, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CONFIG.output_dir / "checkpoints" / "best.pth")
    parser.add_argument("--output", type=Path, default=DEFAULT_CONFIG.output_dir / "plots" / "single_prediction.png")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TinySSD(num_classes=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(args.image).convert("RGB").resize((256, 256))
    tensor = TF.to_tensor(image)
    with torch.no_grad():
        anchors, class_predictions, bbox_predictions = model(tensor.unsqueeze(0).to(device))
        probs = F.softmax(class_predictions, dim=-1).permute(0, 2, 1)
        prediction = multibox_detection(
            probs,
            bbox_predictions,
            anchors,
            score_threshold=args.score_threshold,
            nms_threshold=0.5,
        )[0]
    draw_detections(tensor, None, prediction, args.output, score_threshold=args.score_threshold)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
