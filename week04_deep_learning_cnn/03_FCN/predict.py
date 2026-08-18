"""对任意单张图片进行语义分割，并保存彩色掩码与叠加图。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from config import DEFAULT_CONFIG
from dataset import IMAGENET_MEAN, IMAGENET_STD
from model import FCNResNet18
from utils import class_map_to_rgb, resolve_device, save_overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CONFIG.output_dir / "checkpoints" / "best.pth")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONFIG.output_dir / "plots" / "single_prediction")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FCNResNet18(num_classes=21, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pil_image = Image.open(args.image).convert("RGB")
    image_rgb = np.asarray(pil_image)
    tensor = TF.to_tensor(pil_image)
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    original_h, original_w = tensor.shape[-2:]

    padded_h = math.ceil(original_h / 32) * 32
    padded_w = math.ceil(original_w / 32) * 32
    tensor = F.pad(tensor, (0, padded_w - original_w, 0, padded_h - original_h))

    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        prediction = logits.argmax(dim=1).squeeze(0)[:original_h, :original_w].cpu()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_rgb = class_map_to_rgb(prediction)
    Image.fromarray(mask_rgb).save(args.output_dir / "mask.png")
    save_overlay(image_rgb, prediction, args.output_dir / "overlay.png")
    print(f"saved: {args.output_dir}")


if __name__ == "__main__":
    main()
