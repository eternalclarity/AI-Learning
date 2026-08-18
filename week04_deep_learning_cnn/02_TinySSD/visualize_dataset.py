"""快速查看香蕉数据集真实边界框。"""

from __future__ import annotations

from config import DEFAULT_CONFIG
from dataset import BananaDetectionDataset
from utils import draw_detections


def main() -> None:
    dataset = BananaDetectionDataset(DEFAULT_CONFIG.dataset_dir, is_train=True)
    output_dir = DEFAULT_CONFIG.output_dir / "plots" / "dataset_samples"
    for index in range(min(8, len(dataset))):
        image, target = dataset[index]
        draw_detections(image, target, None, output_dir / f"sample_{index:02d}.png")
    print(f"saved samples to {output_dir}")


if __name__ == "__main__":
    main()
