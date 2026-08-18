"""查看 VOC 输入图像与像素级标签。"""

from __future__ import annotations

from config import DEFAULT_CONFIG
from dataset import VOCSegDataset
from utils import save_prediction_panel


def main() -> None:
    dataset = VOCSegDataset(DEFAULT_CONFIG.voc_dir, is_train=False, crop_size=(320, 480))
    output_dir = DEFAULT_CONFIG.output_dir / "plots" / "dataset_samples"
    for index in range(min(8, len(dataset))):
        image, target = dataset[index]
        save_prediction_panel(image, target, target, output_dir / f"sample_{index:02d}.png")
    print(f"saved samples to {output_dir}")


if __name__ == "__main__":
    main()
