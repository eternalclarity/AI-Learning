"""Pascal VOC2012 语义分割自定义 Dataset。

训练：输入图像和标签同步随机裁剪，并可同步水平翻转。
验证：使用确定性的中心裁剪，保证每次评价使用同一批像素。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from config import DEFAULT_CONFIG


VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
]

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "potted plant", "sheep", "sofa",
    "train", "tv/monitor",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_SORTED_COLOR_CODES = np.array(
    sorted([(r * 256 + g) * 256 + b for r, g, b in VOC_COLORMAP]),
    dtype=np.int64,
)
_COLOR_TO_CLASS = {
    (r * 256 + g) * 256 + b: class_index
    for class_index, (r, g, b) in enumerate(VOC_COLORMAP)
}
_SORTED_CLASSES = np.array([_COLOR_TO_CLASS[int(code)] for code in _SORTED_COLOR_CODES], dtype=np.int64)


def voc_label_indices(rgb_label: Image.Image | np.ndarray) -> torch.Tensor:
    """把 VOC 彩色标签映射为 [H,W] 类别索引。

    VOC 官方边界/void 颜色不属于 21 个语义类别，本项目映射成 255，
    后续 CrossEntropyLoss(ignore_index=255) 会忽略这些像素。
    """
    array = np.asarray(rgb_label, dtype=np.int64)
    pixel_codes = (array[..., 0] * 256 + array[..., 1]) * 256 + array[..., 2]
    positions = np.searchsorted(_SORTED_COLOR_CODES, pixel_codes)
    clipped = np.clip(positions, 0, len(_SORTED_COLOR_CODES) - 1)
    valid = _SORTED_COLOR_CODES[clipped] == pixel_codes
    class_map = np.full(pixel_codes.shape, DEFAULT_CONFIG.ignore_index, dtype=np.int64)
    class_map[valid] = _SORTED_CLASSES[clipped[valid]]
    return torch.from_numpy(class_map).long()


def _crop_pair(
    image: Image.Image,
    label: Image.Image,
    crop_size: tuple[int, int],
    random_crop: bool,
) -> tuple[Image.Image, Image.Image]:
    crop_h, crop_w = crop_size
    width, height = image.size

    if random_crop:
        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)
    else:
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2

    image = TF.crop(image, top, left, crop_h, crop_w)
    label = TF.crop(label, top, left, crop_h, crop_w)
    return image, label


class VOCSegDataset(Dataset):
    """Pascal VOC2012 像素级语义分割数据集。"""

    def __init__(
        self,
        voc_dir: str | Path,
        is_train: bool,
        crop_size: tuple[int, int] = (320, 480),
        horizontal_flip: bool = True,
    ) -> None:
        self.voc_dir = Path(voc_dir)
        self.is_train = is_train
        self.crop_size = crop_size
        self.horizontal_flip = horizontal_flip and is_train

        split_file = self.voc_dir / "ImageSets" / "Segmentation" / (
            "train.txt" if is_train else "val.txt"
        )
        if not split_file.exists():
            raise FileNotFoundError(
                f"未找到 {split_file}。请先运行 python download_voc.py。"
            )

        image_ids = split_file.read_text(encoding="utf-8").split()
        self.image_ids = [image_id for image_id in image_ids if self._large_enough(image_id)]
        print(
            f"VOC {'train' if is_train else 'val'}: "
            f"保留 {len(self.image_ids)} / {len(image_ids)} 张图像"
        )

    def _large_enough(self, image_id: str) -> bool:
        image_path = self.voc_dir / "JPEGImages" / f"{image_id}.jpg"
        with Image.open(image_path) as image:
            width, height = image.size
        crop_h, crop_w = self.crop_size
        return height >= crop_h and width >= crop_w

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[index]
        image_path = self.voc_dir / "JPEGImages" / f"{image_id}.jpg"
        label_path = self.voc_dir / "SegmentationClass" / f"{image_id}.png"

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")
        image, label = _crop_pair(
            image,
            label,
            self.crop_size,
            random_crop=self.is_train,
        )

        if self.horizontal_flip and random.random() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        label_tensor = voc_label_indices(label)
        return image_tensor, label_tensor


def create_dataloaders(
    voc_dir: str | Path | None = None,
    batch_size: int = DEFAULT_CONFIG.batch_size,
    crop_size: tuple[int, int] = (DEFAULT_CONFIG.crop_height, DEFAULT_CONFIG.crop_width),
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    voc_dir = Path(voc_dir or DEFAULT_CONFIG.voc_dir)
    train_dataset = VOCSegDataset(voc_dir, True, crop_size=crop_size)
    val_dataset = VOCSegDataset(voc_dir, False, crop_size=crop_size, horizontal_flip=False)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
