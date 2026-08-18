"""香蕉目标检测数据集与 DataLoader。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from config import DEFAULT_CONFIG


class BananaDetectionDataset(Dataset):
    """读取 D2L 香蕉检测数据集。

    每张图片只有一个香蕉目标，标签格式为：
    [class_id, xmin, ymin, xmax, ymax]。
    坐标会归一化到 [0, 1]。
    """

    def __init__(self, dataset_dir: str | Path, is_train: bool = True) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split_name = "bananas_train" if is_train else "bananas_val"
        self.split_dir = self.dataset_dir / self.split_name
        self.image_dir = self.split_dir / "images"
        self.csv_path = self.split_dir / "label.csv"

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"未找到 {self.csv_path}。请先运行 python download_data.py。"
            )

        self.annotations = pd.read_csv(self.csv_path).set_index("img_name")
        self.image_names = list(self.annotations.index)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_name = self.image_names[index]
        image_path = self.image_dir / image_name
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_tensor = TF.to_tensor(image)

        row = self.annotations.loc[image_name]
        target = torch.tensor(
            [
                float(row["label"]),
                float(row["xmin"]) / width,
                float(row["ymin"]) / height,
                float(row["xmax"]) / width,
                float(row["ymax"]) / height,
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        return image_tensor, target


def create_dataloaders(
    dataset_dir: str | Path | None = None,
    batch_size: int = DEFAULT_CONFIG.batch_size,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """创建训练集与验证集 DataLoader。"""
    dataset_dir = Path(dataset_dir or DEFAULT_CONFIG.dataset_dir)
    train_dataset = BananaDetectionDataset(dataset_dir, is_train=True)
    val_dataset = BananaDetectionDataset(dataset_dir, is_train=False)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
