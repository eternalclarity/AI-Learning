"""FCN 语义分割项目统一配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentationConfig:
    crop_height: int = 320
    crop_width: int = 480
    num_classes: int = 21
    batch_size: int = 4
    num_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    seed: int = 42
    ignore_index: int = 255

    @property
    def project_dir(self) -> Path:
        return Path(__file__).resolve().parent

    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def voc_dir(self) -> Path:
        return self.data_dir / "VOCdevkit" / "VOC2012"

    @property
    def output_dir(self) -> Path:
        return self.project_dir / "outputs"


DEFAULT_CONFIG = SegmentationConfig()
