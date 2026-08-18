"""TinySSD 目标检测项目的统一配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectionConfig:
    """保存训练与数据相关的默认超参数。"""

    image_size: int = 256
    num_classes: int = 1
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 0.2
    weight_decay: float = 5e-4
    iou_threshold: float = 0.5
    nms_threshold: float = 0.5
    score_threshold: float = 0.05
    seed: int = 42

    @property
    def project_dir(self) -> Path:
        return Path(__file__).resolve().parent

    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / "banana-detection"

    @property
    def output_dir(self) -> Path:
        return self.project_dir / "outputs"


DEFAULT_CONFIG = DetectionConfig()
