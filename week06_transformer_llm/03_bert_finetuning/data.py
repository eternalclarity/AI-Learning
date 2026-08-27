"""IMDB raw-text Dataset + 动态 Padding Collator。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Record:
    relative_path: str
    label: int


def discover(data_dir: Path, split: str) -> list[Record]:
    if split not in {"train", "test"}:
        raise ValueError("split 必须为 train/test")
    result = []
    for folder, label in [("neg", 0), ("pos", 1)]:
        path = data_dir / split / folder
        if not path.exists():
            raise FileNotFoundError(f"找不到 {path}；请先运行 download_data.py")
        for file in sorted(path.glob("*.txt")):
            result.append(Record(str(file.relative_to(data_dir)), label))
    return result


def stratified_split(records: Sequence[Record], val_ratio: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    train, val = [], []
    for label in [0, 1]:
        group = [r for r in records if r.label == label]
        rng.shuffle(group)
        n_val = int(round(len(group) * val_ratio))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def save_splits(path: Path, train, val, test, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "train": [r.__dict__ for r in train],
        "val": [r.__dict__ for r in val],
        "test": [r.__dict__ for r in test],
        "config": config,
    }, indent=2), encoding="utf-8")


def load_splits(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: [Record(**x) for x in payload[name]] for name in ["train", "val", "test"]}, payload["config"]


class IMDBTextDataset(Dataset):
    def __init__(self, data_dir: Path, records: Sequence[Record]) -> None:
        self.data_dir = data_dir
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        text = (self.data_dir / record.relative_path).read_text(encoding="utf-8", errors="replace").replace("<br />", " ")
        return {"text": text, "label": record.label, "relative_path": record.relative_path}


class DynamicPaddingCollator:
    """把 tokenizer 延迟到 batch 阶段，padding=True 只补到当前 batch 最长序列。"""

    def __init__(self, tokenizer, max_length: int = 128, pad_to_multiple_of: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: Sequence[dict]):
        encoded = self.tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        encoded["relative_paths"] = [item["relative_path"] for item in batch]
        return encoded


def create_loader(dataset, tokenizer, batch_size: int, shuffle: bool, max_length: int, num_workers: int = 0, pin_memory: bool = False, pad_to_multiple_of: int | None = None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=DynamicPaddingCollator(tokenizer, max_length, pad_to_multiple_of),
    )
