"""IMDB 文本数据处理。

本文件故意不依赖 torchtext，以便完整展示：
文本 -> token -> vocabulary -> token id -> padding -> length -> mask。
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


def basic_tokenize(text: str) -> list[str]:
    """简单、可读、适合教学的 word-level tokenizer。"""
    text = text.replace("<br />", " ")
    text = text.lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[^\w\s]", text)


class Vocab:
    """训练集词表：<pad>=0，<unk>=1。"""

    def __init__(self, tokens: Sequence[str]) -> None:
        self.idx_to_token = list(tokens)
        self.token_to_idx = {
            token: index
            for index, token in enumerate(self.idx_to_token)
        }

        if self.idx_to_token[:2] != [PAD_TOKEN, UNK_TOKEN]:
            raise ValueError("词表前两个 token 必须是 <pad> 和 <unk>。")

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def __getitem__(self, token: str) -> int:
        return self.token_to_idx.get(token, UNK_ID)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self[token] for token in tokens]

    def decode(self, ids: Sequence[int]) -> list[str]:
        result: list[str] = []
        for token_id in ids:
            token_id = int(token_id)
            if 0 <= token_id < len(self.idx_to_token):
                result.append(self.idx_to_token[token_id])
            else:
                result.append(UNK_TOKEN)
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"idx_to_token": self.idx_to_token},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["idx_to_token"])


@dataclass(frozen=True)
class SampleRecord:
    """一条样本的相对路径与标签。"""

    relative_path: str
    label: int


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def discover_imdb_samples(data_dir: Path, split: str) -> list[SampleRecord]:
    """发现官方 train/test 的正负样本；0=negative，1=positive。"""
    if split not in {"train", "test"}:
        raise ValueError("split 必须是 'train' 或 'test'。")

    samples: list[SampleRecord] = []

    for folder_name, label in [("neg", 0), ("pos", 1)]:
        folder = data_dir / split / folder_name

        if not folder.exists():
            raise FileNotFoundError(
                f"找不到目录：{folder}\n请先运行 python download_data.py。"
            )

        for path in sorted(folder.glob("*.txt")):
            samples.append(
                SampleRecord(
                    relative_path=str(path.relative_to(data_dir)),
                    label=label,
                )
            )

    return samples


def stratified_train_val_split(
    samples: Sequence[SampleRecord],
    val_ratio: float,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """按标签分层划分 train/validation。"""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio 必须位于 (0, 1)。")

    rng = random.Random(seed)
    train_records: list[SampleRecord] = []
    val_records: list[SampleRecord] = []

    for label in sorted({sample.label for sample in samples}):
        group = [sample for sample in samples if sample.label == label]
        rng.shuffle(group)
        val_size = int(round(len(group) * val_ratio))
        val_records.extend(group[:val_size])
        train_records.extend(group[val_size:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def build_vocab(
    data_dir: Path,
    train_records: Sequence[SampleRecord],
    max_vocab_size: int = 20_000,
    min_freq: int = 2,
) -> Vocab:
    """只从训练子集统计词频并构建词表。"""
    counter: Counter[str] = Counter()

    for record in train_records:
        text = read_text(data_dir / record.relative_path)
        counter.update(basic_tokenize(text))

    candidates = [
        token
        for token, frequency in counter.most_common()
        if frequency >= min_freq
    ]

    candidates = candidates[: max(0, max_vocab_size - 2)]
    return Vocab([PAD_TOKEN, UNK_TOKEN, *candidates])


def save_split_manifest(
    path: Path,
    train_records: Sequence[SampleRecord],
    val_records: Sequence[SampleRecord],
    test_records: Sequence[SampleRecord],
    seed: int,
    val_ratio: float,
) -> None:
    payload = {
        "seed": seed,
        "val_ratio": val_ratio,
        "train": [record.__dict__ for record in train_records],
        "val": [record.__dict__ for record in val_records],
        "test": [record.__dict__ for record in test_records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_split_manifest(path: Path) -> dict[str, list[SampleRecord]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        split: [SampleRecord(**item) for item in payload[split]]
        for split in ["train", "val", "test"]
    }


class IMDBDataset(Dataset):
    """读取原始评论并返回变长 token-id 序列。"""

    def __init__(
        self,
        data_dir: Path,
        records: Sequence[SampleRecord],
        vocab: Vocab,
        max_length: int = 256,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.records = list(records)
        self.vocab = vocab
        self.max_length = int(max_length)

        if self.max_length <= 0:
            raise ValueError("max_length 必须大于 0。")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        text = read_text(self.data_dir / record.relative_path)
        tokens = basic_tokenize(text)[: self.max_length]

        if not tokens:
            tokens = [UNK_TOKEN]

        ids = self.vocab.encode(tokens)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": len(ids),
            "label": int(record.label),
            "relative_path": record.relative_path,
        }


def collate_batch(batch: Sequence[dict], pad_id: int = PAD_ID) -> dict:
    """把变长样本补齐为一个 batch。"""
    lengths = torch.tensor(
        [int(item["length"]) for item in batch],
        dtype=torch.long,
    )

    max_length = int(lengths.max().item())

    input_ids = torch.full(
        (len(batch), max_length),
        fill_value=pad_id,
        dtype=torch.long,
    )

    for row, item in enumerate(batch):
        sequence = item["input_ids"]
        input_ids[row, : sequence.numel()] = sequence

    attention_mask = input_ids.ne(pad_id)

    labels = torch.tensor(
        [int(item["label"]) for item in batch],
        dtype=torch.long,
    )

    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "attention_mask": attention_mask,
        "labels": labels,
        "relative_paths": [str(item["relative_path"]) for item in batch],
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )


def limit_records(
    records: Sequence[SampleRecord],
    max_samples: int | None,
) -> list[SampleRecord]:
    if max_samples is None:
        return list(records)
    return list(records[: max(0, int(max_samples))])
