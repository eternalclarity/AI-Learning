"""英法翻译数据处理：自己实现 tokenizer、vocab、padding 和 DataLoader。"""

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

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = range(4)


def tokenize(text: str) -> list[str]:
    """接近 D2L 教学风格的 word-level tokenizer；标点单独作为 token。"""
    text = text.replace("\u202f", " ").replace("\xa0", " ").lower().strip()
    return re.findall(r"[a-zà-ÿœæç]+(?:'[a-zà-ÿœæç]+)?|[.!?,;:]", text)


class Vocab:
    def __init__(self, idx_to_token: Sequence[str]) -> None:
        self.idx_to_token = list(idx_to_token)
        self.token_to_idx = {token: i for i, token in enumerate(self.idx_to_token)}
        if self.idx_to_token[:4] != SPECIALS:
            raise ValueError(f"词表前四项必须为 {SPECIALS}")

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def __getitem__(self, token: str) -> int:
        return self.token_to_idx.get(token, UNK_ID)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self[token] for token in tokens]

    def decode(self, ids: Sequence[int], skip_specials: bool = False) -> list[str]:
        result = []
        for token_id in ids:
            token = self.idx_to_token[int(token_id)] if 0 <= int(token_id) < len(self) else UNK
            if skip_specials and token in SPECIALS:
                continue
            result.append(token)
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"idx_to_token": self.idx_to_token}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text(encoding="utf-8"))["idx_to_token"])


def build_vocab(texts: Sequence[str], min_freq: int = 2, max_size: int = 10000) -> Vocab:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    regular = [tok for tok, freq in counter.most_common() if freq >= min_freq]
    regular = regular[: max(0, max_size - len(SPECIALS))]
    return Vocab([*SPECIALS, *regular])


@dataclass(frozen=True)
class Pair:
    source: str
    target: str


def read_raw_pairs(path: Path, max_examples: int | None = None) -> list[Pair]:
    """fra.txt 每行通常为 English<TAB>French<TAB>metadata；只取前两列。"""
    pairs: list[Pair] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        pairs.append(Pair(fields[0].strip(), fields[1].strip()))
        if max_examples is not None and len(pairs) >= max_examples:
            break
    return pairs


def split_pairs(pairs: Sequence[Pair], seed: int = 42, val_ratio: float = 0.1, test_ratio: float = 0.1):
    items = list(pairs)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test = items[:n_test]
    val = items[n_test:n_test + n_val]
    train = items[n_test + n_val:]
    return train, val, test


def save_artifacts(artifact_dir: Path, train, val, test, src_vocab: Vocab, tgt_vocab: Vocab, config: dict) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": [p.__dict__ for p in train],
        "val": [p.__dict__ for p in val],
        "test": [p.__dict__ for p in test],
        "config": config,
    }
    (artifact_dir / "splits.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    src_vocab.save(artifact_dir / "src_vocab.json")
    tgt_vocab.save(artifact_dir / "tgt_vocab.json")


def load_artifacts(artifact_dir: Path):
    payload = json.loads((artifact_dir / "splits.json").read_text(encoding="utf-8"))
    splits = {
        name: [Pair(**item) for item in payload[name]]
        for name in ["train", "val", "test"]
    }
    return splits, Vocab.load(artifact_dir / "src_vocab.json"), Vocab.load(artifact_dir / "tgt_vocab.json"), payload["config"]


class TranslationDataset(Dataset):
    def __init__(self, pairs: Sequence[Pair], src_vocab: Vocab, tgt_vocab: Vocab, max_length: int = 40) -> None:
        self.pairs = list(pairs)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        src_tokens = tokenize(pair.source)[: self.max_length - 1]
        tgt_tokens = tokenize(pair.target)[: self.max_length - 2]

        src_ids = self.src_vocab.encode(src_tokens) + [EOS_ID]
        tgt_ids = [BOS_ID] + self.tgt_vocab.encode(tgt_tokens) + [EOS_ID]

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
            "source_text": pair.source,
            "target_text": pair.target,
        }


def collate_batch(batch: Sequence[dict]):
    max_src = max(item["src"].numel() for item in batch)
    max_tgt = max(item["tgt"].numel() for item in batch)
    src = torch.full((len(batch), max_src), PAD_ID, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt), PAD_ID, dtype=torch.long)
    for i, item in enumerate(batch):
        src[i, : item["src"].numel()] = item["src"]
        tgt[i, : item["tgt"].numel()] = item["tgt"]
    return {
        "src": src,
        "tgt": tgt,
        "source_texts": [item["source_text"] for item in batch],
        "target_texts": [item["target_text"] for item in batch],
    }


def create_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0, pin_memory: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )
