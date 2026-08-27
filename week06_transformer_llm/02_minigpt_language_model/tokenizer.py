"""字符级 tokenizer：把注意力集中到 GPT 架构本身，而不是 tokenizer 工程。"""

from __future__ import annotations

import json
from pathlib import Path

UNK = "<unk>"


class CharTokenizer:
    def __init__(self, chars: list[str]) -> None:
        self.idx_to_token = [UNK, *chars]
        self.token_to_idx = {token: i for i, token in enumerate(self.idx_to_token)}
        self.unk_id = 0

    @classmethod
    def from_training_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.idx_to_token)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_idx.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.idx_to_token[int(i)] if 0 <= int(i) < self.vocab_size else UNK for i in ids)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"chars": self.idx_to_token[1:]}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8"))["chars"])
