"""读取 Tiny Shakespeare 原始文本 → 按 90/10 切分训练集和验证集 → 用训练集建立 Tokenizer → 编码成 token ID → 保存成训练文件"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """切分文本、编码 token，并保存训练数据。"""

    # 创建解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-file", type=Path, default=ROOT / "data" / "raw" / "input.txt")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts") # 处理后文件的保存目录
    parser.add_argument("--train-ratio", type=float, default=0.9) # 训练集占比，默认 90%
    args = parser.parse_args()

    # 切分文本为训练集和验证集
    text = args.raw_file.read_text(encoding="utf-8")
    split = int(len(text) * args.train_ratio)
    train_text = text[:split]
    val_text = text[split:]

    # 建立 vocab tokenizer, 对训练集文本和验证机文本进行 encode
    tokenizer = CharTokenizer.from_training_text(train_text)  # 只根据训练集建立 字符词表 ，避免使用验证集信息，避免验证集的信息提前泄漏进训练阶段
    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)  # 将训练文本编码成 token ID，并转换为 LongTensor
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)   # 将验证文本编码成 token ID，并转换为 LongTensor

    # 把 vocab tokenizer, 训练文本编码, 验证文本编码 存到 /artifact里
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.artifact_dir / "tokenizer.json")
    torch.save(train_ids, args.artifact_dir / "train.pt")
    torch.save(val_ids, args.artifact_dir / "val.pt")
    # 保存数据集统计信息
    (args.artifact_dir / "summary.json").write_text(
        json.dumps({
            "train_chars": len(train_text),
            "val_chars": len(val_text),
            "vocab_size": tokenizer.vocab_size,
            "train_ratio": args.train_ratio,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"train tokens={train_ids.numel()} val tokens={val_ids.numel()} vocab={tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
