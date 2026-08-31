"""下载 Tiny Shakespeare 数据集"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

# Tiny Shakespeare 数据集的下载地址
# 小型语言模型练习数据集, 本质上就是一大段莎士比亚作品文本
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "raw")
    args = parser.parse_args()

    # 创建数据目录
    # parents=True：父目录不存在时一起创建, exist_ok=True：目录已存在时不报错
    args.data_dir.mkdir(parents=True, exist_ok=True)
    path = args.data_dir / "input.txt"

    if path.exists():
        print(f"Dataset already exists: {path}")
        return

    print(f"Downloading: {URL}")
    urllib.request.urlretrieve(URL, path) # 从 URL 下载文件并保存到 path
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
