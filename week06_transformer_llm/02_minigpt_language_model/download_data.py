"""下载 Tiny Shakespeare。"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "raw")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    path = args.data_dir / "input.txt"
    if path.exists():
        print(f"Dataset already exists: {path}")
        return
    print(f"Downloading: {URL}")
    urllib.request.urlretrieve(URL, path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
