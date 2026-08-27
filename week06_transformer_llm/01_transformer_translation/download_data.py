"""下载 D2L 常用英法翻译数据集。"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = "https://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip"
SHA1 = "94646ad1522d915e7b0f9296181140edcf86a4f5"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "raw")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    target = args.data_dir / "fra.txt"
    if target.exists():
        print(f"Dataset already exists: {target}")
        return

    archive = args.data_dir / "fra-eng.zip"
    print(f"Downloading: {URL}")
    urllib.request.urlretrieve(URL, archive)
    digest = hashlib.sha1(archive.read_bytes()).hexdigest()
    if digest != SHA1:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"SHA1 校验失败：expected={SHA1} actual={digest}")

    with zipfile.ZipFile(archive) as zf:
        # 压缩包中通常是 fra-eng/fra.txt。
        member = next(name for name in zf.namelist() if name.endswith("fra.txt"))
        with zf.open(member) as src, target.open("wb") as dst:
            dst.write(src.read())

    archive.unlink(missing_ok=True)
    print(f"Saved: {target}")


if __name__ == "__main__":
    main()
