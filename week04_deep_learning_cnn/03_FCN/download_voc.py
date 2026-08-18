"""下载并解压 Pascal VOC2012 train/val 数据集。

压缩包约 2GB，第一次下载需要较长时间。
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

from config import DEFAULT_CONFIG


DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/VOCtrainval_11-May-2012.tar"
DATA_SHA1 = "4e443f8a2eca6b1dac8a6c57641b67dd40621a49"


def sha1sum(path: Path) -> str:
    sha1 = hashlib.sha1()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()


def download_and_extract(force: bool = False) -> Path:
    data_dir = DEFAULT_CONFIG.data_dir
    voc_dir = DEFAULT_CONFIG.voc_dir
    archive_path = data_dir / "VOCtrainval_11-May-2012.tar"
    data_dir.mkdir(parents=True, exist_ok=True)

    if voc_dir.exists() and not force:
        print(f"VOC2012 已存在：{voc_dir}")
        return voc_dir

    if force and voc_dir.parent.parent.exists():
        shutil.rmtree(voc_dir.parent.parent)

    if not archive_path.exists() or sha1sum(archive_path) != DATA_SHA1:
        print("Pascal VOC2012 压缩包约 2GB，请耐心等待。")
        print(f"正在下载：{DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, archive_path)

    actual = sha1sum(archive_path)
    if actual != DATA_SHA1:
        raise RuntimeError(f"SHA1 校验失败：expected={DATA_SHA1}, actual={actual}")

    print(f"正在解压：{archive_path}")
    destination = data_dir.resolve()
    with tarfile.open(archive_path, "r") as tar:
        # Python 3.10 兼容的路径穿越检查：禁止压缩包成员写出 data 目录。
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise RuntimeError(f"压缩包包含不安全路径：{member.name}")
        tar.extractall(data_dir)

    if not voc_dir.exists():
        raise RuntimeError(f"解压后未找到 {voc_dir}")
    print(f"VOC2012 准备完成：{voc_dir}")
    return voc_dir


if __name__ == "__main__":
    download_and_extract()
