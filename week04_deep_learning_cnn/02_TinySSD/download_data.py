"""下载并解压《动手学深度学习》使用的香蕉目标检测数据集。"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from config import DEFAULT_CONFIG


DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/banana-detection.zip"
DATA_SHA1 = "5de26c8fce5ccdea9f91267273464dc968d20d72"


def sha1sum(path: Path) -> str:
    """计算文件 SHA1，用于检查下载文件是否完整。"""
    sha1 = hashlib.sha1()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()


def download_and_extract(force: bool = False) -> Path:
    """下载并解压数据集，返回 banana-detection 目录。"""
    data_dir = DEFAULT_CONFIG.data_dir
    dataset_dir = DEFAULT_CONFIG.dataset_dir
    archive_path = data_dir / "banana-detection.zip"
    data_dir.mkdir(parents=True, exist_ok=True)

    if dataset_dir.exists() and not force:
        print(f"数据集已存在：{dataset_dir}")
        return dataset_dir

    if force and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    if not archive_path.exists() or sha1sum(archive_path) != DATA_SHA1:
        print(f"正在下载：{DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, archive_path)

    actual_sha1 = sha1sum(archive_path)
    if actual_sha1 != DATA_SHA1:
        raise RuntimeError(
            "数据集压缩包 SHA1 校验失败："
            f"expected={DATA_SHA1}, actual={actual_sha1}"
        )

    print(f"正在解压：{archive_path}")
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(data_dir)

    if not dataset_dir.exists():
        raise RuntimeError(f"解压完成后未找到目录：{dataset_dir}")

    print(f"数据集准备完成：{dataset_dir}")
    return dataset_dir


if __name__ == "__main__":
    download_and_extract()
