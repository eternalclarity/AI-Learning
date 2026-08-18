"""下载并安全解压 Stanford IMDB Large Movie Review Dataset。"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path


DATASET_URL = (
    "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
)


def safe_extract(archive_path: Path, destination: Path) -> None:
    """解压前防止 tar 路径穿越。"""
    destination = destination.resolve()

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()

        for member in members:
            target = (destination / member.name).resolve()

            if destination not in target.parents and target != destination:
                raise RuntimeError(
                    f"Unsafe archive member: {member.name}"
                )

        archive.extractall(destination, members=members)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )

    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)

    extracted_dir = args.data_root / "aclImdb"

    if extracted_dir.exists():
        print(f"Dataset already exists: {extracted_dir}")
        return

    archive_path = args.data_root / "aclImdb_v1.tar.gz"

    print(f"Downloading:\n{DATASET_URL}")

    with urllib.request.urlopen(DATASET_URL) as response:
        with archive_path.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)

    print(f"Downloaded: {archive_path}")

    safe_extract(
        archive_path=archive_path,
        destination=args.data_root,
    )

    print(f"Extracted: {extracted_dir}")

    archive_path.unlink(missing_ok=True)

    print("Archive removed.")


if __name__ == "__main__":
    main()
