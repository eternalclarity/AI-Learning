"""下载 Stanford IMDB Large Movie Review Dataset。"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination, members=members)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "data" / "raw")
    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    extracted = args.data_root / "aclImdb"
    if extracted.exists():
        print(f"Dataset already exists: {extracted}")
        return

    archive = args.data_root / "aclImdb_v1.tar.gz"
    print(f"Downloading: {URL}")
    with urllib.request.urlopen(URL) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output)
    safe_extract(archive, args.data_root)
    archive.unlink(missing_ok=True)
    print(f"Saved: {extracted}")


if __name__ == "__main__":
    main()
