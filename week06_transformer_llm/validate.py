"""快速检查整个 Week 06 工程。

不会下载真实数据，也不会训练正式模型。
"""

from __future__ import annotations

import compileall
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECTS = [
    ROOT / "01_transformer_translation",
    ROOT / "02_minigpt_language_model",
    ROOT / "03_bert_finetuning",
]


def run(command, cwd):
    print(f"\n[{cwd.name}] $ {' '.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    print("1) compileall")
    if not compileall.compile_dir(ROOT, quiet=1):
        raise SystemExit("compileall failed")

    print("2) unit tests")
    for project in PROJECTS:
        run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], project)

    print("3) offline smoke tests")
    run([sys.executable, "smoke_test.py"], PROJECTS[0])
    run([sys.executable, "smoke_test.py"], PROJECTS[1])

    if importlib.util.find_spec("transformers") is not None:
        run([sys.executable, "smoke_test.py"], PROJECTS[2])
    else:
        print("\n[BERT] transformers 未安装：跳过 tiny-BERT smoke_test；安装 requirements 后可运行。")

    print("\nAll available checks passed.")


if __name__ == "__main__":
    main()
