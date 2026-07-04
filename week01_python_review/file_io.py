"""
01_file_io.py

本脚本用于练习 Python 文件读写、pathlib 路径处理、函数封装和异常处理。

功能：
1. 读取一个文本文件；
2. 统计文件中的行数、单词数、字符数；
3. 输出统计结果；
4. 如果示例文件不存在，可以自动创建一个 sample.txt。

运行方式：

python week1_python_review/scripts/01_file_io.py

或者指定文件：

python week1_python_review/scripts/01_file_io.py --file week1_python_review/scripts/sample.txt
"""

from pathlib import Path
import argparse


def create_sample_file(file_path: Path) -> None:
    """
    创建一个示例文本文件。

    参数：
        file_path: 要创建的文件路径
    """
    sample_text = """Python is useful for AI research.
NumPy is used for numerical computing.
Pandas is used for data analysis.
Matplotlib is used for data visualization.
Git is useful for code management.
"""

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        f.write(sample_text)


def analyze_file(file_path: Path) -> dict:
    """
    分析文本文件，统计行数、单词数和字符数。

    参数：
        file_path: 文本文件路径

    返回：
        一个字典，包含 lines、words、chars 三个统计结果
    """
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"当前路径不是文件: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        text = f.read()

    lines = text.splitlines()
    words = text.split()
    chars = len(text)

    result = {
        "file_path": str(file_path),
        "lines": len(lines),
        "words": len(words),
        "chars": chars,
    }

    return result


def print_result(result: dict) -> None:
    """
    格式化输出统计结果。

    参数：
        result: analyze_file 返回的统计结果
    """
    print("=" * 40)
    print("文本文件统计结果")
    print("=" * 40)
    print(f"文件路径: {result['file_path']}")
    print(f"行数: {result['lines']}")
    print(f"单词数: {result['words']}")
    print(f"字符数: {result['chars']}")
    print("=" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(description="文本文件统计工具")

    parser.add_argument(
        "--file",
        type=str,
        default="week1_python_review/scripts/sample.txt",
        help="要分析的文本文件路径",
    )

    parser.add_argument(
        "--create_sample",
        action="store_true",
        help="是否创建示例 sample.txt 文件",
    )

    args = parser.parse_args()

    file_path = Path(args.file)

    if args.create_sample:
        create_sample_file(file_path)
        print(f"示例文件已创建: {file_path}")

    if not file_path.exists():
        print(f"文件不存在，正在自动创建示例文件: {file_path}")
        create_sample_file(file_path)

    try:
        result = analyze_file(file_path)
        print_result(result)
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")


if __name__ == "__main__":
    main()