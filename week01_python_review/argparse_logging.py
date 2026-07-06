"""
argparse_logging.py

练习 argparse 和 logging。

自定义参数运行： python week1_python_review/argparse_logging.py --lr 0.01 --epochs 5 --batch_size 16
"""

import argparse
import logging
import random
from pathlib import Path


def setup_logger(log_file: str = "week1_python_review/scripts/train.log") -> None:
    """
    配置日志系统。
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def set_seed(seed: int) -> None:
    """
    设置随机种子。
    """
    random.seed(seed)


def simulate_training(epochs: int, lr: float, batch_size: int) -> None:
    """
    模拟一个训练过程。
    参数：
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批大小
    """
    logging.info("开始模拟训练")
    logging.info(f"学习率 lr = {lr}")
    logging.info(f"训练轮数 epochs = {epochs}")
    logging.info(f"批大小 batch_size = {batch_size}")

    # 损失值
    loss = 1.0

    for epoch in range(1, epochs + 1):
        # 模拟 loss 逐渐下降
        # 随机噪声 (-0.03, 0.03)
        noise = random.uniform(-0.03, 0.03)
        loss = loss * 0.85 + noise

        if loss < 0:
            loss = 0.01

        logging.info(f"Epoch [{epoch}/{epochs}] - loss: {loss:.4f}")

    logging.info("模拟训练结束")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="argparse 和 logging 练习脚本")

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="学习率，默认值为 0.001",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数，默认值为 10",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="批大小，默认值为 32",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认值为 42",
    )

    parser.add_argument(
        "--log_file",
        type=str,
        default="week1_python_review/scripts/train.log",
        help="日志文件保存路径",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    setup_logger(args.log_file)
    set_seed(args.seed)

    logging.info("程序参数如下：")
    logging.info(args)

    simulate_training(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()