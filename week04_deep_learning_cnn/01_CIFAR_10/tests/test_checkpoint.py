"""检查检查点能否完整保存并重新加载。"""

from __future__ import annotations

import tempfile  # 用于创建自动清理的临时目录。
import unittest  # 使用标准测试框架。
from pathlib import Path  # 用于临时文件路径。

import torch  # 用于模型和优化器。

from models import create_model  # 导入模型工厂。
from utils import load_checkpoint, save_checkpoint  # 导入检查点函数。


class CheckpointTests(unittest.TestCase):
    """测试检查点包含恢复训练所需的关键字段。"""

    def test_save_and_load_checkpoint(self) -> None:
        model = create_model("basic_cnn", use_batch_norm=True, dropout=0.2)  # 创建测试模型。
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 创建测试优化器。
        history = [
            {
                "epoch": 1,
                "train_loss": 2.0,
                "train_accuracy": 0.2,
                "val_loss": 1.9,
                "val_accuracy": 0.25,
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:  # 退出代码块后自动删除临时目录。
            checkpoint_path = Path(temporary_directory) / "model.pth"  # 构造临时检查点路径。
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=1,
                best_val_accuracy=0.25,
                history=history,
                experiment_config={"name": "unit_test"},
                split_config={"val_ratio": 0.1, "seed": 42},
            )

            checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))  # 重新读取检查点。

        self.assertEqual(checkpoint["epoch"], 1)  # 检查 epoch。
        self.assertAlmostEqual(checkpoint["best_val_accuracy"], 0.25)  # 检查最佳指标。
        self.assertEqual(checkpoint["experiment_config"]["name"], "unit_test")  # 检查实验配置。
        self.assertIn("model_state_dict", checkpoint)  # 检查模型参数字典。
        self.assertIn("optimizer_state_dict", checkpoint)  # 检查优化器状态。


if __name__ == "__main__":
    unittest.main()  # 允许直接运行当前测试文件。
