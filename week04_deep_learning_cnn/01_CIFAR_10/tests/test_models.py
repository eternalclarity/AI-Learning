"""使用 Python 标准库 unittest 检查模型输出形状和残差块形状。"""

from __future__ import annotations

import unittest  # 使用标准库测试框架，不额外依赖 pytest。

import torch  # 用于创建虚拟输入。

from models import AVAILABLE_MODELS, create_model  # 导入全部模型名称和模型工厂。
from models.residual_block import BasicResidualBlock  # 单独测试残差块。


class ModelShapeTests(unittest.TestCase):
    """检查所有分类模型都输出 [batch_size, 10]。"""

    def test_all_models_output_ten_logits(self) -> None:
        images = torch.randn(2, 3, 32, 32)  # 创建两张虚拟彩色图片。

        for model_name in AVAILABLE_MODELS:  # 逐个检查注册模型。
            with self.subTest(model=model_name):  # 某一模型失败时仍能明确显示模型名。
                model = create_model(
                    model_name=model_name,
                    num_classes=10,
                    use_batch_norm=True,
                    dropout=0.2,
                )
                model.eval()  # 使用稳定的评估模式。
                with torch.no_grad():  # 形状测试不需要梯度。
                    logits = model(images)
                self.assertEqual(tuple(logits.shape), (2, 10))  # 检查输出形状。

    def test_residual_block_identity_shape(self) -> None:
        block = BasicResidualBlock(32, 32, stride=1)  # 输入输出形状相同，不需要投影捷径。
        x = torch.randn(2, 32, 16, 16)  # 创建虚拟特征图。
        y = block(x)  # 执行前向传播。
        self.assertEqual(tuple(y.shape), (2, 32, 16, 16))  # 形状应保持不变。

    def test_residual_block_projection_shape(self) -> None:
        block = BasicResidualBlock(32, 64, stride=2)  # 通道翻倍且空间尺寸减半。
        x = torch.randn(2, 32, 16, 16)  # 创建虚拟特征图。
        y = block(x)  # 执行前向传播。
        self.assertEqual(tuple(y.shape), (2, 64, 8, 8))  # 检查投影捷径后的输出形状。


if __name__ == "__main__":
    unittest.main()  # 允许直接运行当前测试文件。
