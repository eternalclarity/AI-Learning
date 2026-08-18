"""无需下载 CIFAR-10 的快速环境与模型冒烟测试。"""

from __future__ import annotations

import torch  # 用于随机输入、损失和反向传播。
from torch import nn  # 用于交叉熵损失。

from config import EXPERIMENT_PRESETS  # 导入全部实验预设。
from models import create_model  # 导入模型工厂。


def main() -> None:
    """检查每个模型能否完成一次前向传播和反向传播。"""

    images = torch.randn(4, 3, 32, 32)  # 创建四张虚拟 CIFAR-10 图片。
    labels = torch.randint(low=0, high=10, size=(4,))  # 创建四个随机类别标签。
    criterion = nn.CrossEntropyLoss()  # 创建多分类损失函数。

    checked_models: set[tuple[str, bool, float]] = set()  # 避免相同模型配置重复测试。

    for preset in EXPERIMENT_PRESETS.values():  # 遍历全部实验预设。
        signature = (preset.model_name, preset.use_batch_norm, preset.dropout)  # 生成模型配置签名。
        if signature in checked_models:  # 已经测试过完全相同配置时跳过。
            continue
        checked_models.add(signature)  # 记录当前配置。

        model = create_model(
            model_name=preset.model_name,
            num_classes=10,
            use_batch_norm=preset.use_batch_norm,
            dropout=preset.dropout,
        )
        logits = model(images)  # 执行前向传播。
        loss = criterion(logits, labels)  # 计算损失。
        loss.backward()  # 检查自动求导和参数梯度是否正常。

        assert logits.shape == (4, 10), f"输出形状错误：{preset.name} -> {logits.shape}"
        assert torch.isfinite(loss), f"损失出现 NaN/Inf：{preset.name}"
        print(f"PASS | {preset.name:28} | output={tuple(logits.shape)} | loss={loss.item():.4f}")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()  # 直接运行文件时执行冒烟测试。
