"""打印模型逐层输出形状与参数量，不依赖 torchinfo。"""

from __future__ import annotations

import argparse  # 用于读取模型或实验名称。
import csv  # 用于保存逐层摘要。
from pathlib import Path  # 用于输出路径。
from typing import Any  # 用于 Hook 返回值类型。

import torch  # 用于虚拟输入和前向 Hook。
from torch import nn  # 用于模块类型标注。

from config import EXPERIMENT_PRESETS, get_experiment_preset  # 导入实验配置。
from models import create_model  # 导入模型工厂。
from utils import count_parameters  # 导入参数统计函数。


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Print layer shapes and parameter counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", choices=list(EXPERIMENT_PRESETS.keys()), default="exp1_basic_cnn")
    parser.add_argument("--batch-size", type=int, default=4, help="虚拟输入批大小。")
    parser.add_argument("--output", type=Path, default=None, help="可选 CSV 输出路径。")
    return parser.parse_args()


def summarize_model(model: nn.Module, input_tensor: torch.Tensor) -> list[dict[str, Any]]:
    """使用 forward hook 收集叶子模块的输入输出形状。"""

    rows: list[dict[str, Any]] = []  # 保存每层信息。
    hooks: list[torch.utils.hooks.RemovableHandle] = []  # 保存 Hook 句柄，便于最后移除。

    def register_hook(name: str, module: nn.Module) -> None:
        """给单个叶子模块注册前向 Hook。"""

        def hook_function(
            hooked_module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            input_shape = tuple(inputs[0].shape) if inputs and isinstance(inputs[0], torch.Tensor) else "-"
            if isinstance(output, torch.Tensor):
                output_shape: object = tuple(output.shape)
            else:
                output_shape = str(type(output).__name__)
            parameters = sum(parameter.numel() for parameter in hooked_module.parameters(recurse=False))
            trainable = sum(
                parameter.numel()
                for parameter in hooked_module.parameters(recurse=False)
                if parameter.requires_grad
            )
            rows.append(
                {
                    "layer": name,
                    "type": hooked_module.__class__.__name__,
                    "input_shape": str(input_shape),
                    "output_shape": str(output_shape),
                    "parameters": parameters,
                    "trainable_parameters": trainable,
                }
            )

        hooks.append(module.register_forward_hook(hook_function))  # 保存 Hook 句柄。

    for name, module in model.named_modules():  # 遍历所有模块。
        if name and not list(module.children()):  # 只给不含子模块的叶子层注册 Hook。
            register_hook(name, module)

    model.eval()  # 保证 BatchNorm 和 Dropout 在摘要时使用评估行为。
    with torch.no_grad():  # 不需要构建梯度图。
        model(input_tensor)  # 执行一次前向传播触发所有 Hook。

    for hook in hooks:  # 用完后移除 Hook，避免重复注册。
        hook.remove()

    return rows  # 返回逐层结果。


def main() -> None:
    """创建模型并输出摘要。"""

    args = parse_args()  # 读取参数。
    preset = get_experiment_preset(args.experiment)  # 读取实验配置。
    model = create_model(
        model_name=preset.model_name,
        num_classes=10,
        use_batch_norm=preset.use_batch_norm,
        dropout=preset.dropout,
    )
    dummy_input = torch.randn(args.batch_size, 3, 32, 32)  # 构造 CIFAR-10 形状的虚拟输入。
    rows = summarize_model(model, dummy_input)  # 收集逐层信息。
    totals = count_parameters(model)  # 统计模型总参数量。

    print(f"Model summary for: {args.experiment}")
    print("-" * 120)
    print(f"{'Layer':38} {'Type':20} {'Input':20} {'Output':20} {'Params':>12}")
    print("-" * 120)

    for row in rows:  # 逐行显示摘要。
        print(
            f"{str(row['layer']):38.38} "
            f"{str(row['type']):20.20} "
            f"{str(row['input_shape']):20.20} "
            f"{str(row['output_shape']):20.20} "
            f"{int(row['parameters']):12,}"
        )

    print("-" * 120)
    print(f"Total parameters:     {totals['total_parameters']:,}")
    print(f"Trainable parameters: {totals['trainable_parameters']:,}")

    if args.output is not None:  # 用户指定输出路径时保存 CSV。
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV summary to: {args.output}")


if __name__ == "__main__":
    main()  # 直接运行文件时打印摘要。
