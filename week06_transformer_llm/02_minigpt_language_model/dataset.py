"""
把一整条连续的 token 序列，随机切成 GPT 训练需要的 (x, y) 小批量数据
随机取起点，x 取长度为 block_size 的连续 token，y 则整体右移 1 位
"""

from __future__ import annotations

import torch


def sample_batch(tokens: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    """从连续 token 中随机采样一个训练 batch。"""

    # token 数必须比 block_size 多，否则无法构造输入和右移一位的标签
    if tokens.numel() <= block_size:
        raise ValueError("token 数必须大于 block_size")

    # 随机生成 batch_size 个序列起点
    starts = torch.randint(0, tokens.numel() - block_size - 1, (batch_size,))

    # 从每个起点截取 block_size 个连续 token，组成输入 x
    x = torch.stack([tokens[i:i + block_size] for i in starts.tolist()])
    y = torch.stack([tokens[i + 1:i + block_size + 1] for i in starts.tolist()])

    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
