"""Language Model 的训练样本就是长度 T 的连续 token，以及右移一位的标签。"""

from __future__ import annotations

import torch


def sample_batch(tokens: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    if tokens.numel() <= block_size:
        raise ValueError("token 数必须大于 block_size")
    starts = torch.randint(0, tokens.numel() - block_size - 1, (batch_size,))
    x = torch.stack([tokens[i:i + block_size] for i in starts.tolist()])
    y = torch.stack([tokens[i + 1:i + block_size + 1] for i in starts.tolist()])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
