"""MiniGPT：Pre-Norm Decoder-only Transformer。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from attention import CausalSelfAttention


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    attention_impl: str = "sdpa"


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(
            config.d_model,
            config.num_heads,
            config.dropout,
            config.attention_impl,
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(self, x, past_kv=None, use_cache: bool = False):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # Weight tying：输入 token embedding 与输出 vocabulary projection 共享权重。
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None, past_key_values=None, use_cache: bool = False):
        b, t = input_ids.shape
        if past_key_values is None:
            past_len = 0
            past_key_values = [None] * len(self.blocks)
        else:
            past_len = past_key_values[0][0].size(-2)

        if past_len + t > self.config.block_size:
            raise ValueError("past_len + current_len 超过 block_size；生成代码应重建滑动窗口 cache")

        positions = torch.arange(past_len, past_len + t, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)

        new_cache = [] if use_cache else None
        for block, layer_past in zip(self.blocks, past_key_values):
            x, layer_cache = block(x, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                new_cache.append(layer_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss, new_cache

    def num_parameters(self) -> int:
        # Weight tying 后 state_dict 有共享 storage；numel 按 Parameter identity 去重更准确。
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
