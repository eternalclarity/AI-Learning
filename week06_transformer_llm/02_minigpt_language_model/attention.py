"""同一 Causal Self-Attention 提供 manual 与 PyTorch SDPA 两个后端，并支持 KV Cache。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, impl: str = "sdpa") -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能整除 num_heads")
        if impl not in {"manual", "sdpa"}:
            raise ValueError("impl 必须是 manual 或 sdpa")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout
        self.impl = impl
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)
        self.last_weights: torch.Tensor | None = None

    def _split_qkv(self, x: torch.Tensor):
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # [B,T,H,D] -> [B,H,T,D]
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    @staticmethod
    def _allowed_mask(query_len: int, key_len: int, past_len: int, device) -> torch.Tensor:
        """[Q,K]；Query i 可看 Key <= past_len+i。"""
        q_positions = past_len + torch.arange(query_len, device=device)
        k_positions = torch.arange(key_len, device=device)
        return k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        q, k_new, v_new = self._split_qkv(x)
        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(-2)
            k = torch.cat([past_k, k_new], dim=-2)
            v = torch.cat([past_v, v_new], dim=-2)
        else:
            k, v = k_new, v_new

        q_len = q.size(-2)
        key_len = k.size(-2)
        dropout_p = self.dropout_p if self.training else 0.0

        if self.impl == "manual":
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            allowed = self._allowed_mask(q_len, key_len, past_len, x.device)
            scores = scores.masked_fill(~allowed[None, None, :, :], torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            weights = F.dropout(weights, p=dropout_p, training=self.training)
            y = weights @ v
            self.last_weights = weights.detach()
        else:
            # 无 cache 时直接走高效 causal 路径；有 cache 时显式给 [Q,K] 可见性。
            if past_len == 0:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=dropout_p,
                    is_causal=True,
                )
            else:
                allowed = self._allowed_mask(q_len, key_len, past_len, x.device)
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=allowed,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
            self.last_weights = None

        y = y.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        y = self.resid_dropout(self.out_proj(y))
        new_cache = (k, v) if use_cache else None
        return y, new_cache
