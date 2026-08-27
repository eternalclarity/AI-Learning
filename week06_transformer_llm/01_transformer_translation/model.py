"""从零实现 Encoder-Decoder Transformer。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from masks import make_causal_mask


class MultiHeadAttention(nn.Module):
    """手写 Multi-Head Attention；True mask 表示允许关注。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能整除 num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_weights: torch.Tensor | None = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,C] -> [B,T,H,D] -> [B,H,T,D]
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_valid_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ):
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # [B,H,Q,K]

        if key_valid_mask is not None:
            # [B,K] -> [B,1,1,K]
            scores = scores.masked_fill(~key_valid_mask[:, None, None, :], torch.finfo(scores.dtype).min)

        if attn_mask is not None:
            # 支持 [Q,K] 或可广播到 [B,H,Q,K] 的 bool mask。
            if attn_mask.dtype != torch.bool:
                raise TypeError("attn_mask 必须为 bool，且 True=允许关注")
            while attn_mask.dim() < scores.dim():
                attn_mask = attn_mask.unsqueeze(0)
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = weights @ v  # [B,H,Q,D]
        context = context.transpose(1, 2).contiguous().view(query.size(0), query.size(1), self.d_model)
        output = self.out_proj(context)

        self.last_weights = weights.detach()
        return (output, weights) if need_weights else output


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError("序列长度超过 positional encoding 的 max_len")
        return self.dropout(x + self.pe[:, : x.size(1)].to(dtype=x.dtype))


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AddNorm(nn.Module):
    """经典 Post-Norm：LayerNorm(x + Dropout(sublayer_output))。"""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(y))


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(self, x: torch.Tensor, src_valid_mask: torch.Tensor) -> torch.Tensor:
        y = self.self_attn(x, x, x, key_valid_mask=src_valid_mask)
        x = self.addnorm1(x, y)
        return self.addnorm2(x, self.ffn(x))


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.addnorm3 = AddNorm(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_valid_mask: torch.Tensor,
        src_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        causal = make_causal_mask(x.size(1), device=x.device)
        y = self.self_attn(
            x, x, x,
            key_valid_mask=tgt_valid_mask,
            attn_mask=causal,
        )
        x = self.addnorm1(x, y)

        # Cross-Attention：Query 来自 Decoder；Key/Value 来自 Encoder memory。
        y = self.cross_attn(
            x, memory, memory,
            key_valid_mask=src_valid_mask,
        )
        x = self.addnorm2(x, y)
        return self.addnorm3(x, self.ffn(x))


@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    max_len: int = 128
    pad_id: int = 0


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.src_embedding = nn.Embedding(config.src_vocab_size, config.d_model, padding_idx=config.pad_id)
        self.tgt_embedding = nn.Embedding(config.tgt_vocab_size, config.d_model, padding_idx=config.pad_id)
        self.position = SinusoidalPositionalEncoding(config.d_model, config.max_len, config.dropout)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(config.d_model, config.num_heads, config.d_ff, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(config.d_model, config.num_heads, config.d_ff, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.output = nn.Linear(config.d_model, config.tgt_vocab_size)

    def encode(self, src: torch.Tensor, src_valid_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_embedding(src) * math.sqrt(self.config.d_model)
        x = self.position(x)
        for block in self.encoder_blocks:
            x = block(x, src_valid_mask)
        return x

    def decode(
        self,
        tgt_input: torch.Tensor,
        memory: torch.Tensor,
        tgt_valid_mask: torch.Tensor,
        src_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_embedding(tgt_input) * math.sqrt(self.config.d_model)
        x = self.position(x)
        for block in self.decoder_blocks:
            x = block(x, memory, tgt_valid_mask, src_valid_mask)
        return self.output(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        src_valid_mask: torch.Tensor,
        tgt_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_valid_mask)
        return self.decode(tgt_input, memory, tgt_valid_mask, src_valid_mask)

    @property
    def last_cross_attention(self) -> torch.Tensor | None:
        return self.decoder_blocks[-1].cross_attn.last_weights if self.decoder_blocks else None
