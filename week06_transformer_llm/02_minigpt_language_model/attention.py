"""
因果自注意力实现：
完成多头 Causal Self-Attention，同时支持手写版、PyTorch SDPA 高效版和 KV Cache
输入 x  ->  Linear(C → 3C) ->  Q、K、V  -> 拆成多头  ->  Kᵀ / √D  -> Causal Mask -> softmax -> Attention Weights -> Weights @ V  -> 合并多头  -> out_proj -> Attention 输出 y / new_cache_k,v
(B,T,C)    (B,T,3C)           (B,T,C)    (B,H,T,D)                                                               (B,H,T,D)       (B,T,C)                       (B,T,C)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    """ 因果多头自注意力 """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, impl: str = "sdpa") -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能整除 num_heads")
        if impl not in {"manual", "sdpa"}:
            raise ValueError("impl 必须是 manual 或 sdpa")

        self.d_model = d_model  # 模型隐藏维度
        self.num_heads = num_heads  # Attention Head 数量
        self.head_dim = d_model // num_heads  # 每个 Head 的维度

        self.dropout_p = dropout
        self.impl = impl

        self.qkv = nn.Linear(d_model, 3 * d_model)  # X 一次线性变换同时生成 Q、K、V,   x(256) -> (768)-Q(256),K(256),V(256)
        self.out_proj = nn.Linear(d_model, d_model)  # 多头 Attention 合并后的输出投影

        self.resid_dropout = nn.Dropout(dropout)  # Attention 输出后的 Dropout
        self.last_weights: torch.Tensor | None = None  # 保存最近一次手写 Attention 的权重，便于观察

    def _split_qkv(self, x: torch.Tensor):
        """生成 Q、K、V，并拆分成多个 Attention Head"""

        b, t, _ = x.shape  # 获取 batch_size、序列长度

        qkv = self.qkv(x).view(b, t, 3, self.num_heads, self.head_dim)  # 一次线性映射生成 Q、K、V，再拆成多个 Head
        q, k, v = qkv.unbind(dim=2)     # 沿 QKV 维拆成独立的 q、k、v

        # [B,T,H,D] -> [B,H,T,D]
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    @staticmethod
    def _allowed_mask(query_len: int, key_len: int, past_len: int, device) -> torch.Tensor:
        """
        构造 Causal Mask：当前 Query 只能看到自己及过去的 Key

        :param query_len: 当前要计算的 Query 数量
        :param key_len: 当前一共有多少个 Key
        :param past_len: KV Cache 里已经有多少个历史 token
        :param device: Tensor 放在 CPU 还是 GPU
        :return: Causal Mask
        """

        q_positions = past_len + torch.arange(query_len, device=device)  # 当前 q 在本次 attention中 上下文中的位置
        k_positions = torch.arange(key_len, device=device)    # 本次 attention 中所有 k 在上下文中的位置

        # 生成 Causal Mask 矩阵，只有 key_position <= query_position 时才允许关注
        # k_positions.unsqueeze(0) -> k 为横轴(1, T)； q_positions.unsqueeze(1) -> q 为纵轴 (T, 1)
        return k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):

        q, k_new, v_new = self._split_qkv(x)  # 由当前输入生成新的 Q、K、V

        past_len = 0  # 默认没有历史 KV
        if past_kv is not None:     # 如果存在历史 KV Cache
            past_k, past_v = past_kv    # 取出历史 K、V, past_kv 是一个tuple(K,V), 记录每一层attention之前token的KV值,其中 K/V.shape=[B,H,T,D]，其中K,V可以在生成过程中不断拼接concat
            past_len = past_k.size(-2)  # 获取历史 token 数量，即[B,H,T,D] -> T
            k = torch.cat([past_k, k_new], dim=-2)  # 把当前token生成的新K接上去 [B,H,T+1,D],表示这一层attention的k又加入了新的
            v = torch.cat([past_v, v_new], dim=-2)  # 把当前token生成的新V接上去 [B,H,T+1,D],表示这一层attention的v又加入了新的
        else:
            k, v = k_new, v_new  # 没有 Cache 时直接使用当前 K、V

        q_len = q.size(-2)  # 当前 Query 数量
        key_len = k.size(-2)    # 当前可见的全部 Key 数量
        dropout_p = self.dropout_p if self.training else 0.0  # 训练时使用 Dropout，推理时关闭

        # 手写 Attention
        if self.impl == "manual":
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # 计算缩放点积注意力分数 QK^T / sqrt(d)

            allowed = self._allowed_mask(q_len, key_len, past_len, x.device)  # 生成因果掩码
            scores = scores.masked_fill(~allowed[None, None, :, :], torch.finfo(scores.dtype).min)  # 将未来位置的分数设为极小值

            weights = torch.softmax(scores, dim=-1)  # Softmax 得到 Attention 权重
            weights = F.dropout(weights, p=dropout_p, training=self.training)  # 对 Attention 权重使用 Dropout

            y = weights @ v  # Attention 权重与 V 加权求和
            self.last_weights = weights.detach()  # 保存 注意力权重 用于观察，不参与梯度计算

        # PyTorch SDPA
        else:
            # 无 Cache 时直接使用高效 Causal Attention, 适用训练阶段，无需 KV cache
            if past_len == 0:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=dropout_p,
                    is_causal=True,  # 高效，无需多言
                )
            else:
                # 有 Cache 时显式生成可见性 Mask, 适用生成阶段，可以用 KV cache 加速 自回归生成
                allowed = self._allowed_mask(q_len, key_len, past_len, x.device)  # 这里不能合并，有 KV Cache 时，q_len（2）和 key_len （10）往往不相等，而 PyTorch 的 is_causal=True 对这种 非方阵 attention 采用的是左上对齐的 causal mask，不符合 KV Cache 解码时我们需要的对齐
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=allowed,  # 伟大，无需多言
                    dropout_p=dropout_p,
                    is_causal=False,    # 只针对方阵，也就是无cache，所有k,v都要算的情况
                )
            self.last_weights = None  # SDPA 不直接返回 Attention 权重

        # [B,H,T,D] → [B,T,H,D] → [B,T,d_model], 多头注意力各结果拼接
        y = y.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        y = self.resid_dropout(self.out_proj(y))  # 多头结果进行输出投影，再使用 Dropout

        new_cache = (k, v) if use_cache else None  # 如果启用 Cache，则保存当前层完整 K、V元组，其中 K/V.shape=[B,H,T,D]，其中K,V可以在生成过程中不断拼接concat
        return y, new_cache
