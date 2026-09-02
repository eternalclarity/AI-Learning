"""
MiniGPT：Pre-Norm Decoder-only Transformer
定义了 GPT 配置、FFN、Transformer Block，以及 Decoder-only GPT
input_ids -> Token Embedding  -> +  Position Embedding ->   X    -> Transformer Block × N ->   Y    -> Final LayerNorm -> LM HeadLinear(C → V) -> logits
(B,T)        (B,T,C)                                     (B,T,C)                             (B,T,C)                                              (B,T,V)
"""

from __future__ import annotations

# 用于快速定义只保存配置数据的类
from dataclasses import dataclass

import torch
# 导入 PyTorch 常用函数接口
import torch.nn.functional as F
from torch import nn

from attention import CausalSelfAttention


@dataclass  # 自动为配置类生成 __init__、__repr__ 等方法
class GPTConfig:
    vocab_size: int                 # 词表大小
    block_size: int = 256           # 最大上下文长度, 一次生成当前 q最多和多少个k 做attention
    d_model: int = 256              # token 隐藏向量维度
    num_heads: int = 8              # 多头注意力的 head 数
    num_layers: int = 6             # Transformer Block 数量
    d_ff: int = 1024                # FFN 中间隐藏层维度
    dropout: float = 0.1            # Dropout 概率
    attention_impl: str = "sdpa"    # 注意力实现方式


class FeedForward(nn.Module):
    """ Transformer 中的前馈神经网络 FFN """

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
    """ 单个 Transformer Decoder Block """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)  # Self-Attention 前的 LayerNorm
        self.attn = CausalSelfAttention(         # 创建带 因果掩码 的多头 Self-Attention
            config.d_model,
            config.num_heads,
            config.dropout,
            config.attention_impl,
        )
        self.ln2 = nn.LayerNorm(config.d_model)   # FFN 前的 LayerNorm
        self.ffn = FeedForward(config.d_model, config.d_ff, config.dropout)  # 创建前馈神经网络

    def forward(self, x, past_kv=None, use_cache: bool = False):
        # Pre-Norm → Self-Attention → 残差连接，同时处理 KV Cache
        attn_out, new_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out

        # Pre-Norm → FFN → 残差连接
        x = x + self.ffn(self.ln2(x))

        # 返回当前层输出和当前层新的 KV Cache (K,V)
        return x, new_kv


class GPT(nn.Module):
    """ Decoder-only GPT 模型 """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)      # 将 token ID 映射成 d_model 维向量，一个embedding矩阵，通过查表映射对应的token向量
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)   # 可学习位置嵌入, 将位置 ID 映射成 d_model 维位置向量, 本质也是一个embedding 查表映射，通过给序列长度block_size中每个token的位置编号来查找对应的位置向量，每个位置向量是d_model维度，代表对 对应token的d_model维度都进行编码

        self.dropout = nn.Dropout(config.dropout)

        # 创建 num_layers 个 Transformer Block
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])   # 列表推导式，重复创建 num_layers 个 Block，但不需要循环序号

        # 所有 Transformer Block 后的最终 LayerNorm, 并将 hidden state 映射到整个词表的 logits
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 归遍历模型里的所有子模块, 对所有子模块应用自定义参数初始化
        self.apply(self._init_weights)

        # Weight tying：输入 token embedding 与输出 vocabulary projection 共享权重
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod    # 静态方法，不依赖具体 GPT 对象, 表示这个函数不需要访问 self 或 cls，只是一个放在类里的工具函数
    def _init_weights(module):
        if isinstance(module, nn.Linear):  # 如果当前模块是全连接层
            nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 使用均值 0、标准差 0.02 的正态分布初始化权重
            if module.bias is not None:  # 如果存在 bias，则初始化为 0
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):  # 如果当前模块是 Embedding 层, Linear 还有 bias，而 Embedding 没有 bias
            nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 同样使用正态分布初始化 Embedding 权重

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None, past_key_values=None, use_cache: bool = False):
        # 获取 batch_size 和当前序列长度
        b, t = input_ids.shape

        if past_key_values is None:  # 如果没有历史 KV Cache
            past_len = 0    # 历史序列长度为 0
            past_key_values = [None] * len(self.blocks)  # 为每个 Transformer Block 准备一个空 cache
        else:
            past_len = past_key_values[0][0].size(-2)   # 从第一层缓存中获取历史 token 数量, [(K,V)...] -> K [B,H,T,D] -> T

        # 历史序列 + 当前序列不能超过 最大上下文长度 , 模型允许参与 Attention 的最大 token 序列长度, 每崩一个token就需要和context window 里的所有 k,v 做一次 causal self-atttention
        if past_len + t > self.config.block_size:
            raise ValueError("past_len + current_len 超过 block_size；生成代码应重建滑动窗口 cache")

        # 生成当前 token 对应上下文的位置 ID，因为是当前q和上下文所有k做attention
        positions = torch.arange(past_len, past_len + t, device=input_ids.device)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)

        new_cache = [] if use_cache else None  # 如果启用 cache，则准备保存每一层的新 KV (K,V)

        for block, layer_past in zip(self.blocks, past_key_values):  # past_key_values:[(K,V)...], layer_cache: (K,V)
            x, layer_cache = block(x, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                new_cache.append(layer_cache)

        # hidden state 映射到 vocab_size，得到 logits
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B,T,C) -> (B,T,V)

        loss = None  # 默认不计算 loss
        if targets is not None:  # 如果提供了训练标签
            # logits (B,T,V) -> (B*T, V);  targets (B,T) -> (B*T)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss, new_cache

    def num_parameters(self) -> int:
        """统计模型实际拥有的参数数量 """

        seen = set()  # 保存已经统计过的 Parameter 对象 ID
        total = 0     # 参数总数

        # 遍历模型所有参数
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))    # 记录该参数，避免 Weight Tying 重复统计
                total += p.numel()

        return total
