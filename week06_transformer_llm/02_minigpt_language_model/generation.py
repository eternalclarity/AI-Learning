"""GPT 采样与 KV Cache 推理。"""

from __future__ import annotations

import torch


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None, greedy: bool = False):
    if greedy or temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        threshold = values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_naive(model, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None, greedy: bool = False):
    model.eval()
    ids = input_ids
    for _ in range(max_new_tokens):
        context = ids[:, -model.config.block_size:]
        logits, _, _ = model(context)
        next_id = sample_next(logits[:, -1, :], temperature, top_k, greedy)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


@torch.no_grad()
def generate_cached(model, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None, greedy: bool = False):
    """当 cache 达到 block_size 时重建最后一个滑动窗口，保证 learned position index 不越界。"""
    model.eval()
    ids = input_ids
    cache = None

    for _ in range(max_new_tokens):
        if cache is None:
            context = ids[:, -model.config.block_size:]
            logits, _, cache = model(context, use_cache=True)
        else:
            cache_len = cache[0][0].size(-2)
            if cache_len >= model.config.block_size:
                cache = None
                context = ids[:, -model.config.block_size:]
                logits, _, cache = model(context, use_cache=True)
            else:
                # Cache 已包含此前上下文，只输入最新生成 token。
                logits, _, cache = model(ids[:, -1:], past_key_values=cache, use_cache=True)

        next_id = sample_next(logits[:, -1, :], temperature, top_k, greedy)
        ids = torch.cat([ids, next_id], dim=1)

    return ids
