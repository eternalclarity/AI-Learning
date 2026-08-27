"""Greedy decoding；Encoder 只计算一次。"""

from __future__ import annotations

import torch

from data import BOS_ID, EOS_ID, PAD_ID, Vocab, tokenize
from masks import make_valid_mask


@torch.no_grad()
def greedy_decode(
    model,
    sentence: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_length: int = 40,
):
    model.eval()
    src_tokens = tokenize(sentence)[: max_length - 1]
    src_ids = src_vocab.encode(src_tokens) + [EOS_ID]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_valid = make_valid_mask(src, PAD_ID)

    memory = model.encode(src, src_valid)
    generated = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)

    for _ in range(max_length - 1):
        tgt_valid = make_valid_mask(generated, PAD_ID)
        logits = model.decode(generated, memory, tgt_valid, src_valid)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        if int(next_id.item()) == EOS_ID:
            break

    ids = generated[0].tolist()[1:]  # 去掉 BOS
    if EOS_ID in ids:
        ids = ids[: ids.index(EOS_ID)]
    tokens = tgt_vocab.decode(ids, skip_specials=True)
    return tokens
