"""不依赖 sklearn/sacrebleu 的教学版 BLEU。"""

from __future__ import annotations

import math
from collections import Counter


def _ngrams(tokens: list[str], n: int):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu(prediction: list[str], reference: list[str], max_n: int = 4, smooth: float = 1e-9) -> float:
    """单参考 BLEU；适合教学/项目对比，不声称替代标准评测工具。"""
    if not prediction:
        return 0.0

    pred_len = len(prediction)
    ref_len = len(reference)
    bp = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / max(pred_len, 1))

    log_precisions = []
    usable_orders = 0
    for n in range(1, max_n + 1):
        pred_ng = _ngrams(prediction, n)
        if not pred_ng:
            continue
        ref_ng = _ngrams(reference, n)
        clipped = sum(min(count, ref_ng[gram]) for gram, count in pred_ng.items())
        total = sum(pred_ng.values())
        precision = (clipped + smooth) / (total + smooth)
        log_precisions.append(math.log(precision))
        usable_orders += 1

    if usable_orders == 0:
        return 0.0
    return bp * math.exp(sum(log_precisions) / usable_orders)
