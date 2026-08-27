"""把“迁移学习策略”变成可检查的 requires_grad，而不是停留在概念名称。"""

from __future__ import annotations

import torch
from torch import nn


def _classifier_modules(model):
    result = []
    for name in ["classifier", "score"]:
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            result.append(module)
    if not result:
        raise AttributeError("找不到 classifier/score；默认项目面向 BERT 风格 SequenceClassification 模型")
    return result


def _encoder_layers(base_model):
    # BERT / RoBERTa 风格
    encoder = getattr(base_model, "encoder", None)
    if encoder is not None and hasattr(encoder, "layer"):
        return encoder.layer
    # DistilBERT 风格
    transformer = getattr(base_model, "transformer", None)
    if transformer is not None and hasattr(transformer, "layer"):
        return transformer.layer
    raise AttributeError("无法定位 encoder layers；默认推荐 bert-base-uncased")


def _set_module(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def apply_strategy(model: nn.Module, strategy: str, unfreeze_last_n: int = 3) -> None:
    if strategy not in {"head_only", "last_n", "full"}:
        raise ValueError("strategy 必须是 head_only / last_n / full")

    if strategy == "full":
        _set_module(model, True)
        return

    # 先全部冻结，再按策略精确打开。
    _set_module(model, False)
    for module in _classifier_modules(model):
        _set_module(module, True)

    if strategy == "head_only":
        return

    base = model.base_model
    layers = _encoder_layers(base)
    n = min(max(int(unfreeze_last_n), 0), len(layers))
    if n > 0:
        for layer in list(layers)[-n:]:
            _set_module(layer, True)

    # BERT 的 pooled CLS 还经过 pooler，部分微调时一起适配更合理。
    pooler = getattr(base, "pooler", None)
    if isinstance(pooler, nn.Module):
        _set_module(pooler, True)


def parameter_counts(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": trainable / max(total, 1),
    }
