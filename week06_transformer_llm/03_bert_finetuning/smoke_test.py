"""使用随机初始化的 tiny BERT 做本地 smoke test，不下载预训练权重。"""

from __future__ import annotations

try:
    from transformers import BertConfig, BertForSequenceClassification
except ImportError as exc:
    raise SystemExit("请先 pip install -r requirements.txt") from exc

import torch

from strategies import apply_strategy, parameter_counts


def main() -> None:
    config = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        num_labels=2,
    )
    for strategy in ["head_only", "last_n", "full"]:
        model = BertForSequenceClassification(config)
        apply_strategy(model, strategy, unfreeze_last_n=1)
        counts = parameter_counts(model)
        input_ids = torch.randint(0, 100, (2, 12))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([0, 1])
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        print(strategy, tuple(outputs.logits.shape), counts)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
