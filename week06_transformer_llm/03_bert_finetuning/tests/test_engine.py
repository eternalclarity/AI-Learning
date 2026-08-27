import unittest
from types import SimpleNamespace

import torch
from torch import nn

from engine import evaluate_epoch, train_epoch


class DummyClassifier(nn.Module):
    """模拟 Hugging Face SequenceClassifierOutput 接口，不需要 transformers。"""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 8)
        self.classifier = nn.Linear(8, 2)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(1) / mask.sum(1).clamp_min(1)
        else:
            x = x.mean(1)
        logits = self.classifier(x)
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return SimpleNamespace(logits=logits, loss=loss)


class TestEngine(unittest.TestCase):
    def _loader(self):
        batches = []
        for i in range(3):
            batches.append({
                "input_ids": torch.randint(0, 20, (4, 6)),
                "attention_mask": torch.ones(4, 6, dtype=torch.long),
                "labels": torch.tensor([0, 1, 0, 1]),
                "relative_paths": [f"x{i}_{j}" for j in range(4)],
            })
        return batches

    def test_train_and_eval(self):
        torch.manual_seed(0)
        model = DummyClassifier()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        train_metrics = train_epoch(
            model,
            self._loader(),
            optimizer,
            scheduler,
            torch.device("cpu"),
            grad_accum_steps=2,
            grad_clip=1.0,
            amp=False,
            scaler=None,
        )
        val_metrics = evaluate_epoch(
            model,
            self._loader(),
            torch.device("cpu"),
            amp=False,
        )
        self.assertIn("loss", train_metrics)
        self.assertIn("f1", train_metrics)
        self.assertIn("accuracy", val_metrics)


if __name__ == "__main__":
    unittest.main()
