import unittest
import torch
from torch import nn
from strategies import apply_strategy, parameter_counts


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])


class DummyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(10, 4)
        self.encoder = DummyEncoder()
        self.pooler = nn.Linear(4, 4)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = DummyBase()
        self.classifier = nn.Linear(4, 2)


class TestStrategies(unittest.TestCase):
    def test_head_only(self):
        m = DummyModel()
        apply_strategy(m, "head_only")
        self.assertTrue(all(p.requires_grad for p in m.classifier.parameters()))
        self.assertTrue(all(not p.requires_grad for p in m.base_model.parameters()))

    def test_last_n(self):
        m = DummyModel()
        apply_strategy(m, "last_n", 2)
        self.assertTrue(all(not p.requires_grad for p in m.base_model.encoder.layer[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in m.base_model.encoder.layer[-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in m.base_model.pooler.parameters()))

    def test_full(self):
        m = DummyModel()
        apply_strategy(m, "full")
        counts = parameter_counts(m)
        self.assertEqual(counts["total_parameters"], counts["trainable_parameters"])


if __name__ == "__main__":
    unittest.main()
