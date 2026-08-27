import unittest
import torch
from torch import nn
from lora import LoRALinear


class TestLoRA(unittest.TestCase):
    def test_initially_matches_base(self):
        torch.manual_seed(0)
        base = nn.Linear(6, 4)
        x = torch.randn(3, 6)
        expected = base(x).detach()
        lora = LoRALinear(base, rank=2)
        actual = lora(x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertFalse(base.weight.requires_grad)
        self.assertTrue(lora.lora_a.requires_grad)
        self.assertTrue(lora.lora_b.requires_grad)


if __name__ == "__main__":
    unittest.main()
