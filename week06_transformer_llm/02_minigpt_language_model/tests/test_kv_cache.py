import unittest
import torch
from model import GPT, GPTConfig


class TestKVCache(unittest.TestCase):
    def _run(self, impl: str):
        torch.manual_seed(1)
        model = GPT(GPTConfig(
            vocab_size=20,
            block_size=32,
            d_model=32,
            num_heads=4,
            num_layers=2,
            d_ff=64,
            dropout=0.0,
            attention_impl=impl,
        )).eval()
        ids = torch.randint(0, 20, (1, 8))
        with torch.no_grad():
            full_logits, _, _ = model(ids)
            cache = None
            pieces = []
            for i in range(ids.size(1)):
                logits, _, cache = model(ids[:, i:i+1], past_key_values=cache, use_cache=True)
                pieces.append(logits)
            cached_logits = torch.cat(pieces, dim=1)
        self.assertTrue(torch.allclose(full_logits, cached_logits, atol=1e-5, rtol=1e-4))

    def test_manual_cache_equivalence(self):
        self._run("manual")

    def test_sdpa_cache_equivalence(self):
        self._run("sdpa")


if __name__ == "__main__":
    unittest.main()
