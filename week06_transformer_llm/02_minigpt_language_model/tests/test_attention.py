import unittest
import torch
from attention import CausalSelfAttention


class TestAttention(unittest.TestCase):
    def test_manual_sdpa_match(self):
        torch.manual_seed(0)
        manual = CausalSelfAttention(32, 4, dropout=0.0, impl="manual").eval()
        sdpa = CausalSelfAttention(32, 4, dropout=0.0, impl="sdpa").eval()
        sdpa.load_state_dict(manual.state_dict())
        x = torch.randn(2, 7, 32)
        with torch.no_grad():
            a, _ = manual(x)
            b, _ = sdpa(x)
        self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=1e-4))

    def test_cache_shape(self):
        attn = CausalSelfAttention(32, 4, dropout=0.0, impl="manual").eval()
        x = torch.randn(2, 5, 32)
        with torch.no_grad():
            _, cache = attn(x, use_cache=True)
            _, cache2 = attn(torch.randn(2, 1, 32), past_kv=cache, use_cache=True)
        self.assertEqual(tuple(cache[0].shape), (2, 4, 5, 8))
        self.assertEqual(tuple(cache2[0].shape), (2, 4, 6, 8))


if __name__ == "__main__":
    unittest.main()
