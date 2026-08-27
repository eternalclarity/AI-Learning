import unittest
import torch
from model import MultiHeadAttention, Transformer, TransformerConfig


class TestModel(unittest.TestCase):
    def test_attention_shape(self):
        attn = MultiHeadAttention(32, 4, dropout=0.0)
        q = torch.randn(2, 3, 32)
        kv = torch.randn(2, 5, 32)
        out, weights = attn(q, kv, kv, need_weights=True)
        self.assertEqual(tuple(out.shape), (2, 3, 32))
        self.assertEqual(tuple(weights.shape), (2, 4, 3, 5))

    def test_transformer_shape(self):
        m = Transformer(TransformerConfig(20, 25, d_model=32, num_heads=4, num_layers=1, d_ff=64, dropout=0.0))
        src = torch.randint(1, 20, (2, 6))
        tgt = torch.randint(1, 25, (2, 5))
        logits = m(src, tgt, src.ne(0), tgt.ne(0))
        self.assertEqual(tuple(logits.shape), (2, 5, 25))


if __name__ == "__main__":
    unittest.main()
