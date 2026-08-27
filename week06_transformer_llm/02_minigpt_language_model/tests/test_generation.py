import unittest
import torch
from generation import generate_cached, generate_naive
from model import GPT, GPTConfig


class TestGeneration(unittest.TestCase):
    def test_greedy_naive_cache_same(self):
        torch.manual_seed(2)
        model = GPT(GPTConfig(
            vocab_size=15,
            block_size=24,
            d_model=24,
            num_heads=4,
            num_layers=2,
            d_ff=48,
            dropout=0.0,
            attention_impl="manual",
        )).eval()
        prompt = torch.tensor([[1, 2, 3, 4]])
        a = generate_naive(model, prompt, 10, greedy=True)
        b = generate_cached(model, prompt, 10, greedy=True)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
