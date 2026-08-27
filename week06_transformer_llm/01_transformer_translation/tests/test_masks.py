import unittest
import torch
from masks import make_causal_mask, make_valid_mask


class TestMasks(unittest.TestCase):
    def test_valid_mask(self):
        x = torch.tensor([[1, 2, 0]])
        self.assertEqual(make_valid_mask(x, 0).tolist(), [[True, True, False]])

    def test_causal_mask(self):
        expected = [[True, False, False], [True, True, False], [True, True, True]]
        self.assertEqual(make_causal_mask(3).tolist(), expected)


if __name__ == "__main__":
    unittest.main()
