import unittest
from metrics import bleu


class TestBleu(unittest.TestCase):
    def test_identical(self):
        tokens = ["je", "t", "aime"]
        self.assertAlmostEqual(bleu(tokens, tokens), 1.0, places=6)

    def test_empty(self):
        self.assertEqual(bleu([], ["a"]), 0.0)


if __name__ == "__main__":
    unittest.main()
