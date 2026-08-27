import unittest
from utils import linear_warmup_decay_lambda


class TestScheduler(unittest.TestCase):
    def test_warmup_then_decay(self):
        vals = [linear_warmup_decay_lambda(i, warmup_steps=2, total_steps=6) for i in range(7)]
        self.assertGreater(vals[1], vals[0])
        self.assertAlmostEqual(vals[1], 1.0)
        self.assertGreater(vals[2], vals[3])
        self.assertEqual(vals[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
