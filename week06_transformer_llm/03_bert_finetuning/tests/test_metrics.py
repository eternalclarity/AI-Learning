import unittest
import torch
from metrics import compute_metrics, confusion_matrix_tensor


class TestMetrics(unittest.TestCase):
    def test_binary_metrics(self):
        labels = torch.tensor([0, 0, 1, 1])
        preds = torch.tensor([0, 1, 1, 1])
        m = compute_metrics(labels, preds)
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["specificity"], 0.5)
        self.assertEqual(confusion_matrix_tensor(labels, preds).tolist(), [[1, 1], [0, 2]])


if __name__ == "__main__":
    unittest.main()
