from __future__ import annotations

import unittest

import torch

from sentiment_classification.models import (
    BiLSTMAttentionClassifier,
    BiLSTMClassifier,
    MeanPoolingClassifier,
)


class TestModels(unittest.TestCase):

    def setUp(self) -> None:
        self.input_ids = torch.tensor(
            [
                [2, 3, 4, 0],
                [5, 6, 7, 8],
            ],
            dtype=torch.long,
        )
        self.mask = self.input_ids.ne(0)
        self.lengths = self.mask.sum(dim=1)

    def test_mean_pooling_shape(self) -> None:
        model = MeanPoolingClassifier(
            vocab_size=20,
            embedding_dim=8,
        )

        logits = model(
            self.input_ids,
            self.lengths,
            self.mask,
        )

        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_bilstm_shape(self) -> None:
        model = BiLSTMClassifier(
            vocab_size=20,
            embedding_dim=8,
            hidden_size=6,
        )

        logits = model(
            self.input_ids,
            self.lengths,
            self.mask,
        )

        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_attention_shape_and_mask(self) -> None:
        model = BiLSTMAttentionClassifier(
            vocab_size=20,
            embedding_dim=8,
            hidden_size=6,
            attention_dim=5,
        )

        model.eval()

        with torch.no_grad():
            logits, weights = model(
                self.input_ids,
                self.lengths,
                self.mask,
                return_attention=True,
            )

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(weights.shape), (2, 4))

        self.assertTrue(
            torch.all(weights[~self.mask] < 1e-6)
        )

        self.assertTrue(
            torch.allclose(
                weights.sum(dim=1),
                torch.ones(2),
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
