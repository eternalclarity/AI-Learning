from __future__ import annotations

import unittest

import torch

from sentiment_classification.data import (
    PAD_ID,
    UNK_ID,
    Vocab,
    basic_tokenize,
    collate_batch,
)


class TestTextPipeline(unittest.TestCase):

    def test_tokenizer(self) -> None:
        tokens = basic_tokenize(
            "I don't like this.<br />Really!"
        )

        self.assertEqual(
            tokens,
            ["i", "don't", "like", "this", ".", "really", "!"],
        )

    def test_vocab_unknown(self) -> None:
        vocab = Vocab(["<pad>", "<unk>", "hello"])

        self.assertEqual(vocab["hello"], 2)
        self.assertEqual(vocab["missing"], UNK_ID)

    def test_collate(self) -> None:
        batch = [
            {
                "input_ids": torch.tensor([2, 3, 4]),
                "length": 3,
                "label": 1,
                "relative_path": "a.txt",
            },
            {
                "input_ids": torch.tensor([5]),
                "length": 1,
                "label": 0,
                "relative_path": "b.txt",
            },
        ]

        result = collate_batch(batch)

        self.assertEqual(
            tuple(result["input_ids"].shape),
            (2, 3),
        )
        self.assertEqual(
            int(result["input_ids"][1, 1]),
            PAD_ID,
        )
        self.assertEqual(
            result["attention_mask"].tolist(),
            [
                [True, True, True],
                [True, False, False],
            ],
        )


if __name__ == "__main__":
    unittest.main()
