from __future__ import annotations

import unittest

from sentiment_classification.data import (
    SampleRecord,
    stratified_train_val_split,
)


class TestSplit(unittest.TestCase):

    def test_stratified_split_is_reproducible(self) -> None:
        samples = [
            SampleRecord(
                relative_path=f"neg/{i}.txt",
                label=0,
            )
            for i in range(10)
        ] + [
            SampleRecord(
                relative_path=f"pos/{i}.txt",
                label=1,
            )
            for i in range(10)
        ]

        train_a, val_a = stratified_train_val_split(
            samples,
            val_ratio=0.2,
            seed=42,
        )

        train_b, val_b = stratified_train_val_split(
            samples,
            val_ratio=0.2,
            seed=42,
        )

        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(train_a), 16)
        self.assertEqual(len(val_a), 4)
        self.assertEqual(
            sum(item.label for item in val_a),
            2,
        )


if __name__ == "__main__":
    unittest.main()
