import unittest
import torch
from data import DynamicPaddingCollator


class FakeTokenizer:
    def __call__(self, texts, padding, truncation, max_length, pad_to_multiple_of, return_tensors):
        lengths = [min(len(t.split()), max_length) + 2 for t in texts]
        width = max(lengths)
        ids = torch.zeros(len(texts), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, length in enumerate(lengths):
            ids[i, :length] = 1
            mask[i, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


class TestCollator(unittest.TestCase):
    def test_dynamic_width(self):
        collate = DynamicPaddingCollator(FakeTokenizer(), max_length=20)
        batch = [
            {"text": "a b", "label": 0, "relative_path": "a"},
            {"text": "a b c d e", "label": 1, "relative_path": "b"},
        ]
        out = collate(batch)
        self.assertEqual(tuple(out["input_ids"].shape), (2, 7))
        self.assertEqual(out["labels"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
