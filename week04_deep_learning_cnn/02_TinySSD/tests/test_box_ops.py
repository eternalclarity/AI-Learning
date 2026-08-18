import unittest

import torch

from box_ops import box_iou, multibox_prior, multibox_target, nms
from model import TinySSD


class TestBoxOps(unittest.TestCase):
    def test_iou_identity(self):
        boxes = torch.tensor([[0.1, 0.2, 0.5, 0.6]])
        value = box_iou(boxes, boxes)
        self.assertTrue(torch.allclose(value, torch.ones_like(value)))

    def test_anchor_count(self):
        feature = torch.zeros(1, 8, 4, 5)
        anchors = multibox_prior(feature, [0.2, 0.4], [1, 2, 0.5])
        self.assertEqual(anchors.shape, (1, 4 * 5 * 4, 4))

    def test_target_has_positive_anchor(self):
        feature = torch.zeros(1, 8, 4, 4)
        anchors = multibox_prior(feature, [0.3, 0.5], [1, 2, 0.5])
        labels = torch.tensor([[[0.0, 0.25, 0.25, 0.75, 0.75]]])
        _, mask, cls = multibox_target(anchors, labels)
        self.assertGreater(int((cls > 0).sum()), 0)
        self.assertGreater(float(mask.sum()), 0.0)

    def test_nms_removes_duplicate(self):
        boxes = torch.tensor([[0.1, 0.1, 0.8, 0.8], [0.12, 0.12, 0.79, 0.79], [0.0, 0.0, 0.2, 0.2]])
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, 0.5)
        self.assertEqual(keep.tolist(), [0, 2])

    def test_model_shapes(self):
        model = TinySSD(1)
        anchors, cls, bbox = model(torch.rand(2, 3, 256, 256))
        self.assertEqual(cls.shape[0], 2)
        self.assertEqual(cls.shape[1], anchors.shape[1])
        self.assertEqual(cls.shape[2], 2)
        self.assertEqual(bbox.shape[1], anchors.shape[1] * 4)


if __name__ == "__main__":
    unittest.main()
