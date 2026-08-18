import unittest

import numpy as np
import torch
from PIL import Image

from dataset import VOC_COLORMAP, voc_label_indices
from metrics import SegmentationConfusionMatrix
from model import FCNResNet18


class TestSegmentation(unittest.TestCase):
    def test_color_mapping(self):
        array = np.array([[VOC_COLORMAP[0], VOC_COLORMAP[12], [1, 2, 3]]], dtype=np.uint8)
        label = voc_label_indices(Image.fromarray(array))
        self.assertEqual(label.tolist(), [[0, 12, 255]])

    def test_model_output_shape(self):
        model = FCNResNet18(21, pretrained=False)
        output = model(torch.randn(1, 3, 64, 96))
        self.assertEqual(output.shape, (1, 21, 64, 96))

    def test_perfect_metrics(self):
        target = torch.tensor([[[0, 1], [1, 0]]])
        metric = SegmentationConfusionMatrix(2)
        metric.update(target.clone(), target)
        result = metric.compute()
        self.assertAlmostEqual(result["pixel_accuracy"], 1.0)
        self.assertAlmostEqual(result["mean_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
