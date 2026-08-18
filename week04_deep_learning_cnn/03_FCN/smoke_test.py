"""无需 VOC 数据即可验证 FCN 前向、损失和反向传播。"""

import torch
from torch.nn import functional as F

from metrics import SegmentationConfusionMatrix
from model import FCNResNet18


def main() -> None:
    model = FCNResNet18(num_classes=21, pretrained=False)
    x = torch.randn(1, 3, 64, 96)
    target = torch.randint(0, 21, (1, 64, 96))
    logits = model(x)
    assert logits.shape == (1, 21, 64, 96)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    metric = SegmentationConfusionMatrix(21)
    metric.update(logits.detach(), target)
    result = metric.compute()
    print("logits:", logits.shape)
    print("loss:", float(loss.detach()))
    print("pixel accuracy:", result["pixel_accuracy"])
    print("mIoU:", result["mean_iou"])
    print("FCN smoke test passed.")


if __name__ == "__main__":
    main()
