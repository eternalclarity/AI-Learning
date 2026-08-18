"""无需下载数据即可验证 TinySSD 核心计算是否正常。"""

import torch

from box_ops import multibox_detection, multibox_target
from losses import ssd_loss
from model import TinySSD


def main() -> None:
    model = TinySSD(num_classes=1)
    x = torch.rand(2, 3, 256, 256)
    targets = torch.tensor(
        [
            [[0.0, 0.20, 0.25, 0.70, 0.80]],
            [[0.0, 0.35, 0.20, 0.65, 0.70]],
        ]
    )
    anchors, cls_preds, bbox_preds = model(x)
    bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, targets)
    total, _, _ = ssd_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
    total.mean().backward()

    probs = torch.softmax(cls_preds.detach(), dim=-1).permute(0, 2, 1)
    outputs = multibox_detection(probs, bbox_preds.detach(), anchors, score_threshold=0.9)

    print("anchors:", anchors.shape)
    print("class predictions:", cls_preds.shape)
    print("bbox predictions:", bbox_preds.shape)
    print("loss:", float(total.mean().detach()))
    print("detections per image:", [len(x) for x in outputs])
    print("TinySSD smoke test passed.")


if __name__ == "__main__":
    main()
