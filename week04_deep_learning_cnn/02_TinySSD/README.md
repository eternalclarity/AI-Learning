# Project A · TinySSD 目标检测

> D2L 13.3–13.8 · Bounding Box · Anchor · IoU · Target Assignment · SSD · NMS · AP@0.5

这个项目的目的不是调用 `torchvision.models.detection` 直接得到结果，而是把目标检测中最容易“只会背概念”的部分真正写出来。

## 1. 为什么选 TinySSD + 香蕉数据集

《动手学深度学习》计算机视觉章节按照以下顺序展开：

```text
13.3 目标检测和边界框
→ 13.4 锚框
→ 13.5 多尺度目标检测
→ 13.6 目标检测数据集
→ 13.7 SSD
→ 13.8 R-CNN 系列
```

香蕉数据集只有一个类别、每张图一个目标，所以不会被复杂的数据标注格式淹没，可以集中理解：

```text
图片
→ 多尺度特征图
→ 每个位置生成多个锚框
→ 给锚框分配真实框
→ 分类：背景 / 香蕉
→ 回归：预测边界框偏移量
→ 解码预测框
→ NMS 去重
→ AP@0.5 评价
```

## 2. 你必须真正理解的 8 个问题

1. 图像分类为什么只输出类别，而目标检测还必须输出位置？
2. `[xmin, ymin, xmax, ymax]` 与 `[cx, cy, w, h]` 为什么要互换？
3. IoU 为什么能衡量两个框的重合程度？
4. 为什么一个像素位置要生成多个大小、宽高比不同的锚框？
5. 正锚框、负锚框分别是什么？
6. 为什么 SSD 同时有分类损失和边界框回归损失？
7. 为什么同一个物体会得到多个预测框，NMS 如何去重？
8. 为什么“锚框分类准确率很高”不代表目标检测效果一定好？

## 3. 项目结构

```text
object_detection_tinyssd/
├── data/
├── notes/
│   ├── 01_object_detection_basics.md
│   └── 02_ssd_end_to_end.md
├── notebooks/
│   └── 01_anchor_box_lab.ipynb
├── outputs/
│   ├── checkpoints/
│   ├── plots/
│   └── results/
├── tests/
│   └── test_box_ops.py
├── config.py
├── download_data.py
├── dataset.py
├── box_ops.py
├── model.py
├── losses.py
├── metrics.py
├── engine.py
├── train.py
├── evaluate.py
├── predict.py
├── visualize_dataset.py
├── smoke_test.py
└── README.md
```

## 4. 与 D2L 的对应关系

| 项目代码 | D2L知识 |
|---|---|
| `dataset.py` | 13.6 目标检测数据集 |
| `box_corner_to_center` | 13.3 边界框 |
| `box_iou` | 13.4 锚框与交并比 |
| `multibox_prior` | 13.4 / 13.5 多尺度锚框 |
| `assign_anchor_to_bbox` | 13.4 锚框标注 |
| `multibox_target` | 13.7 SSD训练标签 |
| `TinySSD` | 13.7 SSD |
| `nms` | 13.4 / 13.7 非极大值抑制 |
| `multibox_detection` | 13.7 SSD预测 |

## 5. 本项目相对教材的两个工程扩展

### 5.1 Smooth L1

教材主示例的边界框损失使用 L1；教材练习建议尝试 Smooth L1。本项目直接采用 Smooth L1，使零点附近更加平滑。

### 5.2 AP@0.5

教材 TinySSD 主示例重点观察锚框分类错误率和边界框 MAE，没有完整实现测试集检测评价。本项目额外实现 AP@0.5、Precision、Recall 和匹配框平均 IoU，让训练结果有真正的检测指标。

## 6. 运行顺序

### 第一步：进入目录

```powershell
cd D:\code.py\workspace\AI-Learning\week04_cnn_cifar10\projects\object_detection_tinyssd
conda activate ai
```

### 第二步：不下载数据先检查算法

```powershell
python smoke_test.py
python -m unittest discover -s tests -v
```

### 第三步：做锚框实验

打开：

```text
notebooks/01_anchor_box_lab.ipynb
```

### 第四步：下载香蕉数据集

```powershell
python download_data.py
```

D2L 数据集包含约 1000 张训练图和 100 张验证图，每张图片为 256×256 且有一个香蕉边界框。

### 第五步：先观察数据

```powershell
python visualize_dataset.py
```

### 第六步：训练

```powershell
python train.py --epochs 20 --batch-size 32 --device cuda --amp
```

### 第七步：验证

```powershell
python evaluate.py --device cuda
```

最终重点看：

```text
outputs/results/evaluation.json
outputs/plots/prediction_*.png
```

### 第八步：单张图片预测

```powershell
python predict.py your_image.jpg --device cuda
```

## 7. 推荐学习方式

不要一上来读 `train.py`。

```text
notebook 锚框实验
→ dataset.py
→ box_ops.py
→ model.py
→ losses.py
→ engine.py
→ train.py
→ evaluate.py
```

真正学懂这个项目后，你再看 Faster R-CNN、YOLO、DETR 时，会更容易理解它们到底改变了检测流程中的哪一部分。
