# Project B · Pascal VOC2012 语义分割（ResNet18-FCN）

> D2L 13.9–13.11 · Pixel-wise Classification · VOCSegDataset · Transposed Convolution · FCN · Transfer Learning · mIoU

这个项目从“一个图片一个类别”进一步走到“一个像素一个类别”。

## 1. 学习主线

《动手学深度学习》的分割部分是：

```text
13.9 语义分割和 VOC2012 数据集
→ 13.10 转置卷积
→ 13.11 全卷积网络 FCN
```

项目完整数据流：

```text
VOC RGB 图像
+ 与图像同尺寸的 RGB 标签图
        ↓
同步随机裁剪
        ↓
标签颜色 → 21 类像素索引
        ↓
ResNet18 backbone 提取低分辨率特征
        ↓
1×1 Conv：512 通道 → 21 类
        ↓
ConvTranspose2d：上采样 32 倍
        ↓
[B,21,H,W]
        ↓
CrossEntropyLoss
        ↓
Pixel Accuracy + mIoU
```

## 2. 为什么这个项目比直接用 torchvision FCN 更适合学习

如果直接调用：

```python
torchvision.models.segmentation.fcn_resnet50(...)
```

你能快速得到模型，却不容易看清：

- 为什么分类模型最后的全局池化和全连接层要删掉；
- 为什么 `1×1 Conv` 能得到每个空间位置的类别分数；
- 为什么特征图必须上采样回原图尺寸；
- 转置卷积到底放大了什么；
- 像素标签为什么是 `[B,H,W]` 而不是 `[B]`。

因此本项目按照 D2L 的 ResNet18-FCN 思路自己组装模型。

## 3. 项目结构

```text
semantic_segmentation_fcn/
├── data/
├── notes/
│   ├── 01_semantic_segmentation_basics.md
│   └── 02_fcn_transposed_conv.md
├── notebooks/
│   └── 01_segmentation_label_lab.ipynb
├── outputs/
├── tests/
│   └── test_segmentation.py
├── config.py
├── download_voc.py
├── dataset.py
├── model.py
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
| `VOCSegDataset` | 13.9 自定义语义分割 Dataset |
| 同步裁剪 | 13.9 输入图和标签必须裁同一区域 |
| `voc_label_indices` | 13.9 RGB 标签转类别索引 |
| `bilinear_kernel` | 13.10 / 13.11 双线性初始化 |
| ResNet18 backbone | 13.11 使用预训练网络提取特征 |
| `1×1 Conv` | 13.11 通道变为类别数 |
| `ConvTranspose2d` | 13.10 / 13.11 上采样 |

## 5. 本项目相对教材的工程扩展

### 5.1 Void / 边界像素使用 ignore_index=255

VOC 标签里存在不属于 21 个正式类别的边界像素。本项目不会把这些未知颜色错误当作 background，而是映射到 255，并在交叉熵中忽略。

### 5.2 验证集使用确定性中心裁剪

D2L 教学代码对训练和验证样本都通过同一个随机裁剪接口取固定尺寸。本项目为了让每次验证可复现：

```text
train → random crop + optional horizontal flip
val   → center crop
```

这是工程评价层面的扩展，不改变教材关于“输入和标签必须同步裁剪”的核心思想。

### 5.3 增加 mIoU

像素准确率会被大量背景像素影响，因此项目同时计算每类 IoU 和 mIoU。

## 6. 运行顺序

### 第一步：进入目录

```powershell
cd D:\code.py\workspace\AI-Learning\week04_cnn_cifar10\projects\semantic_segmentation_fcn
conda activate ai
```

### 第二步：无需数据的冒烟测试

```powershell
python smoke_test.py
python -m unittest discover -s tests -v
```

### 第三步：完成像素标签实验

打开：

```text
notebooks/01_segmentation_label_lab.ipynb
```

### 第四步：下载 VOC2012

```powershell
python download_voc.py
```

注意：VOC2012 train/val 压缩包约 2GB。

### 第五步：观察数据

```powershell
python visualize_dataset.py
```

### 第六步：训练

```powershell
python train.py --epochs 5 --batch-size 4 --device cuda --amp
```

第一次运行 `pretrained=True` 会下载 ImageNet 预训练 ResNet18 权重。

如果暂时没有网络：

```powershell
python train.py --no-pretrained --epochs 5 --batch-size 4 --device cuda
```

但学习效果与收敛效果通常不如迁移学习。

### 第七步：验证

```powershell
python evaluate.py --device cuda
```

重点查看：

```text
outputs/results/evaluation.json
outputs/results/per_class_iou.csv
outputs/plots/prediction_*.png
```

### 第八步：任意图片推理

```powershell
python predict.py your_image.jpg --device cuda
```

会保存：

```text
mask.png
正类彩色标签图

overlay.png
语义结果与原图叠加图
```

## 7. 推荐阅读顺序

```text
notebook 标签实验
→ dataset.py
→ model.py
→ metrics.py
→ engine.py
→ train.py
→ evaluate.py
→ predict.py
```

完成后，你应该能清楚区分：目标检测的“框级预测”、语义分割的“像素级预测”和实例分割的“实例级像素预测”。
