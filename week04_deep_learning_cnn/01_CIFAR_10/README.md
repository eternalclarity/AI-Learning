# Week 04 · CNN and CIFAR-10

---

## 1. 产出

### 模型代码

```text
models/
├── mlp.py
├── basic_cnn.py
├── lenet.py
├── vgg_small.py
├── residual_block.py
└── small_resnet.py
```

### 完整训练系统

```text
data_utils.py              # 数据增强、分层划分、DataLoader
engine.py                  # 训练一轮、验证一轮、收集预测
train.py                   # 训练单个实验
run_core_experiments.py    # 顺序训练四组核心实验
compare_experiments.py     # 汇总验证集结果
evaluate.py                # 独立测试集最终评估
model_summary.py           # 逐层输出形状与参数量
utils.py                   # 随机种子、设备、检查点、绘图
smoke_test.py              # 无需下载数据的快速环境检查
```

---

## 2. 项目目录

```text
week04_cnn_cifar10/
├── data/
├── models/
│   ├── __init__.py
│   ├── mlp.py
│   ├── basic_cnn.py
│   ├── lenet.py
│   ├── vgg_small.py
│   ├── residual_block.py
│   └── small_resnet.py
├── outputs/
│   ├── checkpoints/
│   ├── plots/
│   └── results/
├── tests/
│   └── test_models.py
├── config.py
├── data_utils.py
├── engine.py
├── train.py
├── evaluate.py
├── compare_experiments.py
├── run_core_experiments.py
├── model_summary.py
├── smoke_test.py
├── utils.py
├── experiment_report_template.md
├── requirements.txt
└── README.md
```

---

## 3. 环境准备

### 3.1 进入项目

```powershell
cd D:\code.py\workspace\AI-Learning\week04_cnn_cifar10
```

### 3.2 激活 Conda 环境

```powershell
conda activate ai
```

### 3.3 安装依赖

如果你的 `ai` 环境已经安装了支持 CUDA 的 PyTorch，不建议为了执行这一条命令而重新覆盖 PyTorch。

先检查：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

其余依赖可以安装：

```powershell
pip install numpy pandas matplotlib scikit-learn tqdm Pillow jupyter nbformat
```

或者：

```powershell
pip install -r requirements.txt
```

如果 `requirements.txt` 导致原有 CUDA 版 PyTorch 被覆盖，应根据 PyTorch 官网为自己的 CUDA 环境重新安装对应版本。

---

## 4. 第一次运行前的检查

### 4.1 冒烟测试

不下载 CIFAR-10，只用随机张量检查所有模型能否前向传播、计算损失和反向传播：

```powershell
python smoke_test.py
```

### 4.2 单元测试

```powershell
python -m unittest discover -s tests -v
```

测试内容：

- 所有模型输出 `[batch_size, 10]`；
- 普通残差块保持形状；
- 投影残差块正确改变通道和空间尺寸。

### 4.3 查看模型形状

```powershell
python model_summary.py --experiment exp1_basic_cnn
```

保存 CSV：

```powershell
python model_summary.py ^
    --experiment exp1_basic_cnn ^
    --output outputs/results/basic_cnn_model_summary.csv
```

---

## 5. 核心实验设计

| 预设名称 | 模型 | BN | Dropout | 数据增强 |
|---|---|---:|---:|---:|
| `exp1_basic_cnn` | BasicCNN | 否 | 0.0 | 否 |
| `exp2_cnn_bn` | BasicCNN | 是 | 0.0 | 否 |
| `exp3_cnn_bn_dropout` | BasicCNN | 是 | 0.5 | 否 |
| `exp4_cnn_bn_dropout_aug` | BasicCNN | 是 | 0.5 | 是 |

核心四组实验保持：

- 相同数据划分；
- 相同随机种子；
- 相同训练轮数；
- 相同 batch size；
- 相同 Adam 优化器；
- 相同初始学习率；
- 相同模型主体；
- 相同验证集选择规则。

这样才能把结果变化主要归因于 BN、Dropout 或数据增强。

---

## 6. 训练单个实验

### CPU 或自动设备

```powershell
python train.py --experiment exp1_basic_cnn --epochs 20
```

### NVIDIA GPU + AMP

```powershell
python train.py ^
    --experiment exp1_basic_cnn ^
    --epochs 20 ^
    --batch-size 128 ^
    --device cuda ^
    --amp
```

### Windows DataLoader 建议

初学阶段保持：

```text
--num-workers 0
```

确认代码稳定后再尝试：

```text
--num-workers 2
```

### 从最后一个检查点继续训练

```powershell
python train.py ^
    --experiment exp1_basic_cnn ^
    --epochs 30 ^
    --resume outputs/checkpoints/exp1_basic_cnn/last_model.pth
```

这里 `epochs=30` 表示总轮数训练到 30，而不是额外再训练 30 轮。

---

## 7. 一次运行四组核心实验

```powershell
python run_core_experiments.py ^
    --epochs 20 ^
    --batch-size 128 ^
    --device cuda ^
    --amp
```

这个脚本会：

1. 依次训练 Exp 1～Exp 4；
2. 每组保存最佳模型和最后模型；
3. 最后自动运行验证集结果汇总。

四组实验在 RTX 3060 Laptop GPU 上仍需要一定时间。第一次可以先用：

```powershell
python run_core_experiments.py --epochs 2 --batch-size 128 --amp
```

确认完整流程无误后，再删除旧输出并正式训练 20 轮。

---

## 8. 可选模型

### MLP 对照

```powershell
python train.py --experiment exp0_mlp --epochs 20 --amp
```

### LeNet

```powershell
python train.py --experiment exp5_lenet --epochs 20 --amp
```

### Small VGG

```powershell
python train.py --experiment exp6_vgg_small --epochs 20 --amp
```

### Small ResNet

```powershell
python train.py --experiment exp7_small_resnet --epochs 20 --amp
```

可选模型用于理解架构，不应与核心控制变量实验混为同一结论。

---

## 9. 验证集比较

```powershell
python compare_experiments.py
```

输出：

```text
outputs/results/comparison/
├── experiment_comparison.csv
└── validation_accuracy_comparison.png
```

比较后根据：

- 最佳验证准确率；
- 验证损失；
- 泛化差距；
- 训练时间；
- 参数量；
- 训练曲线；

选择最终方案。

---

## 10. 最终测试

训练脚本不会自动查看测试集。

确定最终模型后，只对选中的检查点执行：

```powershell
python evaluate.py ^
    --checkpoint outputs/checkpoints/exp4_cnn_bn_dropout_aug/best_model.pth ^
    --device cuda
```

输出：

```text
outputs/results/exp4_cnn_bn_dropout_aug/final_test/
├── final_test_metrics.json
├── per_class_accuracy.csv
├── test_predictions.csv
├── confusion_matrix.csv
├── confusion_matrix.png
└── sample_predictions.png
```

---

## 11. 每个实验的输出

```text
outputs/checkpoints/<experiment>/
├── best_model.pth
└── last_model.pth
```

```text
outputs/results/<experiment>/
├── history.csv
└── training_summary.json
```

```text
outputs/plots/<experiment>/
└── training_curves.png
```

### `best_model.pth`

保存验证准确率最高轮次的模型。

### `last_model.pth`

保存最后完成轮次的模型，可用于继续训练。

两者都包含：

- 模型参数；
- 优化器状态；
- 调度器状态；
- 当前 epoch；
- 历史指标；
- 模型配置；
- 数据划分配置；
- 类别名称。

---

## 12. Notebook 使用顺序

### CIFAR-10 数据探索

```text
notebooks/01_cifar10_exploration.ipynb
```

学习：

- 数据集形状；
- RGB 通道；
- 类别名称；
- 类别分布；
- 原图与标准化图像；
- 训练/验证划分。

### 手写卷积

```text
notebooks/02_convolution_from_scratch.ipynb
```

学习：

- 二维互相关；
- 边缘检测；
- 与 `nn.Conv2d` 对照；
- padding 和 stride；
- 多通道；
- 最大池化。

---

## 13. 关键设计说明

### 为什么训练集与验证集创建两个 Dataset 对象

同一份官方训练数据需要两套 transform：

```text
训练子集：随机增强 + 标准化
验证子集：确定性转换 + 标准化
```

如果先创建一个 Dataset 再用 `random_split`，两个子集会共享同一个 transform，验证集可能被错误地随机增强。

项目因此创建两个指向同一底层训练数据的 Dataset，再用同一组分层索引组成两个 `Subset`。

### 为什么使用分层划分

每个类别分别按 90%/10% 划分，避免验证集中某些类别数量偶然偏多或偏少。

### 为什么不在模型末尾写 Softmax

`CrossEntropyLoss` 接收 logits，并在内部进行数值稳定的处理。

### 为什么 BN 版本中的卷积 `bias=False`

BatchNorm 的可学习参数 `β` 可以平移输出，卷积偏置通常是冗余的。

### 为什么测试只执行一次

测试集用于估计最终模型对真正未知数据的泛化表现，不参与实验选择。

---

## 14. 推荐学习顺序

```text
notes/00_week4_learning_guide.md
→ notes/01_mlp_and_neural_network_review.md
→ notebooks/01_cifar10_exploration.ipynb
→ notes/02_convolution_basics.md
→ notebooks/02_convolution_from_scratch.ipynb
→ models/basic_cnn.py
→ models/lenet.py
→ notes/03_cnn_architectures.md
→ models/vgg_small.py
→ notes/04_training_techniques.md
→ models/residual_block.py
→ models/small_resnet.py
→ engine.py
→ train.py
→ compare_experiments.py
→ evaluate.py
→ notes/05_experiment_design_and_analysis.md
→ notes/08_week4_self_test.md
```

---

## 15. 参考课程

- 李沐《动手学深度学习》视频：<https://www.bilibili.com/list/1567748478/?sid=358497>
- 《动手学深度学习》中文教材：<https://zh.d2l.ai/>
- 卷积神经网络章节：<https://zh.d2l.ai/chapter_convolutional-neural-networks/index.html>
- 现代卷积神经网络章节：<https://zh.d2l.ai/chapter_convolutional-modern/index.html>

---

## 16. 本周完成标准

你应该能够在不看答案的情况下解释：

1. CNN 为什么比 MLP 更适合图像；
2. 卷积输出形状和参数量怎样计算；
3. 输入通道与输出通道的含义；
4. padding、stride 和 pooling 的作用；
5. LeNet、AlexNet、VGG、ResNet 的演进关系；
6. BatchNorm、Dropout 和数据增强的区别；
7. `model.train()` 与 `model.eval()` 为什么重要；
8. 残差连接为什么要求形状一致；
9. 为什么验证集选方案、测试集最终评估；
10. 怎样根据曲线和混淆矩阵改进模型。
