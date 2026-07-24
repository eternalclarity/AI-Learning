# Week 2：PyTorch 基础入门

---

## 1. 本周学习目标

本周目标不是只会调用 PyTorch API，而是完整理解并跑通一个深度学习任务的基本流程：

```text
数据集
→ Dataset
→ DataLoader
→ 模型定义
→ 前向传播
→ 计算损失
→ 反向传播
→ 参数更新
→ 验证
→ 保存模型
→ 加载模型并测试
```

完成本周代码后，应当能够：

1. 使用 PyTorch 创建和操作 Tensor；
2. 理解 CPU、GPU 与 `device` 的关系；
3. 理解 `requires_grad`、梯度和自动求导；
4. 使用 `loss.backward()` 计算梯度；
5. 使用 `optimizer.step()` 更新参数；
6. 自定义 `Dataset` 并通过 `DataLoader` 分批加载数据；
7. 使用 `nn.Module` 自定义神经网络模型；
8. 编写独立的训练函数和验证函数；
9. 保存并重新加载模型检查点；
10. 绘制 loss 和 accuracy 曲线；
11. 完成 FashionMNIST 图像分类实验。

---

## 2. 本周完成内容

| 学习模块 | 主要内容 | 对应文件 | 本周产出 |
|---|---|---|---|
| Tensor 基础 | 创建、形状、索引、广播、矩阵乘法、GPU 转移 | `tensor_practice.ipynb` | Tensor 操作练习 |
| 自动求导 | `requires_grad`、计算图、`backward()`、梯度清零、参数更新 | `autograd_linear_regression.py` | 手写线性回归 |
| 数据加载 | `Dataset`、`DataLoader`、batch、shuffle | `custom_dataset.py` | 自定义数据集 |
| 模型定义 | `nn.Module`、`forward()`、`nn.Sequential`、MLP | `models/mlp.py` | 自定义 MLP 模型 |
| 完整训练 | FashionMNIST、训练集/验证集划分、训练与验证循环 | `train.py` | FashionMNIST 分类模型 |
| 模型评估 | 加载最佳模型、测试集评估、分类别准确率 | `evaluate.py` | 独立测试脚本 |
| 工具封装 | 随机种子、设备选择、保存检查点、曲线绘制 | `utils.py` | 可复用工具函数 |

---

## 3. 项目结构

```text
week2_pytorch_basic/
├── data/                         # FashionMNIST 数据集下载目录
├── models/
│   ├── __init__.py              # 导出 MLP 模型
│   └── mlp.py                   # MLP 模型定义
├── outputs/
│   ├── checkpoints/             # 保存模型检查点
│   └── plots/                   # 保存实验曲线与预测图片
├── tensor_practice.ipynb        # Tensor 基础练习
├── autograd_linear_regression.py# 自动求导与线性回归
├── custom_dataset.py            # 自定义 Dataset 和 DataLoader
├── train.py                     # FashionMNIST 训练与验证
├── evaluate.py                  # 加载模型并在测试集评估
├── utils.py                     # 公共工具函数
└── README.md                    # 本周学习说明与汇报材料
```

---

## 4.学习顺序：

1. tensor_practice.ipynb；
2. autograd_linear_regression.py；
3. custom_dataset.py；
4. models/mlp.py；
5. utils.py；
6. train.py；
7. evaluate.py；

---

## 5. 核心知识与关键代码

这一部分既用于学习复盘，也可直接作为组会 PPT 的核心参考材料。

### 5.1 Tensor：PyTorch 中的数据载体

Tensor 可以理解为 PyTorch 中的多维数组。它与 NumPy 数组相似，但额外支持：

- GPU 加速；
- 自动求导；
- 神经网络参数管理；
- 深度学习计算图。

### 关键代码

```python
import torch

x = torch.randn(2, 3)

print(x.shape)
print(x.dtype)
print(x.device)
```

### CPU/GPU 转移

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

x = x.to(device)
```

### 核心理解

```text
shape  表示 Tensor 的形状；
dtype  表示元素的数据类型；
device 表示 Tensor 位于 CPU 还是 GPU。
```

模型和输入数据必须在同一个设备上：

```python
model = model.to(device)
images = images.to(device)
labels = labels.to(device)
```

---

### 5.2 自动求导：PyTorch 如何得到梯度

在机器学习中，模型通过降低损失函数来学习参数。

假设：

```text
y = wx + b
```

损失函数衡量预测值与真实值之间的误差。PyTorch 的自动求导系统可以自动计算：

```text
损失函数对参数 w、b 的导数
```

这个导数就是梯度。

#### 最小自动求导示例

```python
x = torch.tensor(2.0, requires_grad=True)

y = x ** 2 + 3 * x + 1

y.backward()

print(x.grad)
```

数学上：

```text
y = x² + 3x + 1
dy/dx = 2x + 3
```

当 `x = 2` 时，梯度为 `7`。

#### 手动线性回归的核心代码

```python
predictions = w * x + b
loss = torch.mean((predictions - y) ** 2)

loss.backward()

with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad

w.grad.zero_()
b.grad.zero_()
```

#### 使用优化器后的标准写法

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

三行代码分别表示：

```text
optimizer.zero_grad()：清空上一轮梯度；
loss.backward()：执行反向传播，计算梯度；
optimizer.step()：根据梯度更新模型参数。
```

---

### 5.3 Dataset 与 DataLoader

#### Dataset 的职责

`Dataset` 负责描述：

```text
数据集中一共有多少条数据；
给定一个 index，应当返回哪条样本和标签。
```

自定义 Dataset 通常需要实现：

```python
class StudentDataset(Dataset):
    def __init__(self):
        ...

    def __len__(self):
        ...

    def __getitem__(self, index):
        ...
```

#### DataLoader 的职责

`DataLoader` 负责：

- 按 batch 分批取数据；
- 打乱训练数据；
- 迭代整个数据集；
- 可选地使用多进程加载数据。

```python
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)
```

#### Dataset 与 DataLoader 的关系

```text
Dataset：定义一条数据怎样读取；
DataLoader：定义数据怎样分批、打乱和迭代。
```

---

### 5.4 使用 nn.Module 自定义 MLP 模型

本周使用多层感知机 MLP 对 FashionMNIST 进行分类。

#### 模型输入与输出

FashionMNIST 图片形状：

```text
[batch_size, 1, 28, 28]
```

经过 `Flatten` 后：

```text
[batch_size, 784]
```

模型最终输出：

```text
[batch_size, 10]
```

其中 10 表示 FashionMNIST 的 10 个类别。

#### 模型结构

```text
输入图片 1×28×28
        ↓ Flatten
784 维向量
        ↓ Linear + ReLU + Dropout
256 维隐藏特征
        ↓ Linear + ReLU + Dropout
128 维隐藏特征
        ↓ Linear
10 维分类 logits
```

#### 核心代码

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)
```

#### `nn.Module` 和 `forward()`

```text
nn.Module：所有 PyTorch 神经网络模型的基础类；
__init__：定义模型包含哪些层；
forward：定义数据依次经过哪些层。
```

调用：

```python
logits = model(images)
```

会自动执行：

```python
model.forward(images)
```

---

### 5.5 FashionMNIST 数据预处理

本项目通过 `torchvision.datasets.FashionMNIST` 自动下载和读取数据。

### transform

```python
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ]
)
```

其中：

```text
ToTensor：将图片转换为 Tensor；
Normalize：对像素值进行标准化。
```

### 数据集划分

```text
FashionMNIST 官方训练集：60,000 张
├── 本项目训练集：54,000 张
└── 本项目验证集：6,000 张

FashionMNIST 官方测试集：10,000 张
└── 最终独立测试
```

三者作用：

```text
训练集：用于反向传播和参数更新；
验证集：用于比较不同 epoch 的效果并保存最佳模型；
测试集：训练完成后进行最终评估。
```

---

### 5.6 完整训练流程

训练阶段的核心代码位于 `train.py` 的 `train_one_epoch()`。

```python
model.train()

for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    logits = model(images)
    loss = loss_fn(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

这段代码完整对应：

```text
读取一个 batch
→ 数据转移到 CPU/GPU
→ 前向传播
→ 计算损失
→ 清空旧梯度
→ 反向传播
→ 更新参数
```

#### 前向传播

```python
logits = model(images)
```

模型根据当前参数输出每张图片属于 10 个类别的分数。

#### 计算损失

```python
loss = loss_fn(logits, labels)
```

本项目使用：

```python
loss_fn = nn.CrossEntropyLoss()
```

`CrossEntropyLoss` 可以直接接收 logits，因此模型最后一层不需要手动添加 `Softmax`。

#### 反向传播

```python
loss.backward()
```

计算损失函数对所有可训练参数的梯度。

#### 参数更新

```python
optimizer.step()
```

本项目使用 Adam 优化器：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)
```

---

### 5.7 验证流程

验证阶段位于 `train.py` 的 `validate()`。

```python
model.eval()

with torch.no_grad():
    for images, labels in validation_loader:
        logits = model(images)
        loss = loss_fn(logits, labels)
```

#### 为什么使用 `model.eval()`

模型中包含 Dropout 等训练行为。

```text
model.train()：启用训练模式；
model.eval()：启用验证或推理模式。
```

#### 为什么使用 `torch.no_grad()`

验证阶段不需要更新参数，因此不需要构建计算图和保存梯度。

这样可以：

- 减少内存占用；
- 提高验证速度；
- 防止无意义的梯度计算。

---

### 5.8 保存最佳模型

每个 epoch 验证结束后，比较当前验证准确率与历史最佳准确率。

```python
if validation_accuracy > best_validation_accuracy:
    best_validation_accuracy = validation_accuracy
    save_checkpoint(...)
```

生成：

```text
outputs/checkpoints/best_model.pth
outputs/checkpoints/last_model.pth
```

二者区别：

```text
best_model.pth：验证集准确率最高的模型；
last_model.pth：最后一个 epoch 的模型。
```

本项目检查点不仅保存模型参数，还保存：

- 当前 epoch；
- `model_state_dict`；
- `optimizer_state_dict`；
- 验证 loss；
- 验证 accuracy；
- 模型结构参数；
- 训练参数；
- 类别名称。

---

### 5.9 加载模型并测试

`evaluate.py` 会从检查点中读取模型配置和模型参数。

```python
checkpoint = load_checkpoint(
    checkpoint_path,
    device,
)

model = MLP(**checkpoint["model_config"])

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()
```

然后在 FashionMNIST 官方测试集上计算：

- Test loss；
- Test accuracy；
- 每个类别的准确率；
- 部分样本预测结果。

---

### 5.10 绘制实验曲线

训练过程中记录：

```python
history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
}
```

训练结束后生成：

```text
outputs/plots/loss_curve.png
outputs/plots/accuracy_curve.png
```

曲线主要用于观察：

```text
loss 是否逐渐下降；
accuracy 是否逐渐提高；
训练集与验证集差距是否过大；
模型是否可能出现过拟合。
```

---

## 7. 运行后生成的文件

```text
outputs/
├── history.json
├── checkpoints/
│   ├── best_model.pth
│   └── last_model.pth
└── plots/
    ├── accuracy_curve.png
    ├── loss_curve.png
    ├── linear_regression_fit.png
    └── sample_predictions.png
```

| 输出文件 | 说明 |
|---|---|
| `history.json` | 保存每个 epoch 的训练和验证指标 |
| `best_model.pth` | 验证准确率最高的模型检查点 |
| `last_model.pth` | 最后一个 epoch 的模型检查点 |
| `loss_curve.png` | 训练集和验证集 loss 曲线 |
| `accuracy_curve.png` | 训练集和验证集 accuracy 曲线 |
| `linear_regression_fit.png` | 自动求导线性回归拟合结果 |
| `sample_predictions.png` | FashionMNIST 测试样本图片 |

---

## 8. FashionMINIST图像分类模型-结果记录

运行训练和测试后，将实际结果补充到这里。不要在未运行实验时编造数据。

### 8.1 实验配置

| 参数 | 实际设置 |
|---|---|
| 数据集 | FashionMNIST |
| 模型 | MLP |
| 隐藏层 | 256、128 |
| Dropout | 0.2 |
| Batch size | 64 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | CrossEntropyLoss |
| 运行设备 | GPU |

### 8.2 最终结果

| 指标 | 实际结果 |
|---|---|
| 最佳 epoch | 待填写 |
| 最佳验证准确率 | 待填写 |
| 测试集 loss | 待填写 |
| 测试集 accuracy | 待填写 |
| 总训练时间 | 待填写 |



---

## 9. 本周重点概念复盘

### 9.1 Batch 与 Epoch

```text
Batch：一次输入模型的一小批数据；
Epoch：模型完整学习一遍训练集。
```

假设：

```text
训练集：54,000 张
batch_size：64
```

一个 epoch 大约包含：

```text
54,000 ÷ 64 ≈ 844 个 batch
```

---

### 9.2 参数、梯度与优化器

```text
参数：模型需要学习的 weight 和 bias；
梯度：loss 对参数变化的敏感程度；
优化器：根据梯度调整参数的工具。
```

---

### 9.3 logits 与概率

模型输出的是 logits：

```text
[batch_size, 10]
```

每个值表示一个类别的原始分数。

预测类别：

```python
predictions = logits.argmax(dim=1)
```

训练时 logits 可直接传入：

```python
nn.CrossEntropyLoss()
```

不需要在模型最后手动使用 Softmax。

---

### 9.4 `state_dict`

`state_dict` 是一个保存模型参数的字典，主要包含各层的：

```text
weight
bias
```

保存：

```python
torch.save(model.state_dict(), "model.pth")
```

加载：

```python
model.load_state_dict(torch.load("model.pth"))
```

本项目使用的是更完整的 checkpoint，在 `model_state_dict` 之外还保存了训练信息。

---

## 10. 本周完成标准

完成第二周学习后，应当可以独立解释以下问题：

1. Tensor 和 NumPy 数组有什么区别？
2. `shape`、`dtype`、`device` 分别是什么？
3. `requires_grad=True` 有什么作用？
4. 梯度是什么？
5. `loss.backward()` 做了什么？
6. 为什么每个 batch 前要清空梯度？
7. `optimizer.step()` 做了什么？
8. Dataset 和 DataLoader 分别负责什么？
9. batch size 与 epoch 分别是什么？
10. `nn.Module` 与 `forward()` 有什么关系？
11. `model.train()` 和 `model.eval()` 有什么区别？
12. 为什么验证时使用 `torch.no_grad()`？
13. 为什么 CrossEntropyLoss 前不需要手动 Softmax？
14. 为什么要区分训练集、验证集和测试集？
15. `best_model.pth` 和 `last_model.pth` 有什么区别？

同时，应当能够独立写出最基本的训练循环：

```python
for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    logits = model(images)
    loss = loss_fn(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 14. 本周总结

本周从 PyTorch 最基础的 Tensor 操作开始，依次学习了自动求导、自定义 Dataset、DataLoader、`nn.Module`、损失函数、优化器、训练模式和验证模式。

最终以 FashionMNIST 为实践任务，完成了：

```text
数据读取
→ 数据分批
→ 模型定义
→ 前向传播
→ 计算损失
→ 反向传播
→ 参数更新
→ 验证
→ 保存最佳模型
→ 加载模型
→ 测试集评估
→ 绘制实验曲线
```

