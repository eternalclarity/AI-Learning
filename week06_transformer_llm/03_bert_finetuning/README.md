# Project 03：BERT Fine-tuning

这个项目专门练真实预训练模型的迁移学习，而不是再从零造一个小 BERT。

## 允许 Hugging Face 做什么

只负责：

```text
AutoTokenizer.from_pretrained(...)
AutoModelForSequenceClassification.from_pretrained(...)
```

即：**Tokenizer + 模型结构 + 预训练权重**。

## 不使用什么

```text
sklearn
Trainer
evaluate library
pandas
```

以下自己写：

```text
Dataset
Dynamic Padding Collator
DataLoader
Freeze/Unfreeze
AdamW
Warmup + Linear Decay
AMP
Gradient Accumulation
Gradient Clipping
Train/Validation Loop
Pure PyTorch Metrics
Checkpoint
Final Test
Confusion Matrix
```

## 数据

继续使用 IMDB，这样可以和 Week 05 的：

```text
BiLSTM
BiLSTM + Attention
```

直接比较。

## 三种策略

### 1. head_only

```text
BERT frozen
Classifier trainable
```

### 2. last_n

```text
最后 N 个 Encoder Layer
+ Pooler
+ Classifier
trainable
```

### 3. full

```text
所有参数 trainable
```

## 默认配置

```text
model_name = bert-base-uncased
max_length = 128
batch_size = 8
grad_accum_steps = 4
effective_batch = 32
epochs = 3
lr = 2e-5
weight_decay = 0.01
warmup_ratio = 0.1
AMP = on（CUDA）
```

如果 3060 Laptop 显存不足，优先：

```text
batch_size: 8 → 4
```

保留 gradient accumulation，而不是一开始就随意换掉整个模型。

## 运行

```powershell
python -m unittest discover -s tests -v
python download_data.py
python prepare_data.py
python smoke_test.py

python train.py --strategy head_only --device cuda --amp
python train.py --strategy last_n --unfreeze-last-n 3 --device cuda --amp
python train.py --strategy full --device cuda --amp

python compare_strategies.py
```

根据 Validation F1 选最终策略，再：

```powershell
python evaluate.py --strategy full --device cuda
```

最后试自己的文本：

```powershell
python predict.py --strategy full --text "This movie was surprisingly good." --device cuda
```

## Dynamic Padding

`Dataset` 返回 raw text；`collate_fn` 才调用 tokenizer：

```text
batch lengths = 47, 82, 61, 100
→ padding=True
→ 只补到当前 batch 最长 100
```

而不是整个 IMDB 全部补到 512。
