# Week 05：NLP 基础与序列建模——Embedding → RNN/LSTM → Attention

本周目标不是提前进入 Transformer，而是把 Transformer 之前最重要的 NLP 地基真正打牢：

```text
原始文本
  ↓
Tokenization
  ↓
Vocabulary
  ↓
Token ID
  ↓
Padding / Length / Mask
  ↓
Embedding
  ↓
RNN
  ↓
GRU / LSTM / BiLSTM
  ↓
Seq2Seq
  ↓
Attention
  ↓
Week 06：Self-Attention / Transformer / BERT / GPT
```

## D2L 对应主线

- 第 8 章：序列模型、文本预处理、语言模型、RNN、BPTT
- 第 9 章：GRU、LSTM、双向 RNN、Encoder-Decoder、Seq2Seq、Beam Search
- 第 10.1～10.4：Attention、评分函数、Bahdanau Attention
- 第 14.1～14.4：Word2Vec 与负采样
- 第 15.1～15.2：情感分析与 RNN

第 10.5～10.7 的 Multi-Head Attention、Self-Attention、Transformer 留到第六周。

## 项目结构

```text
week05_nlp_sequence_attention/
├── notes/
├── notebooks/
├── sentiment_classification/
│   ├── models/
│   └── tests/
├── data/
├── artifacts/
├── outputs/
├── download_data.py
├── prepare_data.py
├── run_core_experiments.py
├── requirements.txt
└── README.md
```

## 主项目：IMDB 情感分类

使用 Stanford Large Movie Review Dataset。

三个公平对比模型：

```text
Model A:
Embedding → Masked Mean Pooling → Linear

Model B:
Embedding → BiLSTM → Linear

Model C:
Embedding → BiLSTM → Additive Attention Pooling → Linear
```

三个模型共享 tokenizer、vocab、数据划分、max_length、embedding_dim、batch size、optimizer、学习率、epoch 和随机种子。

## 安装

```powershell
conda activate ai
cd D:\code.py\workspace\AI-Learning\week05_nlp_sequence_attention
pip install -r requirements.txt
```

## 下载数据

```powershell
python download_data.py
```

## 准备训练/验证/测试划分和词表

```powershell
python prepare_data.py --max-vocab-size 20000 --min-freq 2 --val-ratio 0.2 --seed 42
```

Vocabulary 只从训练子集构建，验证集和测试集不会参与词表统计。

## 先做测试

```powershell
python -m sentiment_classification.smoke_test
python -m unittest discover -s sentiment_classification/tests -v
```

## 快速试训练

```powershell
python -m sentiment_classification.train ^
    --model mean_pooling ^
    --epochs 1 ^
    --batch-size 64 ^
    --max-train-samples 2000 ^
    --max-val-samples 1000 ^
    --device cuda
```

## 正式训练

```powershell
python -m sentiment_classification.train --model mean_pooling --epochs 8 --batch-size 64 --device cuda
python -m sentiment_classification.train --model bilstm --epochs 8 --batch-size 64 --device cuda
python -m sentiment_classification.train --model bilstm_attention --epochs 8 --batch-size 64 --device cuda
```

或：

```powershell
python run_core_experiments.py --epochs 8 --batch-size 64 --device cuda
```

## 比较验证集结果

```powershell
python -m sentiment_classification.compare_models
```

## 最终测试

只对最终选中的模型执行，例如：

```powershell
python -m sentiment_classification.evaluate --model bilstm_attention --device cuda
```

## Attention 可视化

```powershell
python -m sentiment_classification.visualize_attention ^
    --split test ^
    --sample-index 0 ^
    --device cuda
```

## 推荐学习顺序

```text
notes/00
→ notebook 01
→ notes/01
→ notebook 02
→ notes/02
→ notes/03
→ notebook 03
→ notes/04
→ notes/05
→ notebook 04
→ notes/06
→ notebook 05
→ 主项目三个模型
→ 验证集比较
→ 测试集最终评估
→ Attention 可视化
```

## 第五周验收标准

应能独立解释：

1. token、vocab、token id
2. `<pad>`、`<unk>`
3. padding、length、mask
4. `nn.Embedding` 的查表本质
5. RNN hidden state
6. BPTT 与梯度消失/爆炸
7. GRU / LSTM 的门
8. LSTM `h_t` 与 `c_t`
9. BiLSTM 为什么适合分类
10. Encoder / Decoder / Teacher Forcing
11. Query / Key / Value
12. Attention mask
13. Attention 为什么是 Transformer 的直接前置知识

本周暂时不正式进入 Transformer、BERT、GPT、LoRA、RAG 和 Agent。
