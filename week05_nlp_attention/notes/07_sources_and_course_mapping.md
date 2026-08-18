# 07 D2L 课程映射与项目扩展说明

这份文件区分两类内容：

1. **D2L 教材直接支持的理论主线**
2. **本项目为了完整实践而增加的工程扩展**

## 一、D2L 直接对应内容

### 第 8 章：循环神经网络

- 8.1 序列模型
- 8.2 文本预处理
- 8.3 语言模型和数据集
- 8.4 循环神经网络
- 8.5 RNN 从零开始实现
- 8.6 RNN 简洁实现
- 8.7 通过时间反向传播

对应：

```text
notes/01_nlp_text_preprocessing.md
notes/03_rnn.md
notebooks/01_tokenization_vocab.ipynb
notebooks/03_rnn_from_scratch.ipynb
```

### 第 9 章：现代循环神经网络

- 9.1 GRU
- 9.2 LSTM
- 9.4 双向循环神经网络
- 9.5 机器翻译与数据集
- 9.6 Encoder-Decoder
- 9.7 Seq2Seq
- 9.8 Beam Search

对应：

```text
notes/04_gru_lstm.md
notes/05_seq2seq.md
notebooks/04_seq2seq_teacher_forcing.ipynb
models/bilstm.py
```

### 第 10 章：注意力机制

本周重点：

- 10.1 注意力提示
- 10.2 注意力汇聚
- 10.3 注意力评分函数
- 10.4 Bahdanau 注意力

10.3 中把注意力输出描述为 Values 的加权和，并使用 masked softmax 排除无效序列位置。

对应：

```text
notes/06_attention.md
notebooks/05_attention_lab.ipynb
models/attention.py
models/bilstm_attention.py
```

以下留到 Week 06：

```text
10.5 Multi-Head Attention
10.6 Self-Attention + Positional Encoding
10.7 Transformer
```

### 第 14 章：NLP 预训练

本周只学习 Word2Vec 基础：

- 14.1 Word2Vec
- 14.2 近似训练 / Negative Sampling
- 14.3 Word2Vec 数据集
- 14.4 预训练 Word2Vec

对应：

```text
notes/02_embedding_word2vec.md
notebooks/02_embedding_word2vec_lab.ipynb
```

BERT 留到 Week 06。

### 第 15 章：NLP 应用

主项目最接近：

- 15.1 情感分析及数据集
- 15.2 情感分析：使用循环神经网络

---

## 二、本项目额外增加的工程实践

以下内容是为了形成规范项目而增加，并非声称逐行复现 D2L。

### 1. 三模型公平比较

```text
Mean Pooling
BiLSTM
BiLSTM + Attention
```

对应：

```text
忽略词序
→ 显式序列建模
→ 动态注意力汇聚
```

### 2. 官方 test split 封存

只根据 validation 选择模型，test 最后使用一次。

### 3. Vocabulary 只从训练子集建立

防止 validation/test 信息进入训练准备阶段。

### 4. PackedSequence

使用真实 sequence length，避免 LSTM 把 padding 当普通 token 继续递推。

### 5. 多项分类指标

使用 Accuracy / Precision / Recall / Specificity / F1，延续第三周模型评价能力。

### 6. Attention 可视化

用于观察信息汇聚行为；Attention 权重不能简单等价成完整因果解释。

---

## 三、一手资料链接

D2L 中文教材：
https://zh.d2l.ai/

PyTorch Embedding：
https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html

PyTorch LSTM：
https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html

Packed Sequence：
https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html

CrossEntropyLoss：
https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

Stanford IMDB：
https://ai.stanford.edu/~amaas/data/sentiment/
