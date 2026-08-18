# Week 05 学习指南

## 本周知识链

```text
Text → Token → Vocabulary → ID → Padding/Mask
→ Embedding → RNN → GRU/LSTM → Seq2Seq → Attention
```

第六周再进入：

```text
Self-Attention → Multi-Head Attention → Transformer → BERT/GPT
```

## 七天安排

### Day 1：文本预处理
学习 token、vocab、`<pad>`、`<unk>`、padding、length、mask。完成 `01_tokenization_vocab.ipynb`。

### Day 2：Embedding 与 Word2Vec
学习 One-hot、`nn.Embedding`、Skip-Gram、CBOW、Negative Sampling。完成 `02_embedding_word2vec_lab.ipynb`。

### Day 3：RNN
学习 hidden state、时间步参数共享、BPTT、梯度消失/爆炸。完成 `03_rnn_from_scratch.ipynb`。

### Day 4：GRU / LSTM / BiLSTM
重点理解门控、cell state 和双向序列编码；开始训练 Mean Pooling 与 BiLSTM。

### Day 5：Seq2Seq
学习 Encoder、Decoder、`<bos>/<eos>`、Teacher Forcing、Masked Loss、Greedy/Beam Search。完成 `04_seq2seq_teacher_forcing.ipynb`。

### Day 6：Attention
学习 Q/K/V、score、Softmax、mask、Bahdanau Attention。完成 `05_attention_lab.ipynb` 并训练 BiLSTM+Attention。

### Day 7：统一实验
比较三个模型，选最佳模型，只在最后使用 test split，并做 Attention 可视化。

## 必须熟悉的 Shape

```text
Token IDs:       [B, L]
Embedding:       [B, L, E]
BiLSTM output:   [B, L, 2H]
Attention pool:  [B, 2H]
Classifier:      [B, 2]
```

NLP 代码遇到不理解的问题，优先检查 shape。
