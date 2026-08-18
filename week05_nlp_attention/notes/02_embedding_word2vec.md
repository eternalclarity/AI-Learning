# 02 Embedding 与 Word2Vec

## Token ID 没有语义

```text
cat=10, dog=11
```

不表示 dog 比 cat 大 1。

## One-hot 的问题

词表 `V=20000` 时，每个词需要 20000 维稀疏向量，而且不同词之间无法直接表达语义相似性。

## `nn.Embedding`

```python
nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
```

权重：

```text
[V, E]
```

输入：

```text
[B,L]
```

输出：

```text
[B,L,E]
```

本质：

```text
token_id → embedding.weight[token_id]
```

即一个可训练查找表。

## Word2Vec

### Skip-Gram
中心词预测上下文。

### CBOW
上下文预测中心词。

### Negative Sampling
完整 Softmax 每次需要与整个词表比较，代价高；负采样把训练转换为少量正负二分类任务。

## 静态词向量局限

Word2Vec 中同一个词通常只有一个固定向量，无法充分表达一词多义。后续 BERT 会根据上下文生成动态表示。

本周掌握原理即可，不需要大规模复现 Word2Vec 训练。
