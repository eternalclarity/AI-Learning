# 01 NLP 文本预处理

## 1. 为什么文本必须预处理

神经网络不能直接计算字符串：

```text
"I really love this movie"
```

基本流程：

```text
Text → Token → ID → Tensor
```

## 2. Token

最简单的 word-level tokenizer：

```text
"I love deep learning"
→ ["i", "love", "deep", "learning"]
```

Token 不一定等于完整单词；现代 BERT/GPT 常使用 subword。

## 3. Vocabulary

建立映射：

```text
<pad> → 0
<unk> → 1
i     → 2
love  → 3
```

于是：

```text
["i","love"] → [2,3]
```

## 4. `<unk>`

训练词表里没有的新词统一映射成 `<unk>`，避免推理时动态修改词表。

## 5. Padding

句子长度不一样，batch 需要补齐：

```text
[2,3,4,0,0]
[2,8,3,9,10]
```

最终 shape：

```text
[B, L]
```

## 6. Length

```text
[2,3,4,0,0] → length=3
```

RNN 可以利用真实长度跳过 padding。

## 7. Mask

```text
tokens: [2,3,4,0,0]
mask:   [1,1,1,0,0]
```

Pooling/Attention 根据 mask 排除 padding。

## 8. 三者区别

| 名称 | 作用 |
|---|---|
| Padding | 把变长数据补成矩阵 |
| Length | 告诉 RNN 真实序列长度 |
| Mask | 告诉 Attention/Pooling 哪些位置无效 |

## 9. 防止数据泄漏

Vocabulary 只能根据训练子集建立。验证和测试中的未知词使用 `<unk>`。

本项目对应实现：`sentiment_classification/data.py`。
