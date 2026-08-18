# 06 Attention：动态选择重要信息

## 1. 核心流程

```text
Query 与 Keys
→ score
→ Softmax
→ attention weights
→ 对 Values 加权求和
→ output
```

## 2. Q/K/V 直觉

- Query：我现在想找什么？
- Key：每条信息用什么特征来被匹配？
- Value：匹配后真正取出的信息是什么？

## 3. Dot-Product Attention

```text
score(q,k)=q·k
```

Scaled Dot-Product：

```text
score(q,k)=q·k/sqrt(d_k)
```

缩放可以避免高维点积过大导致 Softmax 过度尖锐。

## 4. Additive/Bahdanau Attention

```text
score(q,k)=v^T tanh(W_q q + W_k k)
```

适合 Query/Key 维度不同或需要可学习非线性评分。

## 5. Mask

```python
scores = scores.masked_fill(~mask, -1e9)
weights = softmax(scores)
```

padding 位置权重接近 0。

## 6. 本项目 Attention Pooling

```text
BiLSTM outputs H: [B,L,2H]
learned query
→ attention weights [B,L]
→ Σ α_t h_t
→ context [B,2H]
→ Linear
```

## 7. Attention 权重不是完整因果解释

权重可以帮助诊断模型关注的位置，但不能简单等价于“这个词就是模型预测的唯一原因”。

## 8. 与 Transformer 的衔接

第五周：RNN + Attention。

第六周：去掉 RNN，让 token 直接通过 Self-Attention 相互交互，再扩展成 Multi-Head Attention 和 Transformer。
