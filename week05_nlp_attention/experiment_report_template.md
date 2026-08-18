# Week 05 实验报告模板

## 1. 实验目的

比较以下三种文本分类策略：

1. Embedding + Mean Pooling
2. Embedding + BiLSTM
3. Embedding + BiLSTM + Attention

---

## 2. 数据集

- 数据集：Stanford Large Movie Review Dataset
- 任务：positive / negative 二分类
- 官方 train：25,000
- 官方 test：25,000
- 本项目将官方 train 再划分为 train / validation
- test 只用于最终一次评估

记录你实际的：

```text
train:
validation:
test:
vocab_size:
max_length:
```

---

## 3. 数据处理

说明：

```text
文本 → tokenizer → vocab → id → truncation → padding → mask
```

为什么词表只从训练集建立？

---

## 4. 三个模型

### Model A：Mean Pooling

写出 shape：

```text
[B,L]
→ [B,L,E]
→ [B,E]
→ [B,2]
```

### Model B：BiLSTM

```text
[B,L]
→ [B,L,E]
→ BiLSTM
→ [B,2H]
→ [B,2]
```

### Model C：BiLSTM + Attention

```text
[B,L]
→ [B,L,E]
→ [B,L,2H]
→ attention weights [B,L]
→ [B,2H]
→ [B,2]
```

---

## 5. 实验配置

| 参数 | 值 |
|---|---|
| seed | 42 |
| vocab size | |
| max length | |
| embedding dim | |
| hidden size | |
| batch size | |
| optimizer | Adam |
| learning rate | |
| epochs | |
| grad clip | |

---

## 6. 验证集结果

| Model | Params | Training Time | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Mean Pooling | | | | | | |
| BiLSTM | | | | | | |
| BiLSTM + Attention | | | | | | |

---

## 7. 模型选择

只根据 validation 结果选择最终模型。

说明：

- 为什么选它？
- 是否存在训练更慢但提升有限的情况？
- 参数量增长是否值得？

---

## 8. 最终测试

只有模型确定后再运行 test。

填写：

```text
Accuracy:
Precision:
Recall:
Specificity:
F1:
```

---

## 9. 错误分析

查看：

```text
*_test_predictions.csv
```

回答：

1. 哪些负面评论被预测成正面？
2. 哪些正面评论被预测成负面？
3. 是否存在否定词、反讽、长距离依赖问题？
4. 截断 `max_length` 是否可能丢失关键信息？

---

## 10. Attention 可视化

不要写：

```text
attention 高 = 该词就是模型决策的唯一原因
```

更准确：

```text
Attention 权重是模型信息汇聚过程中的一个可观察信号，
可以辅助分析模型更关注哪些 token。
```

---

## 11. 总结

重点回答：

1. Mean Pooling 忽略词序会带来什么限制？
2. BiLSTM 带来了什么？
3. Attention 又改变了什么？
4. 为什么下一步自然会进入 Self-Attention / Transformer？
