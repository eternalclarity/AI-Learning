# VALIDATION REPORT

交付前完成以下验证。

## 1. Python 语法

```text
python -m compileall -q .
```

通过。

## 2. 冒烟测试

```text
python -m sentiment_classification.smoke_test
```

已检查：

- Mean Pooling 前向/反向
- BiLSTM 前向/反向
- BiLSTM + Attention 前向/反向
- CrossEntropyLoss
- Attention padding mask
- Attention 权重和约为 1

通过。

## 3. 单元测试

```text
python -m unittest discover -s sentiment_classification/tests -v
```

7 项通过：

- tokenizer
- unknown token
- collate / padding / mask
- Mean Pooling shape
- BiLSTM shape
- Attention shape / mask
- stratified split reproducibility

## 4. Notebook

5 个 Notebook 均完成：

- nbformat 读取
- Jupyter execute 实际执行

全部通过。

## 5. 端到端伪数据实验

构造小型正负评论数据后实际走通：

```text
发现样本
→ 分层 train/validation
→ 只用 train 建 vocabulary
→ DataLoader
→ 三模型训练
→ best/last checkpoint
→ 重新加载最佳模型
→ test 指标
→ test predictions CSV
→ confusion matrix
→ Attention token visualization
```

全部通过。

## 6. 正式 IMDB 训练

没有在交付环境中预先跑完整 50,000 条评论的正式结果。

正式实验应该在你的 GPU 环境重新运行，避免把别的环境结果误当成你的实验结果。

因此最终项目中的：

```text
data/
artifacts/
outputs/
```

只保留目录结构。
