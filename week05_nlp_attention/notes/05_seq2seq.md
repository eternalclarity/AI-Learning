# 05 Encoder-Decoder 与 Seq2Seq

## Seq2Seq

输入和输出都是序列，例如机器翻译。

```text
Source → Encoder → Context → Decoder → Target
```

## Encoder
把输入序列编码成上下文表示。

## Decoder
根据上下文逐步生成输出 token。

## 特殊 token

- `<bos>`：开始生成
- `<eos>`：结束生成
- `<pad>`：batch 补齐

## Teacher Forcing

训练时 Decoder 的上一步输入可以使用真实目标 token：

```text
Decoder input: <bos> I love NLP
Target:        I love NLP <eos>
```

## Masked Loss

`<pad>` 不应计算损失，可使用：

```python
nn.CrossEntropyLoss(ignore_index=pad_id)
```

## 推理

训练时能看到真实前一个 token；推理时只能把自己的预测作为下一步输入。

## Greedy vs Beam Search

Greedy 每步选择概率最大 token；Beam Search 同时保留多个候选序列。

## 为什么需要 Attention

早期 Seq2Seq 把整条输入压缩成固定 context vector，长序列信息瓶颈明显；Attention 允许 Decoder 动态查看 Encoder 的不同位置。
