# 04 GRU、LSTM 与 BiLSTM

## 为什么要门控

普通 RNN 没有明确机制控制“记住、忘记、输出”多少信息。

## GRU

两个核心门：

- Update Gate：旧状态保留多少、新状态写入多少
- Reset Gate：计算新候选状态时参考多少过去

## LSTM

增加长期记忆通道 `c_t`。

三个核心门：

- Forget Gate：旧记忆删多少
- Input Gate：新信息写多少
- Output Gate：当前对外输出多少

PyTorch：

```python
output, (h_n, c_n) = lstm(x)
```

`h_n` 是 hidden state，`c_n` 是 cell state，二者不是同一个东西。

## BiLSTM

同时进行：

```text
左 → 右
右 → 左
```

每个位置输出通常是 `2H` 维。

适合文本分类，因为完整句子在编码时已经可见；不适合直接做严格自回归生成，因为 backward 方向会利用未来信息。

本项目用 `pack_padded_sequence` 避免 LSTM 把 `<pad>` 当作真实时间步继续处理。
