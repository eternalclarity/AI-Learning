# 03 RNN：让模型拥有序列记忆

## 核心思想

当前 hidden state 同时依赖：

```text
当前输入 x_t
+
上一时刻 h_(t-1)
```

基本形式：

```text
h_t = tanh(x_t W_xh + h_(t-1) W_hh + b_h)
```

## Hidden State

`h_t` 是模型到当前时间步为止，对历史信息的压缩表示。

## 参数共享

每个时间步使用同一组 `W_xh/W_hh/b_h`，所以序列变长不会增加新的时间步参数。

## PyTorch Shape

```python
nn.RNN(input_size=128, hidden_size=256, batch_first=True)
```

```text
input:  [B,L,128]
output: [B,L,256]
h_n:    [num_layers,B,256]
```

## BPTT

RNN 沿时间展开后反向传播，称 Backpropagation Through Time。

长链乘法会导致：

- 梯度消失
- 梯度爆炸

## Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

主要抑制梯度爆炸，不能根治长期依赖。

## 为什么仍值得学

```text
RNN → 长距离依赖困难 → GRU/LSTM → Attention → Transformer
```

这是理解现代序列模型演进最自然的路线。
