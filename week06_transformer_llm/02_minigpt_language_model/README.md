# Project 02：MiniGPT Language Model

这是 Week 06 的核心项目：**不用 `nn.Transformer`，自己实现 Decoder-only GPT，并把训练与推理阶段真正连起来。**

## 主线

```text
Text
↓ CharTokenizer
Token IDs [B,T]
↓
Token Embedding + Learned Position Embedding
↓
GPT Block × N
  ├─ Pre-LN Causal Multi-Head Self-Attention
  └─ Pre-LN FeedForward
↓
Final LayerNorm
↓
LM Head
↓
Logits [B,T,V]
↓
Next-token CrossEntropyLoss
```

## 两个 Attention 后端

```text
--attention-impl manual
```

自己计算：

```text
QKᵀ / sqrt(D) → causal mask → softmax → V
```

以及：

```text
--attention-impl sdpa
```

使用 `torch.nn.functional.scaled_dot_product_attention`。

两者共用同一套 QKV projection / GPT 结构，因此可以做公平 correctness + performance benchmark。

## KV Cache

实现：

```text
generate_naive()
generate_cached()
```

Cache 每层保存：

```text
K: [B,H,T,D]
V: [B,H,T,D]
```

而不是把“过去 hidden state”当成 Cache 的定义。

## 运行顺序

```powershell
python -m unittest discover -s tests -v
python smoke_test.py
python download_data.py
python prepare_data.py
python train.py --device cuda --attention-impl sdpa --amp
python generate.py --prompt "ROMEO:" --device cuda --use-cache
python benchmark_kv_cache.py --device cuda
python benchmark_attention.py --device cuda
```

先快速跑：

```powershell
python train.py --device cuda --max-steps 200 --eval-interval 100
```

再正式增加 steps。

## 推荐默认小模型

```text
block_size = 256
d_model = 256
num_heads = 8
num_layers = 6
d_ff = 1024
```

3060 Laptop 显存紧张时优先降低：

```text
batch_size
block_size
```

而不是随意改模型所有维度。

## 扩展

项目附带 `lora.py`，用于理解：

```text
Wx + scale · B(Ax)
```

默认不接入主训练，目的是让核心 GPT/KV Cache 路线保持清晰；完成主项目后再尝试替换 Attention projection。
