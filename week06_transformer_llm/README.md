# Week 06：Transformer

```text
                         Transformer
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
      Encoder-Decoder     Decoder-only     Encoder-only
             │                │                │
             ▼                ▼                ▼
        Translation          MiniGPT          BERT
             │                │                │
      Seq2Seq / Cross       Next-token      Pretrain →
        Attention           Prediction      Fine-tuning
        
week06_transformer_practice/
├── 01_transformer_translation/   # 从零实现 Encoder-Decoder Transformer
├── 02_minigpt_language_model/    # 从零实现 Decoder-only GPT + KV Cache
├── 03_bert_finetuning/           # Hugging Face 只加载 tokenizer/预训练权重，训练逻辑纯 PyTorch
├── notebooks/                    # 5 个原理/工程实验
└── README.md
```



# Week 06 文件导航

## Notebooks

| 文件                                 | 实验                                           |
| ------------------------------------ | ---------------------------------------------- |
| `01_mask_lab.ipynb`                  | Padding Mask vs Causal Mask                    |
| `02_multihead_attention_shape.ipynb` | `[B,T,C] → [B,H,T,D]` 完整 shape               |
| `03_manual_vs_sdpa.ipynb`            | 手写 Attention 与 PyTorch SDPA 数值对照        |
| `04_kv_cache_lab.ipynb`              | KV Cache 张量级最小实验                        |
| `05_huggingface_tokenizer_lab.ipynb` | 预训练 tokenizer、动态 padding、attention mask |

## Project 01：Transformer Translation

阅读顺序：

```text
masks.py
→ model.py / MultiHeadAttention
→ EncoderBlock
→ DecoderBlock
→ Transformer
→ train.py
→ inference.py
```

重点不是追求翻译 SOTA，而是把原始 Transformer 的结构真正写出来：

```text
Source Tokens
    ↓
Embedding + Position
    ↓
Encoder Block × N
    ↓
Encoder Memory
    ↓
Decoder Block × N
  ├─ Masked Self-Attention
  ├─ Cross-Attention
  └─ FFN
    ↓
Linear → Target Vocabulary
```

你会自己实现：

- Multi-Head Attention
- Sinusoidal Positional Encoding
- Add & Norm
- Position-wise FFN
- Encoder Block
- Decoder Block
- Source Padding Mask
- Target Padding Mask
- Causal Mask
- Greedy Decoding
- BLEU（纯 Python）
- Attention Map 可视化

`visualize_attention.py` 用最后一层 Cross-Attention 观察 Decoder Query 如何读取 Encoder K/V。

## Project 02：MiniGPT

阅读顺序：

```text
tokenizer.py
→ attention.py
→ model.py
→ generation.py
→ benchmark_kv_cache.py
```

`attention.py` 是本周最值得逐行读的代码之一；同一个接口中同时保留 manual / SDPA / KV Cache。

这是本周最值得深入的项目：

```text
Token IDs [B,T]
      ↓
Token Embedding + Position Embedding
      ↓
Decoder-only Transformer Block × N
      ↓
LayerNorm
      ↓
LM Head: d_model → vocab_size
      ↓
Logits [B,T,V]
      ↓
Next-token CrossEntropyLoss
```

你会实现：

- Manual Causal Self-Attention
- PyTorch SDPA 版本
- Multi-Head Attention shape 变化
- GPT Block
- Weight Tying
- Greedy / Temperature / Top-k Sampling
- Naive Generation
- KV Cache Generation
- Naive vs KV Cache benchmark
- Manual Attention vs SDPA correctness / speed benchmark
- 可选 LoRA Linear 扩展

## Project 03：BERT Fine-tuning

阅读顺序：

```text
data.py / DynamicPaddingCollator
→ strategies.py
→ engine.py
→ train.py
→ evaluate.py
```

重点不是 Hugging Face API，而是观察 `requires_grad`、trainable parameters、gradient accumulation 与显存/效果的关系。

这个项目故意不“从零训练 BERT”，而是学习真实工程最常见的：

```text
Pretrained BERT
      ↓
Classification Head
      ↓
IMDB Sentiment Classification
```

Hugging Face 只负责：

```text
AutoTokenizer
AutoModelForSequenceClassification
预训练权重下载
```

其余全部自己写：

```text
Dataset
Dynamic Padding Collator
DataLoader
Freeze / Unfreeze Strategy
AMP
Gradient Accumulation
AdamW
Training Loop
Validation
Checkpoint
Pure PyTorch Metrics
Final Test
```

比较三种微调策略：

```text
Head Only
Last N Encoder Layers
Full Fine-tuning
```

---

