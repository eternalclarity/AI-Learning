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

## 学习顺序

| 顺序 | 文件                                                         | 学习重点                                                     |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [README.md](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\README.md) | 先建立全局认识：Decoder-only GPT、两种 Attention、KV Cache，以及完整运行顺序。 |
| 3    | [download_data.py:12](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\download_data.py) | 了解原始数据只是 Tiny Shakespeare 的连续文本，内容很简单，快速读完即可。 |
| 4    | [tokenizer.py:11](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\tokenizer.py) | 理解字符级词表、`<unk>`、`encode/decode`、词表保存与加载。重点注意词表只由训练文本创建。 |
| 5    | [prepare_data.py:16](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\prepare_data.py) | 串起数据预处理：90/10 连续切分 → 构造 tokenizer → 编码 → 保存 `train.pt`、`val.pt` 和 `tokenizer.json`。 |
| 6    | [dataset.py:8](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\dataset.py) | 理解语言模型监督信号：`x` 是连续 token，`y` 是整体右移一位的 next-token 标签。建议手算一个长度为 5 的例子。 |
| 7    | [model.py:84](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\model.py) | 第一遍先读 `GPT.forward()`，追踪 `[B,T] → [B,T,D] → [B,T,V] → loss`，暂时不要深入 Attention 数学。 |
| 9    | [attention.py:12](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\attention.py) | 项目最重要的文件。依次研究 QKV 切头、因果 mask、manual Attention、SDPA、`past_kv` 拼接和新 Cache 返回。 |
| 12   | [train.py:36](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\train.py) | 现在再读训练入口：加载制品 → 构造配置 → AdamW → AMP → 梯度裁剪 → 定期评估 → 保存 checkpoint。 |
| 13   | [utils.py:10](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\utils.py) | 辅助文件，配合 `train.py` 阅读即可：随机种子、设备选择、JSON 保存和 CUDA 同步。 |
| 14   | [generation.py:8](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\generation.py) | 先读 `sample_next()`，再比较 `generate_naive()` 与 `generate_cached()`。重点研究 Cache 满 `block_size` 后为什么必须重建滑动窗口。 |
| 15   | [generate.py:16](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\generate.py) | 理解实际推理入口：加载 tokenizer/checkpoint → 恢复 GPT → 编码 prompt → 选择生成函数 → 解码文本。 |
| 19   | [lora.py:12](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\lora.py) | 最后学习扩展内容。理解冻结基础 Linear，以及 `Wx + α/r·B(Ax)`；它目前没有接入主训练链路。 |

这个项目可以分成五层：数据准备、模型实现、训练、生成推理、测试与扩展。

```
download_data.py
        ↓
prepare_data.py ← tokenizer.py
        ↓
train.pt / val.pt / tokenizer.json
        ↓
dataset.py → train.py → model.py → attention.py
                         ↓
                    checkpoints
                         ↓
generate.py → generation.py → 生成文本
```

## 一、数据准备

### 1. `download_data.py`

[download_data.py](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\download_data.py)

职责：下载 Tiny Shakespeare 原始文本。

输入：

```
GitHub 上的 input.txt
```

输出：

```
data/raw/input.txt
```

关键行为：

- 默认下载到项目的 `data/raw`。
- 如果文件已经存在，就不会重复下载。
- 只负责获取原始文本，不做分词或切分。

它是整个项目的数据入口，但代码比较简单。

------

### 2. `tokenizer.py`

[tokenizer.py](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\tokenizer.py)

职责：实现字符级 Tokenizer。

核心类：

```
CharTokenizer
```

主要功能：

- `from_training_text()`：统计训练集里出现过的字符。
- `encode()`：字符转换成整数 ID。
- `decode()`：整数 ID 转换回字符。
- `save()`：保存为 `tokenizer.json`。
- `load()`：恢复 Tokenizer。

词表结构类似：

```
0       → <unk>
1       → 换行
2       → 空格
3       → !
...
```

值得注意：

- 词表只根据训练文本创建，避免验证集信息泄漏。
- 未知字符统一转换成 `unk_id=0`。
- 字符级 Tokenizer 很简单，方便把注意力放在 GPT 模型，而不是 BPE 工程上。

------

### 3. `prepare_data.py`

[prepare_data.py](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\prepare_data.py)

职责：把原始文本转换成训练制品。

执行流程：

```
input.txt
   ↓
按照 90% / 10% 连续切分
   ↓
用 train_text 创建 CharTokenizer
   ↓
encode
   ↓
train.pt + val.pt + tokenizer.json
```

输出：

```
artifacts/
├── tokenizer.json
├── train.pt
├── val.pt
└── summary.json
```

这里采用连续切分，而不是随机打乱字符，因为语言模型需要保留原始文本顺序。

`summary.json` 记录：

- 训练字符数
- 验证字符数
- 词表大小
- 切分比例

------

### 4. `dataset.py`

[dataset.py:8](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\dataset.py)

职责：从一维 Token 序列中随机采样训练 batch。

核心函数：

```
sample_batch(tokens, batch_size, block_size, device)
```

假设一段文本编码后是：

```
[10, 20, 30, 40, 50, 60]
```

当 `block_size=4` 时，一组样本可以是：

```
x = [10, 20, 30, 40]
y = [20, 30, 40, 50]
```

模型在每个位置预测下一个 Token：

```
10 → 20
20 → 30
30 → 40
40 → 50
```

最终输出形状：

```
x: [B,T]
y: [B,T]
```

这也是 next-token CrossEntropyLoss 的监督信号来源。

## 二、模型核心

### 5. `attention.py`

[attention.py:12](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\attention.py)

职责：实现因果多头自注意力，是整个项目最重要的底层文件。

核心类：

```
CausalSelfAttention
```

内部结构：

```
输入 x [B,T,D]
    ↓
qkv Linear
    ↓
Q、K、V [B,H,T,head_dim]
    ↓
Causal Attention
    ↓
合并多头 [B,T,D]
    ↓
out_proj + dropout
```

它提供两个后端：

#### Manual Attention

手动计算：

```
scores = QKᵀ / √head_dim
scores → causal mask
weights = softmax(scores)
output = weights @ V
```

优点：

- 能直接理解公式。
- 能通过 `last_weights` 观察注意力权重。
- 适合教学和调试。

#### SDPA

调用：

```
torch.nn.functional.scaled_dot_product_attention
```

优点：

- 代码更简洁。
- 通常能利用 PyTorch 的底层优化。
- 推荐用于实际训练。

#### KV Cache

如果传入旧 Cache：

```
past_k, past_v
```

就会和当前 Token 产生的新 K/V 拼接：

```
K = concat(past_K, new_K)
V = concat(past_V, new_V)
```

Cache 形状是：

```
[B,H,T,head_dim]
```

不是历史 hidden state。

------

### 6. `model.py`

[model.py](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\model.py)

职责：组装完整 Decoder-only GPT。

这个文件包含四个核心结构。

#### `GPTConfig`

[model.py:15](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\model.py)

保存模型超参数：

```
vocab_size
block_size
d_model
num_heads
num_layers
d_ff
dropout
attention_impl
```

#### `FeedForward`

结构：

```
Linear(D,d_ff)
→ GELU
→ Linear(d_ff,D)
→ Dropout
```

它负责对每个 Token 的表示进行非线性变换。

#### `Block`

一个 GPT Block：

```
x
├─ LayerNorm → CausalSelfAttention → 残差相加
└─ LayerNorm → FeedForward        → 残差相加
```

代码采用 Pre-LN：

```
x = x + self.attn(self.ln1(x))
x = x + self.ffn(self.ln2(x))
```

#### `GPT`

[model.py:60](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\model.py)

完整数据流：

```
input_ids [B,T]
↓
Token Embedding + Position Embedding
↓
GPT Block × N
↓
Final LayerNorm
↓
LM Head
↓
logits [B,T,V]
```

重要实现：

- 使用 Learned Position Embedding。
- LM Head 和 Token Embedding 权重共享。
- 有 targets 时直接计算 CrossEntropyLoss。
- 支持 `past_key_values` 和 `use_cache`。
- 防止位置超过 `block_size`。

------

### 7. `lora.py`

[lora.py:12](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\lora.py)

职责：实现可选的 LoRA Linear。

公式：

```
y = Wx + α/r · B(Ax)
```

实现特点：

- 冻结原始 `base Linear`。
- 只训练低秩矩阵 `A` 和 `B`。
- `B` 初始化为零，因此刚创建时输出和原始 Linear 完全一致。
- 当前没有接入 `train.py`，属于学习扩展。

可以在掌握主模型后，尝试用它替换 Attention 的 QKV 或输出投影。

## 三、训练系统

### 8. `train.py`

[train.py:36](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\train.py)

职责：组织完整训练流程。

执行顺序：

```
设置随机种子和设备
↓
加载 tokenizer、train.pt、val.pt
↓
创建 GPTConfig 和 GPT
↓
创建 AdamW
↓
循环采样 batch
↓
forward → loss → backward
↓
梯度裁剪 → optimizer.step
↓
定期验证
↓
保存 checkpoint、CSV、图表
```

主要功能：

- CUDA AMP 混合精度。
- AdamW 优化器。
- 梯度裁剪。
- 定期计算训练和验证 loss。
- 根据验证 loss 计算 perplexity。
- 保存 `last.pt` 和 `best.pt`。
- 保存训练历史和 loss 曲线。

Checkpoint 内容：

```
{
    "model_state_dict": ...,
    "config": ...,
    "step": ...,
    "val_loss": ...
}
```

#### `estimate_loss()`

[train.py:24](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\train.py)

在多个随机 batch 上计算平均 loss：

- 使用 `torch.no_grad()`。
- 临时切换到 `model.eval()`。
- 计算结束后恢复 `model.train()`。

------

### 9. `utils.py`

[utils.py](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\utils.py)

职责：提供训练和 benchmark 共用的辅助函数。

包含：

- `set_seed()`：设置 Python 和 PyTorch 随机种子。
- `resolve_device()`：处理 `auto`、`cpu`、`cuda`。
- `save_json()`：保存结果文件。
- `synchronize()`：进行 CUDA 同步，保证 benchmark 计时准确。

它不包含核心算法，阅读其他入口文件时遇到再看即可。

## 四、生成与推理

### 10. `generation.py`

[generation.py](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\generation.py)

职责：实现 Token 采样和自回归生成。

#### `sample_next()`

[generation.py:8](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\generation.py)

支持：

- Greedy：直接选择最大概率 Token。
- Temperature：控制分布随机程度。
- Top-k：只保留概率最高的 k 个 Token。
- Multinomial：按概率随机采样。

#### `generate_naive()`

[generation.py:24](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\generation.py)

每生成一个 Token，都重新计算整个上下文：

```
完整上下文 → GPT → 新 Token
完整上下文+新Token → GPT → 下一个 Token
```

优点是逻辑简单，缺点是重复计算很多。

#### `generate_cached()`

[generation.py:36](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\generation.py)

第一次输入完整 prompt，后续只输入最新 Token：

```
prompt → GPT → KV Cache
最新 token + KV Cache → GPT → 更新 Cache
```

当 Cache 长度达到 `block_size` 时，会清空 Cache，并使用最后一个滑动窗口重新构建。

这是因为模型使用 Learned Position Embedding，位置索引不能超过 `block_size`。

------

### 11. `generate.py`

[generate.py:16](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\generate.py)

职责：提供命令行推理入口。

流程：

```
加载 tokenizer
↓
加载 checkpoint
↓
通过 checkpoint config 重建 GPT
↓
加载 model_state_dict
↓
编码 prompt
↓
选择 naive 或 cached
↓
生成 Token
↓
decode 并打印文本
```

它本身没有实现生成算法，而是负责把模型、Tokenizer 和 `generation.py` 连接起来。

## 五、测试文件

### 12. `smoke_test.py`

[smoke_test.py:9](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\smoke_test.py)

职责：快速验证整个核心链路是否能运行。

检查：

- GPT forward 输出形状。
- CrossEntropyLoss 能计算。
- `loss.backward()` 能执行。
- Greedy 模式下 naive/cache 生成完全一致。

它使用很小的随机模型和随机数据，不需要下载训练集。

------

### 13. `tests/__init__.py`

[tests/__init__.py](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\tests\\__init__.py)

空文件，用于把 `tests` 标记为 Python 包。

不包含测试逻辑。

------

### 14. `tests/test_attention.py`

[tests/test_attention.py](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\tests\\test_attention.py)

验证两件事：

1. Manual 和 SDPA 在相同权重、相同输入下输出近似一致。
2. KV Cache 形状随新 Token 正确增长。

例如：

```
第一次 Cache: [2,4,5,8]
增加一个 Token 后: [2,4,6,8]
```

------

### 15. `tests/test_kv_cache.py`

[tests/test_kv_cache.py](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\tests\\test_kv_cache.py)

验证：

```
完整序列一次 forward 的 logits
≈
逐 Token 使用 KV Cache 得到的 logits
```

而且分别验证：

- Manual Attention
- SDPA Attention

它是 KV Cache 正确性的核心测试。

------

### 16. `tests/test_generation.py`

[tests/test_generation.py](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\tests\\test_generation.py)

验证 greedy 模式下：

```
generate_naive()
==
generate_cached()
```

使用 greedy 是为了排除随机采样干扰。如果使用 multinomial，即使概率分布相同，也可能采到不同 Token。

------

### 17. `tests/test_lora.py`

[tests/test_lora.py](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\tests\\test_lora.py)

验证：

- LoRA 刚初始化时输出与原 Linear 相同。
- 原 Linear 参数被冻结。
- LoRA 的 A、B 参数仍然可训练。

## 六、性能测试

### 18. `benchmark_attention.py`

[benchmark_attention.py](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\benchmark_attention.py)

职责：比较 Manual Attention 和 SDPA。

先验证：

```
max_abs_error
```

再测试：

```
manual_ms
sdpa_ms
speedup
```

使用 `torch.cuda.synchronize()` 是因为 CUDA 默认异步执行，不同步会导致计时不准确。

------

### 19. `benchmark_kv_cache.py`

[benchmark_kv_cache.py](D:\\code.py\\workspace\\AI-Learning\\week06_transformer_llm\\02_minigpt_language_model\\benchmark_kv_cache.py)

职责：比较 naive 和 cached 生成。

输出：

- 生成结果是否一致
- 两种方法耗时
- tokens/s
- 加速比
- CUDA 峰值显存

结果保存在：

```
outputs/results/kv_cache_benchmark.json
```

这个项目模型很小，所以短序列下 Cache 的管理开销可能抵消节省的计算量；序列和模型规模增大后，KV Cache 的优势才更明显。

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

## 扩展

项目附带 `lora.py`，用于理解：

```text
Wx + scale · B(Ax)
```

默认不接入主训练，目的是让核心 GPT/KV Cache 路线保持清晰；完成主项目后再尝试替换 Attention projection。
