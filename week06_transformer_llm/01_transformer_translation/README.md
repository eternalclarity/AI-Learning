# Project 01：从零实现 Transformer 机器翻译

这个项目的目标不是调用 `nn.Transformer`，而是把 Encoder-Decoder Transformer 的关键组件全部自己写一遍，并用英法翻译任务验证它真的能训练、解码和可视化 Attention。

## 你会真正实现

```text
MultiHeadAttention
Sinusoidal Positional Encoding
PositionWiseFFN
AddNorm
EncoderBlock
DecoderBlock
TransformerEncoderDecoder
Padding Mask
Causal Mask
Greedy Decode
BLEU
Cross-Attention Visualization
```

## 数据流

```text
Source: "go ."
↓ tokenize / vocab
[B,S]
↓
Encoder
↓
Memory [B,S,C]
↓
Decoder Input: <bos> va !
↓ masked self-attention
↓ cross-attention(memory)
↓
Logits [B,T,V_tgt]
↓
Target: va ! <eos>
```

## 运行

```powershell
python -m unittest discover -s tests -v
python smoke_test.py
python download_data.py
python prepare_data.py --max-examples 20000
python train.py --device cuda --epochs 30 --amp
python translate.py --sentence "go ." --device cuda
python evaluate.py --device cuda
python visualize_attention.py --sentence "i love you ." --device cuda
```

第一次建议把 `--max-examples` 改成 3000、`--epochs` 改成 2，先确保完整流程跑通，再正式训练。

## 三种 Mask

本项目明确区分：

```text
src_valid_mask: [B,S]
    Encoder / Cross-Attention 不看 source <pad>

tgt_valid_mask: [B,T]
    Decoder Self-Attention 不看 target <pad>

causal_mask: [T,T]
    Decoder 位置 t 不允许看未来位置 >t
```

## 为什么不直接用 `nn.Transformer`

学完本项目之后，再看 `nn.Transformer` 才应该觉得它是“把我已经会写的模块封装起来”，而不是一个黑盒。
