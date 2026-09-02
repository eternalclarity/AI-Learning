"""
MiniGPT 的文本生成入口
加载 tokenizer 和训练好的 best.pt → 编码 prompt → 选择普通生成或 KV Cache 生成 → 控制 temperature / top-k / greedy → 最后把 token ID 解码成人类可读文本。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer
from utils import resolve_device

ROOT = Path(__file__).resolve().parent


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None, greedy: bool = False):
    """根据 logits 选择下一个 token。选最大 or 从前 K大中随机选一个 """

    # Greedy (贪心解码)模式或 temperature<=0 时，直接选择最大 logits
    if greedy or temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    # 用 temperature 调整 logits 的尖锐程度, temperature越小，概率更尖锐，更确定； temperature越大，概率更平坦，更随机
    logits = logits / temperature

    # 如果启用了 Top-K Sampling: 只保留概率最高的前 K 个 token，其他 token 全部禁止采样
    if top_k is not None:
        k = min(top_k, logits.size(-1))   # 防止 top_k 超过词表大小
        values, _ = torch.topk(logits, k)  # 找出 logits 最大的前 k 个值
        threshold = values[:, -1].unsqueeze(-1)  # 取第 k 大的 logits 作为保留阈值
        logits = logits.masked_fill(logits < threshold, float("-inf"))  # 将 Top-K 之外的 logits 设为负无穷,Softmax 之后为0

    probs = torch.softmax(logits, dim=-1)  # 将 logits 转换成概率分布
    return torch.multinomial(probs, num_samples=1)  # 按概率随机采样一个 token


@torch.no_grad()
def generate_naive(model, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None, greedy: bool = False):
    """ 普通自回归生成，每一步都重新计算整个上下文 """

    model.eval()
    ids = input_ids  # 保存当前完整 token 序列

    # 自回归生成 max_new_tokens 个 token
    for _ in range(max_new_tokens):
        context = ids[:, -model.config.block_size:]  # 三维切片，只保留所有 batch 最后 block_size 个 token 作为当前上下文
        logits, _, _ = model(context)   # 对整个上下文重新执行 GPT 前向传播
        next_id = sample_next(logits[:, -1, :], temperature, top_k, greedy) # 只使用最后一个位置的 logits 预测下一个 token
        ids = torch.cat([ids, next_id], dim=1)   # 将新 token 拼接到已有序列末尾

    return ids


@torch.no_grad()
def generate_cached(model, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None, greedy: bool = False):
    """ 使用 KV Cache 自回归生成，达到 block_size 后重建滑动窗口 """

    model.eval()
    ids = input_ids  # 保存当前完整 token 序列
    cache = None  # 初始时还没有 KV Cache

    for _ in range(max_new_tokens):
        if cache is None:
            context = ids[:, -model.config.block_size:]  # 取最后 block_size 个 token 作为完整上下文
            logits, _, cache = model(context, use_cache=True)  # 对完整上下文前向传播，同时建立 KV Cache -> KV方阵
        else:
            cache_len = cache[0][0].size(-2)  # 获取当前 KV Cache 已保存的 token 数量
            if cache_len >= model.config.block_size:    # 如果 Cache 已达到最大上下文长度
                cache = None    # 清空旧 Cache
                context = ids[:, -model.config.block_size:]   # 重新取最后一个上下文窗口
                logits, _, cache = model(context, use_cache=True)   # 重新计算该窗口并建立新的 KV Cache
            else:
                # Cache 已保存历史 K/V，只输入最新一个 token
                logits, _, cache = model(ids[:, -1:], past_key_values=cache, use_cache=True)  # KV长方阵

        next_id = sample_next(logits[:, -1, :], temperature, top_k, greedy)  # 使用最后位置 logits 选择下一个 token
        ids = torch.cat([ids, next_id], dim=1)  # 将新 token 拼接到完整序列末尾

    return ids



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="ROMEO:")   # 设置文本生成的初始提示词
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "checkpoints" / "best.pt")  # 设置模型 checkpoint 路径
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")  # 设置 tokenizer 等数据文件目录
    parser.add_argument("--max-new-tokens", type=int, default=300)  # 设置最多生成多少个新 token
    parser.add_argument("--temperature", type=float, default=0.8)  # 设置采样温度，控制生成随机性
    parser.add_argument("--top-k", type=int, default=40)  # 只保留概率最高的前 k 个 token 参与采样
    parser.add_argument("--greedy", action="store_true")  # 是否使用 Greedy Decoding
    parser.add_argument("--use-cache", action="store_true")  # 是否使用 KV Cache 加速生成
    parser.add_argument("--device", type=str, default="auto")  # 设置运行设备，auto 表示自动选择
    args = parser.parse_args()

    device = resolve_device(args.device)
    tokenizer = CharTokenizer.load(args.artifact_dir / "tokenizer.json")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    # 将 prompt 编码成 token ID，并增加 batch 维度 -> 最外层的中括号
    # shape = (1, prompt_len)
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    fn = generate_cached if args.use_cache else generate_naive   # 根据参数选择 KV Cache 或普通生成方式
    output = fn(
        model,
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
