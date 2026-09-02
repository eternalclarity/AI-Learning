"""
MiniGPT 项目的训练主程序：
负责加载数据、创建 GPT、AdamW 优化、AMP 混合精度训练、梯度裁剪、周期性验证、保存最佳模型以及绘制 Loss 曲线
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import trange  # tqdm 的 range，用于控制台显示训练进度条

from dataset import sample_batch
from model import GPT, GPTConfig
from tokenizer import CharTokenizer
from utils import resolve_device, save_json, set_seed

ROOT = Path(__file__).resolve().parent


@torch.no_grad()    # 验证时关闭梯度计算
def estimate_loss(model, tokens, batch_size, block_size, device, eval_batches: int, amp: bool):
    """ 随机采样多个 batch，计算平均 Loss """

    model.eval()    # 切换到评估模式，关闭 Dropout, BatchNorm
    losses = []     # 保存每个验证 batch 的 Loss

    # 随机评估 eval_batches 个 batch
    for _ in range(eval_batches):
        x, y = sample_batch(tokens, batch_size, block_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):   # 如果启用 AMP，则使用 float16 混合精度计算, 让 PyTorch 自动判断哪些运算适合用 FP16，哪些运算应该保持更高精度
            _, loss, _ = model(x, y)
        losses.append(float(loss.item()))

    model.train()   # 验证结束后切回训练模式
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")    # 预处理数据所在目录
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")    # 模型、结果和图片的输出目录
    parser.add_argument("--max-steps", type=int, default=5000)  # 最大训练步数
    parser.add_argument("--eval-interval", type=int, default=250)  # 每隔多少 step 验证一次 -> 相当于epoch
    parser.add_argument("--eval-batches", type=int, default=20)    # 每次验证随机采样多少个 batch
    parser.add_argument("--batch-size", type=int, default=32)   # 每个 batch 的样本数量
    parser.add_argument("--block-size", type=int, default=256)   # 每条训练序列的 token 数量
    parser.add_argument("--d-model", type=int, default=256)   # Transformer 隐藏维度
    parser.add_argument("--num-heads", type=int, default=8)  # Multi-Head Attention 的 head 数
    parser.add_argument("--num-layers", type=int, default=6)  # Transformer Block 层数
    parser.add_argument("--d-ff", type=int, default=1024)  # FFN 中间隐藏层维度
    parser.add_argument("--dropout", type=float, default=0.1)  # Dropout 概率
    parser.add_argument("--attention-impl", choices=["manual", "sdpa"], default="sdpa")   # Attention 实现：手写版或 PyTorch SDPA
    parser.add_argument("--lr", type=float, default=3e-4)  # AdamW 学习率
    parser.add_argument("--weight-decay", type=float, default=0.1)  # AdamW 权重衰减系数
    parser.add_argument("--grad-clip", type=float, default=1.0)  # 梯度裁剪的最大范数
    parser.add_argument("--seed", type=int, default=42)  # 随机种子
    parser.add_argument("--device", type=str, default="auto")  # 训练设备，auto 表示自动选择
    parser.add_argument("--amp", action="store_true")  # 是否开启 AMP 混合精度训练
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    amp = bool(args.amp and device.type == "cuda")   # 只有 CUDA 下才真正启用 AMP

    tokenizer = CharTokenizer.load(args.artifact_dir / "tokenizer.json")    # 加载训练阶段使用的 Tokenizer
    train_tokens = torch.load(args.artifact_dir / "train.pt", map_location="cpu", weights_only=True)    # 加载训练集 token ID，先放在 CPU
    val_tokens = torch.load(args.artifact_dir / "val.pt", map_location="cpu", weights_only=True)     # 加载验证集 token ID，先放在 CPU

    # 创建 GPT 模型配置
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attention_impl=args.attention_impl,
    )
    model = GPT(config).to(device)  # 创建 GPT 并移动到指定设备
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 创建 AdamW 优化器
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp else None  # AMP 模式下创建 GradScaler，防止 float16 梯度下溢

    ckpt_dir = args.output_dir / "checkpoints"  # 模型 checkpoint 保存目录
    result_dir = args.output_dir / "results"    # 数值结果保存目录
    plot_dir = args.output_dir / "plots"        # Loss 曲线保存目录
    for d in [ckpt_dir, result_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 保存每次验证得到的训练指标
    history = []
    best_val = float("inf")
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # 从 step=1 开始执行 max_steps 次训练
    for step in trange(1, args.max_steps + 1, desc="MiniGPT"):
        x, y = sample_batch(train_tokens, args.batch_size, args.block_size, device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            _, loss, _ = model(x, y)  # 前向传播，计算损失

        # 如果启用了 AMP, 因为 FP16 能表示的数值范围比 FP32 小，特别小的梯度可能直接变成 0, GradScaler 会暂时把 loss 放大
        if scaler is not None:
            scaler.scale(loss).backward()   # 放大 Loss 后反向传播，避免 float16 梯度下溢
            scaler.unscale_(optimizer)       # 恢复梯度真实尺度，便于后续梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)   # 将梯度总范数裁剪到 grad_clip 以内
            scaler.step(optimizer)  # 根据缩放后的梯度更新模型参数
            scaler.update()  # 动态调整下一轮的缩放因子
        else:
            loss.backward()  # 普通精度下直接反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 裁剪梯度，防止梯度爆炸
            optimizer.step()   # 更新模型参数
        optimizer.zero_grad(set_to_none=True)   # 清空本轮梯度，为下一步训练做准备

        # 第1步、每隔 eval_interval 步 、最后一步进行验证（相当于epoch)
        if step == 1 or step % args.eval_interval == 0 or step == args.max_steps:
            train_loss = estimate_loss(model, train_tokens, args.batch_size, args.block_size, device, args.eval_batches, amp)
            val_loss = estimate_loss(model, val_tokens, args.batch_size, args.block_size, device, args.eval_batches, amp)

            # 保存本次验证指标
            row = {
                "step": step,   # 当前训练步数
                "train_loss": train_loss,   # 训练集 Loss
                "val_loss": val_loss,   # 验证集 Loss
                "val_perplexity": math.exp(min(val_loss, 20.0)),    # 验证集困惑度 PPL = exp(loss)
            }
            history.append(row)
            print(f"\nstep={step} train={train_loss:.4f} val={val_loss:.4f} ppl={row['val_perplexity']:.2f}")

            # 整理需要保存到 checkpoint 的内容
            payload = {
                "model_state_dict": model.state_dict(),  # 当前模型参数
                "config": config.__dict__,  # 当前模型配置
                "step": step,   # 当前训练步数
                "val_loss": val_loss,   # 当前验证集 Loss
            }
            torch.save(payload, ckpt_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(payload, ckpt_dir / "best.pt")

    elapsed = time.perf_counter() - started   # 计算整个训练过程的总耗时

    # 创建 history.csv 文件
    with (result_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())    # 根据字典字段创建 CSV 写入器
        writer.writeheader()     # 写入 CSV 表头
        writer.writerows(history)   # 写入所有训练记录

    # 保存整个训练实验的摘要信息
    save_json({
        "parameters": model.num_parameters(),   # 模型参数量
        "best_val_loss": best_val,  # 最佳验证集 Loss
        "training_seconds": elapsed,    # 总训练耗时
        "device": str(device),  # 实际训练设备
        "amp": amp,     # 是否启用 AMP
        "config": config.__dict__,  # GPT 模型配置
    }, result_dir / "summary.json")

    # 创建 Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot([r["step"] for r in history], [r["train_loss"] for r in history], label="train")
    plt.plot([r["step"] for r in history], [r["val_loss"] for r in history], label="validation")
    plt.xlabel("Step")
    plt.ylabel("Cross Entropy")
    plt.title("MiniGPT Language Modeling Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_curve.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
