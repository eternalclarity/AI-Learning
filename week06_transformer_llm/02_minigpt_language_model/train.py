"""MiniGPT 训练：随机连续 block、AMP、AdamW、验证 Perplexity。"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import trange

from dataset import sample_batch
from model import GPT, GPTConfig
from tokenizer import CharTokenizer
from utils import resolve_device, save_json, set_seed

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def estimate_loss(model, tokens, batch_size, block_size, device, eval_batches: int, amp: bool):
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = sample_batch(tokens, batch_size, block_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            _, loss, _ = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-impl", choices=["manual", "sdpa"], default="sdpa")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    amp = bool(args.amp and device.type == "cuda")

    tokenizer = CharTokenizer.load(args.artifact_dir / "tokenizer.json")
    train_tokens = torch.load(args.artifact_dir / "train.pt", map_location="cpu", weights_only=True)
    val_tokens = torch.load(args.artifact_dir / "val.pt", map_location="cpu", weights_only=True)

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
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp else None

    ckpt_dir = args.output_dir / "checkpoints"
    result_dir = args.output_dir / "results"
    plot_dir = args.output_dir / "plots"
    for d in [ckpt_dir, result_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    history = []
    best_val = float("inf")
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in trange(1, args.max_steps + 1, desc="MiniGPT"):
        x, y = sample_batch(train_tokens, args.batch_size, args.block_size, device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            _, loss, _ = model(x, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % args.eval_interval == 0 or step == args.max_steps:
            train_loss = estimate_loss(model, train_tokens, args.batch_size, args.block_size, device, args.eval_batches, amp)
            val_loss = estimate_loss(model, val_tokens, args.batch_size, args.block_size, device, args.eval_batches, amp)
            row = {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_perplexity": math.exp(min(val_loss, 20.0)),
            }
            history.append(row)
            print(f"\nstep={step} train={train_loss:.4f} val={val_loss:.4f} ppl={row['val_perplexity']:.2f}")

            payload = {
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "step": step,
                "val_loss": val_loss,
            }
            torch.save(payload, ckpt_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(payload, ckpt_dir / "best.pt")

    elapsed = time.perf_counter() - started
    with (result_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    save_json({
        "parameters": model.num_parameters(),
        "best_val_loss": best_val,
        "training_seconds": elapsed,
        "device": str(device),
        "amp": amp,
        "config": config.__dict__,
    }, result_dir / "summary.json")

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
