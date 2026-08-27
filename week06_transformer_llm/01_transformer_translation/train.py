"""训练从零实现的 Encoder-Decoder Transformer。"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from data import PAD_ID, TranslationDataset, create_loader, load_artifacts
from masks import make_valid_mask
from model import Transformer, TransformerConfig
from utils import count_parameters, resolve_device, save_json, set_seed

ROOT = Path(__file__).resolve().parent


def run_epoch(model, loader, optimizer, device, amp: bool, scaler, grad_clip: float):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, desc="train" if training else "val"):
            src = batch["src"].to(device, non_blocking=True)
            tgt_full = batch["tgt"].to(device, non_blocking=True)
            tgt_input = tgt_full[:, :-1]
            tgt_label = tgt_full[:, 1:]
            src_valid = make_valid_mask(src, PAD_ID)
            tgt_valid = make_valid_mask(tgt_input, PAD_ID)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(src, tgt_input, src_valid, tgt_valid)
                loss_sum = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_label.reshape(-1))
                valid_tokens = tgt_label.ne(PAD_ID).sum()
                loss = loss_sum / valid_tokens.clamp_min(1)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            total_loss += float(loss_sum.detach().item())
            total_tokens += int(valid_tokens.item())

    return total_loss / max(total_tokens, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    splits, src_vocab, tgt_vocab, _ = load_artifacts(args.artifact_dir)
    train_ds = TranslationDataset(splits["train"], src_vocab, tgt_vocab, args.max_length)
    val_ds = TranslationDataset(splits["val"], src_vocab, tgt_vocab, args.max_length)
    train_loader = create_loader(train_ds, args.batch_size, True, args.num_workers, device.type == "cuda")
    val_loader = create_loader(val_ds, args.batch_size, False, args.num_workers, device.type == "cuda")

    config = TransformerConfig(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=max(args.max_length, 64),
        pad_id=PAD_ID,
    )
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None

    ckpt_dir = args.output_dir / "checkpoints"
    result_dir = args.output_dir / "results"
    plot_dir = args.output_dir / "plots"
    for d in [ckpt_dir, result_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    history = []
    best_val = float("inf")
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, amp_enabled, scaler, args.grad_clip)
        val_loss = run_epoch(model, val_loader, None, device, amp_enabled, None, args.grad_clip)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": math.exp(min(val_loss, 20.0)),
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(f"Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f} | ppl={row['val_perplexity']:.2f}")

        payload = {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "epoch": epoch,
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
        "parameters": count_parameters(model),
        "best_val_loss": best_val,
        "training_seconds": elapsed,
        "device": str(device),
        "amp": amp_enabled,
        "config": config.__dict__,
    }, result_dir / "summary.json")

    epochs = [r["epoch"] for r in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["train_loss"] for r in history], label="train")
    plt.plot(epochs, [r["val_loss"] for r in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Token Cross Entropy")
    plt.title("Transformer Translation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_curve.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
