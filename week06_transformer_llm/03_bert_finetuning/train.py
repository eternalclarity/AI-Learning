"""BERT Fine-tuning：Hugging Face 只加载 tokenizer/model；训练循环全部 PyTorch。"""

from __future__ import annotations

import argparse
import csv
import functools
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import IMDBTextDataset, create_loader, load_splits
from engine import evaluate_epoch, optimizer_steps_per_epoch, train_epoch
from strategies import apply_strategy, parameter_counts
from utils import linear_warmup_decay_lambda, resolve_device, save_json, set_seed

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["head_only", "last_n", "full"], required=True)
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--unfreeze-last-n", type=int, default=3)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "aclImdb")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    amp = bool(args.amp and device.type == "cuda")

    splits, _ = load_splits(args.artifact_dir / "splits.json")
    train_records = splits["train"][: args.max_train_samples] if args.max_train_samples else splits["train"]
    val_records = splits["val"][: args.max_val_samples] if args.max_val_samples else splits["val"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = IMDBTextDataset(args.data_dir, train_records)
    val_ds = IMDBTextDataset(args.data_dir, val_records)
    pad_multiple = 8 if device.type == "cuda" else None
    train_loader = create_loader(train_ds, tokenizer, args.batch_size, True, args.max_length, args.num_workers, device.type == "cuda", pad_multiple)
    val_loader = create_loader(val_ds, tokenizer, args.batch_size, False, args.max_length, args.num_workers, device.type == "cuda", pad_multiple)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    apply_strategy(model, args.strategy, args.unfreeze_last_n)
    counts = parameter_counts(model)
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    update_steps_per_epoch = optimizer_steps_per_epoch(len(train_loader), args.grad_accum_steps)
    total_steps = update_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    lr_lambda = functools.partial(
        linear_warmup_decay_lambda,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp else None

    ckpt_dir = args.output_dir / "checkpoints" / args.strategy
    result_dir = args.output_dir / "results"
    plot_dir = args.output_dir / "plots"
    for d in [ckpt_dir, result_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = []
    best_f1 = -1.0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            args.grad_accum_steps,
            args.grad_clip,
            amp,
            scaler,
        )
        val_metrics = evaluate_epoch(model, val_loader, device, amp)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch} | train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        payload = {
            "model_state_dict": model.state_dict(),
            "model_name": args.model_name,
            "strategy": args.strategy,
            "unfreeze_last_n": args.unfreeze_last_n,
            "max_length": args.max_length,
            "epoch": epoch,
            "val_f1": val_metrics["f1"],
        }
        torch.save(payload, ckpt_dir / "last.pt")
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(payload, ckpt_dir / "best.pt")

    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None

    history_path = result_dir / f"{args.strategy}_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    best_row = max(history, key=lambda x: x["val_f1"])
    summary = {
        "strategy": args.strategy,
        "model_name": args.model_name,
        **counts,
        "best_epoch": best_row["epoch"],
        "best_val_accuracy": best_row["val_accuracy"],
        "best_val_precision": best_row["val_precision"],
        "best_val_recall": best_row["val_recall"],
        "best_val_f1": best_row["val_f1"],
        "training_seconds": elapsed,
        "peak_memory_bytes": peak_memory,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "max_length": args.max_length,
        "lr": args.lr,
        "amp": amp,
    }
    save_json(summary, result_dir / f"{args.strategy}_summary.json")

    plt.figure(figsize=(8, 5))
    plt.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="train loss")
    plt.plot([r["epoch"] for r in history], [r["val_loss"] for r in history], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"BERT Fine-tuning: {args.strategy}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"{args.strategy}_loss.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
