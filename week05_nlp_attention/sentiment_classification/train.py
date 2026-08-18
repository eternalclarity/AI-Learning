"""训练 IMDB 三种情感分类模型。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from .data import (
    IMDBDataset,
    Vocab,
    create_dataloader,
    limit_records,
    load_split_manifest,
)
from .engine import run_epoch
from .utils import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    count_parameters,
    create_model,
    plot_training_history,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["mean_pooling", "bilstm", "bilstm_attention"],
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    return parser


def run_training(args: argparse.Namespace) -> dict:
    """训练并只根据 validation F1 保存最佳模型。"""
    set_seed(args.seed)
    device = resolve_device(args.device)

    vocab_path = args.artifact_dir / "vocab.json"
    split_path = args.artifact_dir / "splits.json"

    if not vocab_path.exists() or not split_path.exists():
        raise FileNotFoundError(
            "找不到 vocab.json 或 splits.json。\n"
            "请先运行：python prepare_data.py"
        )

    vocab = Vocab.load(vocab_path)
    split_manifest = load_split_manifest(split_path)

    train_records = limit_records(
        split_manifest["train"],
        args.max_train_samples,
    )
    val_records = limit_records(
        split_manifest["val"],
        args.max_val_samples,
    )

    train_dataset = IMDBDataset(
        data_dir=args.data_dir,
        records=train_records,
        vocab=vocab,
        max_length=args.max_length,
    )
    val_dataset = IMDBDataset(
        data_dir=args.data_dir,
        records=val_records,
        vocab=vocab,
        max_length=args.max_length,
    )

    pin_memory = device.type == "cuda"

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model_config = {
        "vocab_size": len(vocab),
        "embedding_dim": args.embedding_dim,
        "hidden_size": args.hidden_size,
        "attention_dim": args.attention_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "pad_id": 0,
    }

    model = create_model(
        model_name=args.model,
        **model_config,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.CrossEntropyLoss()

    train_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }

    checkpoint_dir = args.output_dir / "checkpoints" / args.model
    result_dir = args.output_dir / "results"
    plot_dir = args.output_dir / "plots"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    best_epoch = -1
    history: list[dict] = []

    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            description=f"Train {epoch}/{args.epochs}",
        )

        val_metrics = run_epoch(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=None,
            grad_clip=None,
            description=f"Val {epoch}/{args.epochs}",
        )

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
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch

            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_f1=best_val_f1,
                model_name=args.model,
                model_config=model_config,
                train_config=train_config,
            )

        save_checkpoint(
            path=checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_f1=best_val_f1,
            model_name=args.model,
            model_config=model_config,
            train_config=train_config,
        )

    elapsed_seconds = time.perf_counter() - start_time

    pd.DataFrame(history).to_csv(
        result_dir / f"{args.model}_history.csv",
        index=False,
    )

    plot_training_history(
        history,
        plot_dir / args.model,
    )

    best_row = max(history, key=lambda item: item["val_f1"])

    summary = {
        "model": args.model,
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_row["val_accuracy"],
        "best_val_precision": best_row["val_precision"],
        "best_val_recall": best_row["val_recall"],
        "best_val_f1": best_row["val_f1"],
        "training_seconds": elapsed_seconds,
        "model_config": model_config,
        "train_config": train_config,
    }

    save_json(
        summary,
        result_dir / f"{args.model}_summary.json",
    )

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation F1: {best_val_f1:.4f}")

    return summary


def main() -> None:
    args = build_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
