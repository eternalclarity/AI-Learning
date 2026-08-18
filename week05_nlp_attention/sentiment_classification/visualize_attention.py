"""可视化 BiLSTM + Attention 模型关注的 token。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .data import (
    IMDBDataset,
    Vocab,
    collate_batch,
    load_split_manifest,
)
from .utils import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    create_model,
    resolve_device,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=25)
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
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    device = resolve_device(args.device)

    checkpoint_path = (
        args.output_dir
        / "checkpoints"
        / "bilstm_attention"
        / "best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "找不到 bilstm_attention/best.pt，请先训练 Attention 模型。"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    vocab = Vocab.load(args.artifact_dir / "vocab.json")
    splits = load_split_manifest(args.artifact_dir / "splits.json")

    dataset = IMDBDataset(
        data_dir=args.data_dir,
        records=splits[args.split],
        vocab=vocab,
        max_length=int(checkpoint["train_config"]["max_length"]),
    )

    sample = dataset[args.sample_index]
    batch = collate_batch([sample])

    input_ids = batch["input_ids"].to(device)
    lengths = batch["lengths"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    model = create_model(
        model_name=checkpoint["model_name"],
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits, weights = model(
            input_ids=input_ids,
            lengths=lengths,
            attention_mask=attention_mask,
            return_attention=True,
        )
        probability = torch.softmax(logits, dim=1)[0]

    true_length = int(lengths[0].item())
    ids = input_ids[0, :true_length].cpu().tolist()
    tokens = vocab.decode(ids)
    weights = weights[0, :true_length].cpu()

    top_k = min(args.top_k, true_length)
    top_values, top_indices = torch.topk(weights, k=top_k)

    top_tokens = [tokens[index] for index in top_indices.tolist()]

    print(f"True label: {sample['label']}")
    print(f"P(positive): {float(probability[1]):.4f}")
    print("\nTop attention tokens:")

    for token, weight in zip(top_tokens, top_values.tolist()):
        print(f"{token:<20} {weight:.6f}")

    top_tokens = top_tokens[::-1]
    top_values_np = top_values.numpy()[::-1]

    plt.figure(figsize=(8, max(5, top_k * 0.25)))
    plt.barh(top_tokens, top_values_np)
    plt.xlabel("Attention Weight")
    plt.title("Top Attention Tokens")
    plt.tight_layout()

    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        plot_dir
        / f"bilstm_attention_{args.split}_{args.sample_index}_attention.png"
    )
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
