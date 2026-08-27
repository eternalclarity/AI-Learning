"""绘制最后一层 Decoder Cross-Attention，帮助理解 Query 来自 Decoder、K/V 来自 Encoder。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data import BOS_ID, EOS_ID, PAD_ID, load_artifacts, tokenize
from masks import make_valid_mask
from model import Transformer, TransformerConfig
from utils import resolve_device

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "checkpoints" / "best.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-length", type=int, default=40)
    args = parser.parse_args()

    device = resolve_device(args.device)
    _, src_vocab, tgt_vocab, _ = load_artifacts(args.artifact_dir)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = Transformer(TransformerConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    src_tokens = tokenize(args.sentence)[: args.max_length - 1]
    src_ids = src_vocab.encode(src_tokens) + [EOS_ID]
    src = torch.tensor(src_ids, device=device).unsqueeze(0)
    src_valid = make_valid_mask(src, PAD_ID)
    memory = model.encode(src, src_valid)

    generated = torch.tensor([[BOS_ID]], device=device)
    for _ in range(args.max_length - 1):
        tgt_valid = make_valid_mask(generated, PAD_ID)
        logits = model.decode(generated, memory, tgt_valid, src_valid)
        next_id = logits[:, -1, :].argmax(-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        if int(next_id.item()) == EOS_ID:
            break

    # 最后一次 decode 后，last_cross_attention 形状 [B,H,T,S]。
    weights = model.last_cross_attention
    if weights is None:
        raise RuntimeError("没有捕获到 cross-attention weights")
    matrix = weights[0].mean(dim=0).detach().cpu()  # 平均所有 Head → [T,S]

    tgt_ids = generated[0].tolist()
    tgt_tokens = tgt_vocab.decode(tgt_ids, skip_specials=False)
    src_labels = src_tokens + ["<eos>"]
    tgt_labels = tgt_tokens[: matrix.size(0)]

    plt.figure(figsize=(max(6, len(src_labels) * 0.6), max(5, len(tgt_labels) * 0.45)))
    plt.imshow(matrix[: len(tgt_labels), : len(src_labels)], aspect="auto")
    plt.colorbar(label="Mean Cross-Attention Weight")
    plt.xticks(range(len(src_labels)), src_labels, rotation=45, ha="right")
    plt.yticks(range(len(tgt_labels)), tgt_labels)
    plt.xlabel("Encoder source positions (K/V)")
    plt.ylabel("Decoder positions (Q)")
    plt.title("Decoder Cross-Attention")
    plt.tight_layout()

    output = ROOT / "outputs" / "plots" / "cross_attention.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=170)
    plt.close()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
