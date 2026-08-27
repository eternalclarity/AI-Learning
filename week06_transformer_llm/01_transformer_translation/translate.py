from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import load_artifacts
from inference import greedy_decode
from model import Transformer, TransformerConfig
from utils import resolve_device

ROOT = Path(__file__).resolve().parent


def load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = Transformer(TransformerConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


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
    model = load_model(args.checkpoint, device)
    tokens = greedy_decode(model, args.sentence, src_vocab, tgt_vocab, device, args.max_length)
    print("source:", args.sentence)
    print("translation:", " ".join(tokens))


if __name__ == "__main__":
    main()
