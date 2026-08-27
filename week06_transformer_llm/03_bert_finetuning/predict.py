from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from strategies import apply_strategy
from utils import resolve_device

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["head_only", "last_n", "full"], required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    device = resolve_device(args.device)
    payload = torch.load(
        args.output_dir / "checkpoints" / args.strategy / "best.pt",
        map_location=device,
        weights_only=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(payload["model_name"], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(payload["model_name"], num_labels=2)
    apply_strategy(model, payload["strategy"], payload["unfreeze_last_n"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()

    inputs = tokenizer(
        args.text,
        truncation=True,
        max_length=payload["max_length"],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    label = "positive" if int(probs.argmax()) == 1 else "negative"
    print(f"label={label}")
    print(f"P(negative)={float(probs[0]):.4f}")
    print(f"P(positive)={float(probs[1]):.4f}")


if __name__ == "__main__":
    main()
