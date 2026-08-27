"""在 test split 上做最终翻译并计算教学版平均 sentence BLEU。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from data import load_artifacts, tokenize
from inference import greedy_decode
from metrics import bleu
from model import Transformer, TransformerConfig
from utils import resolve_device, save_json

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "checkpoints" / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    splits, src_vocab, tgt_vocab, _ = load_artifacts(args.artifact_dir)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = Transformer(TransformerConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    pairs = splits["test"][: args.max_samples] if args.max_samples else splits["test"]
    rows = []
    scores = []
    for pair in pairs:
        pred_tokens = greedy_decode(model, pair.source, src_vocab, tgt_vocab, device, args.max_length)
        ref_tokens = tokenize(pair.target)
        score = bleu(pred_tokens, ref_tokens)
        scores.append(score)
        rows.append({
            "source": pair.source,
            "reference": pair.target,
            "prediction": " ".join(pred_tokens),
            "sentence_bleu": score,
        })

    result_dir = args.output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "translations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["source", "reference", "prediction", "sentence_bleu"])
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "num_samples": len(rows),
        "mean_sentence_bleu": sum(scores) / max(len(scores), 1),
        "note": "教学版单参考 sentence BLEU；用于本项目内部比较，不替代标准翻译基准。",
    }
    save_json(metrics, result_dir / "test_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
