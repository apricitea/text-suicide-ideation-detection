"""Check that the published Protocol C weights reproduce the committed 0.849 F1.

The retrain takes 6,434 seconds, which is too long for a reader to verify a number. This
script closes that gap: download the `model-clean-v1` release asset, point it at the
committed per-row validation texts, and recompute the metric. Inference over 1,586 short
posts takes well under a minute on CPU.

    # one-off: fetch and unpack the weights (441 MB)
    curl -L -o /tmp/clean.tar.gz \\
      https://github.com/apricitea/text-suicide-ideation-detection/releases/download/model-clean-v1/indobert-clean-split-model.tar.gz
    mkdir -p indobert_clean_split/model && tar -xzf /tmp/clean.tar.gz -C indobert_clean_split/model --strip-components=1

    uv run --project eval python eval/verify_clean_retrain.py \\
        --weights indobert_clean_split/model \\
        --predictions indobert_clean_split/predictions.csv

Exit code 0 means the recomputed positive-class F1 matches `metrics.json` to within 0.005.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAX_LENGTH = 128
TOLERANCE = 0.005


def positive_f1(gold: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum((pred == 1) & (gold == 1)))
    fp = int(np.sum((pred == 1) & (gold == 0)))
    fn = int(np.sum((pred == 0) & (gold == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("indobert_clean_split/model"))
    parser.add_argument("--predictions", type=Path, default=Path("indobert_clean_split/predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()

    for path in (args.weights, args.predictions):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 2

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        print(f"needs torch and transformers: {exc}", file=sys.stderr)
        return 2

    table = pd.read_csv(args.predictions)
    gold = table["gold"].astype(int).to_numpy()
    texts = table["text"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(str(args.weights))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.weights))
    model.eval()

    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            encoded = tokenizer(
                batch, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
            )
            logits = model(**encoded).logits
            predictions.extend(torch.argmax(logits, dim=-1).tolist())
    pred = np.asarray(predictions)

    recomputed = positive_f1(gold, pred)
    committed = json.loads((args.predictions.parent / "metrics.json").read_text())["positive_f1"]
    agreement = float(np.mean(pred == table["pred"].astype(int).to_numpy()))

    print(f"rows                : {len(gold)}")
    print(f"recomputed F1       : {recomputed:.4f}")
    print(f"committed F1        : {committed:.4f}")
    print(f"row-level agreement : {agreement:.4f} (vs committed predictions)")
    ok = abs(recomputed - committed) <= args.tolerance
    print("PASS" if ok else "FAIL", f"(tolerance {args.tolerance})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
