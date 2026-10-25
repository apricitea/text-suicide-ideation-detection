"""Fine-tune IndoBERT on a deduplicated split, to close the audit's first limitation.

The audit in ``eval/`` found that the published 80/20 split put 739 of 2,026
validation posts in both halves, and that the reported 0.910 positive-class F1 is
0.877 when scored only on rows with no training twin. That estimate is still
optimistic: the *weights* saw the leaked twins even when the scored rows did not.

This script removes the cause instead of the symptom:

1. deduplicate the corpus by normalised text (the same normalisation the audit uses,
   imported from ``eval/si_eval/data.py`` so the two cannot drift apart);
2. draw the 80/20 split from the deduplicated corpus, so no observation and no
   near-duplicate of one can appear on both sides;
3. fine-tune with the original hyperparameters (seed 1, 3 epochs, max_length 128,
   lr 2e-5) and evaluate once on the held-out half;
4. write metrics, per-row predictions and a manifest.

The number this produces is a clean estimate: smaller training set, wider interval,
no memorisation of duplicated text. Whatever it is, it replaces the caveat.

Usage
-----
    python main/train_clean_split.py --out-dir indobert_clean_split
    python main/train_clean_split.py --limit 400 --max-steps 5 --out-dir /tmp/smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from si_eval.data import clean_split, load_corpus  # noqa: E402

SEED = 1
MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 128


class TweetDataset(Dataset):
    def __init__(self, encodings, labels):  # noqa: ANN001 - torch dataset shape
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                cwd=REPO_ROOT,
            ).stdout.strip()
        )
    except Exception:  # noqa: BLE001 - provenance must not break a long run
        commit, dirty = "unknown", None
    return {"commit": commit, "dirty": dirty}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset" / "final_dataset.xlsx")
    parser.add_argument("--sheet", default="10k")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "indobert_clean_split")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="subsample rows, for smoke tests")
    parser.add_argument("--max-steps", type=int, default=-1, help="stop after N optimizer steps")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    set_seed(SEED)
    started = time.time()

    df = load_corpus(args.dataset, args.sheet)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=SEED).reset_index(drop=True)

    # Deduplicate first, then split: the whole point of this script.
    split = clean_split(df, seed=SEED)
    n_cross = sum(split.val_leaked)
    print(
        f"corpus={len(df)} unique={len(df) - int(pd.Series([t for t in df['text']]).duplicated().sum())} "
        f"train={split.n_train} val={split.n_val} val_pos={sum(split.val_label)} "
        f"cross_split_duplicates={n_cross}",
        flush=True,
    )
    if n_cross:
        raise SystemExit(f"clean split still has {n_cross} cross-split duplicates; refusing to train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(
        list(split.train_text),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    val_enc = tokenizer(
        list(split.val_text),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    train_ds = TweetDataset(train_enc, split.train_label)
    val_ds = TweetDataset(val_enc, split.val_label)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    training_args = TrainingArguments(
        output_dir=str(args.out_dir / "_checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=SEED,
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)
    trainer.train()

    predictions = trainer.predict(val_ds)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = np.asarray(split.val_label)
    proba = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()[:, 1]

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(cm, flush=True)
    print(classification_report(y_true, y_pred, zero_division=0), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "model").mkdir(exist_ok=True)
    trainer.save_model(str(args.out_dir / "model"))
    tokenizer.save_pretrained(str(args.out_dir / "model"))

    metrics = {
        "protocol": "deduplicated corpus, then 80/20 split (no cross-split duplicates)",
        "model_name": MODEL_NAME,
        "seed": SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": MAX_LENGTH,
        "n_train": split.n_train,
        "n_val": split.n_val,
        "n_val_positives": int(y_true.sum()),
        "cross_split_duplicates": int(n_cross),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "positive_f1": report["1"]["f1-score"],
        "positive_precision": report["1"]["precision"],
        "positive_recall": report["1"]["recall"],
        "accuracy": report["accuracy"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "threads": args.threads,
        "wall_clock_seconds": round(time.time() - started, 1),
        "smoke_run": bool(args.limit or args.max_steps > 0),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(
        {"gold": y_true, "pred": y_pred, "prob_positive": proba, "text": list(split.val_text)}
    ).to_csv(args.out_dir / "predictions.csv", index=False)

    manifest = {
        "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset), "sheet": args.sheet},
        "git": git_revision(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "platform": platform.platform(),
        },
        "config": {
            "seed": SEED,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": MAX_LENGTH,
            "limit": args.limit,
            "max_steps": args.max_steps,
        },
        "split": {
            "n_train": split.n_train,
            "n_val": split.n_val,
            "n_val_positives": int(y_true.sum()),
            "cross_split_duplicates": int(n_cross),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"\nclean-split result: positive F1={metrics['positive_f1']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} on {split.n_val} unseen posts "
        f"({int(y_true.sum())} positive), {metrics['wall_clock_seconds']}s"
    )


if __name__ == "__main__":
    main()
