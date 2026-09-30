"""Fine-tune IndoBERT on the same 10k-tweet dataset and split as lstm.ipynb's
'No Imbalanced Class Treatment' scenario, for a direct comparison against the
original FastText+LSTM baseline (F1=0.79 on the positive class).

Unlike the first version of this script, this one persists the trained model
and its metrics. The original trained for ~35 minutes and threw the weights
away, which is why no reproducibility artifact existed for the reported
IndoBERT numbers.

Outputs (under --out-dir, default ./indobert_publish):
    model/            fine-tuned weights + tokenizer, ready to upload
    metrics.json      confusion matrix, per-class report, headline numbers
    predictions.csv   validation-set probabilities and gold labels

    python indobert_finetune.py --out-dir ./indobert_publish
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

SEED = 1  # matches lstm.ipynb's train_test_split random_state
MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 128


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("../dataset/final_dataset.xlsx"))
    parser.add_argument("--sheet", default="10k")
    parser.add_argument("--out-dir", type=Path, default=Path("./indobert_publish"))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    set_seed(SEED)
    print(f"threads={torch.get_num_threads()} cuda={torch.cuda.is_available()}", flush=True)

    df = pd.read_excel(args.dataset, sheet_name=args.sheet)
    X = df["tweet"].astype(str)
    y = df["label"]

    # Same split as the LSTM baseline's "No Imbalanced Class Treatment" scenario.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=0.8, random_state=1
    )
    print(f"train={len(X_train)} val={len(X_val)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(
        list(X_train), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    val_enc = tokenizer(
        list(X_val), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
    )

    train_ds = TweetDataset(train_enc, y_train.tolist())
    val_ds = TweetDataset(val_enc, y_val.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args_out = TrainingArguments(
        output_dir="./indobert_output",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args_out,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()

    preds = trainer.predict(val_ds)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = np.asarray(y_val)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    print(cm)
    print(classification_report(y_true, y_pred))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "model").mkdir(exist_ok=True)
    trainer.save_model(str(args.out_dir / "model"))
    tokenizer.save_pretrained(str(args.out_dir / "model"))

    metrics = {
        "model_name": MODEL_NAME,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "positive_f1": report["1"]["f1-score"],
        "positive_precision": report["1"]["precision"],
        "positive_recall": report["1"]["recall"],
        "accuracy": report["accuracy"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    proba = torch.softmax(torch.tensor(preds.predictions), dim=1).numpy()[:, 1]
    pd.DataFrame(
        {"gold": y_true, "pred": y_pred, "prob_positive": proba, "text": list(X_val)}
    ).to_csv(args.out_dir / "predictions.csv", index=False)

    print(f"\nwrote {args.out_dir}/model, metrics.json, predictions.csv")
    print(f"positive F1 = {metrics['positive_f1']:.4f}  accuracy = {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()