"""Fine-tune IndoBERT on the same 10k-tweet dataset and split as lstm.ipynb's
'No Imbalanced Class Treatment' scenario, for a direct comparison against the
original FastText+LSTM baseline (F1=0.79 on the positive class).
"""

import random

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    set_seed(SEED)

    df = pd.read_excel("../dataset/final_dataset.xlsx", sheet_name="10k")
    X = df["tweet"].astype(str)
    y = df["label"]

    # Same split as the LSTM baseline's "No Imbalanced Class Treatment" scenario.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=0.8, random_state=1
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(
        list(X_train), truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    val_enc = tokenizer(
        list(X_val), truncation=True, padding=True, max_length=128, return_tensors="pt"
    )

    train_ds = TweetDataset(train_enc, y_train.tolist())
    val_ds = TweetDataset(val_enc, y_val.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir="./indobert_output",
        num_train_epochs=3,
        per_device_train_batch_size=16,
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
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()

    preds = trainer.predict(val_ds)
    y_pred = np.argmax(preds.predictions, axis=1)

    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred))


if __name__ == "__main__":
    main()
