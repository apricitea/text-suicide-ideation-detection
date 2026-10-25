"""Dataset loading, split reproduction and duplicate-leakage measurement.

The thesis and the IndoBERT follow-up both evaluate on the same
``train_test_split(train_size=0.8, random_state=1)`` of the ``10k`` sheet with no
stratification and no deduplication. Reproducing that call exactly is what makes
the comparison in this package legitimate: every model is scored on the same rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "dataset" / "final_dataset.xlsx"
DEFAULT_SHEET = "10k"

_URL = re.compile(r"https?://\S+|www\.\S+")
_HANDLE = re.compile(r"@[A-Za-z0-9_]+")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalise(text: str) -> str:
    """Aggressive normalisation used only for duplicate detection.

    Lower-cases, rewrites URLs and handles to fixed tokens, strips punctuation and
    collapses whitespace. Two rows that normalise to the same string are treated as
    the same observation for leakage purposes even when they differ in casing,
    emoji or link targets -- which is exactly the near-duplicate case that inflates
    a random split in a scraped corpus.
    """
    text = _URL.sub(" url ", str(text).lower())
    text = _HANDLE.sub(" user ", text)
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class Split:
    """A reproduced train/validation split plus the leakage measurement."""

    train_text: list[str]
    train_label: list[int]
    val_text: list[str]
    val_label: list[int]
    val_leaked: list[bool]

    @property
    def n_train(self) -> int:
        return len(self.train_text)

    @property
    def n_val(self) -> int:
        return len(self.val_text)

    @property
    def n_leaked(self) -> int:
        return sum(self.val_leaked)

    @property
    def leaked_fraction(self) -> float:
        return self.n_leaked / self.n_val if self.n_val else 0.0

    @property
    def clean_mask(self) -> list[bool]:
        return [not flag for flag in self.val_leaked]


def load_corpus(path: Path | str = DEFAULT_DATASET, sheet: str = DEFAULT_SHEET) -> pd.DataFrame:
    """Load the labelled corpus as ``text``/``label`` columns."""
    df = pd.read_excel(path, sheet_name=sheet)
    missing = {"tweet", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}[{sheet}] is missing columns: {sorted(missing)}")
    out = pd.DataFrame({"text": df["tweet"].astype(str), "label": df["label"].astype(int)})
    return out


def split_like_thesis(df: pd.DataFrame, test_size: float = 0.2, seed: int = 1) -> Split:
    """Reproduce the thesis split and flag validation rows seen during training.

    ``train_test_split`` is called with the same arguments as
    ``main/indobert_finetune.py`` and ``main/lstm.ipynb`` so row order and
    membership match the published experiment.
    """
    x_train, x_val, y_train, y_val = train_test_split(
        df["text"], df["label"], train_size=1.0 - test_size, random_state=seed
    )
    train_text = x_train.tolist()
    val_text = x_val.tolist()
    train_set = {normalise(t) for t in train_text}
    val_leaked = [normalise(t) in train_set for t in val_text]
    return Split(
        train_text=train_text,
        train_label=y_train.tolist(),
        val_text=val_text,
        val_label=y_val.tolist(),
        val_leaked=val_leaked,
    )


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose normalised text was already seen, keeping first occurrence."""
    seen: set[str] = set()
    keep: list[bool] = []
    for text in df["text"]:
        key = normalise(text)
        keep.append(key not in seen)
        seen.add(key)
    return df.loc[keep].reset_index(drop=True)


def dedupe_train(split: Split) -> tuple[list[str], list[int]]:
    """Drop duplicated training rows, keeping labels of first occurrences."""
    seen: set[str] = set()
    texts: list[str] = []
    labels: list[int] = []
    for text, label in zip(split.train_text, split.train_label):
        key = normalise(text)
        if key in seen:
            continue
        seen.add(key)
        texts.append(text)
        labels.append(int(label))
    return texts, labels


def clean_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 1) -> Split:
    """Deduplicate the whole corpus first, then split.

    This is the protocol the leakage audit implies: no observation, and no
    near-duplicate of one, may appear on both sides of the split. It produces a
    smaller validation set, so comparisons against it carry wider intervals.
    """
    return split_like_thesis(dedupe(df), test_size=test_size, seed=seed)
