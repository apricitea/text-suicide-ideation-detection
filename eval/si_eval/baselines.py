"""Cheap classical baselines, trained on exactly the same rows as the transformer.

The point is not to beat IndoBERT. It is to establish what a reasonable
non-neural model achieves on this corpus, so that any claimed transformer gain can
be attributed to the representation rather than to the split, the label
distribution, or the evaluation protocol.

No Indonesian stemming or stopword removal is applied here. That is a deliberate
choice: the thesis arm used Sastrawi/Hunspell preprocessing and cannot be re-run
without its FastText vectors and an unsaved tokenizer, so the honest comparison is
between the transformer and *unpreprocessed* classical models. If anything this
favours the transformer, since it gets to work on raw text as well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline


@dataclass
class Baseline:
    name: str
    pipeline: Pipeline

    def fit(self, texts: list[str], labels: list[int]) -> "Baseline":
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict(texts).astype(int)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        proba = np.asarray(self.pipeline.predict_proba(texts), dtype=float)
        if proba.ndim == 1:  # a degenerate baseline that predicts a constant
            return proba
        return proba[:, 1]


def word_tfidf_logreg() -> Baseline:
    return Baseline(
        "tfidf_word_logreg",
        Pipeline(
            [
                (
                    "vec",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.9,
                        sublinear_tf=True,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=3000, C=4.0, solver="liblinear")),
            ]
        ),
    )


def char_tfidf_logreg() -> Baseline:
    return Baseline(
        "tfidf_char_logreg",
        Pipeline(
            [
                (
                    "vec",
                    TfidfVectorizer(
                        lowercase=True,
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=3000, C=2.0, solver="liblinear")),
            ]
        ),
    )


def word_count_nb() -> Baseline:
    return Baseline(
        "count_word_complementnb",
        Pipeline(
            [
                ("vec", CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)),
                ("clf", ComplementNB(alpha=1.0)),
            ]
        ),
    )


def majority_class() -> Baseline:
    """Predict the training majority class; the reference every metric needs."""

    class _Majority:
        def __init__(self) -> None:
            self.label_ = 0
            self.name = "majority_class"

        def fit(self, texts, labels):  # noqa: ANN001 - sklearn-style duck typing
            self.label_ = int(np.bincount(np.asarray(labels, dtype=int)).argmax())
            return self

        def predict(self, texts):  # noqa: ANN001
            return np.full(len(texts), self.label_, dtype=int)

        def predict_proba(self, texts):  # noqa: ANN001
            return np.full(len(texts), float(self.label_), dtype=float)

    return Baseline("majority_class", _Majority())  # type: ignore[arg-type]


def all_baselines() -> list[Baseline]:
    return [majority_class(), word_count_nb(), word_tfidf_logreg(), char_tfidf_logreg()]


def fit_and_score(baseline: Baseline, train_texts, train_labels, eval_texts, eval_labels) -> dict:
    baseline.fit(list(train_texts), [int(v) for v in train_labels])
    pred = baseline.predict(list(eval_texts))
    prob = baseline.predict_proba(list(eval_texts))
    return {
        "name": baseline.name,
        "pred": pred,
        "prob": prob,
        "positive_f1": float(f1_score(eval_labels, pred, pos_label=1, zero_division=0)),
    }
