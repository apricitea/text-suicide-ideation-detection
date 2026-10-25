"""The re-evaluation itself: leakage, baselines, uncertainty, calibration, errors."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import Baseline, all_baselines
from .data import (
    DEFAULT_DATASET,
    Split,
    clean_split,
    dedupe_train,
    load_corpus,
    normalise,
    split_like_thesis,
)
from .metrics import (
    CLASSIFICATION_METRICS,
    bootstrap_ci,
    calibration,
    mcnemar_test,
    operating_points,
    paired_bootstrap_diff,
    positive_f1,
    risk_coverage,
)

PREDICTIONS_CSV = Path(__file__).resolve().parents[2] / "indobert_publish" / "predictions.csv"

NEGATION = re.compile(r"\b(jangan|tidak|bukan|nggak|gak|enggak|tak|tdk|gausah|jgn)\b", re.I)
FIRST_PERSON = re.compile(r"\b(aku|saya|gue|gw|ane)\b", re.I)
QUESTION = re.compile(r"\?")


def load_indobert_predictions(path: Path | str = PREDICTIONS_CSV) -> pd.DataFrame:
    """Load the committed IndoBERT validation predictions.

    The CSV is written in validation-row order by ``main/indobert_finetune.py``, and
    its ``gold`` column is the split's ``y_val``. Row order, not text, is the
    alignment key: duplicate texts inside the validation set make text-keyed lookup
    ambiguous, and the gold column matches positionally.
    """
    df = pd.read_csv(path)
    for column in ("gold", "pred", "prob_positive"):
        if column not in df.columns:
            raise ValueError(f"{path} is missing column {column!r}")
    return df


def _verify_alignment(split: Split, preds: pd.DataFrame) -> None:
    if len(preds) != split.n_val:
        raise ValueError(
            f"IndoBERT predictions have {len(preds)} rows but the reproduced split has "
            f"{split.n_val}; the split or the predictions file has changed."
        )
    if list(preds["gold"].astype(int)) != list(split.val_label):
        raise ValueError(
            "IndoBERT gold labels do not match the reproduced split in row order; "
            "refusing to compare models evaluated on different rows."
        )


def _subset(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(values)[mask]


def _metric_bundle(y_true, y_pred) -> dict[str, float]:
    return {name: float(fn(y_true, y_pred)) for name, fn in CLASSIFICATION_METRICS.items()}


def _intervals(y_true, y_pred, n_boot: int, seed: int) -> dict[str, dict[str, float | int]]:
    return {
        name: bootstrap_ci(y_true, y_pred, fn, n_boot=n_boot, seed=seed).as_dict()
        for name, fn in CLASSIFICATION_METRICS.items()
    }


def error_buckets(texts: list[str], y_true, y_pred) -> dict[str, dict[str, float]]:
    """False-negative and false-positive rates by simple, interpretable text features.

    These are descriptive, not causal: the buckets overlap and each is small. They
    exist to say where the remaining errors sit, which is the first question a
    reader asks after seeing an aggregate F1.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    lengths = np.array([len(t) for t in texts])

    def rate(mask: np.ndarray) -> dict[str, float]:
        n = int(mask.sum())
        if n == 0:
            return {"n": 0, "error_rate": float("nan"), "fn_rate": float("nan"), "fp_rate": float("nan")}
        fn = int(np.sum(mask & (y_true == 1) & (y_pred == 0)))
        fp = int(np.sum(mask & (y_true == 0) & (y_pred == 1)))
        return {
            "n": n,
            "error_rate": float(np.mean(y_true[mask] != y_pred[mask])),
            "fn_rate": float(fn / max(int(np.sum((y_true == 1) & mask)), 1)),
            "fp_rate": float(fp / max(int(np.sum((y_true == 0) & mask)), 1)),
        }

    masks = {
        "short (<80 chars)": lengths < 80,
        "medium (80-160)": (lengths >= 80) & (lengths < 160),
        "long (>=160)": lengths >= 160,
        "contains negation": np.array([bool(NEGATION.search(t)) for t in texts]),
        "first person": np.array([bool(FIRST_PERSON.search(t)) for t in texts]),
        "contains question mark": np.array([bool(QUESTION.search(t)) for t in texts]),
        "all rows": np.ones(len(texts), dtype=bool),
    }
    return {name: rate(mask) for name, mask in masks.items()}


def run_analysis(
    dataset: Path | str = DEFAULT_DATASET,
    sheet: str = "10k",
    n_boot: int = 2000,
    seed: int = 1,
    quick: bool = False,
) -> dict:
    """Run every comparison and return a JSON-serialisable result tree."""
    df = load_corpus(dataset, sheet)
    split = split_like_thesis(df, seed=seed)
    preds = load_indobert_predictions()
    _verify_alignment(split, preds)

    clean = clean_split(df, seed=seed)
    val_clean_mask = np.array(split.clean_mask)
    y_val = np.array(split.val_label)

    # --- protocol A: reproduce the published evaluation exactly -----------------
    indobert_full_pred = preds["pred"].astype(int).to_numpy()
    indobert_full_prob = preds["prob_positive"].astype(float).to_numpy()

    # --- protocol B: the same model, scored only on rows not seen in training ---
    y_clean = _subset(y_val, val_clean_mask)
    indobert_clean_pred = _subset(indobert_full_pred, val_clean_mask)
    indobert_clean_prob = _subset(indobert_full_prob, val_clean_mask)

    # --- classical baselines, trained on the original and the deduplicated train -
    baseline_rows = []
    baseline_preds: dict[str, dict[str, np.ndarray]] = {}
    train_texts_dedup, train_labels_dedup = dedupe_train(split)

    for baseline in all_baselines():
        for variant, (texts, labels) in {
            "train_as_published": (split.train_text, split.train_label),
            "train_deduplicated": (train_texts_dedup, train_labels_dedup),
        }.items():
            fitted: Baseline = baseline.fit(list(texts), [int(v) for v in labels])
            pred_full = fitted.predict(list(split.val_text))
            prob_full = fitted.predict_proba(list(split.val_text))
            pred_clean = _subset(pred_full, val_clean_mask)
            prob_clean = _subset(prob_full, val_clean_mask)
            baseline_preds[f"{baseline.name}@{variant}"] = {
                "pred_full": pred_full,
                "prob_full": prob_full,
                "pred_clean": pred_clean,
                "prob_clean": prob_clean,
            }
            baseline_rows.append(
                {
                    "model": baseline.name,
                    "training_data": variant,
                    "full_val": _metric_bundle(y_val, pred_full),
                    "clean_val": _metric_bundle(y_clean, pred_clean),
                    "clean_val_f1_ci": bootstrap_ci(
                        y_clean, pred_clean, positive_f1, n_boot=n_boot, seed=seed
                    ).as_dict(),
                    "clean_val_calibration": calibration(y_clean, prob_clean).as_dict(),
                }
            )

    best_baseline_key = max(
        (k for k in baseline_preds if k.endswith("train_deduplicated")),
        key=lambda k: float(np.mean(baseline_preds[k]["pred_clean"] == y_clean)),
    )
    best_baseline = baseline_preds[best_baseline_key]

    # --- statistical comparison on the clean subset -----------------------------
    comparison = {
        "indobert_vs_best_baseline": {
            "baseline": best_baseline_key,
            "mcnemar": mcnemar_test(
                y_clean, indobert_clean_pred, best_baseline["pred_clean"]
            ).as_dict(),
            "paired_bootstrap_f1_diff": paired_bootstrap_diff(
                y_clean,
                indobert_clean_pred,
                best_baseline["pred_clean"],
                n_boot=n_boot,
                seed=seed,
            ),
        }
    }

    # --- calibration and operating points for the deployed transformer ---------
    indobert_calibration_full = calibration(y_val, indobert_full_prob).as_dict()
    indobert_calibration_clean = calibration(y_clean, indobert_clean_prob).as_dict()

    curves = risk_coverage(y_clean, indobert_clean_prob)
    sample_idx = np.unique(np.linspace(0, len(curves["coverage"]) - 1, 200).astype(int))

    # --- error examples ---------------------------------------------------------
    val_text = list(split.val_text)
    clean_text = [val_text[i] for i in range(len(val_text)) if val_clean_mask[i]]
    fn_idx = [
        i
        for i, (t, y, p) in enumerate(zip(clean_text, y_clean, indobert_clean_pred))
        if y == 1 and p == 0
    ]
    fp_idx = [
        i
        for i, (t, y, p) in enumerate(zip(clean_text, y_clean, indobert_clean_pred))
        if y == 0 and p == 1
    ]

    results = {
        "dataset": {
            "path": str(dataset),
            "sheet": sheet,
            "rows": int(len(df)),
            "positives": int(df["label"].sum()),
            "positive_rate": float(df["label"].mean()),
            "exact_duplicate_rows": int(df["text"].duplicated().sum()),
            "duplicate_rows_normalised": int(
                pd.Series([normalise(t) for t in df["text"]]).duplicated().sum()
            ),
            "unique_normalised_texts": int(len({normalise(t) for t in df["text"]})),
        },
        "split": {
            "protocol": "train_test_split(train_size=0.8, random_state=%d)" % seed,
            "n_train": split.n_train,
            "n_val": split.n_val,
            "val_positives": int(y_val.sum()),
            "val_rows_duplicated_in_train": split.n_leaked,
            "val_leaked_fraction": split.leaked_fraction,
            "n_clean_val": int(len(y_clean)),
            "clean_val_positives": int(y_clean.sum()),
            "dedup_train_rows": len(train_texts_dedup),
        },
        "indobert": {
            "full_val": _metric_bundle(y_val, indobert_full_pred),
            "clean_val": _metric_bundle(y_clean, indobert_clean_pred),
            "full_val_intervals": {} if quick else _intervals(y_val, indobert_full_pred, n_boot, seed),
            "clean_val_intervals": {} if quick else _intervals(y_clean, indobert_clean_pred, n_boot, seed),
            "clean_val_calibration": indobert_calibration_clean,
            "full_val_calibration": indobert_calibration_full,
            "operating_points": operating_points(y_clean, indobert_clean_prob),
            "risk_coverage_sampled": {
                "coverage": [curves["coverage"][i] for i in sample_idx],
                "selective_accuracy": [curves["selective_accuracy"][i] for i in sample_idx],
                "positive_precision": [curves["positive_precision"][i] for i in sample_idx],
            },
            "error_buckets_clean": error_buckets(clean_text, y_clean, indobert_clean_pred),
            "n_false_negatives_clean": len(fn_idx),
            "n_false_positives_clean": len(fp_idx),
            "false_negative_examples": [
                {"text": clean_text[i][:220], "p_positive": float(indobert_clean_prob[i])}
                for i in fn_idx[:6]
            ],
            "false_positive_examples": [
                {"text": clean_text[i][:220], "p_positive": float(indobert_clean_prob[i])}
                for i in fp_idx[:6]
            ],
        },
        "baselines": baseline_rows,
        "comparison": comparison,
        "config": {"n_boot": n_boot, "seed": seed, "quick": quick},
    }
    return results
