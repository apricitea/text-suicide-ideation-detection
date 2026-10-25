"""Metrics with uncertainty: bootstrap intervals, paired tests, calibration, selective prediction.

Every function takes plain sequences so the whole module is testable without data.
Positive class is 1 (ideation present) throughout, which is the class the thesis
reports on and the one with operational meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

ArrayLike = Sequence[float] | np.ndarray


def _as_int_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values).astype(int)


def positive_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(f1_score(_as_int_array(y_true), _as_int_array(y_pred), pos_label=1, zero_division=0))


def positive_precision(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(
        precision_score(_as_int_array(y_true), _as_int_array(y_pred), pos_label=1, zero_division=0)
    )


def positive_recall(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(recall_score(_as_int_array(y_true), _as_int_array(y_pred), pos_label=1, zero_division=0))


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(accuracy_score(_as_int_array(y_true), _as_int_array(y_pred)))


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(f1_score(_as_int_array(y_true), _as_int_array(y_pred), average="macro", zero_division=0))


CLASSIFICATION_METRICS: dict[str, Callable[[ArrayLike, ArrayLike], float]] = {
    "positive_f1": positive_f1,
    "positive_precision": positive_precision,
    "positive_recall": positive_recall,
    "accuracy": accuracy,
    "macro_f1": macro_f1,
}


def stratified_indices(y_true: ArrayLike, rng: np.random.Generator) -> np.ndarray:
    """Resample row indices with replacement, keeping the class counts fixed.

    A plain bootstrap on a 19% positive rate occasionally draws a sample with very
    few positives, which makes F1 jump around for reasons that have nothing to do
    with model quality. Resampling within each class keeps the interval honest
    about sample size without adding that artefact.
    """
    y = _as_int_array(y_true)
    parts = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        parts.append(rng.choice(idx, size=idx.size, replace=True))
    return np.concatenate(parts)


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    n_boot: int

    def as_dict(self) -> dict[str, float | int]:
        return {"point": self.point, "low": self.low, "high": self.high, "n_boot": self.n_boot}

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric: Callable[[ArrayLike, ArrayLike], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for a classification metric."""
    y_true_arr = _as_int_array(y_true)
    y_pred_arr = _as_int_array(y_pred)
    point = float(metric(y_true_arr, y_pred_arr))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = stratified_indices(y_true_arr, rng)
        draws[i] = metric(y_true_arr[idx], y_pred_arr[idx])
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(point=point, low=float(low), high=float(high), n_boot=n_boot)


@dataclass(frozen=True)
class McNemar:
    """McNemar test on the discordant pairs of two classifiers."""

    a_correct_b_wrong: int
    a_wrong_b_correct: int
    statistic: float
    p_value: float
    method: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "a_correct_b_wrong": self.a_correct_b_wrong,
            "a_wrong_b_correct": self.a_wrong_b_correct,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "method": self.method,
        }


def mcnemar_test(y_true: ArrayLike, pred_a: ArrayLike, pred_b: ArrayLike) -> McNemar:
    """Two-sided McNemar test, exact when discordant pairs are few.

    The exact binomial form is used below 25 discordant pairs, where the chi-square
    approximation is unreliable; above that the continuity-corrected chi-square is
    standard.
    """
    y = _as_int_array(y_true)
    a = _as_int_array(pred_a)
    b = _as_int_array(pred_b)
    a_ok = a == y
    b_ok = b == y
    n01 = int(np.sum(a_ok & ~b_ok))  # A right, B wrong
    n10 = int(np.sum(~a_ok & b_ok))  # A wrong, B right
    discordant = n01 + n10
    if discordant == 0:
        return McNemar(n01, n10, 0.0, 1.0, "none")
    if discordant < 25:
        p = 2.0 * float(stats.binom.cdf(min(n01, n10), discordant, 0.5))
        return McNemar(n01, n10, float(min(n01, n10)), min(1.0, p), "exact_binomial")
    statistic = (abs(n01 - n10) - 1.0) ** 2 / discordant
    p = float(stats.chi2.sf(statistic, df=1))
    return McNemar(n01, n10, float(statistic), p, "chi2_continuity")


def paired_bootstrap_diff(
    y_true: ArrayLike,
    pred_a: ArrayLike,
    pred_b: ArrayLike,
    metric: Callable[[ArrayLike, ArrayLike], float] = positive_f1,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap the paired difference metric(A) - metric(B) on the same rows."""
    y = _as_int_array(y_true)
    a = _as_int_array(pred_a)
    b = _as_int_array(pred_b)
    observed = float(metric(y, a) - metric(y, b))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = stratified_indices(y, rng)
        draws[i] = metric(y[idx], a[idx]) - metric(y[idx], b[idx])
    low, high = np.quantile(draws, [0.025, 0.975])
    p_two_sided = 2.0 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0)))
    return {
        "difference": observed,
        "low": float(low),
        "high": float(high),
        "p_value": min(1.0, p_two_sided),
        "n_boot": n_boot,
    }


@dataclass(frozen=True)
class Calibration:
    ece: float
    mce: float
    brier: float
    n_bins: int
    bins: list[dict[str, float]]

    def as_dict(self) -> dict[str, object]:
        return {
            "ece": self.ece,
            "mce": self.mce,
            "brier": self.brier,
            "n_bins": self.n_bins,
            "bins": self.bins,
        }


def calibration(y_true: ArrayLike, prob_positive: ArrayLike, n_bins: int = 15) -> Calibration:
    """Equal-width expected/maximum calibration error and the Brier score."""
    y = _as_int_array(y_true)
    p = np.clip(np.asarray(prob_positive, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins: list[dict[str, float]] = []
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        confidence = float(p[mask].mean())
        accuracy_bin = float(y[mask].mean())
        gap = abs(accuracy_bin - confidence)
        ece += n / len(p) * gap
        mce = max(mce, gap)
        bins.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "n": n,
                "mean_confidence": confidence,
                "fraction_positive": accuracy_bin,
            }
        )
    return Calibration(
        ece=float(ece),
        mce=float(mce),
        brier=float(brier_score_loss(y, p)),
        n_bins=n_bins,
        bins=bins,
    )


def risk_coverage(y_true: ArrayLike, prob_positive: ArrayLike) -> dict[str, list[float]]:
    """Selective-prediction curve: accuracy and positive-class precision vs coverage.

    Instances are accepted in descending order of confidence (``max(p, 1-p)``), which
    is the standard selective-classification ordering. Coverage runs from one
    instance to the full set.
    """
    y = _as_int_array(y_true)
    p = np.clip(np.asarray(prob_positive, dtype=float), 0.0, 1.0)
    pred = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1.0 - p)
    order = np.argsort(-confidence, kind="stable")
    y_sorted = y[order]
    pred_sorted = pred[order]
    correct = (y_sorted == pred_sorted).astype(float)
    tp = np.cumsum((pred_sorted == 1) & (y_sorted == 1))
    fp = np.cumsum((pred_sorted == 1) & (y_sorted == 0))
    n = np.arange(1, len(y) + 1)
    coverage = n / len(y)
    with np.errstate(invalid="ignore", divide="ignore"):
        selective_accuracy = np.cumsum(correct) / n
        positive_precision_curve = np.where((tp + fp) > 0, tp / np.maximum(tp + fp, 1), np.nan)
    return {
        "coverage": coverage.tolist(),
        "selective_accuracy": selective_accuracy.tolist(),
        "positive_precision": positive_precision_curve.tolist(),
        "positive_recall": (tp / max(int(y.sum()), 1)).tolist(),
    }


def operating_points(
    y_true: ArrayLike, prob_positive: ArrayLike, targets: Sequence[float] = (0.80, 0.90, 0.95)
) -> list[dict[str, float]]:
    """Smallest alert volume that still reaches each target recall.

    For each target recall, choose the highest threshold (lowest alert volume) that
    meets it and report the resulting precision. This is the number that decides
    whether a screening model is usable, and it is invisible in a single F1.
    """
    y = _as_int_array(y_true)
    p = np.clip(np.asarray(prob_positive, dtype=float), 0.0, 1.0)
    out: list[dict[str, float]] = []
    for target in targets:
        best: dict[str, float] | None = None
        for threshold in np.unique(np.round(p, 6)):
            pred = (p >= threshold).astype(int)
            recall = positive_recall(y, pred)
            if recall + 1e-12 < target:
                continue
            row = {
                "target_recall": float(target),
                "threshold": float(threshold),
                "recall": float(recall),
                "precision": positive_precision(y, pred),
                "flagged_fraction": float(pred.mean()),
            }
            if best is None or row["flagged_fraction"] < best["flagged_fraction"]:
                best = row
        if best is not None:
            out.append(best)
    return out
