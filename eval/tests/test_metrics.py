import numpy as np
import pytest

from si_eval.metrics import (
    accuracy,
    bootstrap_ci,
    calibration,
    mcnemar_test,
    operating_points,
    paired_bootstrap_diff,
    positive_f1,
    positive_precision,
    positive_recall,
    risk_coverage,
)


def test_positive_metrics_on_known_confusion():
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 1, 0]
    # TP=2, FN=1, FP=1, TN=2
    assert positive_recall(y_true, y_pred) == pytest.approx(2 / 3)
    assert positive_precision(y_true, y_pred) == pytest.approx(2 / 3)
    assert positive_f1(y_true, y_pred) == pytest.approx(2 / 3)
    assert accuracy(y_true, y_pred) == pytest.approx(4 / 6)


def test_bootstrap_interval_brackets_point_and_is_reproducible():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=400)
    y_pred = np.where(rng.random(400) < 0.85, y_true, 1 - y_true)
    first = bootstrap_ci(y_true, y_pred, positive_f1, n_boot=300, seed=7)
    second = bootstrap_ci(y_true, y_pred, positive_f1, n_boot=300, seed=7)
    assert first.low <= first.point <= first.high
    assert (first.low, first.high) == (second.low, second.high)
    assert 0.0 <= first.low <= first.high <= 1.0


def test_mcnemar_exact_when_discordant_pairs_are_few():
    y_true = [1] * 10 + [0] * 10
    pred_a = list(y_true)
    pred_b = list(y_true)
    pred_a[0] = 0  # a wrong once
    result = mcnemar_test(y_true, pred_a, pred_b)
    assert result.method == "exact_binomial"
    assert result.a_correct_b_wrong == 0
    assert result.a_wrong_b_correct == 1
    assert 0.0 <= result.p_value <= 1.0


def test_mcnemar_identical_classifiers_is_not_significant():
    y_true = [1, 0, 1, 0]
    result = mcnemar_test(y_true, y_true, y_true)
    assert result.p_value == 1.0


def test_calibration_perfect_and_overconfident():
    perfectly_confident = calibration([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0], n_bins=5)
    assert perfectly_confident.ece == pytest.approx(0.0, abs=1e-9)
    assert perfectly_confident.brier == pytest.approx(0.0, abs=1e-9)

    overconfident = calibration([0, 0, 0, 0], [0.99, 0.99, 0.99, 0.99], n_bins=10)
    assert overconfident.ece == pytest.approx(0.99, abs=1e-9)
    assert overconfident.mce == pytest.approx(0.99, abs=1e-9)


def test_calibration_bins_are_weighted_by_count():
    # 3 confident-and-correct, 1 confident-and-wrong, in one bin
    result = calibration([1, 1, 1, 0], [0.9, 0.9, 0.9, 0.9], n_bins=1)
    assert result.bins[0]["n"] == 4
    assert result.bins[0]["fraction_positive"] == pytest.approx(0.75)
    assert result.ece == pytest.approx(0.15)


def test_risk_coverage_is_monotone_in_coverage():
    probs = [0.1, 0.2, 0.8, 0.9, 0.55]
    labels = [0, 0, 1, 1, 0]
    curve = risk_coverage(labels, probs)
    coverage = curve["coverage"]
    assert coverage[0] == pytest.approx(0.2)
    assert coverage[-1] == pytest.approx(1.0)
    assert all(b > a for a, b in zip(coverage, coverage[1:]))
    # The most confident row (0.9, label 1) must be accepted first.
    assert curve["selective_accuracy"][0] == pytest.approx(1.0)


def test_operating_points_meet_their_recall_target():
    probs = [0.95, 0.9, 0.85, 0.4, 0.3, 0.2, 0.1, 0.05]
    labels = [1, 1, 0, 1, 0, 0, 0, 0]
    points = operating_points(labels, probs, targets=(0.66, 1.0))
    for point in points:
        assert point["recall"] >= point["target_recall"] - 1e-9
        assert 0.0 < point["flagged_fraction"] <= 1.0
    # Perfect recall needs a lower threshold than 2/3 recall.
    assert points[0]["threshold"] > points[1]["threshold"]


def test_paired_bootstrap_detects_identical_models():
    y_true = [1, 0, 1, 0, 1, 0] * 20
    pred = list(y_true)
    pred[0] = 0
    result = paired_bootstrap_diff(y_true, pred, pred, n_boot=50, seed=1)
    assert result["difference"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)
