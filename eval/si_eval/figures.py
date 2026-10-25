"""Figures for the re-evaluation. Matplotlib only, Agg backend, no seaborn."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# The host's ~/.config is not writable for this profile; keep the font cache local.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#1b1b1b"
MUTED = "#8a8579"
ACCENT = "#b8935a"
RED = "#a4443a"
BLUE = "#4a6d8c"
GRID = "#dcd8cf"

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def leakage_figure(results: dict, path: Path) -> Path:
    split = results["split"]
    leaked = split["val_rows_duplicated_in_train"]
    clean = split["n_clean_val"]
    fig, ax = plt.subplots(figsize=(6.4, 1.9))
    ax.barh([0], [leaked], color=RED, height=0.5, label=f"seen in training ({leaked})")
    ax.barh([0], [clean], left=[leaked], color=BLUE, height=0.5, label=f"unseen ({clean})")
    ax.set_yticks([])
    ax.set_xlabel("validation rows")
    ax.set_title(
        f"{leaked} of {split['n_val']} validation rows ({split['val_leaked_fraction']:.1%}) "
        "have a near-duplicate in the training half",
        fontsize=9.5,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def scores_figure(results: dict, path: Path) -> Path:
    rows = [
        ("IndoBERT (published split)", results["indobert"]["full_val"]["positive_f1"], None, RED),
        (
            "IndoBERT (unseen rows)",
            results["indobert"]["clean_val"]["positive_f1"],
            results["indobert"].get("clean_val_intervals", {}).get("positive_f1"),
            RED,
        ),
    ]
    for entry in results["baselines"]:
        if entry["training_data"] != "train_deduplicated":
            continue
        ci = entry.get("clean_val_f1_ci")
        rows.append(
            (
                entry["model"].replace("_", " "),
                entry["clean_val"]["positive_f1"],
                ci,
                BLUE if entry["model"] != "majority_class" else MUTED,
            )
        )
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[3] for r in rows]
    err_low = []
    err_high = []
    for _, value, ci, _ in rows:
        if ci:
            err_low.append(max(0.0, value - float(ci["low"])))
            err_high.append(max(0.0, float(ci["high"]) - value))
        else:
            err_low.append(0.0)
            err_high.append(0.0)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    positions = np.arange(len(rows))
    ax.barh(positions, values, color=colors, height=0.6)
    ax.errorbar(
        values,
        positions,
        xerr=[err_low, err_high],
        fmt="none",
        ecolor=INK,
        elinewidth=1.0,
        capsize=3,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("positive-class F1 (bars: 95% bootstrap interval where computed)")
    ax.grid(axis="y", visible=False)
    ax.set_title(
        "Positive-class F1: published split vs unseen validation rows",
        fontsize=9.5,
        loc="left",
    )
    return _save(fig, path)


def calibration_figure(results: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="perfect calibration")
    bins = results["indobert"]["clean_val_calibration"]["bins"]
    xs = [b["mean_confidence"] for b in bins]
    ys = [b["fraction_positive"] for b in bins]
    sizes = [max(12.0, b["n"] / 8.0) for b in bins]
    ax.scatter(xs, ys, s=sizes, color=RED, alpha=0.85, label="IndoBERT (unseen rows)", zorder=3)
    ax.plot(xs, ys, color=RED, linewidth=1, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability of ideation")
    ax.set_ylabel("observed fraction positive")
    ece = results["indobert"]["clean_val_calibration"]["ece"]
    brier = results["indobert"]["clean_val_calibration"]["brier"]
    ax.set_title(f"Reliability diagram — ECE {ece:.3f}, Brier {brier:.3f}", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, path)


def risk_coverage_figure(results: dict, path: Path) -> Path:
    curve = results["indobert"]["risk_coverage_sampled"]
    coverage = np.array(curve["coverage"])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(coverage, curve["selective_accuracy"], color=RED, label="accuracy on accepted rows")
    precision = np.array(curve["positive_precision"], dtype=float)
    ax.plot(coverage, precision, color=BLUE, label="positive-class precision on accepted rows")
    ax.set_xlabel("coverage (share of validation rows the model commits to)")
    ax.set_ylabel("score")
    ax.set_ylim(0.5, 1.02)
    ax.set_xlim(coverage.min(), 1.0)
    for target in (0.80, 0.90):
        ax.axhline(target, color=GRID, linewidth=0.8, zorder=0)
    ax.set_title(
        "Selective prediction: accuracy is bought by abstaining on the least confident rows",
        fontsize=9.5,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    return _save(fig, path)


def error_bucket_figure(results: dict, path: Path) -> Path:
    buckets = results["indobert"]["error_buckets_clean"]
    names = [k for k in buckets if k != "all rows"]
    fn = [buckets[k]["fn_rate"] * 100 for k in names]
    fp = [buckets[k]["fp_rate"] * 100 for k in names]
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.bar(positions - 0.2, fn, width=0.4, color=RED, label="false-negative rate (missed ideation)")
    ax.bar(positions + 0.2, fp, width=0.4, color=BLUE, label="false-positive rate")
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("% of that bucket")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Where the remaining errors sit (IndoBERT, unseen rows)", fontsize=9.5, loc="left")
    return _save(fig, path)
