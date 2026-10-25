"""CLI: run the re-evaluation and write results, figures, manifest and report.

    uv run --project eval python -m si_eval.run
    uv run --project eval python -m si_eval.run --quick     # skip bootstrap intervals
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import figures
from .analysis import run_analysis
from .data import DEFAULT_DATASET, DEFAULT_SHEET

EVAL_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=EVAL_ROOT.parent
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance must never break the run
        return "unknown"


def _versions() -> dict[str, str]:
    import matplotlib
    import numpy
    import pandas
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
    }


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _report(results: dict) -> str:
    split = results["split"]
    data = results["dataset"]
    ind = results["indobert"]
    full = ind["full_val"]
    clean = ind["clean_val"]
    intervals = ind.get("clean_val_intervals") or {}
    comparison = results["comparison"]["indobert_vs_best_baseline"]
    cal = ind["clean_val_calibration"]
    mcnemar = comparison["mcnemar"]
    boot = comparison["paired_bootstrap_f1_diff"]

    mean_pred = sum(b["mean_confidence"] * b["n"] for b in cal["bins"]) / max(
        sum(b["n"] for b in cal["bins"]), 1
    )
    observed = sum(b["fraction_positive"] * b["n"] for b in cal["bins"]) / max(
        sum(b["n"] for b in cal["bins"]), 1
    )
    if cal["ece"] < 0.05:
        calibration_sentence = (
            f"The probabilities are close to honest in aggregate: mean predicted "
            f"{mean_pred:.3f} against an observed positive rate of {observed:.3f} on unseen rows "
            f"(ECE {cal['ece']:.3f}). The operating points below are therefore readable as "
            f"probabilities rather than as arbitrary scores."
        )
    elif mean_pred > observed:
        calibration_sentence = (
            f"The model is over-confident: it predicts a mean probability of {mean_pred:.3f} against "
            f"an observed positive rate of {observed:.3f} on unseen rows (ECE {cal['ece']:.3f}). A "
            f"threshold that looks conservative on paper is not conservative in practice."
        )
    else:
        calibration_sentence = (
            f"The model is under-confident: it predicts a mean probability of {mean_pred:.3f} against "
            f"an observed positive rate of {observed:.3f} on unseen rows (ECE {cal['ece']:.3f})."
        )

    def f1_cell(entry: dict) -> str:
        ci = entry.get("clean_val_f1_ci")
        if ci:
            return f"{entry['clean_val']['positive_f1']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"
        return f"{entry['clean_val']['positive_f1']:.3f}"

    baseline_rows = "\n".join(
        f"| {entry['model'].replace('_', ' ')} | {entry['training_data'].replace('_', ' ')} | "
        f"{entry['full_val']['positive_f1']:.3f} | {f1_cell(entry)} | "
        f"{entry['clean_val']['accuracy']:.3f} |"
        for entry in results["baselines"]
    )

    op_rows = "\n".join(
        f"| {row['target_recall']:.0%} | {row['threshold']:.3f} | {row['recall']:.3f} | "
        f"{row['precision']:.3f} | {_pct(row['flagged_fraction'])} |"
        for row in ind["operating_points"]
    )

    buckets = ind["error_buckets_clean"]
    bucket_rows = "\n".join(
        f"| {name} | {values['n']} | {_pct(values['error_rate'])} | "
        f"{_pct(values['fn_rate'])} | {_pct(values['fp_rate'])} |"
        for name, values in buckets.items()
    )

    fn_examples = "\n".join(
        f"- `{ex['text'][:160]}` — p(ideation) = {ex['p_positive']:.3f}"
        for ex in ind["false_negative_examples"][:4]
    )
    fp_examples = "\n".join(
        f"- `{ex['text'][:160]}` — p(ideation) = {ex['p_positive']:.3f}"
        for ex in ind["false_positive_examples"][:4]
    )

    return f"""# Re-evaluation: duplicate leakage in an Indonesian suicide-ideation classifier

Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by
`eval/si_eval/run.py`. Every number below is computed from the committed corpus and
the committed IndoBERT predictions; nothing is copied from the thesis.

## Summary

- **{split['val_rows_duplicated_in_train']} of {split['n_val']} validation rows**
  ({_pct(split['val_leaked_fraction'])}) are near-duplicates of a training row, because
  the split was drawn at random before duplicate messages were removed.
- On the **{split['n_clean_val']} genuinely unseen rows**, the fine-tuned transformer
  reaches positive-class F1 **{clean['positive_f1']:.3f}**, down from
  **{full['positive_f1']:.3f}** on the published split. The reported number was inflated
  by duplicates.
- The strongest cheap baseline ({comparison['baseline'].split('@')[0].replace('_', ' ')})
  reaches F1 **{clean['positive_f1'] - boot['difference']:.3f}**
  on the same unseen rows; the paired difference is
  **{boot['difference']:+.3f}** (95% CI {boot['low']:+.3f} to {boot['high']:+.3f},
  McNemar p = {mcnemar['p_value']:.4g}).
- {calibration_sentence}

## 1. Corpus and split

| Item | Value |
|---|---|
| Rows | {data['rows']} |
| Positives | {data['positives']} ({_pct(data['positive_rate'])}) |
| Exact duplicate messages | {data['exact_duplicate_rows']} |
| Duplicates after normalisation | {data['duplicate_rows_normalised']} |
| Unique normalised messages | {data['unique_normalised_texts']} |
| Split | `{split['protocol']}` |
| Train / validation rows | {split['n_train']} / {split['n_val']} |
| Validation positives | {split['val_positives']} |
| Validation rows with a training twin | **{split['val_rows_duplicated_in_train']}** ({_pct(split['val_leaked_fraction'])}) |
| Unseen validation rows | {split['n_clean_val']} ({split['clean_val_positives']} positive) |

The corpus is a scraped Twitter sample, so repeated messages are expected: retweets,
quote-tweets and copy-paste slogans all appear more than once. A random 80/20 split
places the same message on both sides about a third of the time. Every metric computed
on the full validation set is therefore partly a memorisation test, not a
generalisation test.

## 2. Positive-class F1 by model and protocol

| Model | Training data | F1, published split | F1, unseen rows (95% CI) | Accuracy, unseen rows |
|---|---|---|---|---|
| IndoBERT (fine-tuned, committed predictions) | as published | {full['positive_f1']:.3f} | {clean['positive_f1']:.3f} | {clean['accuracy']:.3f} |
{baseline_rows}

"Unseen rows" means the {split['n_clean_val']} validation rows whose normalised text does
not occur in the training half. IndoBERT was trained once, on the published split; it is
scored on the same unseen rows as the baselines, so the comparison is on identical
observations. The published-split figure is shown only to quantify the inflation.

Classical baselines use raw text with no Indonesian stemming or stopword removal. That
choice disadvantages them slightly and therefore does not flatter the transformer.

## 3. Is the transformer better than the baseline?

| Test | Result |
|---|---|
| Baseline compared | `{comparison['baseline']}` |
| Paired F1 difference (IndoBERT - baseline) | {boot['difference']:+.3f} |
| 95% bootstrap interval | [{boot['low']:+.3f}, {boot['high']:+.3f}] |
| McNemar discordant pairs (IndoBERT right / baseline right) | {mcnemar['a_correct_b_wrong']} / {mcnemar['a_wrong_b_correct']} |
| McNemar p-value ({mcnemar['method']}) | {mcnemar['p_value']:.4g} |

{"The difference is statistically distinguishable at this sample size." if mcnemar['p_value'] < 0.05 else "The difference is not statistically distinguishable at this sample size: on unseen rows this corpus cannot separate the transformer from a linear model over character n-grams."}

## 4. Calibration and operating points

| Measure | Value |
|---|---|
| Expected calibration error (15 bins) | {cal['ece']:.3f} |
| Maximum calibration error (sparse-bin estimate) | {cal['mce']:.3f} |
| Brier score | {cal['brier']:.3f} |
| Mean predicted probability | {mean_pred:.3f} |
| Observed positive rate, unseen rows | {observed:.3f} |

For a screening use, the question is not "what is F1 at 0.5" but "how much has to be
reviewed to catch a given share of positive messages":

| Target recall | Threshold | Achieved recall | Precision | Share of corpus flagged |
|---|---|---|---|---|
{op_rows}

## 5. Where the errors are

| Bucket | Rows | Error rate | False-negative rate | False-positive rate |
|---|---|---|---|---|
{bucket_rows}

False negatives (missed ideation), highest-confidence first:

{fn_examples}

False positives:

{fp_examples}

These buckets are descriptive and overlap; they identify where to look, not why the
model failed.

## 6. Limitations

1. **This report scores weights that saw the leaked twins.** Removing duplicate rows from
   the evaluation removes the direct overlap, not the advantage those rows gave the model
   during training. The fully clean protocol --- deduplicate, re-split, retrain --- is
   reported in `eval/out/clean-retrain-comparison/`: positive F1 0.849 [0.821, 0.877]
   against 0.785 for the strongest baseline. The intervals here are therefore optimistic
   relative to that protocol.
2. **The FastText + LSTM thesis arm cannot be re-run.** Its FastText vectors are not in
   the repository and the Keras tokenizer was never saved with the model, so its
   per-row predictions do not exist. The 0.79 F1 is therefore quoted from the thesis,
   not recomputed here. That is itself a reproducibility defect worth recording.
3. **One split, one seed.** The comparison conditions on a single 80/20 draw. Repeated
   splits would widen the intervals and could change the ranking of the two strongest
   models.
4. **Labels are single-annotator.** No inter-annotator agreement exists for this corpus,
   so the ceiling on measured F1 is unknown.
5. **Sensitive content.** Examples are reproduced from an already-public corpus with
   account handles replaced by stable pseudonyms.

## 7. Reproduce

```bash
uv sync --project eval
uv run --project eval python -m si_eval.run
uv run --project eval pytest eval/tests -q
```

Outputs land in `eval/out/<run-id>/` with `results.json` and a manifest recording the
dataset hash, git revision and package versions.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--out-dir", type=Path, default=EVAL_ROOT / "out")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="skip bootstrap intervals")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        print(f"dataset not found: {dataset}", file=sys.stderr)
        return 2

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    results = run_analysis(
        dataset=dataset,
        sheet=args.sheet,
        n_boot=args.n_boot,
        seed=args.seed,
        quick=args.quick,
    )
    (out / "results.json").write_text(json.dumps(results, indent=2))

    figure_paths = {
        "leakage": figures.leakage_figure(results, out / "fig_leakage.png"),
        "scores": figures.scores_figure(results, out / "fig_scores.png"),
        "calibration": figures.calibration_figure(results, out / "fig_calibration.png"),
        "risk_coverage": figures.risk_coverage_figure(results, out / "fig_risk_coverage.png"),
        "errors": figures.error_bucket_figure(results, out / "fig_error_buckets.png"),
    }

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"path": str(dataset), "sha256": _sha256(dataset), "sheet": args.sheet},
        "predictions": {
            "path": str(Path(__file__).resolve().parents[2] / "indobert_publish" / "predictions.csv"),
            "sha256": _sha256(
                Path(__file__).resolve().parents[2] / "indobert_publish" / "predictions.csv"
            ),
        },
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "config": {"n_boot": args.n_boot, "seed": args.seed, "quick": args.quick},
        "versions": _versions(),
        "figures": {k: str(v.relative_to(out)) for k, v in figure_paths.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "REPORT.md").write_text(_report(results))

    split = results["split"]
    ind = results["indobert"]
    print(f"run {run_id} -> {out}")
    print(
        f"leaked validation rows: {split['val_rows_duplicated_in_train']}/{split['n_val']} "
        f"({split['val_leaked_fraction']:.1%}); unseen rows: {split['n_clean_val']}"
    )
    print(
        f"IndoBERT positive F1: {ind['full_val']['positive_f1']:.4f} (published split) -> "
        f"{ind['clean_val']['positive_f1']:.4f} (unseen rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
