# eval — re-evaluation of the suicide-ideation classifiers

This package exists because the repository's headline comparison (FastText + LSTM at F1 0.79
versus fine-tuned IndoBERT at 0.90) was evaluated on a random 80/20 split of a corpus in which
2,161 of 10,128 posts are exact duplicates. A third of the validation set was therefore available
during training.

It answers four questions and nothing else:

1. How much duplicate overlap is there between the training and validation halves? (**739 of
   2,026 validation rows, 36.5%**)
2. What is the transformer's performance on rows it has genuinely never seen, and how does that
   compare with cheap classical baselines trained on the same rows?
3. Is the difference between the models statistically distinguishable at this sample size?
4. Are the predicted probabilities calibrated, and what does a clinically motivated operating
   point cost in review volume?

## Run

```bash
uv sync --project eval
uv run --project eval python -m si_eval.run              # ~2 minutes, 2,000 bootstrap resamples
uv run --project eval python -m si_eval.run --quick      # no intervals, ~10 seconds
uv run --project eval pytest eval/tests -q
```

Outputs land in `out/<run-id>/`: `results.json`, `manifest.json`, `REPORT.md` (generated, not
hand-written) and five figures. The manifest records the corpus SHA-256, the git revision and every
package version.

## Modules

| File | Responsibility |
|---|---|
| `si_eval/data.py` | Corpus loading, exact reproduction of the published split, normalised duplicate detection, deduplicated protocols |
| `si_eval/metrics.py` | Stratified bootstrap intervals, McNemar (exact and chi-square), paired bootstrap differences, ECE/Brier, risk–coverage, operating points |
| `si_eval/baselines.py` | ComplementNB, TF-IDF word and character n-gram logistic regression, majority class |
| `si_eval/analysis.py` | Orchestration: leakage, baselines under both training treatments, calibration, error buckets |
| `si_eval/figures.py` | Five matplotlib figures, no seaborn |
| `si_eval/run.py` | CLI, manifest, generated report |

## Conventions

- Positive class is 1 (ideation present) everywhere; that is the class the thesis reports.
- "Unseen rows" always means validation rows whose normalised text does not occur in training.
- Intervals are 95% percentile bootstraps with class counts held fixed (2,000 resamples).
- Anything that cannot be recomputed (the FastText + LSTM arm) is labelled as quoted, not measured.
