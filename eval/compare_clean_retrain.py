"""Score the clean-split retrain against baselines trained on the same clean split.

``main/train_clean_split.py`` deduplicates the corpus, draws a fresh 80/20 split with no
cross-split duplicates, fine-tunes IndoBERT and writes per-row predictions. This script
puts those predictions next to the classical baselines on the *same* rows, so the
comparison is like-for-like under the clean protocol rather than against the audited
split.

    uv run --project eval python eval/compare_clean_retrain.py \
        --predictions indobert_clean_split/predictions.csv

Writes ``eval/out/clean-retrain-comparison/`` with results.json and COMPARISON.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from si_eval.baselines import all_baselines  # noqa: E402
from si_eval.data import DEFAULT_DATASET, DEFAULT_SHEET, clean_split, load_corpus  # noqa: E402
from si_eval.metrics import (  # noqa: E402
    bootstrap_ci,
    calibration,
    mcnemar_test,
    operating_points,
    paired_bootstrap_diff,
    positive_f1,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("indobert_clean_split/predictions.csv"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/out/clean-retrain-comparison"))
    args = parser.parse_args()

    if not args.predictions.exists():
        print(f"no predictions at {args.predictions}", file=sys.stderr)
        return 2

    corpus = load_corpus(args.dataset, args.sheet)
    split = clean_split(corpus, seed=args.seed)
    preds = pd.read_csv(args.predictions)

    if len(preds) != split.n_val:
        print(
            f"prediction rows {len(preds)} != clean validation rows {split.n_val}; "
            "the split or the predictions file has changed",
            file=sys.stderr,
        )
        return 2
    if list(preds["gold"].astype(int)) != list(split.val_label):
        print("gold labels do not align positionally with the reproduced clean split", file=sys.stderr)
        return 2

    y_true = np.asarray(split.val_label)
    indobert_pred = preds["pred"].astype(int).to_numpy()
    indobert_prob = preds["prob_positive"].astype(float).to_numpy()

    rows = [
        {
            "model": "indobert_clean_split",
            "positive_f1": float(positive_f1(y_true, indobert_pred)),
            "f1_ci": bootstrap_ci(y_true, indobert_pred, positive_f1, n_boot=args.n_boot, seed=args.seed).as_dict(),
        }
    ]
    baseline_preds: dict[str, np.ndarray] = {}
    for baseline in all_baselines():
        baseline.fit(list(split.train_text), [int(v) for v in split.train_label])
        pred = baseline.predict(list(split.val_text))
        baseline_preds[baseline.name] = pred
        rows.append(
            {
                "model": baseline.name,
                "positive_f1": float(positive_f1(y_true, pred)),
                "f1_ci": bootstrap_ci(y_true, pred, positive_f1, n_boot=args.n_boot, seed=args.seed).as_dict(),
            }
        )

    ranked = sorted(rows[1:], key=lambda r: r["positive_f1"], reverse=True)
    best = ranked[0]
    best_pred = baseline_preds[best["model"]]
    comparison = {
        "baseline": best["model"],
        "mcnemar": mcnemar_test(y_true, indobert_pred, best_pred).as_dict(),
        "paired_bootstrap_f1_diff": paired_bootstrap_diff(
            y_true, indobert_pred, best_pred, n_boot=args.n_boot, seed=args.seed
        ),
    }

    results = {
        "protocol": "deduplicated corpus, 80/20 split with no cross-split duplicates",
        "n_train": split.n_train,
        "n_val": split.n_val,
        "n_val_positives": int(y_true.sum()),
        "cross_split_duplicates": int(sum(split.val_leaked)),
        "models": rows,
        "comparison": comparison,
        "indobert_calibration": calibration(y_true, indobert_prob).as_dict(),
        "indobert_operating_points": operating_points(y_true, indobert_prob),
    }

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    def fmt(row: dict) -> str:
        ci = row["f1_ci"]
        return f"{row['positive_f1']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"

    lines = [
        "# Clean-split comparison",
        "",
        f"Deduplicated corpus, then an 80/20 split with no cross-split duplicates: "
        f"{split.n_train} train / {split.n_val} validation ({int(y_true.sum())} positive), "
        f"cross-split duplicates = {int(sum(split.val_leaked))}.",
        "",
        "| Model | Positive-class F1 on the clean validation half (95% CI) |",
        "|---|---|",
    ]
    for row in sorted(rows, key=lambda r: r["positive_f1"], reverse=True):
        lines.append(f"| {row['model'].replace('_', ' ')} | {fmt(row)} |")
    diff = comparison["paired_bootstrap_f1_diff"]
    mc = comparison["mcnemar"]
    lines += [
        "",
        f"IndoBERT vs the strongest baseline (`{best['model']}`): "
        f"**{diff['difference']:+.3f}** F1 "
        f"[{diff['low']:+.3f}, {diff['high']:+.3f}], McNemar p = {mc['p_value']:.4g} "
        f"({mc['a_correct_b_wrong']} / {mc['a_wrong_b_correct']} discordant pairs).",
        "",
        "This is the protocol the audit in `eval/` recommends: deduplicate first, then split.",
        "The training half is smaller than the published one, so the intervals are wider; that",
        "is the cost of removing the duplicate overlap rather than adjusting for it.",
        "",
    ]
    (out / "COMPARISON.md").write_text("\n".join(lines))

    print(f"n_train={split.n_train} n_val={split.n_val} pos={int(y_true.sum())}")
    for row in sorted(rows, key=lambda r: r["positive_f1"], reverse=True):
        print(f"  {row['model']:28s} F1 {fmt(row)}")
    print(
        f"IndoBERT - {best['model']}: {diff['difference']:+.3f} "
        f"[{diff['low']:+.3f}, {diff['high']:+.3f}]  McNemar p={mc['p_value']:.4g}"
    )
    print(f"wrote {out}/COMPARISON.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
