"""Turn the relabel ledger into the numbers that decide whether to run it over the whole corpus.

Reads the machine-label ledger written by ``audit/relabel.py`` plus the workbook, and writes
``audit/out/relabel-v1-report.{md,json}``. Aggregates only: no corpus text is printed or written.

The load-bearing number is the disagreement rate on rows the three coders labelled unanimously and
the machine annotator now calls differently — an upper bound on how often the published label is
wrong, with a Wilson interval. On the rows the coders themselves split 2-1 there is no ground truth
to score against; the report shows which coder the machine sides with instead.

Rows are joined on the *selection* rebuilt by ``audit/relabel.py`` for the same mode (its seed is
fixed), because the ledger's ``i`` is a position within that selection, not a corpus row id.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "audit"))
from relabel import load_frame, select  # noqa: E402

CODERS = ["ACN", "BME", "AIP"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="phase1", choices=["phase1", "full", "unlabeled"])
    ap.add_argument("--ledger", default="audit/out/relabel-v1.jsonl")
    ap.add_argument("--out", default="audit/out/relabel-v1-report.md")
    args = ap.parse_args()

    ledger_path = Path(args.ledger) if Path(args.ledger).is_absolute() else REPO / args.ledger
    raw = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    if not raw:
        raise SystemExit("ledger is empty")
    led = pd.DataFrame(raw).drop_duplicates(subset=["i"], keep="last").rename(
        columns={"label": "machine", "rule": "machine_rule", "confidence": "machine_confidence"})

    ten, cur = load_frame()
    rows = select(args.mode, ten, cur)  # same seed as the ledger's producer
    m = rows.merge(led, left_on="row_index", right_on="i", how="inner")
    m = m.assign(published=m["label"].astype(int), machine=m["machine"].astype(float))

    # The corpus contains exact/near duplicates, so a text can appear on several rows. Count each
    # text once for the agreement figures, and use the repeats as a self-consistency check.
    dup = m[m.duplicated("key", keep=False)]
    duplicate_texts = {
        "rows_annotated_more_than_once": int(len(dup)),
        "distinct_texts_repeat_annotated": int(dup["key"].nunique()),
        "repeat_texts_with_one_machine_label": int((dup.groupby("key")["machine"].nunique() <= 1).sum()),
        "repeat_texts_with_differing_published_labels": int((dup.groupby("key")["published"].nunique() > 1).sum()),
    }
    m = m.drop_duplicates(subset=["key"], keep="first")

    rep: dict = {
        "mode": args.mode,
        "ledger": str(ledger_path.relative_to(REPO)),
        "rows_in_ledger": int(len(led)),
        "rows_selected": int(len(rows)),
        "rows_joined": int(len(m)),
        "models": sorted({str(x) for x in m["model"].dropna().unique()}),
        "prompt_versions": sorted({str(x) for x in m["prompt_version"].dropna().unique()}),
        "undecidable": int(m["machine"].isna().sum()),
        "rule_distribution": dict(Counter(m["machine_rule"].astype(str)).most_common()),
        "confidence": {
            "mean": round(float(m["machine_confidence"].astype(float).mean()), 3),
            "below_0.6": int((m["machine_confidence"].astype(float) < 0.6).sum()),
        },
        "positive_rate": {
            "machine": round(float(m["machine"].mean()), 4),
            "published": round(float(m["published"].mean()), 4),
        },
        "agreement_with_published": {},
        "disputed_rows": {},
        "unanimous_rows_estimate": {},
        "duplicate_texts": duplicate_texts,
    }

    scored = m[m["machine"].notna()]
    for stratum, g in scored.groupby("stratum"):
        agree = int((g["machine"].astype(int) == g["published"]).sum())
        rep["agreement_with_published"][str(stratum)] = {
            "n": int(len(g)), "agree": agree, "agreement": round(agree / len(g), 4)}

    dis = scored[scored["disputed"] == True]  # noqa: E712
    if len(dis):
        rep["disputed_rows"] = {
            "n": int(len(dis)),
            "model_sides_with": {c: round(float((dis["machine"].astype(int) == dis[c]).mean()), 4) for c in CODERS},
            "model_matches_published_majority": round(
                float((dis["machine"].astype(int) == dis["published"]).mean()), 4),
        }

    una = scored[scored["disputed"] != True]  # noqa: E712
    if len(una):
        disagree = int((una["machine"].astype(int) != una["published"]).sum())
        low, high = wilson(disagree, len(una))
        rep["unanimous_rows_estimate"] = {
            "n": int(len(una)),
            "machine_disagrees": disagree,
            "estimated_published_label_error_rate": round(disagree / len(una), 4),
            "wilson_95": [low, high],
            "direction": {
                "machine_1_published_0": int(((una["machine"] == 1) & (una["published"] == 0)).sum()),
                "machine_0_published_1": int(((una["machine"] == 0) & (una["published"] == 1)).sum()),
            },
        }
        # disagreement by class and length, to show where the estimate comes from
        una = una.assign(bucket=pd.cut(una["tweet"].astype(str).str.len(), [0, 80, 160, 10_000],
                                       labels=["short", "medium", "long"]))
        rep["unanimous_rows_estimate"]["by_length"] = {
            str(b): {"n": int(len(g)), "disagree": int((g["machine"].astype(int) != g["published"]).sum())}
            for b, g in una.groupby("bucket", observed=True)
        }
        rep["unanimous_rows_estimate"]["by_published_label"] = {
            str(int(lab)): {"n": int(len(g)), "disagree": int((g["machine"].astype(int) != g["published"]).sum())}
            for lab, g in una.groupby("published")
        }

    out_md = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    lines = [
        "# Re-annotation pass 1 — machine labels vs the published labels",
        "",
        f"Mode `{rep['mode']}` · ledger `{rep['ledger']}` · "
        f"{rep['rows_in_ledger']}/{rep['rows_selected']} rows annotated · "
        f"model {', '.join(rep['models'])} · prompt {', '.join(rep['prompt_versions'])} · "
        f"undecidable {rep['undecidable']}",
        "",
        "## Agreement with the published (three-coder majority) label",
        "",
        "| stratum | n | agree | agreement |",
        "|---|---|---|---|",
    ]
    for stratum, v in rep["agreement_with_published"].items():
        lines.append(f"| {stratum} | {v['n']} | {v['agree']} | {v['agreement']:.2%} |")
    if rep["disputed_rows"]:
        d = rep["disputed_rows"]
        lines += [
            "",
            f"## Rows the coders split 2-1 (n = {d['n']})",
            "",
            f"- the machine matches the published majority in {d['model_matches_published_majority']:.2%} of them",
            "- machine sides with: " + ", ".join(f"{k} {v:.2%}" for k, v in d["model_sides_with"].items()),
        ]
    if rep["unanimous_rows_estimate"]:
        u = rep["unanimous_rows_estimate"]
        lines += [
            "",
            f"## Rows all three coders agreed on (n = {u['n']})",
            "",
            f"- the machine disagrees on {u['machine_disagrees']} "
            f"({u['estimated_published_label_error_rate']:.2%}, Wilson 95% "
            f"[{u['wilson_95'][0]:.2%}, {u['wilson_95'][1]:.2%}])",
            f"- direction: {u['direction']['machine_1_published_0']} machine-positive/published-negative, "
            f"{u['direction']['machine_0_published_1']} machine-negative/published-positive",
            "- by published label: " + ", ".join(f"{k}: {v['disagree']}/{v['n']}"
                                                 for k, v in u["by_published_label"].items()),
            "- by length: " + ", ".join(f"{k}: {v['disagree']}/{v['n']}" for k, v in u["by_length"].items()),
            "",
            "This is an upper bound on how often the published label is wrong: it also counts the places "
            "where the machine annotator is the one who is wrong. Every disagreed row is enumerable for "
            "human adjudication.",
        ]
    lines += [
        "",
        "## Duplicate texts",
        "",
        f"- {duplicate_texts['rows_annotated_more_than_once']} rows share their text with another row "
        f"({duplicate_texts['distinct_texts_repeat_annotated']} distinct texts); the agreement figures above "
        "count each text once",
        f"- of those repeated texts, {duplicate_texts['repeat_texts_with_one_machine_label']} got a single machine "
        f"label and {duplicate_texts['repeat_texts_with_differing_published_labels']} carry differing *published* "
        "labels (same text, different gold label)",
        "",
        "## The machine annotator's own behaviour",
        "",
        f"- positive rate: machine {rep['positive_rate']['machine']:.3f} vs published "
        f"{rep['positive_rate']['published']:.3f}",
        f"- mean confidence {rep['confidence']['mean']}, {rep['confidence']['below_0.6']} rows below 0.6",
        "- rules: " + ", ".join(f"{k} {v}" for k, v in rep["rule_distribution"].items()),
    ]
    out_md.write_text("\n".join(lines) + "\n")
    (out_md.with_suffix(".json")).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2)[:2500])
    print("wrote", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
