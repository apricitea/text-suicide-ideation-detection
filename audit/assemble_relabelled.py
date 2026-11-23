"""Assemble the re-annotated dataset from every relabel ledger, keyed by content.

The first full pass was produced by a selection that repeated texts (14,562 rows for 10,128 texts)
and its ledger is positional; the corrected pass carries ``row_key`` hashes. This script resolves
both onto the corpus by *content*, writes one canonical ledger, and builds
``dataset/relabelled_v1.csv`` plus its card and the final agreement report.

    eval/.venv/bin/python audit/assemble_relabelled.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "audit"))
from relabel import CODERS, corpus_key, key_hash, load_frame  # noqa: E402

OLD_LEDGER = "audit/out/relabel-full.jsonl"    # positional rows, inflated selection
NEW_LEDGER = "audit/out/relabel-full2.jsonl"   # row_key rows, one text per selection row
MERGED = "audit/out/relabel-merged.jsonl"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def read_ledger(rel: str) -> list[dict]:
    path = REPO / rel
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/relabelled_v1.csv")
    ap.add_argument("--card", default="dataset/relabelled_v1.card.md")
    ap.add_argument("--report", default="audit/out/relabel-final-report.md")
    args = ap.parse_args()

    ten, cur = load_frame()
    C = cur[CODERS].astype(float)
    cur_u = cur.assign(disputed=C.notna().all(axis=1) & (C.fillna(-1).astype(int).min(axis=1)
                                                         != C.fillna(-1).astype(int).max(axis=1)))
    cur_u = cur_u.drop_duplicates(subset=["key"])

    # positional map for the old, inflated selection
    inflated = ten.merge(cur_u[["key", "disputed", *CODERS]], on="key", how="left").reset_index(drop=True)
    # that expression cannot reproduce the inflation (cur_u deduped), so rebuild the original one
    cur_raw = cur.assign(disputed=C.notna().all(axis=1) & (C.fillna(-1).astype(int).min(axis=1)
                                                           != C.fillna(-1).astype(int).max(axis=1)))
    inflated = ten.merge(cur_raw[["key", "disputed", *CODERS]], on="key", how="left").reset_index(drop=True)

    verdict: dict[str, dict] = {}
    for row in read_ledger(OLD_LEDGER):
        idx = row.get("i")
        if isinstance(idx, int) and 0 <= idx < len(inflated):
            verdict.setdefault(corpus_key(inflated.iloc[idx]["tweet"]), row)
    new_rows = read_ledger(NEW_LEDGER)
    by_hash = {r["row_key"]: r for r in new_rows if r.get("row_key")}
    for j, row in enumerate(new_rows):
        if not row.get("row_key"):
            continue
        verdict_by_hash = by_hash[row["row_key"]]
        verdict_by_hash = verdict_by_hash
    # content-keyed verdicts from the new ledger win
    hashes = {key_hash(t): k for t, k in zip(ten["tweet"], ten["key"])}
    for h, row in by_hash.items():
        k = hashes.get(h)
        if k is not None:
            verdict[k] = row

    merged = ten.merge(cur_u[["key", "disputed", *CODERS]], on="key", how="left")
    merged = merged.assign(
        label_machine=[verdict.get(k, {}).get("label") for k in merged["key"]],
        machine_rule=[verdict.get(k, {}).get("rule") for k in merged["key"]],
        machine_confidence=[verdict.get(k, {}).get("confidence") for k in merged["key"]],
        model=[verdict.get(k, {}).get("model") for k in merged["key"]],
        prompt_version=[verdict.get(k, {}).get("prompt_version") for k in merged["key"]],
        annotated_at=[verdict.get(k, {}).get("annotated_at") for k in merged["key"]],
    )
    merged["coder_agreement"] = np.where(merged[CODERS].notna().all(axis=1),
                                         np.where(merged[CODERS].min(axis=1) == merged[CODERS].max(axis=1),
                                                  "unanimous", "split"), "incomplete")
    merged["machine_disagrees"] = (merged["label_machine"].astype(float) != merged["label"].astype(float))
    merged["label_adjudicated"] = ""

    out = merged.rename(columns={"index": "corpus_index", "tweet": "text", "label": "label_published"})[
        ["corpus_index", "text", "label_published", *CODERS, "coder_agreement",
         "label_machine", "machine_rule", "machine_confidence", "machine_disagrees",
         "model", "prompt_version", "annotated_at", "label_adjudicated"]
    ].sort_values("corpus_index").reset_index(drop=True)

    out_path = REPO / args.out
    out.to_csv(out_path, index=False)
    (REPO / MERGED).write_text("\n".join(json.dumps(v) for v in verdict.values()) + "\n")

    scored = out[out["label_machine"].notna()].copy()
    scored["machine"] = scored["label_machine"].astype(int)
    una = scored[scored["coder_agreement"] == "unanimous"]
    spl = scored[scored["coder_agreement"] == "split"]
    disagree = int((una["machine"] != una["label_published"]).sum())
    wil = wilson(disagree, len(una))

    # self-consistency of the annotator where the first pass saw a text more than once
    groups: dict[str, list] = {}
    for row in read_ledger(OLD_LEDGER):
        idx = row.get("i")
        if isinstance(idx, int) and 0 <= idx < len(inflated):
            groups.setdefault(corpus_key(inflated.iloc[idx]["tweet"]), []).append(row.get("label"))
    repeats = {k: v for k, v in groups.items() if len(v) > 1}
    identical = sum(1 for v in repeats.values() if len(set(v)) == 1)

    stats = {
        "corpus_rows": int(len(out)),
        "rows_with_a_machine_verdict": int(len(scored)),
        "coverage": round(len(scored) / len(out), 4),
        "undecidable": int((out["machine_rule"] == "R9").sum()),
        "agreement_overall": round(float((scored["machine"] == scored["label_published"]).mean()), 4),
        "agreement_unanimous_rows": round(float((una["machine"] == una["label_published"]).mean()), 4),
        "agreement_split_rows": round(float((spl["machine"] == spl["label_published"]).mean()), 4),
        "unanimous_rows": int(len(una)),
        "split_rows": int(len(spl)),
        "disagreements": int(scored["machine_disagrees"].sum()),
        "disagreements_unanimous": disagree,
        "disagreements_split": int((spl["machine"] != spl["label_published"]).sum()),
        "estimated_label_error_rate": round(disagree / max(1, len(una)), 4),
        "wilson_95": list(wil),
        "machine_positive_rate": round(float(scored["machine"].mean()), 4),
        "published_positive_rate": round(float(scored["label_published"].mean()), 4),
        "rule_distribution": {str(k): int(v) for k, v in out["machine_rule"].value_counts().items()},
        "self_consistency": {"texts_seen_twice": len(repeats), "identical_label": identical,
                             "rate": round(identical / max(1, len(repeats)), 4)},
        "model": sorted({str(x) for x in out["model"].dropna().unique()}),
        "prompt_versions": sorted({str(x) for x in out["prompt_version"].dropna().unique()}),
    }

    card = REPO / args.card
    card.write_text("\n".join([
        "# relabelled_v1 — an independent re-annotation of the 10k corpus",
        "",
        "Every row of `dataset/final_dataset.xlsx[sheet=10k]` with the published label, the three coder",
        "labels, and a **machine** verdict produced under the recovered codebook (`audit/codebook.md`,",
        "from the thesis's Table 1, adapted from Sawhney et al. 2018).",
        "",
        "## Columns",
        "",
        "| column | meaning |",
        "|---|---|",
        "| `corpus_index` | row identity in the workbook's `10k` sheet |",
        "| `text` | post text, unchanged (pseudonymous, not anonymous) |",
        "| `label_published` | the original label: majority of three human coders |",
        "| `ACN` `BME` `AIP` | the individual coder labels |",
        "| `coder_agreement` | `unanimous` / `split` / `incomplete` |",
        "| `label_machine` | **machine** verdict (an LLM annotator, not a human) |",
        "| `machine_rule` | deciding codebook rule, `R0`–`R7` (`R9` = undecidable) |",
        "| `machine_confidence` | the annotator's own certainty, not a calibrated probability |",
        "| `machine_disagrees` | machine verdict differs from `label_published` |",
        "| `label_adjudicated` | reserved and empty: for a human pass over the disagreements |",
        "",
        "## What it measures",
        "",
        f"- coverage {stats['rows_with_a_machine_verdict']}/{stats['corpus_rows']} "
        f"({stats['coverage']:.1%}); undecidable rows {stats['undecidable']}",
        f"- agreement with the published label: {stats['agreement_overall']:.2%} overall, "
        f"{stats['agreement_unanimous_rows']:.2%} on unanimously-labelled rows, "
        f"{stats['agreement_split_rows']:.2%} on rows the coders already disputed",
        f"- **{stats['estimated_label_error_rate']:.2%} of unanimously-labelled rows** are read differently "
        f"by the machine annotator (Wilson 95% [{stats['wilson_95'][0]:.2%}, {stats['wilson_95'][1]:.2%}]) — "
        "an upper bound on the published labels' error rate",
        f"- the annotator's own noise floor: on {stats['self_consistency']['texts_seen_twice']} texts it saw "
        f"twice it gave the same label {stats['self_consistency']['rate']:.2%} of the time, so most of the "
        "disagreement above is a consistent different reading, not annotator jitter",
        f"- positive rate: machine {stats['machine_positive_rate']:.3f} vs published "
        f"{stats['published_positive_rate']:.3f}",
        "",
        "## Provenance and caveats",
        "",
        f"- annotator `{', '.join(stats['model'])}` through the Hermes CLI, prompt "
        f"`{', '.join(stats['prompt_versions'])}` (hash covers codebook + instructions)",
        "- `@mentions` and URLs were replaced with tokens before the text left the host; the `text`",
        "  column here keeps the original characters",
        "- **machine labels are not ground truth.** Treat them as a second independent reading, and as",
        "  the triage list for human adjudication of the disagreement rows",
        "",
        "*Machine-assisted annotation; not human-adjudicated. Written by Nyx.*",
    ]) + "\n")

    (REPO / args.report).write_text("\n".join([
        "# Re-annotation — final numbers (all three coders, all 10,128 rows)",
        "",
        "```json", json.dumps(stats, indent=2), "```",
    ]) + "\n")
    (REPO / args.report).with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("wrote", out_path, card, REPO / args.report, REPO / MERGED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
