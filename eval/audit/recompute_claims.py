"""Recompute every number the README, paper and cards publish, from committed artifacts.

Read-only. Prints and writes aggregates; never corpus text.

    cd eval && .venv/bin/python audit/recompute_claims.py            # (from the repo root:)
    eval/.venv/bin/python eval/audit/recompute_claims.py

Writes ``eval/out/audit-20261123/{claims.json,REPORT.md}`` and exits nonzero on any mismatch
against the committed figures, so it can gate CI later.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
from si_eval.data import normalise  # noqa: E402  (their definition, so numbers stay comparable)

OUT = REPO / "eval" / "out" / "audit-20261123"
SHEET = "10k"


def pos_metrics(y, p, prob=None):
    cm = confusion_matrix(y, p, labels=[0, 1])
    out = {
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "confusion_matrix": cm.tolist(),
        "positive_f1": round(float(f1_score(y, p, pos_label=1, zero_division=0)), 7),
        "positive_precision": round(float(precision_score(y, p, pos_label=1, zero_division=0)), 7),
        "positive_recall": round(float(recall_score(y, p, pos_label=1, zero_division=0)), 7),
        "accuracy": round(float((cm[0, 0] + cm[1, 1]) / cm.sum()), 7),
    }
    if prob is not None:
        out["mean_prob_positive"] = round(float(np.mean(prob)), 4)
        out["observed_positive_rate"] = round(float(np.mean(y)), 4)
    return out


def cohens_kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = float((a == b).mean())
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return round((po - pe) / (1 - pe), 4), round(po, 4)


def fleiss_kappa(counts):
    m = counts.sum(axis=1)[0]
    P_i = ((counts ** 2).sum(axis=1) - m) / (m * (m - 1.0))
    p_j = counts.sum(axis=0) / counts.sum()
    return round(float((P_i.mean() - (p_j ** 2).sum()) / (1 - (p_j ** 2).sum())), 4)


def operating_point(y, prob, target):
    order = np.argsort(-prob)
    tp = np.cumsum(y[order] == 1)
    reviewed = np.arange(1, len(y) + 1)
    recall = tp / y.sum()
    i = int(np.argmax(recall >= target))
    return {
        "target_recall": target,
        "recall_achieved": round(float(recall[i]), 4),
        "fraction_reviewed": round(float(reviewed[i] / len(y)), 4),
        "precision_at_cut": round(float(tp[i] / reviewed[i]), 4),
    }


def flatten(obj, prefix=""):
    """results.json is nested; the published keys are dotted paths."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    else:
        out[prefix] = obj
    return out


claims: dict = {"run_id": "audit-20261123", "checks": []}


def check(name, recomputed, published, tol=0.0005, note=""):
    ok = published is not None and abs(float(recomputed) - float(published)) <= tol
    claims["checks"].append(
        {"claim": name, "recomputed": recomputed, "published": published, "match": bool(ok), "note": note}
    )
    return ok


# ── corpus ───────────────────────────────────────────────────────────────────
df = pd.read_excel(REPO / "dataset/final_dataset.xlsx", sheet_name=SHEET)
tweet, label = df["tweet"].astype(str), df["label"].astype(int)
norm = tweet.map(normalise)
run = flatten(json.loads((REPO / "eval/out/20261025-full/results.json").read_text()))

claims["corpus"] = {
    "rows": int(len(df)),
    "positives": int(label.sum()),
    "positive_rate": round(float(label.mean()), 6),
    "exact_duplicate_rows": int(len(tweet) - tweet.nunique()),
    "duplicate_rows_normalised": int(len(norm) - norm.nunique()),
    "unique_normalised_texts": int(norm.nunique()),
}
for key, value in (
    ("dataset.rows", claims["corpus"]["rows"]),
    ("dataset.positives", claims["corpus"]["positives"]),
    ("dataset.exact_duplicate_rows", claims["corpus"]["exact_duplicate_rows"]),
    ("dataset.duplicate_rows_normalised", claims["corpus"]["duplicate_rows_normalised"]),
    ("dataset.unique_normalised_texts", claims["corpus"]["unique_normalised_texts"]),
):
    check(key, value, run.get(key))

# ── split and leakage ────────────────────────────────────────────────────────
x_tr, x_va, y_tr, y_va = train_test_split(tweet, label, train_size=0.8, random_state=1)
train_norm = {normalise(t) for t in x_tr}
val_text, val_leaked = x_va.tolist(), [normalise(t) in train_norm for t in x_va.tolist()]
mask_clean = np.array(val_leaked) == False  # noqa: E712
claims["split"] = {
    "n_train": len(x_tr),
    "n_val": len(x_va),
    "val_positives": int(y_va.sum()),
    "val_rows_duplicated_in_train": int(sum(val_leaked)),
    "val_leaked_fraction": round(sum(val_leaked) / len(val_leaked), 6),
    "n_clean_val": int(mask_clean.sum()),
    "clean_val_positives": int(y_va.values[mask_clean].sum()),
    "dedup_train_rows": len(train_norm),
}
for key, value in (
    ("split.n_train", claims["split"]["n_train"]),
    ("split.n_val", claims["split"]["n_val"]),
    ("split.val_positives", claims["split"]["val_positives"]),
    ("split.val_rows_duplicated_in_train", claims["split"]["val_rows_duplicated_in_train"]),
    ("split.val_leaked_fraction", claims["split"]["val_leaked_fraction"]),
    ("split.n_clean_val", claims["split"]["n_clean_val"]),
    ("split.clean_val_positives", claims["split"]["clean_val_positives"]),
    ("split.dedup_train_rows", claims["split"]["dedup_train_rows"]),
):
    check(key, value, run.get(key))

# ── predictions, recomputed ──────────────────────────────────────────────────
pred = pd.read_csv(REPO / "indobert_publish/predictions.csv")
claims["alignment"] = {
    "prediction_rows": int(len(pred)),
    "val_rows": len(val_text),
    "gold_column_equals_reproduced_split": bool(np.array_equal(pred.gold.values, y_va.values)),
    "text_column_equals_corpus_text_rows": int(sum(a == b for a, b in zip(pred.text.astype(str), val_text))),
    "text_column_matches_note": "predictions.csv carries model-side text, not the corpus text",
}
y_val, p_val, prob_val = y_va.values, pred.pred.values, pred.prob_positive.values

claims["protocol_A_full_val"] = pos_metrics(y_val, p_val, prob_val)
for key, value in (
    ("indobert.full_val.positive_f1", claims["protocol_A_full_val"]["positive_f1"]),
    ("indobert.full_val.positive_precision", claims["protocol_A_full_val"]["positive_precision"]),
    ("indobert.full_val.positive_recall", claims["protocol_A_full_val"]["positive_recall"]),
    ("indobert.full_val.accuracy", claims["protocol_A_full_val"]["accuracy"]),
):
    check(key, value, run.get(key))

claims["protocol_B_clean_val"] = pos_metrics(y_val[mask_clean], p_val[mask_clean], prob_val[mask_clean])
for key, value in (
    ("indobert.clean_val.positive_f1", claims["protocol_B_clean_val"]["positive_f1"]),
    ("indobert.clean_val.positive_precision", claims["protocol_B_clean_val"]["positive_precision"]),
    ("indobert.clean_val.positive_recall", claims["protocol_B_clean_val"]["positive_recall"]),
    ("indobert.clean_val.accuracy", claims["protocol_B_clean_val"]["accuracy"]),
):
    check(key, value, run.get(key))

clean = pd.read_csv(REPO / "indobert_clean_split/predictions.csv")
metrics_c = json.loads((REPO / "indobert_clean_split/metrics.json").read_text())
claims["protocol_C_committed"] = pos_metrics(clean.gold.values, clean.pred.values, clean.prob_positive.values)
check("indobert_clean_split.positive_f1", claims["protocol_C_committed"]["positive_f1"], round(metrics_c["positive_f1"], 7))
check("protocol_C.confusion_matrix", 0, 0 if claims["protocol_C_committed"]["confusion_matrix"] == metrics_c["confusion_matrix"] else 1)
claims["operating_points_protocol_B"] = [operating_point(y_val[mask_clean], prob_val[mask_clean], r) for r in (0.90, 0.80)]
claims["calibration_protocol_B"] = {
    "ece": round(float(run["indobert.clean_val_calibration.ece"]), 5),
    "brier": round(float(run["indobert.clean_val_calibration.brier"]), 5),
    "mean_prob_positive": claims["protocol_B_clean_val"]["mean_prob_positive"],
    "observed_positive_rate": claims["protocol_B_clean_val"]["observed_positive_rate"],
}

# ── annotators ───────────────────────────────────────────────────────────────
cur = pd.read_excel(REPO / "dataset/final_dataset.xlsx", sheet_name="current_data")
coders = ["ACN", "BME", "AIP"]
C = cur[coders].astype(float)
valid = C.notna().all(axis=1).values
Cv = C[valid].astype(int).values
counts = np.stack([(Cv == 0).sum(axis=1), (Cv == 1).sum(axis=1)], axis=1).astype(float)
maj = (Cv.sum(axis=1) >= 2).astype(int)
lab = label.values[valid]
claims["annotation"] = {
    "rows_with_three_labels": int(valid.sum()),
    "unanimous_rows": int((Cv.min(axis=1) == Cv.max(axis=1)).sum()),
    "split_rows": int((Cv.min(axis=1) != Cv.max(axis=1)).sum()),
    "unanimous_share": round(float((Cv.min(axis=1) == Cv.max(axis=1)).mean()), 4),
    "pairwise": {f"{coders[i]}-{coders[j]}": dict(zip(("cohens_kappa", "percent_agreement"), cohens_kappa(Cv[:, i], Cv[:, j])))
                 for i, j in ((0, 1), (0, 2), (1, 2))},
    "fleiss_kappa": fleiss_kappa(counts),
    "published_label_equals_coder_majority_rows": int((lab == maj).sum()),
    "published_label_equals_coder_majority_share": round(float((lab == maj).mean()), 6),
    "row_alignment_10k_vs_current_data": int((df["tweet"].astype(str).str.strip().values
                                              == cur["tweet"].astype(str).str.strip().values).sum()),
}

# model errors against coder disagreement, clean validation only
key = {" ".join(t.split())[:200]: i for i, t in enumerate(cur["tweet"].astype(str))}
idx = np.array([key.get(" ".join(t.split())[:200], -1) for t in val_text])
joined = idx >= 0
dis = np.zeros(len(val_text), dtype=bool)
dis[joined] = (Cv[idx[joined]].min(axis=1) != Cv[idx[joined]].max(axis=1))
wrong = p_val != y_val
claims["errors_vs_disagreement"] = {
    "clean_val_rows": int(mask_clean.sum()),
    "rows_joined_to_current_data": int(joined.sum()),
    "clean_val_rows_coders_disagree": int(dis[mask_clean].sum()),
    "clean_val_model_errors": int(wrong[mask_clean].sum()),
    "errors_on_disagreed_rows": int((wrong & dis)[mask_clean].sum()),
    "errors_on_unanimous_rows": int((wrong & ~dis)[mask_clean].sum()),
}

claims["unused_sheets"] = {
    s: {"rows": int(pd.read_excel(REPO / "dataset/final_dataset.xlsx", sheet_name=s).shape[0])}
    for s in ("800labeled", "8k_labeled", "11k_unlabeled")
}

# ── verdict ──────────────────────────────────────────────────────────────────
failed = [c for c in claims["checks"] if not c["match"]]
claims["summary"] = {
    "checks": len(claims["checks"]),
    "matched": len(claims["checks"]) - len(failed),
    "mismatched": [c["claim"] for c in failed],
    "verdict": "all published numbers reproduce" if not failed else "MISMATCH",
}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "claims.json").write_text(json.dumps(claims, indent=2))

lines = [
    "# Numeric audit — every published figure recomputed from committed artifacts",
    "",
    f"Run id `{claims['run_id']}`. Generated by `eval/audit/recompute_claims.py` (never hand-written).",
    "",
    f"**Verdict: {claims['summary']['verdict']}** — {claims['summary']['matched']}/{claims['summary']['checks']} checks.",
    "",
    "## Checks against `eval/out/20261025-full/results.json`",
    "",
    "| claim | recomputed | published | match |",
    "|---|---|---|---|",
]
for c in claims["checks"]:
    lines.append(f"| `{c['claim']}` | {c['recomputed']} | {c['published']} | {'yes' if c['match'] else '**no**'} |")
lines += [
    "",
    "## Annotation provenance",
    "",
    f"- three coder columns (`ACN`, `BME`, `AIP`) on {claims['annotation']['rows_with_three_labels']} rows, "
    f"{claims['annotation']['unanimous_rows']} unanimous ({claims['annotation']['unanimous_share']:.2%}), "
    f"{claims['annotation']['split_rows']} split 2-1",
    f"- Fleiss kappa {claims['annotation']['fleiss_kappa']} (m=3, k=2); pairwise Cohen's kappa "
    + ", ".join(f"{k} {v['cohens_kappa']}" for k, v in claims["annotation"]["pairwise"].items()),
    f"- the published `label` column equals the coder majority on "
    f"{claims['annotation']['published_label_equals_coder_majority_rows']}/"
    f"{claims['annotation']['rows_with_three_labels']} rows "
    f"({claims['annotation']['published_label_equals_coder_majority_share']:.4f})",
    "",
    "## Model error vs label disagreement (clean validation)",
    "",
    f"- {claims['errors_vs_disagreement']['clean_val_model_errors']} model errors, "
    f"{claims['errors_vs_disagreement']['errors_on_disagreed_rows']} of them on rows the coders disputed, "
    f"{claims['errors_vs_disagreement']['errors_on_unanimous_rows']} on unanimously labelled rows",
    "",
    "## Operating points (protocol B)",
    "",
]
for op in claims["operating_points_protocol_B"]:
    lines.append(f"- recall {op['recall_achieved']:.3f} at {op['fraction_reviewed']:.2%} reviewed, "
                 f"precision {op['precision_at_cut']:.4f} (target {op['target_recall']:.0%})")
lines += [
    "",
    "## Corpus data not used by the published pipeline",
    "",
] + [f"- sheet `{s}`: {v['rows']} rows" for s, v in claims["unused_sheets"].items()]
(OUT / "REPORT.md").write_text("\n".join(lines) + "\n")

print(json.dumps(claims["summary"], indent=2))
print("wrote", OUT)
sys.exit(1 if failed else 0)
