"""The committed re-annotation artifact must match its own report.

`dataset/relabelled_v1.csv` and `audit/out/relabel-final-report.json` are generated together by
`audit/assemble_relabelled.py`. If someone regenerates one without the other — or edits the CSV —
these assertions fail instead of leaving a card that describes a file it no longer matches.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "dataset" / "relabelled_v1.csv"
REPORT = REPO / "audit" / "out" / "relabel-final-report.json"


def test_dataset_and_report_exist():
    assert CSV.exists(), "dataset/relabelled_v1.csv is missing"
    assert REPORT.exists(), "audit/out/relabel-final-report.json is missing"


def test_row_count_and_coverage():
    d = pd.read_csv(CSV)
    r = json.loads(REPORT.read_text())
    assert len(d) == r["corpus_rows"] == 10128
    scored = d["label_machine"].notna().sum()
    assert scored == r["rows_with_a_machine_verdict"]
    assert scored / len(d) > 0.999


def test_agreement_figures_match_the_report():
    d = pd.read_csv(CSV)
    r = json.loads(REPORT.read_text())
    s = d[d["label_machine"].notna()].copy()
    s["machine"] = s["label_machine"].astype(int)
    assert round(float((s["machine"] == s["label_published"]).mean()), 4) == r["agreement_overall"]
    una = s[s["coder_agreement"] == "unanimous"]
    disagree = int((una["machine"] != una["label_published"]).sum())
    assert disagree == r["disagreements_unanimous"]
    assert round(disagree / len(una), 4) == r["estimated_label_error_rate"]


def test_every_row_keeps_its_published_label_and_coders():
    d = pd.read_csv(CSV)
    assert d["label_published"].isin([0, 1]).all()
    complete = d[["ACN", "BME", "AIP"]].notna().all(axis=1)
    assert complete.sum() >= 10126  # two workbook rows are missing a coder label
    assert (d.loc[complete, "coder_agreement"] != "incomplete").all()
    assert (d.loc[complete, "coder_agreement"].isin(["unanimous", "split"])).all()
    assert d["label_adjudicated"].isna().all() or (d["label_adjudicated"].fillna("") == "").all()
