"""The published label is the three coders' majority vote, and agreement is a documented number.

Pins the claims the README, paper and both cards now make: the corpus carries three independent
coder columns, the label equals their majority, and the agreement figures are reproducible. If a
future edit reloads the workbook, changes the label column or drops a coder column, this fails
instead of silently invalidating the limitation text.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from si_eval.data import DEFAULT_DATASET

CODERS = ["ACN", "BME", "AIP"]


@pytest.fixture(scope="module")
def frames():
    ten = pd.read_excel(DEFAULT_DATASET, sheet_name="10k")
    try:
        cur = pd.read_excel(DEFAULT_DATASET, sheet_name="current_data")
    except ValueError:  # pragma: no cover - workbook lost its annotation sheet
        pytest.skip("current_data sheet is absent")
    return ten, cur


def test_three_coder_columns_exist(frames):
    _, cur = frames
    for coder in CODERS:
        assert coder in cur.columns, f"coder column {coder} missing"
    assert len(cur) == 10_128


def test_rows_align_between_sheets(frames):
    ten, cur = frames
    assert (ten["tweet"].astype(str).str.strip().values
            == cur["tweet"].astype(str).str.strip().values).all()


def test_published_label_is_the_coder_majority(frames):
    ten, cur = frames
    C = cur[CODERS].astype(float)
    valid = C.notna().all(axis=1).values
    Cv = C[valid].astype(int).values
    majority = (Cv.sum(axis=1) >= 2).astype(int)
    label = ten["label"].astype(int).values[valid]
    assert (label == majority).all(), "published label is not the coder majority"
    assert valid.sum() == 10_127


def test_agreement_numbers_are_stable(frames):
    _, cur = frames
    C = cur[CODERS].astype(float)
    valid = C.notna().all(axis=1).values
    Cv = C[valid].astype(int).values
    unanimous = (Cv.min(axis=1) == Cv.max(axis=1))
    assert unanimous.sum() == 9_751
    assert (Cv.min(axis=1) != Cv.max(axis=1)).sum() == 376
    counts = np.stack([(Cv == 0).sum(axis=1), (Cv == 1).sum(axis=1)], axis=1).astype(float)
    m = counts.sum(axis=1)[0]
    P_bar = float((((counts**2).sum(axis=1) - m) / (m * (m - 1.0))).mean())
    p_j = counts.sum(axis=0) / counts.sum()
    kappa = (P_bar - float((p_j**2).sum())) / (1 - float((p_j**2).sum()))
    assert kappa == pytest.approx(0.9229, abs=0.0005)
