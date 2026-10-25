"""Tests for the serving demo.

Two levels:

* the HTTP contract (health, validation, 503 when weights are absent) using a stub
  model, so these run anywhere;
* the real-weight check, skipped unless ``indobert_publish/model`` is present. It posts
  strings whose probabilities are already committed in ``indobert_publish/predictions.csv``
  and fails if serving drifts from the audited numbers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from demo import app as demo_app

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "indobert_publish" / "model"
HAS_WEIGHTS = (WEIGHTS / "config.json").exists()


class StubModel:
    def __call__(self, **kwargs):  # noqa: ANN003 - mimic transformers output shape
        raise AssertionError("stub model should not be evaluated in contract tests")


@pytest.fixture
def client_without_weights(monkeypatch):
    monkeypatch.setattr(demo_app, "MODEL_DIR", Path("/nonexistent/model"))
    demo_app._state.update({"model": None, "tokenizer": None, "audit": {}})
    with TestClient(demo_app.app) as client:
        yield client


def test_health_reports_missing_weights_honestly(client_without_weights):
    body = client_without_weights.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_predict_returns_503_rather_than_a_fabricated_score(client_without_weights):
    response = client_without_weights.post("/predict", json={"text": "halo"})
    assert response.status_code == 503
    assert "MODEL_DIR" in response.json()["detail"]


def test_empty_text_is_rejected(client_without_weights):
    assert client_without_weights.post("/predict", json={"text": ""}).status_code == 422


def test_index_renders_without_weights(monkeypatch):
    monkeypatch.setattr(demo_app, "MODEL_DIR", Path("/nonexistent/model"))
    demo_app._state.update({"model": None, "tokenizer": None, "audit": {}})
    with TestClient(demo_app.app) as client:
        page = client.get("/")
    assert page.status_code == 200
    assert "Not a clinical or diagnostic tool" in page.text


def test_interpret_uses_the_committed_operating_points():
    demo_app._state["audit"] = {
        "operating_points": [
            {"target_recall": 0.80, "threshold": 0.985, "precision": 0.92, "flagged_fraction": 0.176},
            {"target_recall": 0.90, "threshold": 0.087, "precision": 0.836, "flagged_fraction": 0.218},
        ]
    }
    assert demo_app.interpret(0.99)["decision_at_0_5"] == "ideation_present"
    assert demo_app.interpret(0.99)["nearest_audited_operating_point"]["target_recall"] == 0.80
    assert demo_app.interpret(0.2)["decision_at_0_5"] == "no_ideation"


@pytest.mark.skipif(not HAS_WEIGHTS, reason="fine-tuned weights not present in this checkout")
def test_real_weights_reproduce_the_committed_probabilities():
    pytest.importorskip("torch", reason="deep-learning stack not installed")
    pytest.importorskip("transformers", reason="deep-learning stack not installed")
    os.environ["MODEL_DIR"] = str(WEIGHTS)
    demo_app.MODEL_DIR = WEIGHTS
    demo_app._state.update({"model": None, "tokenizer": None, "audit": {}})
    demo_app.load_model()
    assert demo_app._state["model"] is not None
    for text, expected in demo_app.SELF_TEST_CASES:
        assert abs(demo_app.predict(text) - expected) <= 0.02
