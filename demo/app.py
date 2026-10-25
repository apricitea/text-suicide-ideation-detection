"""Serving demo for the re-evaluated Indonesian suicide-ideation classifier.

This exists because the repository's previous public demo was a 2023 Streamlit app
serving Keras ``.h5`` files trained on the leaky split, and it stopped scheduling.
A demo that contradicts the audit is worse than no demo.

What it serves is exactly the model the audit measured: the fine-tuned
``indobenchmark/indobert-base-p1`` weights from ``main/indobert_finetune.py``, whose
committed per-row validation predictions are in ``indobert_publish/predictions.csv``.

Endpoints
---------
GET  /health   liveness plus whether weights actually loaded
POST /predict  {"text": "..."} -> probability, decision at 0.5, audited operating point
GET  /         a one-page form, no CDN, no analytics

Run locally
-----------
    MODEL_DIR=indobert_publish/model uvicorn demo.app:app --port 8000
    python demo/app.py --selftest          # no server, prints probabilities

Research artefact only. Not a clinical or diagnostic tool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# torch and transformers are imported inside the functions that need them, so the HTTP
# contract (health, validation, 503-when-weights-absent) is testable on a machine
# without a 2 GB deep-learning stack installed.

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "indobert_publish" / "model"))
RESULTS_JSON = Path(
    os.environ.get("RESULTS_JSON", REPO_ROOT / "eval" / "out" / "20261025-full" / "results.json")
)
MAX_LENGTH = 128

# Two rows from the committed validation predictions, used as a smoke test. The
# probabilities are what indobert_publish/predictions.csv recorded for these exact
# strings, so a serving regression shows up as a mismatch, not as a suspiciously
# round number.
SELF_TEST_CASES = [
    ("cape banget anjing. gue mau bundir aja lah bangsat", 0.9985),
    ("gue udah gakuat hidup lagi sedih banget bjir slalu diasingkan mending bundir aja", 0.9986),
    ("tak tahu la kenapa hati ni jadi si keras ni mungkin dah mati rasa", 0.0005),
    ('Dibalik kata "udah gak usah dibahas" ada rasa kecewa yang paling dalam.', 0.0006),
]

_state: dict[str, Any] = {"model": None, "tokenizer": None, "audit": {}, "error": None}


def load_audit() -> dict:
    """Pull the measured operating points and metrics from the committed run, if present."""
    if not RESULTS_JSON.exists():
        return {}
    data = json.loads(RESULTS_JSON.read_text())
    ind = data.get("indobert", {})
    split = data.get("split", {})
    return {
        "unseen_rows": split.get("n_clean_val"),
        "leaked_rows": split.get("val_rows_duplicated_in_train"),
        "val_rows": split.get("n_val"),
        "f1_clean": ind.get("clean_val", {}).get("positive_f1"),
        "f1_published": ind.get("full_val", {}).get("positive_f1"),
        "f1_clean_ci": ind.get("clean_val_intervals", {}).get("positive_f1"),
        "ece": ind.get("clean_val_calibration", {}).get("ece"),
        "operating_points": ind.get("operating_points", []),
    }


def load_model() -> None:
    """Load weights if both the stack and the checkpoint are present; never raise.

    A missing ``transformers`` install, a corrupt checkpoint, or an absent weights
    directory must leave the service up and honest: ``/health`` says why, and
    ``/predict`` returns 503 instead of inventing a score. That is what lets the
    HTTP contract be tested in a CI container without the 2 GB deep-learning stack.
    """
    _state["model"] = None
    _state["tokenizer"] = None
    _state["error"] = None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # ImportError, or a broken torch/CUDA install
        _state["error"] = f"{type(exc).__name__}: {exc}"
        return

    if not (MODEL_DIR / "config.json").exists():
        _state["error"] = f"no config.json under {MODEL_DIR}"
        return
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        model.eval()
    except Exception as exc:
        _state["error"] = f"{type(exc).__name__}: {exc}"
        return
    _state["tokenizer"] = tokenizer
    _state["model"] = model


def predict(text: str) -> float:
    import torch

    tokenizer = _state["tokenizer"]
    model = _state["model"]
    if tokenizer is None or model is None:
        raise RuntimeError("model weights are not loaded")
    encoded = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits
    return float(torch.softmax(logits, dim=-1)[0, 1])


def interpret(probability: float) -> dict:
    """Place the score against the operating points measured on unseen rows."""
    audit = _state.get("audit") or {}
    points = audit.get("operating_points") or []
    chosen = None
    for point in points:
        if probability >= float(point["threshold"]):
            chosen = point
            break
    if chosen is None and points:
        chosen = points[-1]
    return {
        "decision_at_0_5": "ideation_present" if probability >= 0.5 else "no_ideation",
        "nearest_audited_operating_point": chosen,
        "note": (
            "Thresholds come from the 1,287 unseen validation rows, not from this text. "
            "A high score means the post resembles labelled ideation examples; it is not "
            "an assessment of any person."
            if points
            else "No committed run found beside this deployment, so operating points are omitted."
        ),
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    _state["audit"] = load_audit()
    load_model()
    yield


app = FastAPI(title="Indonesian suicide-ideation classifier (research demo)", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "model_dir": str(MODEL_DIR),
        "audit_loaded": bool(_state.get("audit")),
        "load_error": _state.get("error"),
    }


@app.post("/predict")
def predict_endpoint(request: PredictRequest) -> dict:
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"model not loaded ({_state.get('error') or 'weights absent'}). "
                f"Mount weights under {MODEL_DIR} or set MODEL_DIR."
            ),
        )
    probability = predict(request.text)
    return {
        "probability_positive": probability,
        **interpret(probability),
        "model": "indobenchmark/indobert-base-p1, fine-tuned (seed 1, 3 epochs)",
        "audit": _state.get("audit", {}),
    }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Indonesian suicide-ideation classifier — research demo</title>
<style>
 body{margin:0;background:#141413;color:#e6e2d8;font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif}
 main{max-width:720px;margin:0 auto;padding:48px 20px 72px}
 h1{font-size:22px;margin:0 0 6px;font-weight:600}
 p.sub{color:#9c968a;margin:0 0 28px;font-size:13.5px}
 textarea{width:100%;min-height:110px;background:#1c1c1b;color:#e6e2d8;border:1px solid #343330;
   border-radius:6px;padding:12px;font:inherit;resize:vertical}
 button{margin-top:12px;background:#b8935a;color:#141413;border:0;border-radius:5px;
   padding:10px 18px;font-weight:600;cursor:pointer}
 button[disabled]{opacity:.5;cursor:default}
 #out{margin-top:22px;border-top:1px solid #2c2b28;padding-top:18px;display:none}
 .score{font-size:30px;font-weight:700;color:#c9a877}
 .warn{background:#2a1f1c;border:1px solid #5a3a33;border-radius:6px;padding:12px 14px;
   font-size:12.5px;color:#d8b7ae;margin-bottom:26px}
 table{border-collapse:collapse;margin-top:10px;font-size:12.5px;color:#aba692}
 td,th{border-bottom:1px solid #2c2b28;padding:5px 12px 5px 0;text-align:left}
</style></head><body><main>
<h1>Indonesian suicide-ideation classifier</h1>
<p class="sub">Research demo for a re-evaluated undergraduate thesis model. Every number
below comes from the committed run, not from this page.</p>
<div class="warn"><b>Not a clinical or diagnostic tool.</b> This is a text classifier
evaluated on __N_UNSEEN__ unseen social-media posts. It must not be used to assess a person.
If you are in crisis, contact local emergency services.</div>
<textarea id="t" placeholder="Paste an Indonesian-language post…"></textarea>
<button id="b">Classify</button>
<div id="out">
  <div>p(ideation) = <span class="score" id="p"></span></div>
  <div id="dec" style="color:#9c968a;font-size:13px"></div>
  <table><tr><th>Measured on unseen rows</th><th></th></tr>
  <tr><td>Positive-class F1 (published split → unseen)</td><td>__F1_PUB__ → __F1_CLEAN__</td></tr>
  <tr><td>Validation posts also in training</td><td>__LEAKED__ of __N_VAL__</td></tr>
  <tr><td>Expected calibration error</td><td>__ECE__</td></tr></table>
</div>
<script>
const b=document.getElementById('b'),t=document.getElementById('t'),o=document.getElementById('out');
b.onclick=async()=>{
  if(!t.value.trim())return; b.disabled=true; b.textContent='…';
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text:t.value})});
    const d=await r.json();
    if(!r.ok){document.getElementById('p').textContent='error';document.getElementById('dec').textContent=d.detail||'';o.style.display='block';return}
    document.getElementById('p').textContent=d.probability_positive.toFixed(3);
    const pt=d.nearest_audited_operating_point;
    document.getElementById('dec').textContent=
      (d.decision_at_0_5==='ideation_present'?'above':'below')+' the 0.5 threshold'+
      (pt?` · at the audited ${Math.round(pt.target_recall*100)}%-recall threshold (${pt.threshold.toFixed(3)}): precision ${pt.precision.toFixed(2)} on ${Math.round(pt.flagged_fraction*100)}% of posts flagged`:'');
    o.style.display='block';
  }finally{b.disabled=false;b.textContent='Classify'}
};
</script></main></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Render the form with the committed audit numbers substituted in.

    Deliberately string replacement rather than str.format: the page contains CSS and
    JavaScript whose braces would be parsed as format fields.
    """
    audit = _state.get("audit") or {}
    ci = audit.get("f1_clean_ci") or {}
    f1_clean = audit.get("f1_clean")
    if isinstance(f1_clean, float):
        if ci.get("low") is not None and ci.get("high") is not None:
            rendered = f"{f1_clean:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"
        else:
            rendered = f"{f1_clean:.3f}"
    else:
        rendered = "n/a"
    values = {
        "__N_UNSEEN__": str(audit.get("unseen_rows", "n/a")),
        "__F1_PUB__": f"{audit['f1_published']:.3f}" if isinstance(audit.get("f1_published"), float) else "n/a",
        "__F1_CLEAN__": rendered,
        "__LEAKED__": str(audit.get("leaked_rows", "n/a")),
        "__N_VAL__": str(audit.get("val_rows", "n/a")),
        "__ECE__": f"{audit['ece']:.3f}" if isinstance(audit.get("ece"), float) else "n/a",
    }
    html = PAGE
    for token, value in values.items():
        html = html.replace(token, value)
    return html


def selftest() -> int:
    load_model()
    if _state["model"] is None:
        print(f"FAIL: no weights at {MODEL_DIR}")
        return 2
    failures = 0
    for text, expected in SELF_TEST_CASES:
        got = predict(text)
        ok = abs(got - expected) <= 0.02
        failures += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} p={got:.4f} expected≈{expected:.4f}  {text[:60]}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="score the fixed cases and exit")
    args = parser.parse_args()
    if args.selftest:
        load_audit_ok = load_audit()
        _state["audit"] = load_audit_ok
        raise SystemExit(selftest())
    print("run with: uvicorn demo.app:app --port 8000", file=sys.stderr)
