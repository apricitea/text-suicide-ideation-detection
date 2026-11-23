"""Re-annotate corpus rows against audit/codebook.md, one batched call at a time.

Model calls go through the Hermes CLI (`hermes -z`) with an explicit provider/model, as the
owner instructed. Results append to a JSONL file, so a run is resumable and a timeout costs a
batch, never the run.

    # 376 rows the three coders split 2-1, plus a stratified sample of unanimous rows
    eval/.venv/bin/python audit/relabel.py --mode phase1 --out audit/out/relabel-v1.jsonl

    # after phase 1 looks sane
    eval/.venv/bin/python audit/relabel.py --mode full --out audit/out/relabel-v1.jsonl
    eval/.venv/bin/python audit/relabel.py --mode unlabeled --out audit/out/unlabeled-machine.jsonl

Privacy: URLs and @handles are stripped before anything leaves the host (the corpus has them;
the thesis collected no usernames but scraped text still carries mentions and links).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from si_eval.data import normalise  # noqa: E402

MODEL = "deepseek/deepseek-v4.1-flash-1017"
PROVIDER = "nous"
XL = REPO / "dataset/final_dataset.xlsx"
CODEBOOK = REPO / "audit" / "codebook.md"

_URL = re.compile(r"https?://\S+|www\.\S+")
_HANDLE = re.compile(r"@[A-Za-z0-9_]+")

SYSTEM = """You annotate Indonesian tweets for presence of suicidal ideation, using one fixed codebook.

Codebook (label 1 = presence):
R1 explicit or serious display of the author's suicidal ideation
R2 the author discusses a suicide plan and/or previous attempts
R3 the author asks for advice or ways to hurt or kill himself/herself
R4 implicit intent to die/end life if a condition is fulfilled, no will to continue living
R0 no presence or relevance to suicidal ideation
R5 suicide-related news or information, statistics, reporting
R6 condolence or suicide awareness content
R7 song lyrics, film titles/quotes, discussion of fictional works

Boundaries: the label is about the AUTHOR's own expressed state, not the topic. Figurative,
sarcastic, joking, quoted or third-person uses are 0 (R0/R5/R6/R7) however grim. Conditional and
implicit first-person forms are 1 (R4). If a text could fairly be read both ways, choose the
reading of the author as a person (R1/R4 over R0) - the original annotation favoured recall.

Output: one JSON object per input row, on its own line, nothing else. No prose, no code fences,
no repetition of the tweet text.
Format: {"i": <row number>, "label": 0 or 1, "rule": "R0".."R7" or "R9", "confidence": <0.0-1.0>}
Use "R9" and "label": null only when you genuinely cannot decide.

Rows follow, each as "<n>. <tweet>".
"""


def strip_private(text: str) -> str:
    text = _URL.sub(" [url] ", str(text))
    text = _HANDLE.sub(" [user] ", text)
    return " ".join(text.split())[:600]


def load_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    ten = pd.read_excel(XL, sheet_name="10k")
    cur = pd.read_excel(XL, sheet_name="current_data")
    ten["key"] = ten["tweet"].astype(str).map(lambda t: " ".join(t.split())[:200])
    cur["key"] = cur["tweet"].astype(str).map(lambda t: " ".join(t.split())[:200])
    return ten, cur


def select(mode: str, ten: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with columns: row_index, tweet, stratum (>0 selects the row)."""
    coders = ["ACN", "BME", "AIP"]
    C = cur[coders].astype(float)
    valid = C.notna().all(axis=1)
    Cv = C.fillna(-1).astype(int)
    cur = cur.assign(disputed=valid & (Cv.min(axis=1) != Cv.max(axis=1)))
    merged = ten.merge(cur[["key", "disputed", "ACN", "BME", "AIP"]], on="key", how="left")

    if mode == "phase1":
        rng = np.random.default_rng(1)
        disputed = merged[merged["disputed"] == True]  # noqa: E712
        rest = merged[merged["disputed"] != True]  # noqa: E712
        # stratify the unanimous rows by class and length bucket, deterministically
        rest = rest.assign(bucket=pd.cut(rest["tweet"].astype(str).str.len(), [0, 80, 160, 10_000], labels=["short", "medium", "long"]))
        strata = list(rest.groupby(["label", "bucket"], observed=True))
        picks = [g.sample(n=min(len(g), max(1, 300 // max(1, len(strata)))), random_state=1) for _, g in strata]
        sample = pd.concat(picks)
        out = pd.concat([disputed, sample]).drop_duplicates(subset=["key"])
        out = out.assign(stratum=np.where(out["disputed"] == True, "disputed", "unanimous_sample"))  # noqa: E712
    elif mode == "full":
        out = merged.assign(stratum="full")
    elif mode == "unlabeled":
        raw = pd.read_excel(XL, sheet_name="11k_unlabeled", header=None)
        col = raw.iloc[:, 0].astype(str)
        col = col[col.str.split().str.len() > 3]
        out = pd.DataFrame({"tweet": col, "stratum": "unlabeled"})
        out["row_index"] = range(len(out))
        return out.reset_index(drop=True)
    else:
        raise SystemExit(f"unknown mode {mode}")
    out = out.reset_index(drop=True)
    out["row_index"] = range(len(out))
    return out


def call_model(prompt: str, timeout: int) -> tuple[str, float]:
    started = time.time()
    proc = subprocess.run(
        ["hermes", "-z", prompt, "-m", MODEL, "--provider", PROVIDER],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"hermes exited {proc.returncode}: {proc.stderr.strip()[:200]}")
    return proc.stdout, time.time() - started


def parse_rows(stdout: str, batch_ids: list[int]) -> list[dict]:
    found: dict[int, dict] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("i"), int) and obj["i"] in batch_ids:
            found[obj["i"]] = obj
    return list(found.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="phase1", choices=["phase1", "full", "unlabeled"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    out_path = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_version = hashlib.sha256((CODEBOOK.read_text() + SYSTEM).encode()).hexdigest()[:12]

    ten, cur = load_frame()
    rows = select(args.mode, ten, cur)
    done: dict[int, dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                obj = json.loads(line)
                done[obj["i"]] = obj
            except ValueError:
                continue
    todo = rows[~rows["row_index"].isin(done)]
    print(f"mode={args.mode} rows={len(rows)} already_done={len(done)} todo={len(todo)} "
          f"batch={args.batch_size} model={MODEL} prompt={prompt_version}", flush=True)

    batches_done = 0
    with out_path.open("a") as fh:
        for start in range(0, len(todo), args.batch_size):
            if args.max_batches and batches_done >= args.max_batches:
                break
            chunk = todo.iloc[start:start + args.batch_size]
            body = "\n".join(f"{int(r.row_index)}. {strip_private(r.tweet)}" for r in chunk.itertuples())
            try:
                stdout, seconds = call_model(SYSTEM + "\n" + body, args.timeout)
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                print(f"  batch {batches_done}: {type(exc).__name__} {exc} — continuing", flush=True)
                batches_done += 1
                continue
            got = parse_rows(stdout, [int(i) for i in chunk.row_index])
            for obj in got:
                obj.update(model=MODEL, provider=PROVIDER, prompt_version=prompt_version,
                           annotated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                fh.write(json.dumps(obj) + "\n")
                done[obj["i"]] = obj
            fh.flush()
            batches_done += 1
            print(f"  batch {batches_done}: +{len(got)}/{len(chunk)} rows in {seconds:.1f}s "
                  f"(total {len(done)})", flush=True)

    # ── agreement summary, aggregates only ──────────────────────────────────
    labelled = pd.DataFrame([v for v in done.values() if v.get("label") is not None])
    if len(labelled):
        m = rows.merge(labelled, left_on="row_index", right_on="i", how="inner")
        summary = {
            "rows_annotated": int(len(labelled)),
            "undecidable": int(sum(1 for v in done.values() if v.get("label") is None)),
            "model_positive_rate": round(float(labelled.label.mean()), 4),
        }
        if "label_x" in m or "label" in m:
            pub = m["label_x"] if "label_x" in m else m.get("label")
            if pub is not None:
                summary["agreement_with_published_label"] = round(float((m["label_y"] == pub).mean()), 4)
        if "disputed" in m:
            for stratum, g in m.groupby("stratum"):
                if "label_x" in g and "label_y" in g:
                    summary[f"agreement_{stratum}"] = round(float((g["label_x"] == g["label_y"]).mean()), 4)
                summary[f"n_{stratum}"] = int(len(g))
            ds = m[m["stratum"] == "disputed"] if "stratum" in m else m.iloc[0:0]
            if len(ds) and {"ACN", "BME", "AIP", "label_y"} <= set(ds.columns):
                for coder in ("ACN", "BME", "AIP"):
                    summary[f"model_sides_with_{coder}_when_disputed"] = round(
                        float((ds["label_y"] == ds[coder]).mean()), 4)
        print("SUMMARY " + json.dumps(summary, indent=2), flush=True)
        (out_path.parent / (out_path.stem + "-summary.json")).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
