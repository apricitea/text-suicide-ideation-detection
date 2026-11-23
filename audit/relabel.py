"""Re-annotate corpus rows against audit/codebook.md, one batched call at a time.

Model calls go through the Hermes CLI (`hermes -z`) with an explicit provider/model, as the owner
instructed. Every written row carries a content-derived ``row_key``, so a run is resumable and a
*new* run can skip texts a previous ledger already covered — resume must never depend on row
positions, because a selection change renumbers every row.

    # 376 rows the three coders split 2-1, plus a stratified sample of unanimous rows
    eval/.venv/bin/python audit/relabel.py --mode phase1 --out audit/out/relabel-v1.jsonl

    # one row per corpus row; skip texts an earlier ledger already annotated
    eval/.venv/bin/python audit/relabel.py --mode full --out audit/out/relabel-full2.jsonl \
        --skip-keys-from audit/out/relabel-full.done-keys.json

    # the unlabelled pool on the 11k sheet
    eval/.venv/bin/python audit/relabel.py --mode unlabeled --out audit/out/unlabeled-machine.jsonl

Privacy: URLs and @handles are stripped before anything leaves the host (the corpus has them; the
thesis collected no usernames, but scraped text still carries mentions and links).
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
CODERS = ["ACN", "BME", "AIP"]

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


def corpus_key(text: str) -> str:
    """The workbook join key: whitespace-collapsed, first 200 characters."""
    return " ".join(str(text).split())[:200]


def key_hash(text: str) -> str:
    """Content-derived row identity, stable across selection changes."""
    return hashlib.sha1(corpus_key(text).encode("utf-8")).hexdigest()[:16]


def load_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    ten = pd.read_excel(XL, sheet_name="10k")
    cur = pd.read_excel(XL, sheet_name="current_data")
    for frame in (ten, cur):
        frame["key"] = frame["tweet"].astype(str).map(corpus_key)
    return ten, cur


def select(mode: str, ten: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with columns row_index, row_key, tweet, label, stratum, disputed, coders."""
    C = cur[CODERS].astype(float)
    valid = C.notna().all(axis=1)
    Cv = C.fillna(-1).astype(int)
    # One coder row per text: the workbook repeats texts, and joining on the raw key would multiply
    # corpus rows (the first full pass did exactly that — 14,562 selection rows for 10,128 texts).
    cur_u = cur.assign(disputed=valid & (Cv.min(axis=1) != Cv.max(axis=1))).drop_duplicates(subset=["key"])
    merged = ten.merge(cur_u[["key", "disputed", *CODERS]], on="key", how="left")

    if mode == "phase1":
        disputed = merged[merged["disputed"] == True]  # noqa: E712
        rest = merged[merged["disputed"] != True]  # noqa: E712
        rest = rest.assign(bucket=pd.cut(rest["tweet"].astype(str).str.len(), [0, 80, 160, 10_000],
                                         labels=["short", "medium", "long"]))
        strata = list(rest.groupby(["label", "bucket"], observed=True))
        picks = [g.sample(n=min(len(g), max(1, 300 // max(1, len(strata)))), random_state=1) for _, g in strata]
        out = pd.concat([disputed, pd.concat(picks)]).drop_duplicates(subset=["key"])
        out = out.assign(stratum=np.where(out["disputed"] == True, "disputed", "unanimous_sample"))  # noqa: E712
    elif mode == "full":
        out = merged.assign(stratum="full")
    elif mode == "unlabeled":
        raw = pd.read_excel(XL, sheet_name="11k_unlabeled", header=None)
        col = raw.iloc[:, 0].astype(str)
        col = col[col.str.split().str.len() > 3]
        out = pd.DataFrame({"tweet": col, "stratum": "unlabeled"})
    else:
        raise SystemExit(f"unknown mode {mode}")

    out = out.reset_index(drop=True)
    out["row_index"] = range(len(out))
    out["row_key"] = [key_hash(t) for t in out["tweet"]]
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


def load_ledger(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        out[obj.get("row_key") or f"i:{obj.get('i')}"] = obj
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="phase1", choices=["phase1", "full", "unlabeled"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = all")
    ap.add_argument("--skip-keys-from", default="",
                    help="JSON file holding {'done_keys': [corpus texts]} to skip by content hash")
    ap.add_argument("--skip-ledger", default="",
                    help="another ledger whose rows are skipped by their row_key")
    args = ap.parse_args()

    out_path = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_version = hashlib.sha256((CODEBOOK.read_text() + SYSTEM).encode()).hexdigest()[:12]

    ten, cur = load_frame()
    rows = select(args.mode, ten, cur)
    done = load_ledger(out_path)

    skip: set[str] = set()
    if args.skip_keys_from:
        payload = json.loads(Path(args.skip_keys_from).read_text())
        keys = payload["done_keys"] if isinstance(payload, dict) else payload
        skip |= {key_hash(k) for k in keys}
    if args.skip_ledger:
        p = Path(args.skip_ledger) if Path(args.skip_ledger).is_absolute() else REPO / args.skip_ledger
        skip |= {v["row_key"] for v in load_ledger(p).values() if v.get("row_key")}

    todo = rows[~rows["row_key"].isin(skip) & ~rows["row_key"].isin(done)]
    # one call per distinct text: repeated texts share a row_key and will inherit the same verdict
    todo = todo.drop_duplicates(subset=["row_key"])
    print(f"mode={args.mode} rows={len(rows)} skipped={len(rows) - len(todo)} todo={len(todo)} "
          f"batch={args.batch_size} model={MODEL} prompt={prompt_version}", flush=True)

    batches = 0
    with out_path.open("a") as fh:
        for start in range(0, len(todo), args.batch_size):
            if args.max_batches and batches >= args.max_batches:
                break
            chunk = todo.iloc[start:start + args.batch_size]
            body = "\n".join(f"{int(r.row_index)}. {strip_private(r.tweet)}" for r in chunk.itertuples())
            try:
                stdout, seconds = call_model(SYSTEM + "\n" + body, args.timeout)
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                print(f"  batch {batches}: {type(exc).__name__} {exc} — continuing", flush=True)
                batches += 1
                continue
            got = parse_rows(stdout, [int(i) for i in chunk.row_index])
            by_index = {int(r.row_index): r.row_key for r in chunk.itertuples()}
            for obj in got:
                obj.update(model=MODEL, provider=PROVIDER, prompt_version=prompt_version,
                           row_key=by_index.get(int(obj["i"]), ""),
                           annotated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                fh.write(json.dumps(obj) + "\n")
                done[obj["row_key"] or f"i:{obj['i']}"] = obj
            fh.flush()
            batches += 1
            print(f"  batch {batches}: +{len(got)}/{len(chunk)} in {seconds:.1f}s (ledger {len(done)})",
                  flush=True)

    labelled = [v for v in done.values() if v.get("label") is not None]
    if labelled:
        summary = {
            "rows_annotated": len(labelled),
            "undecidable": sum(1 for v in done.values() if v.get("label") is None),
            "model_positive_rate": round(float(np.mean([v["label"] for v in labelled])), 4),
            "rule_distribution": {str(k): int(v) for k, v in pd.Series([v.get("rule") for v in labelled]).value_counts().items()},
            "model": MODEL,
            "prompt_version": prompt_version,
        }
        print("SUMMARY " + json.dumps(summary, indent=2), flush=True)
        (out_path.parent / (out_path.stem + "-summary.json")).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
