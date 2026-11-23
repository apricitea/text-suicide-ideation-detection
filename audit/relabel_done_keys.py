"""Map an existing relabel ledger onto corpus texts, so a re-run can skip what is already annotated.

The first full-corpus pass used a selection that duplicated rows for repeated texts, and its
ledger's ``i`` is a position in that inflated selection. This script rebuilds that mapping, writes
the set of already-annotated corpus keys (``audit/out/*.done-keys.json``) and reports the machine
annotator's self-consistency on the texts it saw more than once.

    eval/.venv/bin/python audit/relabel_done_keys.py --ledger audit/out/relabel-full.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "audit"))
from relabel import load_frame  # noqa: E402


def key_of(text: str) -> str:
    return " ".join(str(text).split())[:200]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="audit/out/relabel-full.jsonl")
    args = ap.parse_args()
    ledger = Path(args.ledger) if Path(args.ledger).is_absolute() else REPO / args.ledger

    ten, cur = load_frame()
    # rebuild exactly what relabel.select() built for the first full pass (its 'disputed' column
    # included), so the ledger's positional index maps back onto corpus texts
    coders = ["ACN", "BME", "AIP"]
    C = cur[coders].astype(float)
    valid = C.notna().all(axis=1)
    Cv = C.fillna(-1).astype(int)
    cur = cur.assign(disputed=valid & (Cv.min(axis=1) != Cv.max(axis=1)))
    inflated = ten.merge(cur[["key", "disputed", *coders]], on="key", how="left").reset_index(drop=True)

    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    seen: dict[int, dict] = {r["i"]: r for r in rows}
    key_of_i = {i: key_of(inflated.iloc[i]["tweet"]) for i in seen if i < len(inflated)}
    done_keys = sorted(set(key_of_i.values()))

    groups: dict[str, list] = {}
    for i, row in seen.items():
        if i in key_of_i:
            groups.setdefault(key_of_i[i], []).append(row.get("label"))
    repeats = {k: v for k, v in groups.items() if len(v) > 1}
    identical = sum(1 for v in repeats.values() if len(set(v)) == 1)

    corpus_keys = set(ten["key"])
    missing = sorted(corpus_keys - set(done_keys))
    out = ledger.parent / (ledger.stem + ".done-keys.json")
    out.write_text(json.dumps({
        "annotations_in_ledger": len(seen),
        "inflated_selection_rows": int(len(inflated)),
        "corpus_rows": int(len(ten)),
        "distinct_corpus_texts_annotated": len(done_keys),
        "corpus_texts_still_missing": len(missing),
        "self_consistency": {"texts_annotated_more_than_once": len(repeats),
                             "machine_labels_identical": identical},
        "done_keys": done_keys,
    }))
    (ledger.parent / (ledger.stem + ".missing-keys.json")).write_text(json.dumps(missing))
    print(json.dumps({k: v for k, v in json.loads(out.read_text()).items() if k != "done_keys"}, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
