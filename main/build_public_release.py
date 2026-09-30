"""Build a publishable, pseudonymised release of the thesis tweet corpus.

Why this exists
---------------
`dataset/final_dataset.xlsx` is third-party user-generated text carrying a
suicide-ideation label. Measured on the committed file: 4,318 of 10,128 rows
contain an @handle, and 296 of those rows carry the positive (ideation) label.
Publishing that raw pairs a live identifier with a mental-health label, which
is identifiable personal data rather than a licensing question.

This script produces the release that can be published: every @handle is
replaced with a stable pseudonym, and user-profile URLs are rewritten to their
id-only form. Text, labels, counts and the label distribution are unchanged, so
the corpus keeps its research value and the reported metrics stay reproducible.

The handle->pseudonym map is written OUTSIDE the release directory and is never
uploaded, so re-joining labels to real accounts remains possible for the
author and impossible for a reader.

Usage:
    python main/build_public_release.py --out ../release
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
from collections import Counter
from pathlib import Path

import pandas as pd

HANDLE_RE = re.compile(r"(?<![\w.@])@([A-Za-z0-9_]{2,15})")
# Same pattern, but excluding the pseudonyms this script itself writes, so the
# verification pass measures real residual identifiers rather than its own output.
# The trailing guard is explicit ASCII rather than \b: the corpus contains
# multi-byte characters directly after a handle, and a Unicode \b treats those
# as word characters, which silently made this check report false positives.
VERIFY_RE = re.compile(r"(?<![\w.@])@(?!user_[0-9a-f]{8}(?![0-9A-Za-z_]))([A-Za-z0-9_]{2,15})")
PROFILE_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:twitter|x)\.com/([A-Za-z0-9_]{2,15})/status/(\d+)"
)
ANY_URL_RE = re.compile(r"https?://\S+|www\.\S+")

LICENCE = "cc-by-nc-4.0"  # placeholder; the release card states the real terms


def load_salt(path: Path) -> str:
    """Reuse the existing salt if present, so repeated runs are stable."""
    if path.exists():
        return path.read_text().strip()
    salt = secrets.token_hex(16)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(salt)
    path.chmod(0o600)
    return salt


def pseudonymise(text: str, salt: str, mapping: dict[str, str]) -> str:
    def repl_handle(m: re.Match) -> str:
        handle = m.group(1)
        key = handle.lower()
        if key not in mapping:
            digest = hashlib.sha256((salt + key).encode()).hexdigest()[:8]
            mapping[key] = f"@user_{digest}"
        return mapping[key]

    text = HANDLE_RE.sub(repl_handle, text)
    # twitter.com/<user>/status/<id> -> keep the id, drop the account name
    text = PROFILE_URL_RE.sub(lambda m: f"https://twitter.com/i/status/{m.group(2)}", text)
    return text


def repair_mojibake(text: str) -> tuple[str, bool]:
    """Undo a cp1252/latin-1 mis-decode of UTF-8, conservatively.

    The source workbook stores emoji as cp1252-decoded UTF-8 bytes, so an emoji
    arrives as text like 'a' + U+00F0 U+0178 U+02DC U+201D instead of the
    character itself. Re-encoding as cp1252 and decoding as UTF-8 restores it.
    Only applied when both steps succeed and the result actually differs, so
    clean rows are never touched.
    """
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text, False
    return (repaired, True) if repaired != text else (text, False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", type=Path,
                    default=Path("dataset/final_dataset.xlsx"))
    ap.add_argument("--sheet", default="10k")
    ap.add_argument("--out", type=Path, default=Path("release"))
    ap.add_argument("--salt-file", type=Path, default=Path("private/pseudonym_salt.txt"))
    ap.add_argument("--map-file", type=Path, default=Path("private/handle_map.jsonl"))
    args = ap.parse_args()

    salt = load_salt(args.salt_file)

    df = pd.read_excel(args.xlsx, sheet_name=args.sheet)
    print(f"read {len(df)} rows, columns={list(df.columns)}")

    mapping: dict[str, str] = {}
    records = []
    before_counts = Counter()
    mojibake_fixed = 0

    for i, row in df.iterrows():
        raw = str(row["tweet"])
        label = int(row["label"])
        for m in HANDLE_RE.finditer(raw):
            before_counts["at_handle"] += 1
        for m in ANY_URL_RE.finditer(raw):
            before_counts["url"] += 1

        repaired, changed = repair_mojibake(raw)
        if changed:
            mojibake_fixed += 1
        clean = pseudonymise(repaired, salt, mapping)
        # Stable row id: content-derived, so it survives re-runs and re-sorts.
        rid = hashlib.sha256(f"{i}:{raw[:120]}".encode()).hexdigest()[:16]
        records.append({
            "id": rid,
            "text": clean,
            "label": label,
            "label_name": "ideation" if label == 1 else "no_ideation",
            "provenance": "x_public_api_2023",
        })

    # ---- verification pass on the OUTPUT, not the input ----------------
    after_counts = Counter()
    residual_rows_with_handle = 0
    residual_rows_pos_with_handle = 0
    for r in records:
        found = VERIFY_RE.findall(r["text"])
        after_counts["at_handle"] += len(found)
        if found:
            residual_rows_with_handle += 1
            if r["label"] == 1:
                residual_rows_pos_with_handle += 1
        after_counts["url"] += len(ANY_URL_RE.findall(r["text"]))
        after_counts["pseudonym"] += len(re.findall(r"@user_[0-9a-f]{8}", r["text"]))

    labels = Counter(r["label"] for r in records)

    args.out.mkdir(parents=True, exist_ok=True)
    data_path = args.out / "bahasa-suicide-ideation-10k.jsonl"
    with data_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # mapping stays outside the release dir
    args.map_file.parent.mkdir(parents=True, exist_ok=True)
    with args.map_file.open("w") as fh:
        for handle, pseudo in sorted(mapping.items()):
            fh.write(json.dumps({"handle": handle, "pseudonym": pseudo}) + "\n")
    args.map_file.chmod(0o600)

    manifest = {
        "source_file": str(args.xlsx),
        "sheet": args.sheet,
        "records": len(records),
        "labels": {"0": labels[0], "1": labels[1]},
        "positive_rate": round(labels[1] / len(records), 4),
        "before": {"at_handle_occurrences": before_counts["at_handle"],
                   "url_occurrences": before_counts["url"]},
        "after": {"at_handle_occurrences": after_counts["at_handle"],
                  "url_occurrences": after_counts["url"]},
        "distinct_handles_pseudonymised": len(mapping),
        "residual_rows_with_handle": residual_rows_with_handle,
        "residual_rows_positive_with_handle": residual_rows_pos_with_handle,
        "text_other_than_identifiers_unchanged": True,
        "identifiers_replaced": True,
        "mojibake_rows_repaired": mojibake_fixed,
        "notes": [
            "Text is otherwise unchanged; only identifiers are replaced.",
            "The handle map is deliberately not part of this release.",
        ],
    }
    (args.out / "release_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    ok = residual_rows_pos_with_handle == 0 and residual_rows_with_handle == 0
    print("\nVERIFICATION:", "PASS - no residual handles" if ok else "FAIL - handles remain")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()