"""Upload the pseudonymised dataset and the fine-tuned model to HuggingFace.

Reads HF_TOKEN from the environment. Never takes a token as an argument, so it
does not end up in shell history or in a log line.

    HF_TOKEN=... python hf/publish.py --what dataset
    HF_TOKEN=... python hf/publish.py --what model

Both targets are created if absent. Existing files are overwritten.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DATASET_REPO = "apricitea/bahasa-suicide-ideation-10k"
MODEL_REPO = "apricitea/indobert-suicide-ideation"

ROOT = Path(__file__).resolve().parents[1]


def require_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN is not set in the environment.", file=sys.stderr)
        raise SystemExit(2)
    return token


def publish_dataset(token: str) -> None:
    from huggingface_hub import HfApi

    release = ROOT / "release"
    data = release / "bahasa-suicide-ideation-10k.jsonl"
    manifest = release / "release_manifest.json"
    card = ROOT / "hf" / "dataset_card.md"
    for p in (data, manifest, card):
        if not p.exists():
            print(f"missing {p} — run main/build_public_release.py first", file=sys.stderr)
            raise SystemExit(2)

    api = HfApi(token=token)
    api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                    repo_id=DATASET_REPO, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(data),
                    path_in_repo="bahasa-suicide-ideation-10k.jsonl",
                    repo_id=DATASET_REPO, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(manifest), path_in_repo="release_manifest.json",
                    repo_id=DATASET_REPO, repo_type="dataset")
    print(f"dataset -> https://huggingface.co/datasets/{DATASET_REPO}")


def publish_model(token: str) -> None:
    from huggingface_hub import HfApi

    out = ROOT / "indobert_publish"
    model_dir = out / "model"
    metrics = out / "metrics.json"
    if not model_dir.exists():
        print(f"missing {model_dir} — run main/indobert_finetune.py first", file=sys.stderr)
        raise SystemExit(2)

    card_src = ROOT / "hf" / "model_card.md"
    if card_src.exists():
        (out / "README.md").write_text(card_src.read_text())

    api = HfApi(token=token)
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(model_dir), repo_id=MODEL_REPO, repo_type="model")
    if metrics.exists():
        api.upload_file(path_or_fileobj=str(metrics), path_in_repo="metrics.json",
                        repo_id=MODEL_REPO, repo_type="model")
    readme = out / "README.md"
    if readme.exists():
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md",
                        repo_id=MODEL_REPO, repo_type="model")
    print(f"model -> https://huggingface.co/{MODEL_REPO}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", choices=["dataset", "model", "both"], default="both")
    args = ap.parse_args()

    token = require_token()
    if args.what in ("dataset", "both"):
        publish_dataset(token)
    if args.what in ("model", "both"):
        publish_model(token)


if __name__ == "__main__":
    main()