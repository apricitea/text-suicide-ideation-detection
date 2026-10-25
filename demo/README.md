# demo — serving the model the audit actually measured

The repository's earlier public demo was a 2023 Streamlit app loading Keras `.h5` files
that were trained on the leaky split, and it stopped scheduling on HuggingFace
(`RUNTIME_ERROR: not enough hardware capacity`). This replaces it with a small service
that serves the re-evaluated transformer and states its measured behaviour on the page.

## What it serves

`indobenchmark/indobert-base-p1`, fine-tuned by `main/indobert_finetune.py` (seed 1, 3
epochs, `max_length=128`). These are the weights behind
`indobert_publish/predictions.csv`, so the demo and the audit cannot drift apart.

Weights are **not** in git (476 MB). They are a GitHub release asset:

- `model-v1` → `indobert-suicide-ideation-model.tar.gz` (461,903,383 bytes)
- SHA-256 `fe65a0ea18a5f67569f347c4a84de070d9f8ca49cdacca1b8f66ac6c11cbff76`

The container verifies that checksum before extracting. A corrupted or substituted
artifact fails the build rather than serving wrong probabilities.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /health` | liveness, whether weights loaded, whether the committed audit run was found |
| `POST /predict` | `{"text": "..."}` → probability, decision at 0.5, nearest audited operating point |
| `GET /` | one-page form; no CDN, no analytics, no external font |

The page restates the audit numbers (`0.910 → 0.877` on unseen rows, 739/2,026 leaked,
ECE 0.046) and a not-a-clinical-tool banner, because a demo without its caveats is how
the original claim survived as long as it did.

## Verify

```bash
# weights on disk, no server
MODEL_DIR=indobert_publish/model python demo/app.py --selftest

# service
MODEL_DIR=indobert_publish/model uvicorn demo.app:app --port 8000
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"text":"gue udah gakuat hidup lagi sedih banget"}'
```

`--selftest` scores four strings whose probabilities are already committed in
`indobert_publish/predictions.csv` and fails if any drifts by more than 0.02.

## Container

```bash
docker build -f demo/Dockerfile -t si-demo .
docker run --rm -p 8000:8000 si-demo
```

CI (`.github/workflows/demo-image.yml`) builds the image, waits for `/health`, and posts
a real prediction, so "the demo works" is a build result rather than a claim.

## Deploying it publicly

Not done here, and deliberately not claimed anywhere in the repository. A public URL
needs one of: a HuggingFace token to recreate the Space (`hf auth login`, then a Docker
Space pointing at this image), a machine with a public address, or a container host. The
image is ready; the account step is the owner's.
