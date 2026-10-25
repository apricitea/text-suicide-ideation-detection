# Serving image for the re-evaluated IndoBERT classifier.
#
# Weights are not committed to git (476 MB). They are a GitHub release asset so the
# image builds reproducibly from a public URL and a checksum, without needing a
# HuggingFace account or a token at build time.
#
#   docker build -f demo/Dockerfile -t si-demo .
#   docker run --rm -p 8000:8000 si-demo
#   curl -s localhost:8000/health

FROM python:3.11-slim

ARG WEIGHTS_URL=https://github.com/apricitea/text-suicide-ideation-detection/releases/download/model-v1/indobert-suicide-ideation-model.tar.gz
ARG WEIGHTS_SHA256=fe65a0ea18a5f67569f347c4a84de070d9f8ca49cdacca1b8f66ac6c11cbff76

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/model \
    RESULTS_JSON=/app/results.json \
    OMP_NUM_THREADS=2

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first, so the CPU wheel satisfies the transformers dependency and
# pip never pulls the multi-gigabyte CUDA build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY demo/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Weights: verify the checksum before extracting, so a corrupted or substituted
# artifact fails the build instead of serving wrong probabilities.
RUN mkdir -p /app/model \
 && curl -fsSL "$WEIGHTS_URL" -o /tmp/weights.tar.gz \
 && echo "${WEIGHTS_SHA256}  /tmp/weights.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/weights.tar.gz -C /app/model --strip-components=1 \
 && rm /tmp/weights.tar.gz \
 && test -f /app/model/config.json

COPY eval/out/20261025-full/results.json /app/results.json
COPY demo/app.py /app/app.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
