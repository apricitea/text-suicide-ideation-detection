# Indonesian suicide-ideation classification — thesis model, transformer follow-up, and a leakage audit

[![CI](https://github.com/apricitea/text-suicide-ideation-detection/actions/workflows/eval.yml/badge.svg)](https://github.com/apricitea/text-suicide-ideation-detection/actions/workflows/eval.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Weights](https://img.shields.io/badge/weights-release-blue.svg)](https://github.com/apricitea/text-suicide-ideation-detection/releases/tag/model-v1)
[![Demo image](https://github.com/apricitea/text-suicide-ideation-detection/actions/workflows/demo-image.yml/badge.svg)](https://github.com/apricitea/text-suicide-ideation-detection/actions/workflows/demo-image.yml)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/main.pdf)

An Indonesian-language text classifier for suicide-ideation content, built first as an
undergraduate thesis (FastText embeddings + LSTM) and later extended with a fine-tuned IndoBERT on
the same split. This repository now also contains a **re-evaluation of both results**, because the
original comparison turned out to be inflated by duplicate rows.

Research and educational work. **Not a clinical or diagnostic tool** — model outputs must not be
substituted for professional mental-health assessment.

---

## The finding

The corpus is a scraped X sample. Of 10,128 posts, 2,161 are exact duplicates, and the 80/20 split
was drawn at random without deduplicating first. The consequence:

**739 of the 2,026 validation posts (36.5%) also appear, near-verbatim, in the training half.**

Three protocols, same corpus, in increasing order of cleanliness:

| Protocol | Validation rows | IndoBERT positive F1 | Strongest cheap baseline |
|---|---|---|---|
| **A. Published split, as reported** | 2,026 | **0.910** | — |
| **B. Same weights, unseen rows only** | 1,287 | **0.877** [0.845, 0.904] | 0.787 [0.744, 0.827] (char n-grams) |
| **C. Fully clean: dedupe, re-split, retrain** | 1,586 | **0.849** [0.821, 0.877] | 0.785 [0.749, 0.819] (word n-grams) |

- **A → B** is what the duplicate overlap was worth: 3.3 F1 points of the published number came
  from rows the model had already seen.
- **B → C** is what the leaked twins were worth to the *weights*: retraining on a deduplicated
  corpus costs another 2.8 points, on a smaller training set (6,340 rows instead of 8,102).
- Under the clean protocol the transformer's advantage over a linear model is
  **+0.064 F1** [+0.029, +0.105], McNemar p = 0.015 — smaller than the +0.090 visible on
  unseen rows of the leaky split, and smaller again than the 0.11 the published pair implied.
  The direction survives; the size does not.
- Baselines also reorder between protocols (word n-grams overtake character n-grams under C),
  which is a reminder that a 0.01 gap between baselines is not a finding.
- Some probabilities are close to honest in aggregate: mean predicted 0.206 against an observed
  positive rate of 0.201, ECE 0.046, Brier 0.046. That makes an operating point meaningful.
- **Operating point (what deployment actually costs):** to catch 90% of ideation-positive posts
  you must review **21.8% of the corpus at 83.6% precision**. At 80% recall: 17.6% reviewed,
  92.0% precision. These come from protocol B; protocol C is not recalibrated yet.

Full write-up: [`paper/main.pdf`](paper/main.pdf) · generated report and run manifest:
[`eval/out/`](eval/out) · evaluation code: [`eval/`](eval).

![Positive-class F1 by protocol](eval/out/20261025-full/fig_scores.png)

![Duplicate leakage in the published validation split](eval/out/20261025-full/fig_leakage.png)

---

## Reproducing it

The evaluation has its own dependency set — no TensorFlow, no torch:

```bash
uv sync --project eval
uv run --project eval python -m si_eval.run          # full run, 2,000 bootstrap resamples
uv run --project eval pytest eval/tests -q           # metric and split tests
```

A run writes `results.json`, `manifest.json` (corpus SHA-256, git revision, package versions), five
figures and a generated `REPORT.md` under `eval/out/<run-id>/`. The committed run is
`20261025-full`. `--quick` skips the bootstrap intervals and finishes in seconds; CI runs that
path on every push.

Protocol C is reproducible end to end, with a manifest recording the corpus hash, git revision and
library versions:

```bash
.venv/bin/python main/train_clean_split.py --out-dir indobert_clean_split   # ~110 min on 4 CPU threads
uv run --project eval python eval/compare_clean_retrain.py \
    --predictions indobert_clean_split/predictions.csv
```

The retrain writes `metrics.json`, `manifest.json` and per-row `predictions.csv`; the comparison
trains each baseline on the same deduplicated split and writes
`eval/out/clean-retrain-comparison/COMPARISON.md`.

To re-run the original protocol-A fine-tune (needs the `transformer` dependency group):

```bash
poetry install --with transformer
cd main && poetry run python indobert_finetune.py
```

That script saves the weights, the tokenizer and per-row validation predictions. The original
thesis run did not, which is why the FastText + LSTM arm cannot be recomputed here.

---

## What is in the repository

| Path | Contents |
|---|---|
| `eval/` | The re-evaluation: leakage measurement, four classical baselines, bootstrap intervals, McNemar tests, calibration, selective prediction, error buckets, tests, CI |
| `paper/` | LaTeX write-up of the audit and its results (6 pages, compiled PDF committed) |
| `main/` | Original pipeline: tweet crawling, stemming/stopword comparison (Sastrawi, Hunspell, Stanza), LSTM notebook, IndoBERT fine-tuning script, workbook pseudonymiser |
| `dataset/` | The final labelled dataset used for the reported models |
| `release/` | Pseudonymised 10k JSONL release + manifest (built by `main/build_public_release.py`) |
| `demo/` | FastAPI service + Dockerfile serving the audited weights; CI builds the image and checks a real prediction against the committed one |
| `indobert_publish/` | Committed IndoBERT metrics and per-row validation predictions; weights are the `model-v1` release asset |
| `indobert_clean_split/` | Protocol C: deduplicated-split retrain — metrics, manifest and per-row predictions (weights not committed) |
| `eval/compare_clean_retrain.py` | Scores protocol C against baselines trained on the same deduplicated split |
| `hf/` | Dataset and model cards, upload script |
| `hunspell-id-main/` | Indonesian Hunspell dictionary used in the stemming comparison |

The fine-tuned weights are **not** on HuggingFace: the previously advertised
`apricitea/indobert-suicide-ideation` repository does not exist (the URL returns 401), so that
claim has been removed from this README, the profile and the CV. They are published as a GitHub
release asset instead — [`model-v1`](https://github.com/apricitea/text-suicide-ideation-detection/releases/tag/model-v1),
461,903,383 bytes, SHA-256 `fe65a0ea18a5f67569f347c4a84de070d9f8ca49cdacca1b8f66ac6c11cbff76`.
`demo/` serves them and verifies that checksum at build time; `hf/publish.py` will upload to the
Hub instead if you supply a token.

The old Streamlit Space at `apricitea/suicide-detection` is a 2023 LSTM app. It serves models
trained on the leaky split and it currently fails to schedule
(`RUNTIME_ERROR: not enough hardware capacity`), so it is no longer linked as the demo.

Pre-trained Indonesian FastText vectors are **not** included: download `cc.id.300.bin` from
[fastText's crawl vectors](https://fasttext.cc/docs/en/crawl-vectors.html) if you want to re-run
the LSTM notebook.

Python floor is `>=3.10`. `cyhunspell` (used only in the Hunspell arm of the stemming comparison)
has no wheels past Python 3.9, so it lives in its own optional `hunspell` group — install it
separately under a 3.9 interpreter if you need that comparison.

---

## Limitations

1. **Protocol C is a single run, one seed.** The retrain cost 6,434 s on four CPU threads and was
   run once (seed 1). A second seed would widen the 0.849 interval and could move it by a point;
   the interval reported is sampling uncertainty on the validation set, not run-to-run variance.
   Protocol C also has not been recalibrated, so the operating-point table still comes from B.
2. **The FastText + LSTM thesis arm cannot be re-run.** Its FastText vectors are not in the
   repository and the Keras tokenizer was never saved with the model, so no per-row predictions
   exist. The 0.79 figure is quoted from the thesis, not recomputed. That is a reproducibility
   defect in the original work, recorded rather than hidden.
3. **One split, one seed.** Repeated splits would widen the intervals and could reorder the two
   strongest baselines.
4. **Labels are single-annotator.** No inter-annotator agreement study exists, so the label
   ceiling — and therefore how much a 0.09 F1 difference means — is unknown.
5. **The leakage rate is corpus-specific.** The method transfers to other scraped corpora; the
   36.5% does not.

---

## Data provenance

`dataset/final_dataset.xlsx` is a corpus of Indonesian-language posts collected from X (Twitter)
through the public API and labelled for suicide-ideation content. The file is committed because
Twitter's free developer API no longer permits historical queries, so the training data cannot be
re-fetched for reproducibility.

The post text is third-party user-generated content. It is **not our work**, we assert no licence
over it, and it is included here for research reproducibility only. The labels, preprocessing
pipeline and models in this repository are our own work, released under the MIT licence in
`LICENSE`.

### Identifiers in the corpus were pseudonymised

The corpus originally carried live account names. Measured across all six sheets of the workbook:
27,559 `@handle` occurrences covering 7,602 distinct accounts; in the `10k` training sheet alone,
4,318 of 10,128 rows contained a handle and **296 of those rows carried the positive (ideation)
label.** An account name next to a mental-health label is identifiable personal data, so before
publishing the workbook every handle was replaced with a stable pseudonym
(`@user_` + 8 hex characters of `sha256(salt ‖ handle)`), and `twitter.com/<user>/status/<id>` URLs
were rewritten to their id-only form.

Text and labels are otherwise unchanged, so this file is still the corpus the reported metrics were
computed from, apart from the handle substitution. The salt and the handle → pseudonym map are kept
out of the repository (`private/`, gitignored), so the mapping is reversible for us and not for a
reader.

Reproduce or verify it with:

```bash
python main/pseudonymise_workbook.py --check    # measure handles, write nothing
python main/pseudonymise_workbook.py --write    # apply the transform
```

The transform verifies its own output and refuses to write if any handle survives.

Two things were deliberately left alone:

- **Emoji mojibake.** The workbook mis-decoded UTF-8 as cp1252, so emoji are stored as sequences
  like `ðŸ˜”`. 1,450 rows in the `10k` sheet are affected. A repair exists in
  `main/build_public_release.py --repair-mojibake` but is off by default, so this file matches the
  metrics reported above.
- **Three corporate email addresses** appearing inside post text (`info@careerbridge.coach`,
  `help@dana.id`). These are business support addresses, not personal accounts, and the handle
  transform does not touch text preceded by a word character.

---

## Citation

```bibtex
@misc{nataputra2026leakage,
  title  = {What a 0.91 F1 Actually Measures: Duplicate Leakage in a Scraped Indonesian Mental-Health Corpus},
  author = {Nataputra, Alvin C.},
  year   = {2026},
  note   = {Research note, \url{https://github.com/apricitea/text-suicide-ideation-detection}}
}
```
