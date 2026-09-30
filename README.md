# Suicide Ideation Detection — FastText+LSTM (thesis) vs. Fine-Tuned IndoBERT

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face Space](https://img.shields.io/badge/demo-HuggingFace-yellow.svg)](https://huggingface.co/spaces/apricitea/suicide-detection)

Originally my undergraduate thesis (FastText embeddings + LSTM). A deployed
version of that model is on [HuggingFace](https://huggingface.co/apricitea); full write-up
of the process is on [Medium](https://medium.com/@apricitea). Since then I added a
transformer baseline (fine-tuned IndoBERT) on the exact same train/val split, to see how
much a pretrained language model buys over word embeddings + a small recurrent net.

This is a research/educational project on Indonesian-language text classification, not a
clinical or diagnostic tool — model outputs should not be treated as a substitute for
professional mental health assessment.

## Results

Both trained and evaluated on the same 80/20 split (`random_state=1`) of the 10,128-tweet
dataset, binary label (suicide ideation present / not), no class-imbalance treatment.

| Model | Precision (positive) | Recall (positive) | F1 (positive) | Accuracy |
|---|---|---|---|---|
| FastText + LSTM (original thesis) | 0.74 | 0.85 | 0.79 | 0.91 |
| Fine-tuned IndoBERT (`indobenchmark/indobert-base-p1`) | 0.91 | 0.90 | **0.90** | 0.96 |

The transformer's contextual embeddings resolve a lot of what static FastText vectors
miss — sarcasm, negation, and multi-word idiomatic expressions common in Indonesian social
media text. The original thesis also tried class-weighting and ADASYN oversampling to
handle the 8:2 class imbalance; both *reduced* F1 relative to no treatment (better recall,
much worse precision) — see `main/lstm.ipynb`. IndoBERT wasn't given any imbalance
treatment either, and didn't need it.

Fine-tuned IndoBERT weights aren't committed to git (too large). They are
published on HuggingFace at
[`apricitea/indobert-suicide-ideation`](https://huggingface.co/apricitea/indobert-suicide-ideation),
together with `metrics.json` and the validation predictions.

`main/indobert_finetune.py` now saves the model it trains. The original version
trained for ~35 minutes and discarded the weights, which is why the numbers above
had no artifact behind them for so long. Re-run on this corpus it gives
**F1 0.9098 / accuracy 0.9650** — consistent with the reported 0.90 / 0.96 but
marginally higher, not bit-identical: the original ran a 2023-era stack on a GPU,
this run is torch 2.14 / transformers 5.17 on CPU. The committed run is a CPU run
— 4 threads, no GPU, ~2 hours.

## Reproducing it

Dependencies are managed with Poetry. Twitter's free Developer API no longer supports
querying historical tweets, so `dataset/final_dataset.xlsx` is included directly instead
of a live crawl.

```bash
poetry install --with transformer  # only needed for main/indobert_finetune.py
cd main && poetry run python indobert_finetune.py
```

Python floor is `>=3.10` for current dependency versions. `cyhunspell` (used only in the Hunspell
arm of the stemming comparison) has no wheels past Python 3.9, so it's split into its own
optional `hunspell` group — install it separately under a 3.9 interpreter if you need that
specific comparison arm; everything else works on 3.10/3.11.

## Repository contents

- `dataset/` — the final dataset used to train the deployed model
- `hunspell-id-main/` — tools for building a Hunspell stemmer and stopword-removal
  function
- `main/` — the pipeline: (1) Twitter data crawling (`twitter_crawl_data.ipynb`),
  (2) comparing three stemming/stopword-removal tools (Sastrawi, Hunspell, Stanza),
  (3) preprocessing, word clouds, loading the embedding matrix, model training, and
  metric evaluation across scenarios (`lstm.ipynb`), (4) the IndoBERT comparison
  (`indobert_finetune.py`)

**Not included**: pre-trained FastText word vectors for Indonesian — download them from
[fastText's crawl vectors](https://fasttext.cc/docs/en/crawl-vectors.html) (search for
"Indonesian" and grab the `.bin` file).

---

## Data provenance

`dataset/final_dataset.xlsx` is a corpus of Indonesian-language posts collected from
X (Twitter) through the public API and labelled for suicide-ideation content. The file is
committed because Twitter's free developer API no longer permits historical queries, so
the training data cannot be re-fetched for reproducibility.

The post text is third-party user-generated content. It is **not our work**, we assert no
licence over it, and it is included here for research reproducibility only. The labels,
preprocessing pipeline and models in this repository are our own work, released under the
MIT licence in `LICENSE`.

### Identifiers in the corpus were pseudonymised

The corpus originally carried live account names. Measured across all six sheets of the
workbook, there were 27,559 `@handle` occurrences covering 7,602 distinct accounts; in the
`10k` training sheet alone, 4,318 of 10,128 rows contained a handle and **296 of those rows
carried the positive (ideation) label.** An account name next to a mental-health label is
identifiable personal data, so before publishing the workbook every handle was replaced
with a stable pseudonym (`@user_` + 8 hex characters of `sha256(salt ‖ handle)`), and
`twitter.com/<user>/status/<id>` URLs were rewritten to their id-only form.

Text and labels are otherwise unchanged, so this file is still the corpus the reported
metrics were computed from, apart from the handle substitution. The salt and the
handle → pseudonym map are kept out of the repository (`private/`, gitignored), so the
mapping is reversible for us and not for a reader.

Reproduce or verify it with:

```bash
python main/pseudonymise_workbook.py --check    # measure handles, write nothing
python main/pseudonymise_workbook.py --write    # apply the transform
```

The transform verifies its own output and refuses to write if any handle survives.

Two things were deliberately left alone:

- **Emoji mojibake.** The workbook mis-decoded UTF-8 as cp1252, so emoji are stored as
  sequences like `ðŸ˜”`. 1,450 rows in the `10k` sheet are affected. A repair exists in
  `main/build_public_release.py --repair-mojibake` but is off by default, so that this
  file matches the metrics reported below.
- **Three corporate email addresses** appearing inside post text
  (`info@careerbridge.coach`, `help@dana.id`). These are business support addresses, not
  personal accounts, and the handle transform does not touch text preceded by a word
  character.
