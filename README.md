# Suicide Ideation Detection — FastText+LSTM (thesis) vs. Fine-Tuned IndoBERT

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

Fine-tuned IndoBERT weights aren't committed (too large for git) — rerun
`main/indobert_finetune.py` to reproduce, takes ~35 min on an RTX 3050 6GB.

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
