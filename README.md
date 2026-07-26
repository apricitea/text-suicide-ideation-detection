# Suicide Ideation Detection using FastText Word Embeddings and LSTM

Originally my undergraduate thesis, documented here for reproducibility. A deployed
version of the model is on [HuggingFace](https://huggingface.co/apricitea); full write-up
of the process is on [Medium](https://medium.com/@apricitea).

This is a research/educational project on Indonesian-language text classification, not a
clinical or diagnostic tool — model outputs should not be treated as a substitute for
professional mental health assessment.

## Reproducing it

Dependencies are managed with Poetry. Twitter's free Developer API no longer supports
querying historical tweets, so `dataset/final_dataset.xlsx` is included directly instead
of a live crawl.

## Repository contents

- `dataset/` — the final dataset used to train the deployed model
- `hunspell-id-main/` — tools for building a Hunspell stemmer and stopword-removal
  function
- `main/` — the pipeline: (1) Twitter data crawling (`twitter_crawl_data.ipynb`),
  (2) comparing three stemming/stopword-removal tools (Sastrawi, Hunspell, Stanza),
  (3) preprocessing, word clouds, loading the embedding matrix, model training, and
  metric evaluation across scenarios

**Not included**: pre-trained FastText word vectors for Indonesian — download them from
[fastText's crawl vectors](https://fasttext.cc/docs/en/crawl-vectors.html) (search for
"Indonesian" and grab the `.bin` file).
