# Re-evaluation: duplicate leakage in an Indonesian suicide-ideation classifier

Generated 2026-10-25 16:33 UTC by
`eval/si_eval/run.py`. Every number below is computed from the committed corpus and
the committed IndoBERT predictions; nothing is copied from the thesis.

## Summary

- **739 of 2026 validation rows**
  (36.5%) are near-duplicates of a training row, because
  the split was drawn at random before duplicate messages were removed.
- On the **1287 genuinely unseen rows**, the fine-tuned transformer
  reaches positive-class F1 **0.877**, down from
  **0.910** on the published split. The reported number was inflated
  by duplicates.
- The strongest cheap baseline (tfidf char logreg)
  reaches F1 **0.787**
  on the same unseen rows; the paired difference is
  **+0.090** (95% CI +0.049 to +0.131,
  McNemar p = 0.0006329).
- The probabilities are close to honest in aggregate: mean predicted 0.206 against an observed positive rate of 0.201 on unseen rows (ECE 0.046). The operating points below are therefore readable as probabilities rather than as arbitrary scores.

## 1. Corpus and split

| Item | Value |
|---|---|
| Rows | 10128 |
| Positives | 1967 (19.4%) |
| Exact duplicate messages | 2161 |
| Duplicates after normalisation | 2202 |
| Unique normalised messages | 7926 |
| Split | `train_test_split(train_size=0.8, random_state=1)` |
| Train / validation rows | 8102 / 2026 |
| Validation positives | 392 |
| Validation rows with a training twin | **739** (36.5%) |
| Unseen validation rows | 1287 (259 positive) |

The corpus is a scraped Twitter sample, so repeated messages are expected: retweets,
quote-tweets and copy-paste slogans all appear more than once. A random 80/20 split
places the same message on both sides about a third of the time. Every metric computed
on the full validation set is therefore partly a memorisation test, not a
generalisation test.

## 2. Positive-class F1 by model and protocol

| Model | Training data | F1, published split | F1, unseen rows (95% CI) | Accuracy, unseen rows |
|---|---|---|---|---|
| IndoBERT (fine-tuned, committed predictions) | as published | 0.910 | 0.877 | 0.950 |
| majority class | train as published | 0.000 | 0.000 [0.000, 0.000] | 0.799 |
| majority class | train deduplicated | 0.000 | 0.000 [0.000, 0.000] | 0.799 |
| count word complementnb | train as published | 0.784 | 0.745 [0.707, 0.785] | 0.893 |
| count word complementnb | train deduplicated | 0.787 | 0.762 [0.727, 0.796] | 0.895 |
| tfidf word logreg | train as published | 0.848 | 0.786 [0.745, 0.824] | 0.921 |
| tfidf word logreg | train deduplicated | 0.834 | 0.766 [0.722, 0.807] | 0.915 |
| tfidf char logreg | train as published | 0.832 | 0.802 [0.761, 0.839] | 0.928 |
| tfidf char logreg | train deduplicated | 0.820 | 0.787 [0.744, 0.827] | 0.923 |

"Unseen rows" means the 1287 validation rows whose normalised text does
not occur in the training half. IndoBERT was trained once, on the published split; it is
scored on the same unseen rows as the baselines, so the comparison is on identical
observations. The published-split figure is shown only to quantify the inflation.

Classical baselines use raw text with no Indonesian stemming or stopword removal. That
choice disadvantages them slightly and therefore does not flatter the transformer.

## 3. Is the transformer better than the baseline?

| Test | Result |
|---|---|
| Baseline compared | `tfidf_char_logreg@train_deduplicated` |
| Paired F1 difference (IndoBERT - baseline) | +0.090 |
| 95% bootstrap interval | [+0.049, +0.131] |
| McNemar discordant pairs (IndoBERT right / baseline right) | 67 / 32 |
| McNemar p-value (chi2_continuity) | 0.0006329 |

The difference is statistically distinguishable at this sample size.

## 4. Calibration and operating points

| Measure | Value |
|---|---|
| Expected calibration error (15 bins) | 0.046 |
| Maximum calibration error (sparse-bin estimate) | 0.817 |
| Brier score | 0.046 |
| Mean predicted probability | 0.206 |
| Observed positive rate, unseen rows | 0.201 |

For a screening use, the question is not "what is F1 at 0.5" but "how much has to be
reviewed to catch a given share of positive messages":

| Target recall | Threshold | Achieved recall | Precision | Share of corpus flagged |
|---|---|---|---|---|
| 80% | 0.985 | 0.803 | 0.920 | 17.6% |
| 90% | 0.087 | 0.907 | 0.836 | 21.8% |
| 95% | 0.006 | 0.954 | 0.767 | 25.0% |

## 5. Where the errors are

| Bucket | Rows | Error rate | False-negative rate | False-positive rate |
|---|---|---|---|---|
| short (<80 chars) | 323 | 5.3% | 6.6% | 4.6% |
| medium (80-160) | 390 | 4.9% | 13.3% | 2.9% |
| long (>=160) | 574 | 4.9% | 16.7% | 3.0% |
| contains negation | 381 | 5.2% | 15.7% | 3.6% |
| first person | 539 | 8.0% | 12.3% | 6.2% |
| contains question mark | 176 | 2.8% | 13.8% | 0.7% |
| all rows | 1287 | 5.0% | 11.6% | 3.3% |

False negatives (missed ideation), highest-confidence first:

- `Bunuh aja aku tolong, gamau hidup tp gamau masuk neraka` — p(ideation) = 0.004
- `Sebenernyaa, aku jadi paham kenapa org bisa selingkuh dan bunuh diri. Kalo boleh milih pun aku bakal milih keduanya. Tapi hebatnya allah ngasih aku hati yg tulu` — p(ideation) = 0.001
- `@user_0520332b dulu pas putus cinta mau bunuh diri sampe keracunan wkwkkwkw ngakak pas inget lg mikir kok dlu gua aneh bgt sih‚` — p(ideation) = 0.002
- `Siapa tau ada yg mau bunuh diri. Skuy dengar lagu ini trus kita bunuh diri bareng » https://t.co/vpK7Cjk7FL` — p(ideation) = 0.002

False positives:

- `Ini udh gabisa dibiarkaaan gw udh gakuat lg hidup tanpa wifiðŸ˜­ðŸ˜­ðŸ˜­ðŸ˜­ðŸ˜­ðŸ˜­ðŸ˜­` — p(ideation) = 0.900
- `@user_1539fc94 Ada konten tiktok bilangnya gini "jgn ajarin aku sbr krn selama ini alesan aku hidup cuma krn tau kalo bunuh diri itu dosa"` — p(ideation) = 0.992
- `"jangan bikin alasan aku bunuh diri fikri" IDIH, NAJIEZ` — p(ideation) = 0.954
- `9. Masyarakat sekitar juga sulit dan pemerintah daeraha kurang perhatian

Kalau dipaparin banyak lagi. Bahkan aku yang lihat kondisi orang tersebut kayak gitu s` — p(ideation) = 0.817

These buckets are descriptive and overlap; they identify where to look, not why the
model failed.

## 6. Limitations

1. **This report scores weights that saw the leaked twins.** Removing duplicate rows from
   the evaluation removes the direct overlap, not the advantage those rows gave the model
   during training. The fully clean protocol --- deduplicate, re-split, retrain --- is
   reported in `eval/out/clean-retrain-comparison/`: positive F1 0.849 [0.821, 0.877]
   against 0.785 for the strongest baseline. The intervals here are therefore optimistic
   relative to that protocol.
2. **The FastText + LSTM thesis arm cannot be re-run.** Its FastText vectors are not in
   the repository and the Keras tokenizer was never saved with the model, so its
   per-row predictions do not exist. The 0.79 F1 is therefore quoted from the thesis,
   not recomputed here. That is itself a reproducibility defect worth recording.
3. **One split, one seed.** The comparison conditions on a single 80/20 draw. Repeated
   splits would widen the intervals and could change the ranking of the two strongest
   models.
4. **Labels are single-annotator.** No inter-annotator agreement exists for this corpus,
   so the ceiling on measured F1 is unknown.
5. **Sensitive content.** Examples are reproduced from an already-public corpus with
   account handles replaced by stable pseudonyms.

## 7. Reproduce

```bash
uv sync --project eval
uv run --project eval python -m si_eval.run
uv run --project eval pytest eval/tests -q
```

Outputs land in `eval/out/<run-id>/` with `results.json` and a manifest recording the
dataset hash, git revision and package versions.
