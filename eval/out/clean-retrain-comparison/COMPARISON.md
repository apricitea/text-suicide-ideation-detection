# Clean-split comparison

Deduplicated corpus, then an 80/20 split with no cross-split duplicates: 6340 train / 1586 validation (311 positive), cross-split duplicates = 0.

| Model | Positive-class F1 on the clean validation half (95% CI) |
|---|---|
| indobert clean split | 0.849 [0.821, 0.877] |
| tfidf word logreg | 0.785 [0.749, 0.819] |
| tfidf char logreg | 0.775 [0.737, 0.812] |
| count word complementnb | 0.734 [0.701, 0.769] |
| majority class | 0.000 [0.000, 0.000] |

IndoBERT vs the strongest baseline (`tfidf_word_logreg`): **+0.064** F1 [+0.029, +0.105], McNemar p = 0.01532 (76 / 48 discordant pairs).

This is the protocol the audit in `eval/` recommends: deduplicate first, then split.
The training half is smaller than the published one, so the intervals are wider; that
is the cost of removing the duplicate overlap rather than adjusting for it.
