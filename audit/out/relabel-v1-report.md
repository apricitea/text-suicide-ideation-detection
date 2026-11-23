# Re-annotation pass 1 — machine labels vs the published labels

Mode `phase1` · ledger `audit/out/relabel-v1.jsonl` · 658/658 rows annotated · model deepseek/deepseek-v4.1-flash-1017 · prompt 01d25055c81b · undecidable 0

## Agreement with the published (three-coder majority) label

| stratum | n | agree | agreement |
|---|---|---|---|
| disputed | 375 | 354 | 94.40% |
| unanimous_sample | 283 | 265 | 93.64% |

## Rows the coders split 2-1 (n = 375)

- the machine matches the published majority in 94.40% of them
- machine sides with: ACN 94.67%, BME 48.00%, AIP 51.20%

## Rows all three coders agreed on (n = 283)

- the machine disagrees on 18 (6.36%, Wilson 95% [4.06%, 9.83%])
- direction: 16 machine-positive/published-negative, 2 machine-negative/published-positive
- by published label: 0: 16/146, 1: 2/137
- by length: short: 6/93, medium: 8/95, long: 4/95

This is an upper bound on how often the published label is wrong: it also counts the places where the machine annotator is the one who is wrong. Every disagreed row is enumerable for human adjudication.

## The machine annotator's own behaviour

- positive rate: machine 0.385 vs published 0.334
- mean confidence 0.842, 26 rows below 0.6
- rules: R0 353, R1 184, R4 43, R7 29, R2 19, R5 16, R6 7, R3 7
