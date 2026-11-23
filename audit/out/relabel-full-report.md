# Re-annotation pass 1 — machine labels vs the published labels

Mode `full` · ledger `audit/out/relabel-full.jsonl` · 10398/14562 rows annotated · model deepseek/deepseek-v4.1-flash-1017 · prompt 01d25055c81b · undecidable 2

## Agreement with the published (three-coder majority) label

| stratum | n | agree | agreement |
|---|---|---|---|
| full | 6140 | 5713 | 93.05% |

## Rows the coders split 2-1 (n = 225)

- the machine matches the published majority in 94.67% of them
- machine sides with: ACN 94.67%, BME 51.11%, AIP 48.89%

## Rows all three coders agreed on (n = 5915)

- the machine disagrees on 415 (7.02%, Wilson 95% [6.39%, 7.70%])
- direction: 378 machine-positive/published-negative, 37 machine-negative/published-positive
- by published label: 0: 378/4726, 1: 37/1189
- by length: short: 142/1506, medium: 123/1726, long: 150/2683

This is an upper bound on how often the published label is wrong: it also counts the places where the machine annotator is the one who is wrong. Every disagreed row is enumerable for human adjudication.

## Duplicate texts

- 6251 rows share their text with another row (1995 distinct texts); the agreement figures above count each text once
- of those repeated texts, 1957 got a single machine label and 2 carry differing *published* labels (same text, different gold label)

## The machine annotator's own behaviour

- positive rate: machine 0.259 vs published 0.202
- mean confidence 0.823, 367 rows below 0.6
- rules: R0 3855, R1 1031, R4 425, R7 340, R5 251, R6 105, R2 86, R3 47, R9 2
