# relabelled_v1 — an independent re-annotation of the 10k corpus

Every row of `dataset/final_dataset.xlsx[sheet=10k]` with the published label, the three coder
labels, and a **machine** verdict produced under the recovered codebook (`audit/codebook.md`,
from the thesis's Table 1, adapted from Sawhney et al. 2018).

## Columns

| column | meaning |
|---|---|
| `corpus_index` | row identity in the workbook's `10k` sheet |
| `text` | post text, unchanged (pseudonymous, not anonymous) |
| `label_published` | the original label: majority of three human coders |
| `ACN` `BME` `AIP` | the individual coder labels |
| `coder_agreement` | `unanimous` / `split` / `incomplete` |
| `label_machine` | **machine** verdict (an LLM annotator, not a human) |
| `machine_rule` | deciding codebook rule, `R0`–`R7` (`R9` = undecidable) |
| `machine_confidence` | the annotator's own certainty, not a calibrated probability |
| `machine_disagrees` | machine verdict differs from `label_published` |
| `label_adjudicated` | reserved and empty: for a human pass over the disagreements |

## What it measures

- coverage 10125/10128 (100.0%); undecidable rows 3
- agreement with the published label: 92.72% overall, 92.67% on unanimously-labelled rows, 94.13% on rows the coders already disputed
- **7.33% of unanimously-labelled rows** are read differently by the machine annotator (Wilson 95% [6.83%, 7.87%]) — an upper bound on the published labels' error rate
- the annotator's own noise floor: on 2004 texts it saw twice it gave the same label 98.05% of the time, so most of the disagreement above is a consistent different reading, not annotator jitter
- positive rate: machine 0.257 vs published 0.194

## Provenance and caveats

- annotator `deepseek/deepseek-v4.1-flash-1017` through the Hermes CLI, prompt `01d25055c81b` (hash covers codebook + instructions)
- `@mentions` and URLs were replaced with tokens before the text left the host; the `text`
  column here keeps the original characters
- **machine labels are not ground truth.** Treat them as a second independent reading, and as
  the triage list for human adjudication of the disagreement rows

*Machine-assisted annotation; not human-adjudicated. Written by Nyx.*
