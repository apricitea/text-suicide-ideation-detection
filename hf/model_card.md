---
license: mit
language:
  - id
library_name: transformers
pipeline_tag: text-classification
base_model: indobenchmark/indobert-base-p1
datasets:
  - apricitea/bahasa-suicide-ideation-10k
tags:
  - suicide-ideation
  - mental-health
  - indonesian
  - indobert
  - text-classification
metrics:
  - f1
  - accuracy
  - precision
  - recall
---

# IndoBERT — Indonesian suicide-ideation detection

A fine-tuned `indobenchmark/indobert-base-p1` binary classifier for detecting
suicidal ideation in Indonesian social-media text.

**This is a research artifact, not a clinical or diagnostic tool.** It scores
text; it does not assess a person. Do not use it to make decisions about anyone,
and do not present its output to someone in distress as an assessment.

## Results

Evaluated on the held-out 20% of the thesis corpus (2,026 posts), the same split
the FastText+LSTM baseline used (`random_state=1`, no class-imbalance treatment):

| Model | Precision (pos.) | Recall (pos.) | F1 (pos.) | Accuracy |
|---|---|---|---|---|
| FastText + LSTM (baseline) | 0.74 | 0.85 | 0.79 | 0.91 |
| **IndoBERT (this model)** | **0.906** | **0.913** | **0.910** | **0.965** |

Confusion matrix (rows = gold, cols = predicted):

|  | pred 0 | pred 1 |
|---|---|---|
| **gold 0** | 1597 | 37 |
| **gold 1** | 34 | 358 |

So 34 of 392 ideation posts are missed and 37 of 1,634 non-ideation posts are
flagged. The baseline's F1 of 0.79 came mostly from worse precision (0.74) — it
raised false alarms, which for a screening-style task is the failure mode that
matters.

## What this reproduces

The repository previously reported IndoBERT at F1 0.90 / accuracy 0.96 with no
artifact behind it — the training script discarded the weights. This checkpoint
is a full retrain from the committed corpus with the hyperparameters that script
specified, run on CPU (4 threads, no GPU), and it lands at **F1 0.9098 /
accuracy 0.9650**, i.e. it reproduces the reported figures.

Full per-class report, confusion matrix and the training configuration are in
[`metrics.json`](metrics.json). The 2,026 validation predictions (gold, predicted
label, positive probability, text) are in `predictions.csv` in the repository.

## Training

| | |
|---|---|
| Base model | `indobenchmark/indobert-base-p1` |
| Seed | 1 (matches the LSTM split) |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Max sequence length | 128 |
| Train / validation | 8,102 / 2,026 |
| Hardware | CPU, 4 threads, no GPU |

No class-imbalance treatment was applied, deliberately: the thesis tested
class-weighting and ADASYN on the LSTM arm and both *reduced* positive-class F1.
IndoBERT needed none.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

name = "apricitea/indobert-suicide-ideation"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name)

text = "aku udah capek banget, pengen ngilang aja"
inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
print(probs[1].item())  # P(ideation)
```

## Limitations

- **Single annotator, no inter-annotator agreement.** Some label noise is
  certain; idiom, sarcasm and quoted song lyrics are the hardest cases.
- **Indonesian social-media register only**, collected in 2023. Expect
  degradation on formal Indonesian, other languages, and clinical text.
- **Emoji in the training corpus are mangled** (a cp1252 mis-decode in the
  source workbook, ~1,450 rows). The model therefore learned from damaged emoji,
  which is a real capability gap given how much sentiment emoji carry.
- **Not a screening instrument.** Recall of 0.913 means roughly 1 in 12 positive
  posts is missed; it is not suitable for any safety-critical use.
- The probability output is not calibrated; do not read it as a risk level.

## Citation

Cite the GitHub repository and the dataset. A DOI is intended.