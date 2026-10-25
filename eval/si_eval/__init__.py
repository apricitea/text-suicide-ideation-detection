"""Reproducible evaluation of the Indonesian suicide-ideation classifiers.

The package answers four questions about the thesis result (FastText + LSTM,
positive-class F1 0.79) and its later transformer comparison (fine-tuned IndoBERT,
F1 0.90):

1. How much of the reported validation performance comes from duplicate rows that
   appear in both the training and validation halves of the split?
2. Once those rows are removed, how do the transformers compare with cheap
   classical baselines trained on exactly the same data?
3. Are the predicted probabilities calibrated, and what happens to precision when
   the classifier is used at a clinically plausible operating point?
4. Which errors survive, and are the differences between models statistically
   distinguishable at this sample size?
"""

__version__ = "0.1.0"
