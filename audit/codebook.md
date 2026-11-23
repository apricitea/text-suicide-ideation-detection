# Annotation codebook — presence of suicidal ideation (v1, recovered from the thesis)

Recovered from `undergraduate_thesis_acn.pdf` §2.1 Table 1 ("Data labeling rules"), which adapts
Sawhney et al. (2018). This is the rule set the corpus's three coders (`ACN`, `BME`, `AIP`) were
given; the published `label` column is their majority vote. Any re-annotation must use this
codebook, not a fresh intuition, or it cannot be compared with the published labels.

## Label 1 — presence of suicidal ideation

| id | rule |
|---|---|
| R1 | The tweet conveys an **explicit or serious display** of suicidal ideation by its author (first-person wanting to die / not wanting to live / wanting it to end). |
| R2 | Suicide **plan** and/or **previous attempts** are discussed by the author. |
| R3 | The author **asks for advice or ways** to hurt or kill himself/herself. |
| R4 | The author **implicitly states intent to die/end their life if a condition is fulfilled**, and displays **no will to continue living**. |

## Label 0 — absence of suicidal ideation

| id | rule |
|---|---|
| R0 | Default for tweets with **no presence or relevance** to suicidal ideation. |
| R5 | Tweets emphasising **suicide-related news or information** (headlines, statistics, reporting). |
| R6 | Tweets pertaining to **condolence and suicide awareness**. |
| R7 | Tweets containing **song lyrics, movie titles/quotes, or discussion of fictional works**. |

## Boundary decisions the rules imply

- **Author's own state, not the topic.** A tweet *about* suicide that does not express the author's
  ideation is 0 (R5–R7), however grim it is.
- **Figurative and hyperbolic uses are 0.** Thesis example: `nekat mau belajar ukulele agaknya
  bunuh diri` → 0. Idiom, sarcasm, jokes and quoted speech do not express ideation.
- **Third-person and reported content is 0**: news, campaign material, statistics, advice *to others*,
  helpline posts.
- **Conditional and implicit forms are 1** (R4): e.g. "kalau X terjadi, aku sudah tidak ada lagi",
  "tidak kuat lagi hidup", "ingin semuanya selesai".
- **Ambiguity**: when the text could be read as either, prefer the reading of the *author as a
  person* (R1 over R0) — that is what the original annotation did, and it is the recall-favouring
  direction the thesis explicitly chose ("FP is considered a better case compared to FN").

## Lexical cues (from the crawl keyword list — cues, never sufficient)

`bunuh diri`, `ingin bunuh diri`, `mau bunuh diri`, `cara bunuh diri`, `pikiran bunuh diri`,
`rencana bunuh diri`, `ide bunuh diri`, `mengakhiri hidup`, `akhirin hidup`, `tidak ingin
melanjutkan hidup`, `tidak ada gunanya hidup`, `tidak ada alasan untuk hidup`, `tidak kuat hidup`,
`tidur selamanya`, `mau mati`, `ingin mati`, `lebih baik mati`, `mati sendirian`, `siap untuk mati`,
`ingin semuanya selesai`, `mengambil nyawaku sendiri`, `melukai pergelangan tangan`, `cutting`.

## Output contract for a re-annotation pass

One JSON object per row, no free text reproduction of the tweet:

```json
{"i": 1234, "label": 1, "rule": "R4", "confidence": 0.72}
```

`label` is the codebook verdict; `rule` names the deciding rule (R0–R7); `confidence` is the
annotator's own certainty, not a probability that the codebook is right. Any row where the
annotator cannot pick a rule is `label: null` with `rule: "R9"` (undecidable) — those rows are the
ones a human must see.
