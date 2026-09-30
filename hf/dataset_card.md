---
license: other
license_name: research-use-only
license_link: https://github.com/apricitea/text-suicide-ideation-detection#data-provenance
language:
  - id
task_categories:
  - text-classification
size_categories:
  - 10K<n<100K
tags:
  - suicide-ideation
  - mental-health
  - indonesian
  - twitter
pretty_name: Indonesian suicide-ideation detection (10,128 posts, pseudonymised)
configs:
  - config_name: default
    data_files:
      - split: train
        path: bahasa-suicide-ideation-10k.jsonl
---

# Indonesian suicide-ideation detection — 10,128 posts (pseudonymised)

The annotated corpus behind the undergraduate thesis
[text-suicide-ideation-detection](https://github.com/apricitea/text-suicide-ideation-detection):
Indonesian-language posts labelled for the presence of suicidal ideation.

This is a **research/educational dataset, not a clinical tool.** Labels are an
annotation of text, not a diagnosis, and no model trained on it should be used
to assess a real person.

## Contents

`bahasa-suicide-ideation-10k.jsonl` — one JSON object per line:

| field | meaning |
|---|---|
| `id` | stable content-derived row id |
| `text` | post text, identifiers replaced (see below) |
| `label` | `1` = ideation present, `0` = not present |
| `label_name` | `ideation` / `no_ideation` |
| `provenance` | `x_public_api_2023` |

| | |
|---|---|
| records | 10,128 |
| positive (ideation) | 1,967 (19.4%) |
| negative | 8,161 (80.6%) |
| language | Indonesian (`id`), informal/social-media register |

This is the same text as the `10k` sheet of the thesis workbook, verified
row-for-row (10,128/10,128 identical in text and label).

The 8:2 imbalance is inherent to the source and is **not** corrected here. The
thesis studied class-weighting and ADASYN as treatments; both *reduced*
positive-class F1 on the LSTM arm. Decide the imbalance treatment yourself and
report it.

## Identifiers were removed, not the text

Publishing this raw would pair a live account name with a mental-health label.
So every handle is replaced with a stable pseudonym: `@user_` + 8 hex characters
of `sha256(salt ‖ handle)`, and `twitter.com/<user>/status/<id>` URLs are
rewritten to their id-only form. Text and labels are otherwise untouched.

Within the 10k training sheet:

| | before | after |
|---|---|---|
| `@handle` occurrences | 6,988 | 0 |
| rows containing a handle | 4,318 | 0 |
| rows with a handle **and** the positive label | 296 | **0** |

Across all six sheets of the source workbook the same transform replaces 27,559
handle occurrences (7,602 distinct accounts). The salt and the handle→pseudonym
map are deliberately **not** part of this release, so the mapping cannot be
reversed from here.

Verification is done on the output, not the input: the build fails loudly if any
handle survives, and an independent stdlib scan of the released workbook reports
0 residual handles.

### What was not changed

- **Emoji mojibake is left as-is.** The source workbook mis-decoded UTF-8 as
  cp1252, so emoji arrive as sequences like `ðŸ˜”` rather than the character.
  1,450 rows in the 10k sheet are affected. Repairing it is implemented and
  available (`build_public_release.py --repair-mojibake`), but it is **off by
  default** so that this release stays aligned with the metrics the repository
  reports; repairing the emoji changes the text a model sees.
- **Three email addresses remain** (`info@careerbridge.coach`, `help@dana.id`).
  These are corporate support addresses published in the source posts, not
  personal accounts, and the handle transform deliberately does not touch
  anything preceded by a word character. Treat them accordingly.

## Provenance and rights

The post text is **third-party user-generated content** collected from the
public X/Twitter API in 2023. It is not the author's work and **no licence is
asserted over the text.** It is redistributed here, pseudonymised, for research
reproducibility: X's free API no longer permits retrieving historical posts, so
without the text the reported results could not be checked at all. The
annotations, pseudonymisation and pipeline are the author's own work.

If you are the author of a post included here and want it removed, open an issue
on the GitHub repository.

## Limitations

- **Pseudonymisation is not anonymisation.** Post text can still identify
  someone (self-disclosure, rare phrasing, quoted content), and the labels
  concern mental health. Treat the file as sensitive.
- Single annotator, no inter-annotator agreement. Idiom, sarcasm and quoted
  lyrics are the hardest cases; some label noise is certain.
- Indonesian social-media register only. Do not expect transfer to formal
  Indonesian, other languages, or clinical text.
- 2023 collection window; platform vocabulary drifts.
- Emoji are mangled in the source and were not repaired (above).

## Citation

Cite the GitHub repository and this release; see it for the current citation.
A DOI is intended.