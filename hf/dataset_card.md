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

This is a **research/educational dataset, not a clinical tool.** Labels are the
author's annotation of text, not a diagnosis, and no model trained on it should
be used to assess a real person.

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

The 8:2 imbalance is inherent to the source and is **not** corrected here — the
thesis explicitly studied class-weighting and ADASYN as treatments, and both
*reduced* positive-class F1 on the LSTM arm. Anyone training on this should
decide the imbalance treatment themselves and report it.

## Identifiers were removed, not the text

Publishing this raw would have paired a live account name with a mental-health
label. Measured on the source workbook, 4,318 of 10,128 rows contained an
`@handle`, and **296 of those carried the positive label.**

So every handle is replaced with a stable pseudonym (`@user_` + 8 hex characters
of `sha256(salt ‖ handle)`) and `twitter.com/<user>/status/<id>` URLs are rewritten
to their id-only form. Text, labels and counts are otherwise untouched.

| | before | after |
|---|---|---|
| `@handle` occurrences | 6,987 | 0 |
| distinct handles | 2,943 | 2,943 pseudonyms |
| rows containing a handle | 4,318 | 0 |
| rows with a handle **and** the positive label | 296 | **0** |

The salt and the handle→pseudonym map are deliberately **not** part of this
release, so the mapping cannot be reversed from here. The build is reproducible
with [`main/build_public_release.py`](https://github.com/apricitea/text-suicide-ideation-detection/blob/main/main/build_public_release.py),
and the release manifest records the counts above.

Emoji were also repaired: the source workbook mis-decoded UTF-8 as cp1252, so
1,450 rows contained mojibake (`ðŸ˜”`) instead of the intended character.
Affected rows are restored (that example becomes 😔); rows that were already
clean are byte-identical to the source.

## Provenance and rights

The post text is **third-party user-generated content** collected from the
public X/Twitter API in 2023. It is not the author's work, and **no licence is
asserted over the text.** It is redistributed here, pseudonymised, for research
reproducibility, because X's free API no longer permits retrieving historical
posts — without including the text the reported results could not be checked at
all. The annotations, pseudonymisation and pipeline are the author's own work.

If you are the author of a post included here and want it removed, open an issue
on the GitHub repository and it will be dropped in the next revision.

## Limitations

- **Pseudonymisation is not anonymisation.** The post text itself can still
  identify someone (self-disclosure, rare phrasing, quoted content), and the
  labels concern mental health. Treat it as sensitive.
- Single annotator, no inter-annotator agreement. Idiom, sarcasm and quoted
  lyrics are the hardest cases; some label noise is certain.
- Indonesian social-media register only. Do not expect it to transfer to formal
  Indonesian, other languages, or clinical text.
- 2023 collection window. Vocabulary and platform norms drift.

## Citation

See the GitHub repository for the current citation. A DOI is intended; until one
exists, cite the repository and the release manifest.