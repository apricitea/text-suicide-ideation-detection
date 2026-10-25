import pytest

from si_eval.data import (
    DEFAULT_DATASET,
    clean_split,
    dedupe,
    dedupe_train,
    load_corpus,
    normalise,
    split_like_thesis,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_DATASET.exists(), reason="corpus not present in this checkout"
)


def test_normalise_collapses_links_handles_and_case():
    assert normalise("Halo @Someone https://t.co/x ABC!!!") == "halo user url abc"
    # Different handles and different link targets normalise to the same string.
    assert normalise("Halo @Someone https://t.co/x ABC") == normalise("halo @other http://a.b ABC")
    assert normalise("Aku   mau") == "aku mau"


def test_split_reproduces_the_published_sizes():
    df = load_corpus()
    split = split_like_thesis(df)
    assert split.n_train == 8102
    assert split.n_val == 2026
    assert sum(split.val_label) == 392
    assert split.n_leaked > 0


def test_leakage_is_measured_not_assumed():
    df = load_corpus()
    split = split_like_thesis(df)
    # The thesis corpus is a scraped Twitter sample; a third of the validation rows
    # repeating a training message is the defect this package exists to quantify.
    assert 0.25 < split.leaked_fraction < 0.45
    assert split.n_leaked + sum(split.clean_mask) == split.n_val


def test_dedupe_removes_only_repeats():
    df = load_corpus()
    deduped = dedupe(df)
    assert len(deduped) < len(df)
    keys = [normalise(t) for t in deduped["text"]]
    assert len(keys) == len(set(keys))


def test_clean_split_has_no_cross_split_duplicates():
    df = load_corpus()
    clean = clean_split(df)
    train_keys = {normalise(t) for t in clean.train_text}
    assert not any(normalise(t) in train_keys for t in clean.val_text)


def test_dedupe_train_leaves_no_internal_duplicates():
    df = load_corpus()
    split = split_like_thesis(df)
    texts, labels = dedupe_train(split)
    assert len(texts) == len(labels)
    assert len({normalise(t) for t in texts}) == len(texts)
    assert len(texts) < len(split.train_text)
