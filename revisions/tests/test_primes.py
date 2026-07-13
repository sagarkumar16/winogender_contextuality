"""
Tests for the referent-only null primes (revisions/primes.py).

The design requirement is exact: the prime must contain the referents and NO pronoun. These
tests pin that, plus the grammatical-case handling that makes the generated sentences
well-formed, and they run over the real WinoPron items when those are available.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from revisions.primes import (
    PRIME_STYLES,
    build_referent_primes,
    conjunction_prime,
    find_inanimate_pronouns,
    find_pronouns,
    referent_np,
    repeated_np_prime,
)

PAIRS_TSV = Path(__file__).resolve().parents[2] / "data" / "interim" / "winopron_pairs.tsv"


@pytest.fixture
def toy_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "template_1": "The technician told the customer that BLANK could pay with cash.",
                "differences_1": "['he', 'she']",
                "case_1": "$NOM_PRONOUN",
                "referent_1": "customer",
                "template_2": "The technician told the customer that BLANK had completed the repair.",
                "differences_2": "['he', 'she']",
                "case_2": "$NOM_PRONOUN",
                "referent_2": "technician",
            },
            {
                "template_1": "The customer told the technician that BLANK fridge was broken.",
                "differences_1": "['his', 'her']",
                "case_1": "$POSS_PRONOUN",
                "referent_1": "customer",
                "template_2": "The customer told the technician that BLANK charges were too high.",
                "differences_2": "['his', 'her']",
                "case_2": "$POSS_PRONOUN",
                "referent_2": "technician",
            },
        ]
    )


# --------------------------------------------------------------------------------------
# Noun-phrase construction
# --------------------------------------------------------------------------------------


def test_referent_np_by_case():
    assert referent_np("customer", "$NOM_PRONOUN") == "the customer"
    assert referent_np("customer", "$ACC_PRONOUN") == "the customer"
    assert referent_np("customer", "$POSS_PRONOUN") == "the customer's"


def test_referent_np_rejects_unknown_case():
    with pytest.raises(ValueError):
        referent_np("customer", "$REFLEXIVE_PRONOUN")


def test_repeated_np_prime_nominative():
    out = repeated_np_prime(
        "The technician told the customer that BLANK had completed the repair.",
        "technician",
        "$NOM_PRONOUN",
    )
    assert out == "The technician told the customer that the technician had completed the repair."
    assert find_pronouns(out) == []


def test_repeated_np_prime_possessive_gets_genitive():
    """A possessive pronoun ('his fridge') must become a genitive NP ('the customer's fridge')."""
    out = repeated_np_prime(
        "The customer told the technician that BLANK fridge was broken.",
        "customer",
        "$POSS_PRONOUN",
    )
    assert out == "The customer told the technician that the customer's fridge was broken."
    assert find_pronouns(out) == []


def test_sentence_initial_blank_is_capitalised():
    out = repeated_np_prime("BLANK paid with cash.", "customer", "$NOM_PRONOUN")
    assert out == "The customer paid with cash."
    assert not out.startswith("the ")


def test_conjunction_prime_names_both_referents_and_no_pronoun():
    out = conjunction_prime("technician", "customer")
    assert "technician" in out and "customer" in out
    assert find_pronouns(out) == []


# --------------------------------------------------------------------------------------
# Pronoun detection
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pronoun", ["he", "she", "him", "her", "his", "they", "them", "their", "xe", "xyr"])
def test_find_pronouns_detects_personal_pronouns(pronoun):
    assert find_pronouns(f"The technician said {pronoun} was late.") == [pronoun]


def test_find_pronouns_is_word_bounded():
    # "the" contains "he"; "shell" contains "she". Neither is a pronoun.
    assert find_pronouns("The shell of the theremin.") == []


def test_inanimate_pronouns_are_tracked_separately():
    s = "The patient found it hard to eat enough."
    assert find_pronouns(s) == []  # not a gender-priming token
    assert find_inanimate_pronouns(s) == ["it"]


# --------------------------------------------------------------------------------------
# Table construction
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("style", PRIME_STYLES)
def test_build_produces_two_rows_per_pair_with_no_personal_pronouns(toy_pairs, style):
    df = build_referent_primes(toy_pairs, style=style)

    assert len(df) == 2 * len(toy_pairs)
    assert not df.has_personal_pronoun.any()
    assert set(df.free_slot) == {1, 2}


def test_row_order_and_sent_order_mapping(toy_pairs):
    """
    Row 2k is the free_slot=2 item (sentence order [0,1], free sentence = template_2) and row
    2k+1 is free_slot=1 ([1,0]). revisions.referent_null.null_row_for depends on this.
    """
    df = build_referent_primes(toy_pairs)

    for k in range(len(toy_pairs)):
        first, second = df.iloc[2 * k], df.iloc[2 * k + 1]

        assert first.pair_index == k and first.free_slot == 2
        assert first.sent_order == "[0, 1]"
        assert first.template == toy_pairs.template_2[k]

        assert second.pair_index == k and second.free_slot == 1
        assert second.sent_order == "[1, 0]"
        assert second.template == toy_pairs.template_1[k]


def test_prime_comes_from_the_other_sentence(toy_pairs):
    """The prime is built from the sentence that is NOT being completed."""
    df = build_referent_primes(toy_pairs)

    row = df.iloc[0]  # free = template_2, so the prime derives from template_1
    assert row.prime_referent == toy_pairs.referent_1[0]
    assert "could pay with cash" in row.prime  # template_1's content
    assert "BLANK" not in row.prime


def test_free_sentence_still_has_its_blank(toy_pairs):
    df = build_referent_primes(toy_pairs)
    assert df.template.str.contains("BLANK").all()


def test_unknown_style_rejected(toy_pairs):
    with pytest.raises(ValueError):
        build_referent_primes(toy_pairs, style="nonsense")


# --------------------------------------------------------------------------------------
# The real dataset
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(not PAIRS_TSV.exists(), reason="winopron_pairs.tsv not available")
def test_no_personal_pronouns_in_any_real_prime():
    """The whole point of the condition, checked against every real WinoPron item."""
    pairs = pd.read_csv(PAIRS_TSV, sep="\t")
    df = build_referent_primes(pairs, style="repeated_np", max_index=180)

    offenders = df[df.has_personal_pronoun]
    assert len(offenders) == 0, f"{len(offenders)} primes contain a personal pronoun:\n{offenders.prime.head()}"
    assert len(df) == 360


@pytest.mark.skipif(not PAIRS_TSV.exists(), reason="winopron_pairs.tsv not available")
def test_real_primes_mention_their_referent():
    pairs = pd.read_csv(PAIRS_TSV, sep="\t")
    df = build_referent_primes(pairs, style="repeated_np", max_index=180)

    for _, row in df.iterrows():
        assert str(row.prime_referent) in row.prime
