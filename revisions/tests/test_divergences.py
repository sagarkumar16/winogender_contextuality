"""
Tests for the divergence machinery and the count/probability extraction (revisions/common.py),
plus the three-condition row alignment used by Experiment 3.
"""

from __future__ import annotations

import numpy as np
import pytest

from revisions.common import (
    FemaleCounts,
    bernoulli_kl,
    female_generation_counts,
    female_internal_prob,
    female_option,
    kl_from_counts,
    male_option,
)
from revisions.fixtures import null_records, refnull_records, steering_records, unprimed_records
from revisions.referent_null import null_row_for

# --------------------------------------------------------------------------------------
# Bernoulli KL
# --------------------------------------------------------------------------------------


def test_kl_is_zero_for_identical_distributions():
    assert bernoulli_kl(0.3, 0.3) == pytest.approx(0.0)


def test_kl_hand_computed():
    # KL(0.5 || 0.25) in bits = 0.5*log2(2) + 0.5*log2(1.5/0.75) ... compute directly:
    # 0.5*log2(0.5/0.25) + 0.5*log2(0.5/0.75) = 0.5*1 + 0.5*log2(2/3)
    expected = 0.5 * np.log2(0.5 / 0.25) + 0.5 * np.log2(0.5 / 0.75)
    assert bernoulli_kl(0.5, 0.25) == pytest.approx(expected)


def test_kl_is_nonnegative():
    rng = np.random.default_rng(3)
    for _ in range(200):
        p, q = rng.uniform(0.01, 0.99, size=2)
        assert bernoulli_kl(p, q) >= -1e-12


def test_kl_is_asymmetric():
    # Note (0.9, 0.1) would NOT work: swapping p and 1-p leaves Bernoulli KL invariant.
    assert bernoulli_kl(0.5, 0.25) != pytest.approx(bernoulli_kl(0.25, 0.5))


# --------------------------------------------------------------------------------------
# Add-half smoothing
# --------------------------------------------------------------------------------------


def test_smoothing_matches_the_published_formula():
    """(k + 0.5) / (n + 1) -- the rule used by get_model_divergences in the paper notebook."""
    c = FemaleCounts(successes=10, total=40)
    assert c.smoothed_prob == pytest.approx(10.5 / 41)
    assert c.raw_prob == pytest.approx(0.25)


def test_smoothing_keeps_kl_finite_at_the_boundary():
    """A cell with zero female generations would make an unsmoothed KL infinite."""
    zero = FemaleCounts(successes=0, total=50)
    base = FemaleCounts(successes=25, total=50)

    assert np.isfinite(kl_from_counts(zero, base))
    assert kl_from_counts(zero, base) > 0


def test_kl_from_counts_is_nan_without_data():
    assert np.isnan(kl_from_counts(FemaleCounts(0, 0), FemaleCounts(10, 20)))
    assert np.isnan(kl_from_counts(FemaleCounts(10, 20), FemaleCounts(0, 0)))


# --------------------------------------------------------------------------------------
# Option resolution -- the ordering hazard in the null runs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options, expected_f, expected_m",
    [
        (["he", "she"], "she", "he"),
        (["she", "he"], "she", "he"),  # reversed presentation order
        (["him", "her"], "her", "him"),
        (["her", "him"], "her", "him"),
        (["his", "her"], "her", "his"),
    ],
)
def test_female_option_resolved_by_string_not_position(options, expected_f, expected_m):
    """
    The null runs store `pronouns_2` in PRESENTED order, which is reversed half the time. Taking
    position [1] would call 'he' the female pronoun in those records; we resolve by string.
    """
    assert female_option(options) == expected_f
    assert male_option(options) == expected_m


def test_female_counts_are_order_invariant():
    """The same generations counted under either presentation order give the same successes."""
    forward = null_records(0, n_female=30, n_male=20, n_order=0, pronouns=["he", "she"])
    reversed_ = null_records(0, n_female=30, n_male=20, n_order=1, pronouns=["she", "he"])

    a = female_generation_counts(forward)
    b = female_generation_counts(reversed_)

    assert (a.successes, a.total) == (30, 50)
    assert (b.successes, b.total) == (30, 50)


def test_generations_outside_the_option_set_are_dropped():
    records = null_records(0, n_female=10, n_male=10)
    records += [
        {**records[0], "measurement": {"BLANK": "None"}},
        {**records[0], "measurement": {"BLANK": "potato"}},
    ]

    counts = female_generation_counts(records)
    assert counts.total == 20  # the two invalid generations are excluded


# --------------------------------------------------------------------------------------
# Internal probabilities
# --------------------------------------------------------------------------------------


def test_female_internal_prob_recovers_the_pinned_value():
    """fixtures.logits_for(p) writes logits whose softmax is exactly p at the female slot."""
    table = np.array([[10.0, 10.0], [10.0, 10.0]])
    records = steering_records(0, table, sent_order=[0, 1], j=0, internal_p=0.73)
    assert female_internal_prob(records) == pytest.approx(0.73, abs=1e-9)


def test_female_internal_prob_is_nan_without_logits():
    records = unprimed_records(0, n_female=5, n_male=5, sent_order=[0, 1])
    assert np.isnan(female_internal_prob(records))


# --------------------------------------------------------------------------------------
# Row alignment between the pair-indexed and slot-indexed runs
# --------------------------------------------------------------------------------------


def test_null_row_mapping():
    """
    Null-style runs are indexed per sentence slot: row 2k is pair k's [0,1] item (free sentence
    = template_2) and row 2k+1 is its [1,0] item. Experiment 3 joins on this.
    """
    assert null_row_for(0, free_slot=2) == 0
    assert null_row_for(0, free_slot=1) == 1
    assert null_row_for(5, free_slot=2) == 10
    assert null_row_for(5, free_slot=1) == 11


def test_null_and_refnull_records_are_distinguishable():
    """The two null-style conditions must not be confusable by their sent_order tag."""
    null = null_records(0, 10, 10, prime_tag="null_0")
    refnull = refnull_records(0, 10, 10)

    assert {r["context"]["sent_order"][0] for r in null} == {"null_0"}
    assert {r["context"]["sent_order"][0] for r in refnull} == {"refnull"}
