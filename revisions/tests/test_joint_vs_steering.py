"""
Tests that the joint-measurement pipeline agrees with the steering one where it must.

The central check the reviewers will care about: the joint and steering ΔC are computed from
different record schemas (BLANK1/BLANK2 generations vs a fixed prime + one generation), by
different extraction code. When the underlying 2x2 contingency tables are the same, the two
must return the SAME ΔC. If they don't, one of the two extractors is misreading its schema.

Also checks agreement with the published estimator (calculate_sentence_dc_fraction) in the
regime where the published estimator's known defect is inert -- see revisions/README.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from winogender_contextuality.modeling.contextuality import (
    calculate_sentence_dc_fraction,
    sentence_order_single_results,
)

from revisions.cbd import context_stats_from_table, delta_c_rank2, is_contextual
from revisions.fixtures import joint_records, steering_records
from revisions.joint_measurement import (
    joint_item_stats,
    joint_table,
    steering_item_stats,
    steering_table,
)

FORWARD = [0, 1]
REVERSE = [1, 0]


def _build(fwd_table: np.ndarray, rev_table: np.ndarray, idx: int = 0):
    """Joint and steering record sets encoding the SAME two contingency tables."""
    joint, steering = [], []
    for table, sent_order in ((fwd_table, FORWARD), (rev_table, REVERSE)):
        for k in (0, 1):  # both option-order conditions
            joint += joint_records(idx, table, sent_order=sent_order, i=k, j=k)
            steering += steering_records(idx, table, sent_order=sent_order, j=k)
    return joint, steering


def test_joint_table_round_trips_through_the_schema():
    """joint_table() must recover the exact table the records were generated from."""
    table = np.array([[7.0, 13.0], [21.0, 9.0]])
    records = joint_records(0, table, sent_order=FORWARD, i=0, j=0)
    assert joint_table(records) == pytest.approx(table)


def test_steering_table_round_trips_through_the_schema():
    """steering_table() must recover the table, reading the prime out of pnoun_order[0]."""
    table = np.array([[7.0, 13.0], [21.0, 9.0]])
    records = steering_records(0, table, sent_order=FORWARD, j=0)
    assert steering_table(records) == pytest.approx(table)


def test_steering_table_ignores_unprimed_records():
    from revisions.fixtures import unprimed_records

    table = np.array([[10.0, 10.0], [10.0, 10.0]])
    records = steering_records(0, table, sent_order=FORWARD, j=0)
    records += unprimed_records(0, n_female=25, n_male=25, sent_order=FORWARD)

    # The 50 unprimed records have pnoun_order[0] = None and must not enter the table.
    assert steering_table(records).sum() == 40


@pytest.mark.parametrize(
    "fwd, rev",
    [
        (np.array([[50.0, 0.0], [0.0, 50.0]]), np.array([[0.0, 50.0], [50.0, 0.0]])),  # ΔC = 2
        (np.array([[25.0, 25.0], [25.0, 25.0]]), np.array([[25.0, 25.0], [25.0, 25.0]])),  # ΔC = 0
        (np.array([[30.0, 12.0], [8.0, 40.0]]), np.array([[11.0, 33.0], [29.0, 17.0]])),  # generic
        (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[4.0, 3.0], [2.0, 1.0]])),  # tiny counts
    ],
)
def test_joint_and_steering_agree_when_tables_coincide(fwd, rev):
    """
    The invariant: same tables in, same ΔC out -- across two different record schemas and two
    different extraction paths.
    """
    joint, steering = _build(fwd, rev)

    for condition in ("mfirst", "ffirst"):
        j = joint_item_stats(0, joint, condition)["delta_c_joint"]
        s = steering_item_stats(0, steering, condition)["delta_c_steering"]

        expected = delta_c_rank2(context_stats_from_table(fwd), context_stats_from_table(rev))

        assert j == pytest.approx(expected)
        assert s == pytest.approx(expected)
        assert j == pytest.approx(s)


def test_joint_matches_hand_computed_delta_c():
    """Maximally contextual by hand (see test_cbd): ΔC = 2."""
    fwd = np.array([[50.0, 0.0], [0.0, 50.0]])
    rev = np.array([[0.0, 50.0], [50.0, 0.0]])
    joint, _ = _build(fwd, rev)

    assert joint_item_stats(0, joint, "mfirst")["delta_c_joint"] == pytest.approx(2.0)


def test_qq_statistic_from_records():
    """
    q = P_fwd(agree) - P_rev(agree).
    fwd agrees 100/100 of the time, rev 0/100 -> q = 1.
    """
    fwd = np.array([[50.0, 0.0], [0.0, 50.0]])
    rev = np.array([[0.0, 50.0], [50.0, 0.0]])
    joint, _ = _build(fwd, rev)

    assert joint_item_stats(0, joint, "mfirst")["qq"] == pytest.approx(1.0)


def test_agrees_with_published_estimator_where_they_coincide():
    """
    The published estimator builds its forward correlation from marginals (V1, V2) -- the prime
    marginals of BOTH contexts -- where the CbD definition calls for (V1, W2): the prime marginal
    and the *generation* marginal of the forward context. The two expressions coincide exactly
    when V2 == W2, i.e. when the forward generation marginal equals the reverse prime marginal.

    Primes are balanced by design (V2 = 0.5), so we construct a forward context whose generation
    marginal is also 0.5. There the published and corrected estimators must agree.
    """
    # forward: prime marginal 0.5, generation marginal 0.5
    fwd = np.array([[25.0, 25.0], [25.0, 25.0]])
    # reverse: prime marginal 0.5, generation marginal 0.5
    rev = np.array([[50.0, 0.0], [0.0, 50.0]])

    _, steering = _build(fwd, rev)

    ours = steering_item_stats(0, steering, "mfirst")["delta_c_steering"]

    dd = sentence_order_single_results(0, steering, mode="generation", pnoun_order=0)
    published = calculate_sentence_dc_fraction(dd, mode="generation")

    assert ours == pytest.approx(published)
    assert ours == pytest.approx(1.0)  # |0 - 1| - 0


def test_diverges_from_published_estimator_when_generation_marginal_is_skewed():
    """
    The companion to the test above: when the forward generation marginal is NOT 0.5, the
    published estimator's (V1, V2) substitution bites and the two disagree -- here they even
    disagree about whether the item is contextual at all.

    By hand, with the prime marginals balanced at 0.5 (as the design guarantees):

        forward: p_prime = 0.5, p_gen = 0.3, p_joint = 0.20
            correct   <R_A R_B>_fwd = 4(0.20) - 2(0.5) - 2(0.3) + 1 = +0.2
            published                = 4(0.20) - 2(0.5) - 2(0.5) + 1 = -0.2   <-- uses V2, not W2
        reverse: p_prime = 0.5, p_gen = 0.5, p_joint = 0.40
            both      <R_A R_B>_rev  = 4(0.40) - 2(0.5) - 2(0.5) + 1 = +0.6
        Δ0 = |0 - 0| + |0 - (2*0.3 - 1)| = 0.4   (both agree on this)

        correct   ΔC = |0.2 - 0.6| - 0.4 =  0.0   -> NOT contextual
        published ΔC = |-0.2 - 0.6| - 0.4 = +0.4  -> contextual

    This is a regression guard on the discrepancy documented in revisions/README.md: if someone
    later fixes contextuality.py, this test starts failing, which is the signal to retire the
    `delta_c_steering_published` column.
    """
    fwd = np.array([[40.0, 10.0], [30.0, 20.0]])  # p_gen = 0.3, p_joint = 0.2
    rev = np.array([[40.0, 10.0], [10.0, 40.0]])  # p_gen = 0.5, p_joint = 0.4

    _, steering = _build(fwd, rev)

    ours = steering_item_stats(0, steering, "mfirst")["delta_c_steering"]

    dd = sentence_order_single_results(0, steering, mode="generation", pnoun_order=0)
    published = calculate_sentence_dc_fraction(dd, mode="generation")

    assert ours == pytest.approx(0.0)
    assert published == pytest.approx(0.4)
    assert ours != pytest.approx(published)

    # The verdicts actually differ: the published estimator calls this item contextual.
    # Note ours is exactly 0 in exact arithmetic but ~1e-16 in floating point, which is why
    # verdicts go through is_contextual() rather than a bare `> 0`.
    assert not is_contextual(ours)
    assert is_contextual(published)
