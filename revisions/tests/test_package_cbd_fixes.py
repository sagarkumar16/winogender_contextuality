"""
Regression tests for the three CbD fixes in winogender_contextuality/modeling/contextuality.py.

Each is cross-checked against revisions/cbd.py, which is an independent implementation of the
same definitions. The two were written from the definitions rather than from each other, so
agreement is real evidence, not a tautology.

Covers:
  * cbd_s1_4cycle              -- s1 must maximise over ODD-sign patterns only
  * calculate_pronouns_nc_fraction -- the rank-4 system's contexts must be walked in CYCLE order
  * calculate_sentence_nc_fraction -- the forward correlation must use (V1, W2), not (V1, V2)
"""

from __future__ import annotations

import numpy as np
import pytest

from winogender_contextuality.modeling.contextuality import (
    calculate_pronouns_nc_fraction,
    calculate_sentence_nc_fraction,
    cbd_s1_4cycle,
)

from revisions.cbd import context_stats_from_table, delta_c_cyclic, s1_cyclic

# --------------------------------------------------------------------------------------
# cbd_s1_4cycle
# --------------------------------------------------------------------------------------


def test_s1_4cycle_rejects_even_sign_patterns():
    """
    For x = (1, 1, -1, -1) the even pattern (+,+,-,-) would give 4, but it has TWO minus signs
    and is not admissible. The best ODD pattern gives 2.

    The old fourth term, |w + x - y - z|, is exactly that inadmissible pattern, and returned 4.
    """
    assert cbd_s1_4cycle(1, 1, -1, -1) == pytest.approx(2.0)


def test_s1_4cycle_pr_box_still_maximal():
    assert cbd_s1_4cycle(1, 1, 1, -1) == pytest.approx(4.0)


def test_s1_4cycle_tsirelson():
    r = 1 / np.sqrt(2)
    assert cbd_s1_4cycle(r, r, r, -r) == pytest.approx(2 * np.sqrt(2))


@pytest.mark.parametrize("seed", range(25))
def test_s1_4cycle_matches_independent_implementation(seed):
    """Against revisions.cbd.s1_cyclic, which enumerates the odd-sign patterns explicitly."""
    rng = np.random.default_rng(seed)
    corrs = rng.uniform(-1, 1, size=4)
    assert cbd_s1_4cycle(*corrs) == pytest.approx(s1_cyclic(corrs))


# --------------------------------------------------------------------------------------
# calculate_pronouns_nc_fraction -- rank-4 cycle ordering
# --------------------------------------------------------------------------------------


def _scenario_rows(tables: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """
    Build the array calculate_pronouns_nc_fraction consumes: one row per option-order pair, in
    the PRODUCT order MeasurementScenario uses -- (0,0), (0,1), (1,0), (1,1) -- each row the
    flattened 2x2 joint over (sentence-1 pronoun, sentence-2 pronoun), normalised.
    """
    rows = []
    for key in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        t = np.asarray(tables[key], dtype=float)
        rows.append((t / t.sum()).reshape(-1))
    return np.array(rows)


def _expected_delta_c(tables: dict[tuple[int, int], np.ndarray]) -> float:
    """
    The same system scored by revisions.cbd, with the contexts placed in cycle order and the
    marginals swapped where the lower-indexed content variable is the sentence-2 pronoun.

        c0 = (0,0) first=q0(s1)  c1 = (1,0) first=q1(s2) -> transpose
        c2 = (1,1) first=q2(s1)  c3 = (0,1) first=q3(s2) -> transpose
    """
    specs = [((0, 0), False), ((1, 0), True), ((1, 1), False), ((0, 1), True)]
    contexts = []
    for key, transpose in specs:
        t = np.asarray(tables[key], dtype=float)
        contexts.append(context_stats_from_table(t.T if transpose else t))
    return delta_c_cyclic(contexts)


def test_pronouns_nc_fraction_pr_box_is_maximally_contextual():
    """
    A PR box on the option-order system: three contexts perfectly correlated, one perfectly
    anti-correlated, all marginals 0.5. s1 = 4, n - 2 = 2, disturbance = 0 -> delta_c = 2.

    The anti-correlated context must be placed at a CYCLE position; here it is options (1,1).
    """
    pos = np.array([[50.0, 0.0], [0.0, 50.0]])
    neg = np.array([[0.0, 50.0], [50.0, 0.0]])
    tables = {(0, 0): pos, (1, 0): pos, (1, 1): neg, (0, 1): pos}

    assert calculate_pronouns_nc_fraction(_scenario_rows(tables)) == pytest.approx(2.0)


def test_pronouns_nc_fraction_uniform_is_not_contextual():
    uniform = np.array([[25.0, 25.0], [25.0, 25.0]])
    tables = {k: uniform for k in [(0, 0), (0, 1), (1, 0), (1, 1)]}

    # s1 = 0 -> delta_c = 0 - 2 - 0 = -2
    assert calculate_pronouns_nc_fraction(_scenario_rows(tables)) == pytest.approx(-2.0)


def test_pronouns_nc_fraction_consistent_deterministic_is_not_contextual():
    """All four contexts perfectly correlated: s1 = 2, so delta_c = 2 - 2 - 0 = 0."""
    pos = np.array([[50.0, 0.0], [0.0, 50.0]])
    tables = {k: pos for k in [(0, 0), (0, 1), (1, 0), (1, 1)]}

    assert calculate_pronouns_nc_fraction(_scenario_rows(tables)) == pytest.approx(0.0)


@pytest.mark.parametrize("seed", range(25))
def test_pronouns_nc_fraction_matches_independent_implementation(seed):
    """
    Arbitrary data, including disturbance (unequal marginals across contexts) -- which is what
    the cycle ordering actually governs. Walking the rows in product order instead mispairs the
    disturbance terms and this diverges.
    """
    rng = np.random.default_rng(seed)
    tables = {
        key: rng.integers(1, 40, size=(2, 2)).astype(float)
        for key in [(0, 0), (0, 1), (1, 0), (1, 1)]
    }

    got = calculate_pronouns_nc_fraction(_scenario_rows(tables))
    assert got == pytest.approx(_expected_delta_c(tables))


def test_pronouns_nc_fraction_never_exceeds_two():
    rng = np.random.default_rng(0)
    for _ in range(100):
        tables = {
            key: rng.integers(1, 40, size=(2, 2)).astype(float)
            for key in [(0, 0), (0, 1), (1, 0), (1, 1)]
        }
        assert calculate_pronouns_nc_fraction(_scenario_rows(tables)) <= 2.0 + 1e-9


# --------------------------------------------------------------------------------------
# calculate_sentence_nc_fraction -- the (V1, V2) defect
# --------------------------------------------------------------------------------------


def _joint_data_dict(fwd: np.ndarray, rev: np.ndarray, pronouns=("he", "she")) -> dict:
    """
    A sentence_order_results()-shaped dict from two 2x2 tables indexed
    [sentence1_pronoun_is_female, sentence2_pronoun_is_female].
    """
    male, female = pronouns

    def trials(table):
        p1, p2 = [], []
        for a in (0, 1):
            for b in (0, 1):
                for _ in range(int(table[a, b])):
                    p1.append(female if a else male)
                    p2.append(female if b else male)
        return p1, p2

    f1, f2 = trials(fwd)
    r1, r2 = trials(rev)

    return {
        "forward": {"pnoun_1": f1, "pnoun_2": f2, "pronouns": [female, female]},
        "reverse": {"pnoun_1": r1, "pnoun_2": r2, "pronouns": [female, female]},
    }


def test_sentence_nc_fraction_maximally_contextual():
    """Perfect positive correlation one way, perfect negative the other, no disturbance: 2."""
    fwd = np.array([[50.0, 0.0], [0.0, 50.0]])
    rev = np.array([[0.0, 50.0], [50.0, 0.0]])

    assert calculate_sentence_nc_fraction(_joint_data_dict(fwd, rev)) == pytest.approx(2.0)


def test_sentence_nc_fraction_skewed_marginal():
    """
    The case the (V1, V2) defect got wrong. Sentence-1 marginals are 0.5 in both contexts; the
    forward sentence-2 marginal is 0.3.

        forward: <VW> = 4(0.20) - 2(0.5) - 2(0.3) + 1 = +0.2   (old code used V2 = 0.5 -> -0.2)
        reverse: <VW> = 4(0.40) - 2(0.5) - 2(0.5) + 1 = +0.6
        disturbance = |0 - 0| + |0 - (2(0.3) - 1)| = 0.4
        delta_c = |0.2 - 0.6| - 0.4 = 0.0   -> NOT contextual   (old code gave +0.4)
    """
    fwd = np.array([[40.0, 10.0], [30.0, 20.0]])
    rev = np.array([[40.0, 10.0], [10.0, 40.0]])

    assert calculate_sentence_nc_fraction(_joint_data_dict(fwd, rev)) == pytest.approx(0.0)


@pytest.mark.parametrize("seed", range(25))
def test_sentence_nc_fraction_matches_independent_implementation(seed):
    rng = np.random.default_rng(seed)
    fwd = rng.integers(1, 40, size=(2, 2)).astype(float)
    rev = rng.integers(1, 40, size=(2, 2)).astype(float)

    got = calculate_sentence_nc_fraction(_joint_data_dict(fwd, rev))
    expected = delta_c_cyclic(
        [context_stats_from_table(fwd), context_stats_from_table(rev.T)]
    )
    assert got == pytest.approx(expected)
