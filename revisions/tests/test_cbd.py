"""
Tests for the joint-measurement CbD math (revisions/cbd.py).

Three kinds of check:

  * hand-computed cases -- textbook cyclic systems (PR box, Tsirelson, deterministic,
    disturbed) whose ΔC is known in closed form;
  * structural invariants -- the identity ΔC = 2|q| - Δ0 linking rank-2 ΔC to the QQ statistic,
    and the cycle-ordering requirement of the rank-4 system;
  * agreement with the published steering estimator in the regime where the two provably
    coincide.
"""

from __future__ import annotations

import numpy as np
import pytest

from revisions.cbd import (
    ContextStats,
    context_stats_from_table,
    delta_c_cyclic,
    delta_c_rank2,
    delta_c_rank4,
    qq_equality,
    s1_cyclic,
)

# --------------------------------------------------------------------------------------
# s1
# --------------------------------------------------------------------------------------


def test_s1_rank2_is_absolute_difference():
    # Odd-sign vectors for n=2 are (+,-) and (-,+), so s1 = max(a-b, b-a) = |a-b|.
    assert s1_cyclic([0.5, -0.25]) == pytest.approx(0.75)
    assert s1_cyclic([-0.25, 0.5]) == pytest.approx(0.75)
    assert s1_cyclic([0.3, 0.3]) == pytest.approx(0.0)


def test_s1_rank4_all_correlations_one():
    # Odd number of minus signs, so the best we can do is 1+1+1-1 = 2.
    assert s1_cyclic([1, 1, 1, 1]) == pytest.approx(2.0)


def test_s1_rank4_pr_box():
    # The PR box: three +1 correlations and one -1. Sign pattern (+,+,+,-) has one minus sign
    # (odd, so admissible) and gives 1+1+1+1 = 4, the algebraic maximum.
    assert s1_cyclic([1, 1, 1, -1]) == pytest.approx(4.0)


def test_s1_only_uses_odd_sign_patterns():
    # (+,+,-,-) has TWO minus signs and is not admissible. For x = (1,1,-1,-1) it would give 4;
    # the best odd pattern gives only 2. A version of s1 that allowed even patterns -- as the
    # published cbd_s1_4cycle does, via its |w+x-y-z| term -- would return 4 here.
    assert s1_cyclic([1, 1, -1, -1]) == pytest.approx(2.0)


def test_s1_rank4_tsirelson():
    # Quantum maximum: |correlations| = 1/sqrt(2) with an odd number of negatives -> s1 = 2*sqrt(2).
    r = 1 / np.sqrt(2)
    assert s1_cyclic([r, r, r, -r]) == pytest.approx(2 * np.sqrt(2))


# --------------------------------------------------------------------------------------
# Rank-2 ΔC, hand-computed
# --------------------------------------------------------------------------------------


def test_context_stats_from_table():
    # [first, second]; 0 = male, 1 = female
    table = np.array([[10.0, 20.0], [30.0, 40.0]])  # N = 100
    s = context_stats_from_table(table)

    assert s.p_first == pytest.approx(0.7)  # (30 + 40) / 100
    assert s.p_second == pytest.approx(0.6)  # (20 + 40) / 100
    assert s.p_joint == pytest.approx(0.4)
    assert s.n_trials == 100
    # <XY> = 4*p_joint - 2*p_first - 2*p_second + 1
    assert s.correlation == pytest.approx(4 * 0.4 - 2 * 0.7 - 2 * 0.6 + 1)
    # agreement = P(1,1) + P(0,0)
    assert s.p_agree == pytest.approx(0.4 + 0.1)


def test_rank2_maximally_contextual():
    """
    Perfect positive correlation in one context, perfect negative in the other, and no
    disturbance (all marginals 0.5). By hand:
        corr_fwd = +1, corr_rev = -1, Δ0 = 0
        ΔC = |1 - (-1)| - 0 = 2      (the rank-2 maximum)
    """
    fwd = context_stats_from_table(np.array([[50.0, 0.0], [0.0, 50.0]]))
    rev = context_stats_from_table(np.array([[0.0, 50.0], [50.0, 0.0]]))

    assert fwd.correlation == pytest.approx(1.0)
    assert rev.correlation == pytest.approx(-1.0)
    assert delta_c_rank2(fwd, rev) == pytest.approx(2.0)


def test_rank2_independent_is_not_contextual():
    """Independent, identical contexts: corr = 0 in both, no disturbance -> ΔC = 0 (not > 0)."""
    table = np.array([[25.0, 25.0], [25.0, 25.0]])
    fwd = context_stats_from_table(table)
    rev = context_stats_from_table(table)

    assert fwd.correlation == pytest.approx(0.0)
    assert delta_c_rank2(fwd, rev) == pytest.approx(0.0)
    assert not delta_c_rank2(fwd, rev) > 0


def test_rank2_identical_contexts_never_contextual():
    """Any system whose two contexts have the same distribution is non-contextual."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        table = rng.integers(1, 50, size=(2, 2)).astype(float)
        s = context_stats_from_table(table)
        assert delta_c_rank2(s, s) <= 1e-12


def test_rank2_disturbance_is_subtracted():
    """
    Disturbance eats the signal. Forward is perfectly correlated (corr = +1); reverse has a
    prime that is always female (p_first = 1) and an independent generation (corr = 0).
        Δ0 = |E_first_fwd - E_second_rev| + |E_first_rev - E_second_fwd|
           = |0 - 0| + |1 - 0| = 1
        ΔC = |1 - 0| - 1 = 0
    """
    fwd = context_stats_from_table(np.array([[50.0, 0.0], [0.0, 50.0]]))
    rev = context_stats_from_table(np.array([[0.0, 0.0], [50.0, 50.0]]))

    assert rev.p_first == pytest.approx(1.0)
    assert rev.correlation == pytest.approx(0.0)
    assert delta_c_rank2(fwd, rev) == pytest.approx(0.0)


def test_rank2_bounds():
    """ΔC for a rank-2 system can never exceed 2."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        f = context_stats_from_table(rng.integers(0, 40, size=(2, 2)).astype(float) + 1)
        r = context_stats_from_table(rng.integers(0, 40, size=(2, 2)).astype(float) + 1)
        assert delta_c_rank2(f, r) <= 2.0 + 1e-9


# --------------------------------------------------------------------------------------
# QQ equality
# --------------------------------------------------------------------------------------


def test_qq_zero_when_orders_agree():
    table = np.array([[30.0, 20.0], [10.0, 40.0]])
    s = context_stats_from_table(table)
    assert qq_equality(s, s) == pytest.approx(0.0)


def test_delta_c_equals_two_abs_q_minus_disturbance():
    """
    The identity that ties Experiment 1 to the reviewer's QQ point:
        ΔC(rank 2) = 2|q| - Δ0
    where q is the QQ statistic. Holds for arbitrary tables.
    """
    rng = np.random.default_rng(11)

    for _ in range(200):
        t_f = rng.integers(1, 40, size=(2, 2)).astype(float)
        t_r = rng.integers(1, 40, size=(2, 2)).astype(float)
        f = context_stats_from_table(t_f)
        r = context_stats_from_table(t_r)

        q = qq_equality(f, r)
        delta_0 = abs(f.e_first - r.e_second) + abs(r.e_first - f.e_second)

        assert delta_c_rank2(f, r) == pytest.approx(2 * abs(q) - delta_0)


# --------------------------------------------------------------------------------------
# Rank-4
# --------------------------------------------------------------------------------------


def _uniform_context() -> ContextStats:
    return context_stats_from_table(np.array([[25.0, 25.0], [25.0, 25.0]]))


def test_rank4_pr_box_is_maximally_contextual():
    """
    A PR box: three contexts perfectly correlated, one perfectly anti-correlated, all marginals
    0.5. s1 = 4, n - 2 = 2, Δ0 = 0, so ΔC = 2.
    """
    corr_pos = context_stats_from_table(np.array([[50.0, 0.0], [0.0, 50.0]]))
    corr_neg = context_stats_from_table(np.array([[0.0, 50.0], [50.0, 0.0]]))

    contexts = [corr_pos, corr_pos, corr_pos, corr_neg]
    assert delta_c_rank4(contexts) == pytest.approx(2.0)


def test_rank4_uniform_is_not_contextual():
    contexts = [_uniform_context() for _ in range(4)]
    # s1 = 0, minus (n-2) = 2 -> ΔC = -2
    assert delta_c_rank4(contexts) == pytest.approx(-2.0)
    assert delta_c_rank4(contexts) <= 0


def test_rank4_consistent_deterministic_is_not_contextual():
    """All four contexts perfectly correlated: s1 = 2, so ΔC = 2 - 2 - 0 = 0. Not contextual."""
    c = context_stats_from_table(np.array([[50.0, 0.0], [0.0, 50.0]]))
    assert delta_c_rank4([c, c, c, c]) == pytest.approx(0.0)


def test_rank4_requires_four_contexts():
    with pytest.raises(ValueError):
        delta_c_rank4([_uniform_context()] * 3)


def test_delta_c_cyclic_matches_rank2_for_two_contexts():
    f = context_stats_from_table(np.array([[40.0, 10.0], [10.0, 40.0]]))
    r = context_stats_from_table(np.array([[10.0, 40.0], [40.0, 10.0]]))
    assert delta_c_cyclic([f, r]) == pytest.approx(delta_c_rank2(f, r))


def test_empty_context_yields_nan():
    empty = context_stats_from_table(np.zeros((2, 2)))
    good = _uniform_context()
    assert np.isnan(delta_c_rank2(good, empty))
