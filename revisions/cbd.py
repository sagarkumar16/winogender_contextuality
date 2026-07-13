"""
Contextuality-by-Default for cyclic systems, in the **joint-measurement** formulation.

The published analysis uses a *steering* formulation: the priming pronoun is fixed by the
experimenter, so one of the two random variables in each context is deterministic and only
the free pronoun is measured. Reviewer 1 asked for the formulation used by the QQ equality
and by the social-science contextuality literature, in which **both** variables are measured
jointly in each context. That is what this module computes.

Both formulations are instances of the same object -- a cyclic system of rank n -- so a
single implementation serves both, and the two ΔC values are directly comparable. The
existing steering code (winogender_contextuality.modeling.contextuality) is left untouched
and is imported by revisions.joint_measurement for the side-by-side.

Definitions (Dzhafarov, Kujala & Cervantes; Kujala & Dzhafarov 2016)
-------------------------------------------------------------------
A cyclic system of rank n has content variables q_0 .. q_{n-1} and contexts c_0 .. c_{n-1},
where context c_i jointly measures q_i and q_{i+1 mod n}. With ±1-valued outcomes,

    ΔC = s1({<R_i R_{i+1}>_{c_i}}) - (n - 2) - Δ0

    s1(x) = max over sign vectors λ ∈ {-1,+1}^n with an ODD number of -1s of Σ λ_i x_i
    Δ0    = Σ_q | <R_q>_{c}  -  <R_q>_{c'} |   (q's two contexts; the inconsistent-
                                                connectedness / "disturbance" correction)

The system is contextual iff ΔC > 0.

For n = 2 this collapses to

    ΔC = |<R_0 R_1>_{c_0} - <R_0 R_1>_{c_1}| - |<R_0>_{c_0} - <R_0>_{c_1}|
                                             - |<R_1>_{c_1} - <R_1>_{c_0}|

which is the shape the published steering estimator already has.

Relation to the QQ equality
---------------------------
For two binary questions asked in both orders, the QQ statistic is
q = P_AB(agree) - P_BA(agree). Since <XY> = 2·P(agree) - 1 for ±1 variables,
the rank-2 s1 term is exactly |2q|, so

    ΔC(rank 2) = 2·|q| - Δ0

i.e. the joint ΔC *is* the QQ order effect, corrected for inconsistent connectedness.
qq_equality() below reports q directly.

- EMNLP revisions, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

# Import rather than redefine: these are the published estimator's primitives.
from winogender_contextuality.modeling.contextuality import cbd_expectation, cbd_correlation

__all__ = [
    "ContextStats",
    "CONTEXTUAL_TOL",
    "is_contextual",
    "s1_cyclic",
    "delta_c_cyclic",
    "delta_c_rank2",
    "delta_c_rank4",
    "qq_equality",
    "context_stats_from_table",
    "bootstrap_delta_c",
]

# ΔC > 0 is the contextuality criterion, but it is a knife edge: an exactly non-contextual
# system evaluates to 0 in exact arithmetic and to ~1e-16 in floating point, so a bare `> 0`
# test reports contextuality from rounding noise alone. Every verdict goes through
# is_contextual() instead.
CONTEXTUAL_TOL = 1e-9


def is_contextual(delta_c: float, tol: float = CONTEXTUAL_TOL) -> bool:
    """Verdict for a ΔC value, guarded against floating-point noise at the ΔC = 0 boundary."""
    import math

    if delta_c is None or (isinstance(delta_c, float) and math.isnan(delta_c)):
        return False
    return bool(delta_c > tol)


@dataclass(frozen=True)
class ContextStats:
    """
    One context of a cyclic system: the two marginals and the joint, all as probabilities
    of the *positive* outcome (here: the female pronoun).

    p_first  -- P(R_{q_i} = 1 | c_i)      (q_i is the context's lower-indexed content variable)
    p_second -- P(R_{q_{i+1}} = 1 | c_i)
    p_joint  -- P(R_{q_i} = 1 and R_{q_{i+1}} = 1 | c_i)
    n_trials -- trials behind the estimate (bookkeeping only)
    """

    p_first: float
    p_second: float
    p_joint: float
    n_trials: int = 0

    @property
    def correlation(self) -> float:
        return float(cbd_correlation(self.p_first, self.p_second, self.p_joint))

    @property
    def e_first(self) -> float:
        return float(cbd_expectation(self.p_first))

    @property
    def e_second(self) -> float:
        return float(cbd_expectation(self.p_second))

    @property
    def p_agree(self) -> float:
        """P(both female) + P(both male) -- the QQ 'agreement' probability."""
        return float(self.p_joint + (1.0 - self.p_first - self.p_second + self.p_joint))


def context_stats_from_table(table: np.ndarray, positive: int = 1) -> ContextStats:
    """
    Build a ContextStats from a 2x2 contingency table of raw counts (or probabilities),
    indexed [outcome_of_first, outcome_of_second] with 0 = male, 1 = female.
    """
    table = np.asarray(table, dtype=float)
    if table.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 table, got shape {table.shape}")

    total = table.sum()
    if total <= 0:
        return ContextStats(np.nan, np.nan, np.nan, 0)

    p = table / total
    return ContextStats(
        p_first=float(p[positive, :].sum()),
        p_second=float(p[:, positive].sum()),
        p_joint=float(p[positive, positive]),
        n_trials=int(round(total)),
    )


def s1_cyclic(correlations) -> float:
    """
    s1 = max over sign vectors with an ODD number of -1s of Σ λ_i x_i.

    Because the odd-sign vectors come in ± pairs, this equals the max of |Σ λ_i x_i| over
    the vectors with exactly one -1 when n is even. We enumerate honestly instead of relying
    on that, so the function is correct for any rank.
    """
    x = np.asarray(correlations, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("A cyclic system needs at least 2 contexts.")

    best = -np.inf
    for signs in product([1.0, -1.0], repeat=n):
        if sum(1 for s in signs if s < 0) % 2 == 1:  # odd number of minus signs
            best = max(best, float(np.dot(signs, x)))
    return best


def delta_c_cyclic(contexts: list[ContextStats]) -> float:
    """
    ΔC for a cyclic system of rank n = len(contexts).

    `contexts` must be given in cycle order: context i measures content variables
    (q_i, q_{i+1 mod n}), so q_i is the `p_first` of context i and the `p_second` of
    context i-1. ΔC > 0 means contextual.
    """
    n = len(contexts)
    stats = list(contexts)

    correlations = [c.correlation for c in stats]
    if any(np.isnan(v) for v in correlations):
        return float("nan")

    s1 = s1_cyclic(correlations)

    # Each content variable q appears as `first` in context q and `second` in context q-1.
    delta_0 = sum(
        abs(stats[q].e_first - stats[(q - 1) % n].e_second) for q in range(n)
    )

    return float(s1 - (n - 2) - delta_0)


def delta_c_rank2(forward: ContextStats, reverse: ContextStats) -> float:
    """
    ΔC for the two-context system used in the paper: content variables are the pronouns of
    sentence A and sentence B; contexts are the two sentence orders.

    In `forward` the first content variable is A's pronoun; in `reverse` the roles swap, so
    `forward.p_first` and `reverse.p_second` both refer to sentence A.
    """
    return delta_c_cyclic([forward, reverse])


def delta_c_rank4(contexts: list[ContextStats]) -> float:
    """
    ΔC for the rank-4 (CHSH-shaped) system: content variables are
        q0 = sentence-A pronoun measured with options in male-first order
        q1 = sentence-B pronoun, male-first
        q2 = sentence-A pronoun, female-first
        q3 = sentence-B pronoun, female-first
    Contexts, in cycle order, are (q0,q1), (q1,q2), (q2,q3), (q3,q0) -- i.e.
        (A_mf, B_mf), (B_mf, A_fm), (A_fm, B_fm), (B_fm, A_mf).

    Note the cycle order: consecutive contexts must SHARE a content variable. Enumerating
    the four option-order combinations in the natural (0,0), (0,1), (1,0), (1,1) order does
    not do that, and pairs contexts that share nothing.
    """
    if len(contexts) != 4:
        raise ValueError(f"Rank-4 system needs exactly 4 contexts, got {len(contexts)}")
    return delta_c_cyclic(contexts)


def qq_equality(forward: ContextStats, reverse: ContextStats) -> float:
    """
    The QQ (question-order) statistic q = P_AB(agree) - P_BA(agree).

    Quantum-question-order models predict q = 0 exactly. Equivalent to half the rank-2 s1
    term: ΔC(rank 2) = 2|q| - Δ0.
    """
    return float(forward.p_agree - reverse.p_agree)


def bootstrap_delta_c(
    tables: list[np.ndarray],
    n_boot: int = 1000,
    seed: int = 0,
    estimator=delta_c_cyclic,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for ΔC, resampling trials within each context independently.

    `tables` are the per-context 2x2 count tables in cycle order.
    Returns (lo, hi) of the 95% interval; (nan, nan) if any context has no trials.
    """
    rng = np.random.default_rng(seed)
    tables = [np.asarray(t, dtype=float) for t in tables]

    if any(t.sum() <= 0 for t in tables):
        return (float("nan"), float("nan"))

    draws = []
    for _ in range(n_boot):
        resampled = []
        for t in tables:
            n = int(round(t.sum()))
            flat = (t / t.sum()).ravel()
            counts = rng.multinomial(n, flat).reshape(2, 2)
            resampled.append(context_stats_from_table(counts))
        draws.append(estimator(resampled))

    draws = np.asarray(draws, dtype=float)
    draws = draws[~np.isnan(draws)]
    if draws.size == 0:
        return (float("nan"), float("nan"))

    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
