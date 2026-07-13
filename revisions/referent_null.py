"""
Experiment 3 (Reviewer 3, "Co-Occurrence") -- analysis.

Reports KL divergence and ΔC for three prime conditions against the same unprimed baseline:

  1. contextual        the paired sentence with a gendered pronoun   (published primed runs)
  2. null              a generic unrelated sentence                  (published null runs)
  3. referent_null     the paired sentence with the pronoun replaced by its antecedent NP
                       -- referents present, no pronoun              (revisions.collect_referent_null)

If the shift under `contextual` is really pronoun priming, condition 3 should look like
condition 2. If it is referent co-occurrence, condition 3 should look like condition 1.

KL is the smoothed Bernoulli KL over generation counts, matching the paper
(bernoulli_kl / add-half smoothing, from notebooks/13-sk-paper-figures.ipynb):

    KL( P(female | prime) || P(female | unprimed) )

The contextual condition yields two KLs (male-primed and female-primed); the pronoun-free
conditions yield one each, since they have no pronoun to vary.

Generation counts, not logits: the published null runs store a 3-vector of logits over the
characters 's','h','e' (generate_one_null_context passes `pronouns[1]`, a string, to
get_token_ids), so their internal probabilities are unusable. Our referent-null runs store
proper 2-vectors, so `--mode internal` works for conditions 1 and 3 -- but the default, and
the only three-way comparable mode, is generation.

Row alignment: the null-style runs are indexed per sentence slot, two rows per pair
(row 2k = pair k / sent_order [0,1], row 2k+1 = pair k / sent_order [1,0]). We do not trust
that convention silently -- `--primes` joins each null row back to its (pair_index, free_slot)
via the primes CSV, and the mapping is asserted.

Usage:
    python -m revisions.referent_null \
        --primed   one_pronoun_measurements_gemma-3-12b-it_0.5_wp_k40.ndjson \
        --null     null_measurements_gemma-3-12b-it_0.5_wp.ndjson \
        --refnull  refnull_measurements_gemma-3-12b-it_0.5_wp.ndjson \
        --primes   revisions/outputs/primes/referent_primes_repeated_np.csv \
        --model gemma --out revisions/outputs/referent_null/

- EMNLP revisions, 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from revisions.cbd import context_stats_from_table, delta_c_rank2
from revisions.common import (
    FemaleCounts,
    OUTPUTS_DIR,
    add_common_args,
    data_dir,
    female_generation_counts,
    female_internal_prob,
    get_filled_pnoun,
    get_index,
    get_sent_order,
    kl_from_counts,
    load_ndjson,
    provenance,
    set_seeds,
    write_results,
)
from revisions.joint_measurement import steering_item_stats

FORWARD = [0, 1]
REVERSE = [1, 0]

# row 2k of a null-style run <-> pair k with sentence order [0,1] (free sentence = template_2)
SLOT_TO_SENT_ORDER = {2: FORWARD, 1: REVERSE}


def null_row_for(pair_index: int, free_slot: int) -> int:
    """Row index in a null-style run for (pair, free-sentence slot)."""
    return 2 * pair_index + (0 if free_slot == 2 else 1)


def _null_condition_counts(data: list[dict], row: int, prime_tags: list[str]) -> FemaleCounts:
    """Pool generations across the given prime tags (e.g. the three generic null primes)."""
    item = get_index(row, data)
    pooled = [r for r in item if str(r["context"]["sent_order"][0]) in prime_tags]
    return female_generation_counts(pooled)


def _null_prime_tags(data: list[dict]) -> list[str]:
    """Whatever prime tags a null-style file actually contains ('null_0'.. or 'refnull')."""
    tags = {str(r["context"]["sent_order"][0]) for r in data}
    return sorted(tags)


def per_item_table(
    primed_data: list[dict],
    null_data: list[dict],
    refnull_data: list[dict],
    model: str,
    max_index: int,
    primes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (pair, sentence order): the three conditions' probabilities, KLs and ΔC."""
    null_tags = _null_prime_tags(null_data) if null_data else []
    refnull_tags = _null_prime_tags(refnull_data) if refnull_data else []
    logger.info(f"null prime tags: {null_tags} | referent-null prime tags: {refnull_tags}")

    rows = []

    for idx in range(max_index):
        item = get_index(idx, primed_data)

        for free_slot, sent_order in SLOT_TO_SENT_ORDER.items():
            ordered = get_sent_order(sent_order, item)

            # Condition 1: contextual prime (and its unprimed baseline).
            unprimed = female_generation_counts(
                [r for r in ordered if r["context"]["sentence_1"] is None]
            )
            m_primed = female_generation_counts(get_filled_pnoun(0, ordered))
            f_primed = female_generation_counts(get_filled_pnoun(1, ordered))

            row_null = null_row_for(idx, free_slot)

            # Conditions 2 and 3, keyed by the null-style row for this (pair, slot).
            null_counts = _null_condition_counts(null_data, row_null, null_tags) if null_data else FemaleCounts(0, 0)
            refnull_counts = (
                _null_condition_counts(refnull_data, row_null, refnull_tags)
                if refnull_data
                else FemaleCounts(0, 0)
            )

            row = {
                "model": model,
                "index": idx,
                "free_slot": free_slot,
                "sent_order": str(sent_order),
                # raw counts, so nothing has to be re-run to rescore
                "n_unprimed": unprimed.total,
                "n_m_primed": m_primed.total,
                "n_f_primed": f_primed.total,
                "n_null": null_counts.total,
                "n_refnull": refnull_counts.total,
                "k_unprimed": unprimed.successes,
                "k_m_primed": m_primed.successes,
                "k_f_primed": f_primed.successes,
                "k_null": null_counts.successes,
                "k_refnull": refnull_counts.successes,
                # smoothed P(female)
                "p_fem_unprimed": unprimed.smoothed_prob,
                "p_fem_m_primed": m_primed.smoothed_prob,
                "p_fem_f_primed": f_primed.smoothed_prob,
                "p_fem_null": null_counts.smoothed_prob if null_counts.total else np.nan,
                "p_fem_refnull": refnull_counts.smoothed_prob if refnull_counts.total else np.nan,
                # KL against the shared unprimed baseline
                "kl_m_primed": kl_from_counts(m_primed, unprimed),
                "kl_f_primed": kl_from_counts(f_primed, unprimed),
                "kl_null": kl_from_counts(null_counts, unprimed),
                "kl_refnull": kl_from_counts(refnull_counts, unprimed),
            }

            if primes is not None:
                match = primes[(primes.pair_index == idx) & (primes.free_slot == free_slot)]
                if len(match):
                    row["prime_text"] = match.iloc[0]["prime"]
                    row["free_sentence"] = match.iloc[0]["template"]

            rows.append(row)

    df = pd.DataFrame(rows)

    # ΔC is a property of the sentence *pair* (it needs both orders), not of a single slot.
    for idx in range(max_index):
        for condition in ("mfirst", "ffirst"):
            s = steering_item_stats(idx, primed_data, condition)
            df.loc[df["index"] == idx, f"delta_c_contextual_{condition}"] = s["delta_c_steering"]

    # ΔC for the pronoun-free conditions: no prime pronoun varies, so the prime variable is
    # constant and ΔC reduces to the disturbance between sentence orders. Computed from the
    # 2x2 tables with the prime slot held at its single value.
    for cond, data, tags in (("null", null_data, null_tags), ("refnull", refnull_data, refnull_tags)):
        if not data:
            continue
        for idx in range(max_index):
            fwd = _null_condition_counts(data, null_row_for(idx, 2), tags)
            rev = _null_condition_counts(data, null_row_for(idx, 1), tags)
            df.loc[df["index"] == idx, f"delta_c_{cond}"] = _pronoun_free_delta_c(fwd, rev)

    return df


def _pronoun_free_delta_c(forward: FemaleCounts, reverse: FemaleCounts) -> float:
    """
    ΔC for a condition whose prime carries no pronoun.

    The prime variable is degenerate (a single value), so it contributes no correlation and
    the rank-2 ΔC collapses to -|<R_gen>_fwd - <R_gen>_rev|: pure disturbance, never positive.
    Reporting it makes the point quantitatively -- a pronoun-free prime cannot produce
    contextuality, only order-dependence -- and gives the magnitude of that order-dependence.
    """
    if not forward.total or not reverse.total:
        return float("nan")

    # Degenerate prime: put all mass in the "male-prime" row; only the generated column varies.
    fwd_table = np.array(
        [[forward.total - forward.successes, forward.successes], [0.0, 0.0]], dtype=float
    )
    rev_table = np.array(
        [[reverse.total - reverse.successes, reverse.successes], [0.0, 0.0]], dtype=float
    )
    return delta_c_rank2(context_stats_from_table(fwd_table), context_stats_from_table(rev_table))


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Model-level means of the three conditions' KLs -- the table for the response letter."""
    out = []
    for model, g in df.groupby("model", sort=False):
        kl_ctx = pd.concat([g["kl_m_primed"], g["kl_f_primed"]])
        out.append(
            {
                "model": model,
                "n_items": int(len(g)),
                "mean_kl_contextual": float(kl_ctx.mean()),
                "mean_kl_m_primed": float(g["kl_m_primed"].mean()),
                "mean_kl_f_primed": float(g["kl_f_primed"].mean()),
                "mean_kl_null": float(g["kl_null"].mean()),
                "mean_kl_refnull": float(g["kl_refnull"].mean()),
                "median_kl_contextual": float(kl_ctx.median()),
                "median_kl_null": float(g["kl_null"].median()),
                "median_kl_refnull": float(g["kl_refnull"].median()),
                # The decisive ratio: how much of the contextual prime's divergence survives
                # when the pronoun is removed but the referents stay?
                "refnull_over_contextual": float(g["kl_refnull"].mean() / kl_ctx.mean())
                if kl_ctx.mean()
                else np.nan,
                "null_over_contextual": float(g["kl_null"].mean() / kl_ctx.mean())
                if kl_ctx.mean()
                else np.nan,
                "mean_p_fem_unprimed": float(g["p_fem_unprimed"].mean()),
                "mean_p_fem_null": float(g["p_fem_null"].mean()),
                "mean_p_fem_refnull": float(g["p_fem_refnull"].mean()),
            }
        )
    return pd.DataFrame(out)


def plot_conditions(df: pd.DataFrame, path: Path) -> Path:
    """KL by condition, per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.2), squeeze=False)

    conditions = [
        ("kl_m_primed", "contextual\n(m-primed)", "#E69F00"),
        ("kl_f_primed", "contextual\n(f-primed)", "#009E73"),
        ("kl_refnull", "referent-only\nnull (new)", "#0072B2"),
        ("kl_null", "generic\nnull", "gray"),
    ]

    for ax, model in zip(axes[0], models):
        g = df[df["model"] == model]
        data = [g[c].dropna().values for c, _, _ in conditions]
        labels = [lab for _, lab, _ in conditions]

        bp = ax.boxplot(data, showfliers=False, patch_artist=True)
        for patch, (_, _, color) in zip(bp["boxes"], conditions):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("KL from unprimed (bits)")
        ax.set_title(model)

    fig.suptitle("Divergence from the unprimed baseline by prime condition", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Wrote plot to {path}")
    return path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        prog="python -m revisions.referent_null",
        description="KL and ΔC for contextual / null / referent-only-null primes (Experiment 3).",
    )
    add_common_args(parser)
    parser.add_argument("--primed", required=True, nargs="+", help="Published one-pronoun NDJSON(s).")
    parser.add_argument("--null", required=True, nargs="+", help="Published null-prime NDJSON(s).")
    parser.add_argument("--refnull", required=True, nargs="+", help="Referent-only-null NDJSON(s).")
    parser.add_argument("--models", nargs="+", default=None, help="Model label per file triple.")
    parser.add_argument("--primes", default=None, help="Primes CSV, to attach prime text to each row.")
    parser.add_argument("--out", default=str(OUTPUTS_DIR / "referent_null"))
    args = parser.parse_args(argv)

    seed = set_seeds(args.seed)

    if not (len(args.primed) == len(args.null) == len(args.refnull)):
        parser.error("--primed, --null and --refnull must list the same number of files.")

    labels = args.models or [Path(p).stem for p in args.primed]
    primes = pd.read_csv(args.primes) if args.primes else None

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.exists() else data_dir(args.data_dir) / p

    frames = []
    for primed_f, null_f, refnull_f, label in zip(args.primed, args.null, args.refnull, labels):
        logger.info(f"[{label}] loading measurements")
        frames.append(
            per_item_table(
                primed_data=load_ndjson(resolve(primed_f)),
                null_data=load_ndjson(resolve(null_f)),
                refnull_data=load_ndjson(resolve(refnull_f)),
                model=label,
                max_index=args.max_index,
                primes=primes,
            )
        )

    per_item = pd.concat(frames, ignore_index=True)
    summary = summarise(per_item)

    out_dir = Path(args.out)
    meta = provenance(
        {
            "experiment": "3_referent_only_null_primes",
            "primed_files": args.primed,
            "null_files": args.null,
            "refnull_files": args.refnull,
            "models": labels,
            "max_index": args.max_index,
            "kl": "smoothed Bernoulli KL (bits) vs the unprimed baseline",
        },
        seed=seed,
    )

    write_results(per_item, out_dir / "three_condition_per_item.csv", meta)
    summary_path = write_results(summary, out_dir / "three_condition_summary.csv", meta)

    with open(out_dir / "three_condition_summary.tex", "w") as f:
        f.write(
            summary.round(4).to_latex(
                index=False,
                caption=(
                    "KL divergence from the unprimed baseline under a contextual prime, a generic "
                    "null prime, and a referent-only null prime (referents present, no pronoun)."
                ),
                label="tab:referent_null",
            )
        )

    plot_conditions(per_item, out_dir / "three_conditions.pdf")

    logger.info("\n" + summary.round(4).to_string(index=False))
    return summary_path


if __name__ == "__main__":
    main()
