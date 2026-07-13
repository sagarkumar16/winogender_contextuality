"""
Experiment 4 (Reviewer 2, "Plain Probabilities") -- per-item table.

Figure 1a-e aggregates over items. This emits the underlying per-item plain probabilities and
their differences, as CSV (for reuse) and LaTeX (a longtable for the supplement).

One row per (model, item, sentence order) -- the same unit Figure 1 plots as a single marker.
Columns:

    p_fem_unprimed / _m_primed / _f_primed      P(female pronoun), generation frequency
    ci_lo / ci_hi                                Wilson 95% intervals on those
    k_* / n_*                                    successes and valid trials behind each
    shift_m_primed = p_m_primed - p_unprimed     the plain probability DIFFERENCES that
    shift_f_primed = p_f_primed - p_unprimed     Figure 1 aggregates
    spread          = p_f_primed - p_m_primed    total swing between the two primes
    p_fem_*_internal                             the same quantities from the model's logits
    ordered                                      is p_m <= p_unprimed <= p_f? (the expected
                                                 monotone pattern; Figure 1's colour coding)

Probabilities are reported RAW (k/n) and smoothed (add-half). Raw is what Figure 1 plots;
smoothed is what the KL analyses use, and is reported so the two can be reconciled.

As a check, the mean shifts are compared against the published aggregator
(analysis.mean_completion_shift, imported and run unmodified) and a warning is logged if they
disagree.

Usage:
    python -m revisions.per_item_table \
        --measurements one_pronoun_measurements_gemma-3-12b-it_0.5_wp_k40.ndjson \
        --models gemma --out revisions/outputs/per_item/

- EMNLP revisions, 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.stats.proportion import proportion_confint

# The published aggregator, used to cross-check the per-item numbers.
from winogender_contextuality.modeling.analysis import mean_completion_shift

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
    load_ndjson,
    provenance,
    set_seeds,
    write_results,
)

SENT_ORDERS = {"forward": [0, 1], "reverse": [1, 0]}


def _wilson(counts: FemaleCounts) -> tuple[float, float]:
    if counts.total == 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(counts.successes, counts.total, alpha=0.05, method="wilson")
    return (float(lo), float(hi))


def build_table(data: list[dict], model: str, max_index: int) -> pd.DataFrame:
    """One row per (item, sentence order)."""
    rows = []

    for idx in range(max_index):
        item = get_index(idx, data)

        for order_name, sent_order in SENT_ORDERS.items():
            ordered = get_sent_order(sent_order, item)
            if not ordered:
                continue

            unprimed_recs = [r for r in ordered if r["context"]["sentence_1"] is None]
            m_recs = get_filled_pnoun(0, ordered)
            f_recs = get_filled_pnoun(1, ordered)

            unprimed = female_generation_counts(unprimed_recs)
            m_primed = female_generation_counts(m_recs)
            f_primed = female_generation_counts(f_recs)

            ctx = ordered[0]["context"]

            p_u, p_m, p_f = unprimed.raw_prob, m_primed.raw_prob, f_primed.raw_prob

            u_lo, u_hi = _wilson(unprimed)
            m_lo, m_hi = _wilson(m_primed)
            f_lo, f_hi = _wilson(f_primed)

            rows.append(
                {
                    "model": model,
                    "index": idx,
                    "sent_order": order_name,
                    "case": ctx.get("case_2"),
                    "free_sentence": ctx.get("sentence_2"),
                    "options": ", ".join(map(str, ctx.get("pronouns_2", []))),
                    # counts
                    "k_unprimed": unprimed.successes,
                    "n_unprimed": unprimed.total,
                    "k_m_primed": m_primed.successes,
                    "n_m_primed": m_primed.total,
                    "k_f_primed": f_primed.successes,
                    "n_f_primed": f_primed.total,
                    # raw probabilities (what Figure 1 plots)
                    "p_fem_unprimed": p_u,
                    "p_fem_m_primed": p_m,
                    "p_fem_f_primed": p_f,
                    # Wilson 95% CIs
                    "ci_lo_unprimed": u_lo,
                    "ci_hi_unprimed": u_hi,
                    "ci_lo_m_primed": m_lo,
                    "ci_hi_m_primed": m_hi,
                    "ci_lo_f_primed": f_lo,
                    "ci_hi_f_primed": f_hi,
                    # smoothed (what the KL analyses use)
                    "p_fem_unprimed_smoothed": unprimed.smoothed_prob,
                    "p_fem_m_primed_smoothed": m_primed.smoothed_prob,
                    "p_fem_f_primed_smoothed": f_primed.smoothed_prob,
                    # internal (logit-derived)
                    "p_fem_unprimed_internal": female_internal_prob(unprimed_recs),
                    "p_fem_m_primed_internal": female_internal_prob(m_recs),
                    "p_fem_f_primed_internal": female_internal_prob(f_recs),
                    # the differences this table exists to report
                    "shift_m_primed": p_m - p_u,
                    "shift_f_primed": p_f - p_u,
                    "spread": p_f - p_m,
                    "ordered": bool(p_m <= p_u <= p_f)
                    if not any(np.isnan(v) for v in (p_m, p_u, p_f))
                    else False,
                }
            )

    return pd.DataFrame(rows)


def cross_check(df: pd.DataFrame, data: list[dict], model: str, max_index: int) -> dict:
    """Compare our mean shifts against the published aggregator."""
    check = {}
    try:
        mean_m, mean_f, _, _ = mean_completion_shift(data, mode="generation", max_index=max_index)
        ours_m = float(df["shift_m_primed"].mean(skipna=True))
        ours_f = float(df["shift_f_primed"].mean(skipna=True))

        check = {
            "published_mean_mshift": float(mean_m),
            "published_mean_fshift": float(mean_f),
            "our_mean_mshift": ours_m,
            "our_mean_fshift": ours_f,
            "abs_diff_mshift": abs(float(mean_m) - ours_m),
            "abs_diff_fshift": abs(float(mean_f) - ours_f),
        }

        tol = 1e-6
        if check["abs_diff_mshift"] > tol or check["abs_diff_fshift"] > tol:
            logger.warning(
                f"[{model}] per-item mean shifts differ from analysis.mean_completion_shift: "
                f"m {ours_m:.6f} vs {mean_m:.6f}, f {ours_f:.6f} vs {mean_f:.6f}. "
                "Small gaps are expected -- the published aggregator drops items where ANY of the "
                "three probabilities is NaN (is_valid_tuple), while this table keeps a row whenever "
                "that row's own probability exists."
            )
        else:
            logger.success(f"[{model}] mean shifts match analysis.mean_completion_shift exactly.")

    except Exception as e:
        logger.warning(f"[{model}] cross-check against the published aggregator failed: {e}")

    return check


LATEX_COLUMNS = [
    "index",
    "sent_order",
    "case",
    "p_fem_unprimed",
    "p_fem_m_primed",
    "p_fem_f_primed",
    "shift_m_primed",
    "shift_f_primed",
    "spread",
]

LATEX_HEADER = [
    "Item",
    "Order",
    "Case",
    r"$P(f\mid\varnothing)$",
    r"$P(f\mid m)$",
    r"$P(f\mid f)$",
    r"$\Delta_m$",
    r"$\Delta_f$",
    "Spread",
]


def to_latex(df: pd.DataFrame, path: Path, model: str) -> Path:
    """A supplement-ready longtable of the plain probability differences."""
    tex = df[LATEX_COLUMNS].copy()
    tex["case"] = tex["case"].astype(str).str.replace("$", "", regex=False).str.replace("_PRONOUN", "", regex=False)
    tex.columns = LATEX_HEADER

    body = tex.to_latex(
        index=False,
        longtable=True,
        escape=False,
        float_format="%.3f",
        caption=(
            f"Per-item plain probabilities of female-pronoun generation for {model}. "
            r"$P(f\mid\varnothing)$ is the unprimed baseline; $P(f\mid m)$ and $P(f\mid f)$ are "
            r"the masculine- and feminine-primed conditions. "
            r"$\Delta_m$ and $\Delta_f$ are the plain probability differences from baseline, "
            r"and Spread is $P(f\mid f) - P(f\mid m)$. "
            "Figure 1 aggregates these rows."
        ),
        label=f"tab:per_item_{model}",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(body)

    logger.success(f"Wrote LaTeX longtable ({len(tex)} rows) to {path}")
    return path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        prog="python -m revisions.per_item_table",
        description="Per-item plain probability differences, as CSV and LaTeX (Experiment 4).",
    )
    add_common_args(parser)
    parser.add_argument("--measurements", required=True, nargs="+", help="One-pronoun NDJSON(s).")
    parser.add_argument("--models", nargs="+", default=None, help="Model label per file.")
    parser.add_argument("--out", default=str(OUTPUTS_DIR / "per_item"))
    parser.add_argument("--no-latex", action="store_true", help="Skip the LaTeX longtables.")
    args = parser.parse_args(argv)

    seed = set_seeds(args.seed)
    labels = args.models or [Path(p).stem for p in args.measurements]
    if len(labels) != len(args.measurements):
        parser.error("--models must have one label per --measurements file.")

    out_dir = Path(args.out)
    frames, checks = [], {}

    for path, label in zip(args.measurements, labels):
        p = Path(path)
        if not p.exists():
            p = data_dir(args.data_dir) / path

        logger.info(f"[{label}] loading {p.name}")
        data = load_ndjson(p)

        df = build_table(data, model=label, max_index=args.max_index)
        checks[label] = cross_check(df, data, label, args.max_index)
        frames.append(df)

        if not args.no_latex:
            to_latex(df, out_dir / f"per_item_probabilities_{label}.tex", label)

    combined = pd.concat(frames, ignore_index=True)

    meta = provenance(
        {
            "experiment": "4_per_item_plain_probabilities",
            "measurement_files": args.measurements,
            "models": labels,
            "max_index": args.max_index,
            "cross_check_vs_published_aggregator": checks,
        },
        seed=seed,
    )

    csv_path = write_results(combined, out_dir / "per_item_probabilities.csv", meta)

    logger.info(
        "\n"
        + combined.groupby("model")[["shift_m_primed", "shift_f_primed", "spread"]]
        .mean()
        .round(4)
        .to_string()
    )
    return csv_path


if __name__ == "__main__":
    main()
