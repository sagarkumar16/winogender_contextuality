"""
Experiment 2 (Reviewer 2, "Quantization") -- analysis.

Two jobs.

1. COMPARISON (default). Quantized vs unquantized runs of the same model on the same items:
   per-item plain probabilities, KL divergences from the unprimed baseline, and ΔC. Answers
   "do the paper's conclusions survive without quantization?"

2. SANITY CHECK (--sanity-check). gemma-3-12b-it was already run unquantized, so re-running it
   through the new pipeline must reproduce the existing numbers. Two checks, with very
   different expectations:

     internal (logit-derived) probabilities are DETERMINISTIC given the same weights and
        prompt, so they must agree to numerical tolerance. This is the real test of the
        pipeline. Disagreement here means the pipeline changed something.

     generation frequencies are SAMPLED (temperature 0.5, ~50 draws per cell), so they cannot
        match exactly. Expecting them to would be wrong. We instead ask whether each item's
        two counts are consistent with a common binomial rate -- a two-proportion test -- and
        report the fraction of items that are not, which under the null should sit near the
        alpha level.

   Any item failing either check is written to a discrepancies CSV and logged.

Usage:
    # comparison
    python -m revisions.quantization_compare --models gemma llama8b \
        --quantized   q_gemma.ndjson q_llama8b.ndjson \
        --unquantized u_gemma.ndjson u_llama8b.ndjson \
        --out revisions/outputs/quantization/

    # gemma sanity check
    python -m revisions.quantization_compare --sanity-check \
        --new      revisions/outputs/measurements/one_pronoun_measurements_gemma-3-12b-it_0.5_wp_unquant.ndjson \
        --existing one_pronoun_measurements_gemma-3-12b-it_0.5_wp_k40.ndjson

- EMNLP revisions, 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from revisions.cbd import is_contextual
from revisions.common import (
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
from revisions.joint_measurement import steering_item_stats, steering_published

FORWARD = [0, 1]
REVERSE = [1, 0]
SENT_ORDERS = {"forward": FORWARD, "reverse": REVERSE}

# Internal probabilities are deterministic; this tolerance only absorbs bf16/kernel jitter.
INTERNAL_TOL = 1e-3


def item_rows(data: list[dict], model: str, arm: str, max_index: int) -> pd.DataFrame:
    """Per (item, sentence order): plain probabilities, counts and KLs for one run."""
    rows = []

    for idx in range(max_index):
        item = get_index(idx, data)

        for order_name, sent_order in SENT_ORDERS.items():
            ordered = get_sent_order(sent_order, item)

            unprimed_recs = [r for r in ordered if r["context"]["sentence_1"] is None]
            m_recs = get_filled_pnoun(0, ordered)
            f_recs = get_filled_pnoun(1, ordered)

            unprimed = female_generation_counts(unprimed_recs)
            m_primed = female_generation_counts(m_recs)
            f_primed = female_generation_counts(f_recs)

            rows.append(
                {
                    "model": model,
                    "arm": arm,
                    "index": idx,
                    "sent_order": order_name,
                    "k_unprimed": unprimed.successes,
                    "n_unprimed": unprimed.total,
                    "k_m_primed": m_primed.successes,
                    "n_m_primed": m_primed.total,
                    "k_f_primed": f_primed.successes,
                    "n_f_primed": f_primed.total,
                    "p_fem_unprimed": unprimed.smoothed_prob,
                    "p_fem_m_primed": m_primed.smoothed_prob,
                    "p_fem_f_primed": f_primed.smoothed_prob,
                    "p_fem_unprimed_internal": female_internal_prob(unprimed_recs),
                    "p_fem_m_primed_internal": female_internal_prob(m_recs),
                    "p_fem_f_primed_internal": female_internal_prob(f_recs),
                    "kl_m_primed": kl_from_counts(m_primed, unprimed),
                    "kl_f_primed": kl_from_counts(f_primed, unprimed),
                }
            )

    df = pd.DataFrame(rows)

    for idx in range(max_index):
        for condition in ("mfirst", "ffirst"):
            s = steering_item_stats(idx, data, condition)
            df.loc[df["index"] == idx, f"delta_c_{condition}"] = s["delta_c_steering"]
            df.loc[df["index"] == idx, f"delta_c_published_{condition}"] = steering_published(
                idx, data, condition
            )

    return df


def compare_arms(quant: pd.DataFrame, unquant: pd.DataFrame) -> pd.DataFrame:
    """Join the two arms per (item, sentence order) and difference the quantities of interest."""
    keys = ["model", "index", "sent_order"]
    cols = [
        c
        for c in quant.columns
        if c.startswith(("p_fem", "kl_", "delta_c_"))
    ]

    merged = quant[keys + cols].merge(
        unquant[keys + cols], on=keys, suffixes=("_quant", "_unquant")
    )

    for c in cols:
        merged[f"diff_{c}"] = merged[f"{c}_unquant"] - merged[f"{c}_quant"]

    return merged


def summarise(merged: pd.DataFrame) -> pd.DataFrame:
    """Per-model: how much does removing quantization move each quantity?"""
    out = []
    for model, g in merged.groupby("model", sort=False):
        row = {"model": model, "n_items": int(len(g))}

        for c in ["p_fem_unprimed", "p_fem_m_primed", "p_fem_f_primed", "kl_m_primed", "kl_f_primed"]:
            row[f"mean_{c}_quant"] = g[f"{c}_quant"].mean()
            row[f"mean_{c}_unquant"] = g[f"{c}_unquant"].mean()
            row[f"mean_abs_diff_{c}"] = g[f"diff_{c}"].abs().mean()

        for c in ["delta_c_mfirst", "delta_c_ffirst"]:
            row[f"mean_{c}_quant"] = g[f"{c}_quant"].mean()
            row[f"mean_{c}_unquant"] = g[f"{c}_unquant"].mean()
            ctx_q = g[f"{c}_quant"].map(is_contextual)
            ctx_u = g[f"{c}_unquant"].map(is_contextual)
            row[f"frac_contextual_{c}_quant"] = float(ctx_q.mean())
            row[f"frac_contextual_{c}_unquant"] = float(ctx_u.mean())
            # Does the contextual/non-contextual verdict flip when quantization is removed?
            row[f"frac_verdict_flip_{c}"] = float((ctx_q != ctx_u).mean())

        out.append(row)

    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# Sanity check
# --------------------------------------------------------------------------------------


def two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided p-value that two binomial counts share a rate (normal approximation)."""
    if n1 == 0 or n2 == 0:
        return float("nan")

    p_pool = (k1 + k2) / (n1 + n2)
    if p_pool in (0.0, 1.0):
        return 1.0

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0

    from scipy.stats import norm

    z = (k1 / n1 - k2 / n2) / se
    return float(2 * (1 - norm.cdf(abs(z))))


def sanity_check(
    new_data: list[dict],
    existing_data: list[dict],
    max_index: int,
    alpha: float = 0.05,
    internal_tol: float = INTERNAL_TOL,
) -> tuple[pd.DataFrame, dict]:
    """
    Does the new pipeline reproduce an existing unquantized run?

    Internal probabilities must match to `internal_tol` (deterministic). Generation counts are
    compared with a two-proportion test; the fraction of items below `alpha` should sit near
    `alpha` itself if the two runs are draws from the same model.
    """
    new_df = item_rows(new_data, model="new", arm="new", max_index=max_index)
    old_df = item_rows(existing_data, model="existing", arm="existing", max_index=max_index)

    merged = new_df.merge(
        old_df.drop(columns=["model", "arm"]),
        on=["index", "sent_order"],
        suffixes=("_new", "_old"),
    )

    conditions = ["unprimed", "m_primed", "f_primed"]

    for cond in conditions:
        merged[f"internal_absdiff_{cond}"] = (
            merged[f"p_fem_{cond}_internal_new"] - merged[f"p_fem_{cond}_internal_old"]
        ).abs()
        merged[f"internal_mismatch_{cond}"] = merged[f"internal_absdiff_{cond}"] > internal_tol

        merged[f"gen_p_value_{cond}"] = [
            two_proportion_p(
                row[f"k_{cond}_new"], row[f"n_{cond}_new"], row[f"k_{cond}_old"], row[f"n_{cond}_old"]
            )
            for _, row in merged.iterrows()
        ]
        merged[f"gen_mismatch_{cond}"] = merged[f"gen_p_value_{cond}"] < alpha

    internal_cols = [f"internal_mismatch_{c}" for c in conditions]
    gen_cols = [f"gen_mismatch_{c}" for c in conditions]

    merged["any_internal_mismatch"] = merged[internal_cols].any(axis=1)
    merged["any_gen_mismatch"] = merged[gen_cols].any(axis=1)

    have_internal = merged[[f"p_fem_{c}_internal_new" for c in conditions]].notna().any(axis=1).any()

    report = {
        "n_items": int(len(merged)),
        "internal_probs_available": bool(have_internal),
        "internal_tolerance": internal_tol,
        "n_internal_mismatch": int(merged["any_internal_mismatch"].sum()),
        "max_internal_absdiff": float(
            merged[[f"internal_absdiff_{c}" for c in conditions]].max().max()
        ),
        "alpha": alpha,
        "n_generation_mismatch": int(merged["any_gen_mismatch"].sum()),
        "frac_generation_mismatch": float(merged["any_gen_mismatch"].mean()),
        "mean_abs_delta_c_diff": float(
            (merged["delta_c_mfirst_new"] - merged["delta_c_mfirst_old"]).abs().mean()
        ),
    }

    # Verdict
    if not have_internal:
        logger.warning(
            "No internal probabilities in one of the runs -- the deterministic check could not "
            "run. (The published NULL runs store character logits, not pronoun logits; use the "
            "primed runs for this check.)"
        )
    elif report["n_internal_mismatch"] == 0:
        logger.success(
            f"SANITY CHECK PASSED: internal probabilities agree to within {internal_tol} on all "
            f"{report['n_items']} items (max |diff| = {report['max_internal_absdiff']:.2e})."
        )
    else:
        logger.error(
            f"SANITY CHECK FAILED: {report['n_internal_mismatch']}/{report['n_items']} items differ "
            f"in their DETERMINISTIC internal probabilities by more than {internal_tol} "
            f"(max |diff| = {report['max_internal_absdiff']:.2e}). The new pipeline is not "
            "reproducing the existing run."
        )

    expected = alpha
    observed = report["frac_generation_mismatch"]
    if observed > 3 * expected:
        logger.warning(
            f"Generation frequencies differ on {observed:.1%} of items at alpha={alpha} "
            f"(sampling noise alone would give ~{expected:.0%}). Worth inspecting: could be a "
            "seed/temperature/n_runs difference rather than a pipeline error."
        )
    else:
        logger.success(
            f"Generation frequencies differ on {observed:.1%} of items at alpha={alpha}, "
            f"consistent with sampling noise (~{expected:.0%} expected)."
        )

    return merged, report


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        prog="python -m revisions.quantization_compare",
        description="Quantized vs unquantized comparison, and the gemma reproduction check (Experiment 2).",
    )
    add_common_args(parser)
    parser.add_argument("--quantized", nargs="+", default=None, help="Quantized-arm NDJSON(s).")
    parser.add_argument("--unquantized", nargs="+", default=None, help="Unquantized-arm NDJSON(s).")
    parser.add_argument("--models", nargs="+", default=None, help="Model label per file pair.")
    parser.add_argument("--sanity-check", action="store_true", help="Run the reproduction check instead.")
    parser.add_argument("--new", default=None, help="Sanity check: NDJSON from the new pipeline.")
    parser.add_argument("--existing", default=None, help="Sanity check: the published NDJSON.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Sanity check: two-proportion alpha.")
    parser.add_argument(
        "--internal-tol",
        type=float,
        default=INTERNAL_TOL,
        help="Sanity check: tolerance on deterministic internal probabilities.",
    )
    parser.add_argument("--out", default=str(OUTPUTS_DIR / "quantization"))
    args = parser.parse_args(argv)

    seed = set_seeds(args.seed)
    out_dir = Path(args.out)

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.exists() else data_dir(args.data_dir) / p

    # ---------------- sanity check ----------------
    if args.sanity_check:
        if not (args.new and args.existing):
            parser.error("--sanity-check requires --new and --existing.")

        logger.info(f"Sanity check: new={args.new}  existing={args.existing}")
        merged, report = sanity_check(
            new_data=load_ndjson(resolve(args.new)),
            existing_data=load_ndjson(resolve(args.existing)),
            max_index=args.max_index,
            alpha=args.alpha,
            internal_tol=args.internal_tol,
        )

        meta = provenance(
            {
                "experiment": "2_unquantized",
                "check": "gemma_reproduction_sanity_check",
                "new_file": str(args.new),
                "existing_file": str(args.existing),
                **report,
            },
            seed=seed,
        )

        write_results(merged, out_dir / "sanity_check_per_item.csv", meta)

        discrepancies = merged[merged["any_internal_mismatch"] | merged["any_gen_mismatch"]]
        path = write_results(discrepancies, out_dir / "sanity_check_discrepancies.csv", meta)
        logger.info(f"{len(discrepancies)} item(s) flagged; report: {report}")
        return path

    # ---------------- comparison ----------------
    if not (args.quantized and args.unquantized):
        parser.error("Pass --quantized and --unquantized (or use --sanity-check).")
    if len(args.quantized) != len(args.unquantized):
        parser.error("--quantized and --unquantized must list the same number of files.")

    labels = args.models or [Path(p).stem for p in args.quantized]

    q_frames, u_frames = [], []
    for q_f, u_f, label in zip(args.quantized, args.unquantized, labels):
        logger.info(f"[{label}] quantized={Path(q_f).name}  unquantized={Path(u_f).name}")
        q_frames.append(item_rows(load_ndjson(resolve(q_f)), label, "quantized", args.max_index))
        u_frames.append(item_rows(load_ndjson(resolve(u_f)), label, "unquantized", args.max_index))

    quant = pd.concat(q_frames, ignore_index=True)
    unquant = pd.concat(u_frames, ignore_index=True)

    merged = compare_arms(quant, unquant)
    summary = summarise(merged)

    meta = provenance(
        {
            "experiment": "2_unquantized",
            "quantized_files": args.quantized,
            "unquantized_files": args.unquantized,
            "models": labels,
            "max_index": args.max_index,
        },
        seed=seed,
    )

    write_results(pd.concat([quant, unquant], ignore_index=True), out_dir / "quantization_per_item_raw.csv", meta)
    write_results(merged, out_dir / "quantization_per_item_diff.csv", meta)
    summary_path = write_results(summary, out_dir / "quantization_summary.csv", meta)

    with open(out_dir / "quantization_summary.tex", "w") as f:
        f.write(
            summary.round(4).to_latex(
                index=False,
                caption="Quantized vs unquantized: plain probabilities, KL divergences and $\\Delta C$.",
                label="tab:quantization",
            )
        )

    logger.info("\n" + summary.round(4).to_string(index=False))
    return summary_path


if __name__ == "__main__":
    main()
