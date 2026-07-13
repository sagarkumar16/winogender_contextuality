"""
Experiment 1 (Reviewer 1, "Steering") -- analysis.

Computes ΔC under the joint-measurement formulation and puts it next to the published
steering ΔC, per item and per condition.

Three ΔC columns are emitted, and the distinction between them matters:

  delta_c_steering_published
      From the package estimator, contextuality.calculate_sentence_dc_fraction.

  delta_c_steering
      The SAME steering data, scored with the CbD rank-2 estimator in revisions.cbd -- the same
      one used for the joint column. This is the apples-to-apples baseline: it isolates the
      effect of the *measurement protocol* (steering vs joint) rather than the effect of a
      change in estimator.

      These two columns used to disagree, because the package estimator built its forward
      correlation from marginals (V1, V2) instead of (V1, W2). That is now fixed, so the two are
      independent implementations of the same estimator and must agree -- a test asserts they do
      on random tables. The pair is kept as a cross-check: divergence means one has drifted.
      See revisions/README.md, "Read this first".

  delta_c_joint
      The joint-measurement ΔC: both pronouns generated in one pass (BLANK1/BLANK2), so both
      random variables are actually measured in each context. Same estimator as
      delta_c_steering, so the two are directly comparable.

Also reported:
  qq            the QQ (question-order) statistic q = P_fwd(agree) - P_rev(agree). The rank-2
                ΔC is 2|q| - Δ0, so the joint analysis literally contains the QQ test, with a
                correction for inconsistent connectedness.
  delta_c_joint_rank4_{forward,reverse}
                the CHSH-shaped rank-4 system over option orders, within each sentence order.

Raw per-item 2x2 contingency tables are written alongside the summary so nothing needs
re-running to rescore.

Usage:
    python -m revisions.joint_measurement \
        --steering  one_pronoun_measurements_Llama-3.2-1B-Instruct_0.5_wp.ndjson \
        --joint     joint_measurements_Llama-3.2-1B-Instruct_0.5_wp.ndjson \
        --model llama1b --out revisions/outputs/joint/

- EMNLP revisions, 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Published steering estimator -- imported, never edited.
from winogender_contextuality.modeling.contextuality import (
    calculate_sentence_dc_fraction,
    sentence_order_single_results,
)

from revisions.cbd import (
    ContextStats,
    bootstrap_delta_c,
    context_stats_from_table,
    delta_c_cyclic,
    delta_c_rank2,
    is_contextual,
    qq_equality,
)
from revisions.common import (
    OUTPUTS_DIR,
    add_common_args,
    data_dir,
    female_option,
    get_index,
    get_sent_order,
    load_ndjson,
    provenance,
    set_seeds,
    write_results,
)

FORWARD = [0, 1]
REVERSE = [1, 0]

# The paper's two conditions: the option list is presented male-first or female-first.
CONDITIONS = {"mfirst": 0, "ffirst": 1}


# --------------------------------------------------------------------------------------
# Joint measurements: 2x2 tables over (BLANK1, BLANK2)
# --------------------------------------------------------------------------------------


def joint_table(records: list[dict]) -> np.ndarray:
    """
    Contingency table of joint generations, indexed [BLANK1_is_female, BLANK2_is_female].

    BLANK1 belongs to the sentence presented first, so `pronouns_1` gives its option set and
    `pronouns_2` gives BLANK2's -- in both sentence orders. Generations outside the offered
    options are dropped, as everywhere else in the pipeline.
    """
    table = np.zeros((2, 2), dtype=float)
    if not records:
        return table

    for r in records:
        ctx = r["context"]
        opts1, opts2 = ctx["pronouns_1"], ctx["pronouns_2"]
        fem1 = str(female_option(opts1)).lower()
        fem2 = str(female_option(opts2)).lower()
        valid1 = {str(o).lower() for o in opts1}
        valid2 = {str(o).lower() for o in opts2}

        m = r.get("measurement")
        if not isinstance(m, dict):
            continue
        try:
            b1 = str(m["BLANK1"]).lower()
            b2 = str(m["BLANK2"]).lower()
        except (KeyError, TypeError):
            continue

        if b1 not in valid1 or b2 not in valid2:
            continue

        table[int(b1 == fem1), int(b2 == fem2)] += 1

    return table


def _option_order(records: list[dict], i: int, j: int) -> list[dict]:
    """Records whose option orders are (i, j) for (BLANK1, BLANK2)."""
    return [r for r in records if list(r["context"]["pnoun_order"]) == [i, j]]


def joint_item_stats(idx: int, data: list[dict], condition: str) -> dict:
    """
    Joint ΔC for one item under one condition.

    Rank-2 system: content variables are the pronouns of sentence A (template_1) and
    sentence B (template_2); the two contexts are the two sentence orders. Within a context,
    `first` is BLANK1 and `second` is BLANK2 -- which is exactly the cyclic convention, since
    reversing the sentence order also swaps which content variable BLANK1 refers to.

    The condition fixes both blanks to the same option order (male-first / female-first), the
    closest analogue of the paper's single free-measurement setting. Option order is treated
    properly as a measurement setting by the rank-4 system below.
    """
    k = CONDITIONS[condition]
    item = get_index(idx, data, filter_none=False)

    fwd_recs = _option_order(get_sent_order(FORWARD, item), k, k)
    rev_recs = _option_order(get_sent_order(REVERSE, item), k, k)

    fwd_table = joint_table(fwd_recs)
    rev_table = joint_table(rev_recs)

    fwd = context_stats_from_table(fwd_table)
    rev = context_stats_from_table(rev_table)

    return {
        "delta_c_joint": delta_c_rank2(fwd, rev),
        "qq": qq_equality(fwd, rev),
        "joint_n_forward": int(fwd_table.sum()),
        "joint_n_reverse": int(rev_table.sum()),
        "joint_p_fem_b1_forward": fwd.p_first,
        "joint_p_fem_b2_forward": fwd.p_second,
        "joint_p_fem_b1_reverse": rev.p_first,
        "joint_p_fem_b2_reverse": rev.p_second,
        "_tables": {"forward": fwd_table, "reverse": rev_table},
    }


def joint_rank4(idx: int, data: list[dict], sent_order: list[int]) -> float:
    """
    ΔC for the rank-4 (CHSH-shaped) system within one sentence order.

    Content variables: q0 = BLANK1 with options male-first, q1 = BLANK2 male-first,
    q2 = BLANK1 female-first, q3 = BLANK2 female-first. Contexts in CYCLE order --
    consecutive contexts must share a content variable:

        c0 = (q0,q1) = options (0,0)   first=BLANK1, second=BLANK2
        c1 = (q1,q2) = options (1,0)   first=BLANK2, second=BLANK1   -> transpose
        c2 = (q2,q3) = options (1,1)   first=BLANK1, second=BLANK2
        c3 = (q3,q0) = options (0,1)   first=BLANK2, second=BLANK1   -> transpose

    Enumerating the option pairs in the natural (0,0),(0,1),(1,0),(1,1) order instead would
    place (0,1) next to (1,0), two contexts that share no content variable.
    """
    recs = get_sent_order(sent_order, get_index(idx, data, filter_none=False))

    specs = [((0, 0), False), ((1, 0), True), ((1, 1), False), ((0, 1), True)]
    contexts = []
    for (i, j), transpose in specs:
        table = joint_table(_option_order(recs, i, j))
        if transpose:
            table = table.T
        contexts.append(context_stats_from_table(table))

    return delta_c_cyclic(contexts)


# --------------------------------------------------------------------------------------
# Steering measurements: the same rank-2 estimator, applied to the primed runs
# --------------------------------------------------------------------------------------


def steering_table(records: list[dict]) -> np.ndarray:
    """
    Contingency table for a steering context, indexed [prime_is_female, generated_is_female].

    In a steering context the first content variable is the *prime* (set by the experimenter,
    hence deterministic within a sub-run) and the second is the generated pronoun. This holds
    in both sentence orders: `sentence_1` is always the primed sentence, so `pronouns_1` is
    the prime's option set and `pronouns_2` the free sentence's.
    """
    table = np.zeros((2, 2), dtype=float)

    for r in records:
        ctx = r["context"]
        prime = ctx["pnoun_order"][0]
        if prime is None:  # unprimed baseline record
            continue

        fem_prime = str(female_option(ctx["pronouns_1"])).lower()
        fem_gen = str(female_option(ctx["pronouns_2"])).lower()
        valid_gen = {str(o).lower() for o in ctx["pronouns_2"]}

        m = r.get("measurement")
        if not isinstance(m, dict):
            continue
        gen = str(m.get("BLANK", "")).lower()
        if gen not in valid_gen:
            continue

        table[int(str(prime).lower() == fem_prime), int(gen == fem_gen)] += 1

    return table


def steering_item_stats(idx: int, data: list[dict], condition: str) -> dict:
    """Steering ΔC for one item, scored with the same CbD estimator as the joint column."""
    k = CONDITIONS[condition]
    item = get_index(idx, data, filter_none=False)
    primed = [r for r in item if r["context"]["pnoun_order"][1] == k]

    fwd_table = steering_table(get_sent_order(FORWARD, primed))
    rev_table = steering_table(get_sent_order(REVERSE, primed))

    fwd = context_stats_from_table(fwd_table)
    rev = context_stats_from_table(rev_table)

    return {
        "delta_c_steering": delta_c_rank2(fwd, rev),
        "steering_n_forward": int(fwd_table.sum()),
        "steering_n_reverse": int(rev_table.sum()),
        "steering_p_fem_gen_forward": fwd.p_second,
        "steering_p_fem_gen_reverse": rev.p_second,
        "_tables": {"forward": fwd_table, "reverse": rev_table},
    }


def steering_published(idx: int, data: list[dict], condition: str, mode: str = "generation") -> float:
    """The published estimator, called unmodified."""
    try:
        dd = sentence_order_single_results(
            idx, data, mode=mode, pnoun_order=CONDITIONS[condition]
        )
        return float(calculate_sentence_dc_fraction(dd, mode=mode))
    except Exception as e:
        logger.debug(f"published steering ΔC failed at idx {idx} ({condition}): {e}")
        return float("nan")


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def compare(
    steering_data: list[dict],
    joint_data: list[dict],
    model: str,
    max_index: int,
    bootstrap: int = 0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-item, per-condition comparison table plus the raw 2x2 tables."""
    rows, raw = [], []

    for idx in range(max_index):
        for condition in CONDITIONS:
            j = joint_item_stats(idx, joint_data, condition)
            s = steering_item_stats(idx, steering_data, condition)

            j_tables = j.pop("_tables")
            s_tables = s.pop("_tables")

            row = {
                "model": model,
                "index": idx,
                "condition": condition,
                "delta_c_steering_published": steering_published(idx, steering_data, condition),
                **s,
                **j,
                "delta_c_joint_rank4_forward": joint_rank4(idx, joint_data, FORWARD),
                "delta_c_joint_rank4_reverse": joint_rank4(idx, joint_data, REVERSE),
            }

            if bootstrap:
                lo, hi = bootstrap_delta_c(
                    [j_tables["forward"], j_tables["reverse"]], n_boot=bootstrap, seed=seed + idx
                )
                row["delta_c_joint_ci_lo"], row["delta_c_joint_ci_hi"] = lo, hi
                lo, hi = bootstrap_delta_c(
                    [s_tables["forward"], s_tables["reverse"]], n_boot=bootstrap, seed=seed + idx
                )
                row["delta_c_steering_ci_lo"], row["delta_c_steering_ci_hi"] = lo, hi

            rows.append(row)

            for formulation, tables in (("joint", j_tables), ("steering", s_tables)):
                for order, t in tables.items():
                    raw.append(
                        {
                            "model": model,
                            "index": idx,
                            "condition": condition,
                            "formulation": formulation,
                            "sent_order": order,
                            # counts of (first_is_female, second_is_female)
                            "n_mm": t[0, 0],
                            "n_mf": t[0, 1],
                            "n_fm": t[1, 0],
                            "n_ff": t[1, 1],
                        }
                    )

    return pd.DataFrame(rows), pd.DataFrame(raw)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Model x condition summary: mean ΔC, how often each formulation calls an item contextual."""
    out = []
    for (model, condition), g in df.groupby(["model", "condition"], sort=False):
        both = g.dropna(subset=["delta_c_steering", "delta_c_joint"])
        ctx_steer = g["delta_c_steering"].map(is_contextual)
        ctx_joint = g["delta_c_joint"].map(is_contextual)
        out.append(
            {
                "model": model,
                "condition": condition,
                "n_items": int(len(g)),
                "mean_delta_c_steering_published": g["delta_c_steering_published"].mean(),
                "mean_delta_c_steering": g["delta_c_steering"].mean(),
                "mean_delta_c_joint": g["delta_c_joint"].mean(),
                "frac_contextual_steering": float(ctx_steer.mean()),
                "frac_contextual_joint": float(ctx_joint.mean()),
                "frac_agree_contextual": float(
                    (
                        both["delta_c_steering"].map(is_contextual)
                        == both["delta_c_joint"].map(is_contextual)
                    ).mean()
                )
                if len(both)
                else np.nan,
                "pearson_r": float(both["delta_c_steering"].corr(both["delta_c_joint"]))
                if len(both) > 1
                else np.nan,
                "mean_qq": g["qq"].mean(),
                "mean_abs_qq": g["qq"].abs().mean(),
            }
        )
    return pd.DataFrame(out)


def plot_comparison(df: pd.DataFrame, path: Path) -> Path:
    """Scatter of steering vs joint ΔC per item, plus the paired distributions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(df["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 2)))
    for c, model in zip(colors, models):
        g = df[df["model"] == model].dropna(subset=["delta_c_steering", "delta_c_joint"])
        ax.scatter(g["delta_c_steering"], g["delta_c_joint"], s=14, alpha=0.6, label=model, color=c)

    lims = [
        np.nanmin([df["delta_c_steering"].min(), df["delta_c_joint"].min(), 0]) - 0.05,
        np.nanmax([df["delta_c_steering"].max(), df["delta_c_joint"].max(), 0]) + 0.05,
    ]
    ax.plot(lims, lims, ls="--", lw=0.8, color="gray", zorder=0)
    ax.axhline(0, lw=0.6, color="k", zorder=0)
    ax.axvline(0, lw=0.6, color="k", zorder=0)
    ax.set_xlabel(r"$\Delta C$ (steering)")
    ax.set_ylabel(r"$\Delta C$ (joint)")
    ax.set_title("Per-item contextuality: steering vs joint\n(> 0 = contextual)")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    labels, data = [], []
    for model in models:
        for condition in CONDITIONS:
            g = df[(df["model"] == model) & (df["condition"] == condition)]
            for col, name in (("delta_c_steering", "steer"), ("delta_c_joint", "joint")):
                vals = g[col].dropna().values
                if len(vals):
                    data.append(vals)
                    labels.append(f"{model}\n{condition}\n{name}")

    if data:
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.axhline(0, ls="--", lw=0.8, color="red")
    ax.set_ylabel(r"$\Delta C$")
    ax.set_title("Distribution by model and condition")
    ax.tick_params(axis="x", labelsize=6)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Wrote plot to {path}")
    return path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        prog="python -m revisions.joint_measurement",
        description="Compare steering and joint-measurement contextuality (Experiment 1).",
    )
    add_common_args(parser)
    parser.add_argument("--steering", required=True, nargs="+", help="One-pronoun (steering) NDJSON(s).")
    parser.add_argument("--joint", required=True, nargs="+", help="Joint two-pronoun NDJSON(s).")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model label per --steering/--joint pair (defaults to the filename stem).",
    )
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap resamples for ΔC CIs (0 = off).")
    parser.add_argument("--out", default=str(OUTPUTS_DIR / "joint"), help="Output directory.")
    args = parser.parse_args(argv)

    seed = set_seeds(args.seed)

    if len(args.steering) != len(args.joint):
        parser.error("--steering and --joint must list the same number of files, pairwise by model.")

    labels = args.models or [Path(p).stem for p in args.joint]
    if len(labels) != len(args.joint):
        parser.error("--models must have one label per file pair.")

    out_dir = Path(args.out)
    per_item, raw_all = [], []

    for steering_f, joint_f, label in zip(args.steering, args.joint, labels):
        s_path = Path(steering_f)
        j_path = Path(joint_f)
        if not s_path.is_absolute() and not s_path.exists():
            s_path = data_dir(args.data_dir) / steering_f
        if not j_path.is_absolute() and not j_path.exists():
            j_path = data_dir(args.data_dir) / joint_f

        logger.info(f"[{label}] steering={s_path.name}  joint={j_path.name}")
        steering_data = load_ndjson(s_path)
        joint_data = load_ndjson(j_path)
        logger.info(f"[{label}] {len(steering_data)} steering records, {len(joint_data)} joint records")

        df, raw = compare(
            steering_data,
            joint_data,
            model=label,
            max_index=args.max_index,
            bootstrap=args.bootstrap,
            seed=seed,
        )
        per_item.append(df)
        raw_all.append(raw)

    per_item_df = pd.concat(per_item, ignore_index=True)
    raw_df = pd.concat(raw_all, ignore_index=True)
    summary_df = summarise(per_item_df)

    meta = provenance(
        {
            "experiment": "1_joint_measurement",
            "steering_files": [str(p) for p in args.steering],
            "joint_files": [str(p) for p in args.joint],
            "models": labels,
            "max_index": args.max_index,
            "bootstrap": args.bootstrap,
        },
        seed=seed,
    )

    write_results(per_item_df, out_dir / "joint_vs_steering_per_item.csv", meta)
    write_results(raw_df, out_dir / "joint_vs_steering_raw_counts.csv", meta)
    summary_path = write_results(summary_df, out_dir / "joint_vs_steering_summary.csv", meta)

    with open(out_dir / "joint_vs_steering_summary.tex", "w") as f:
        f.write(
            summary_df.round(4).to_latex(
                index=False,
                caption="Contextuality under the steering and joint-measurement formulations.",
                label="tab:joint_vs_steering",
            )
        )

    plot_comparison(per_item_df, out_dir / "joint_vs_steering.pdf")

    logger.info("\n" + summary_df.round(4).to_string(index=False))
    return summary_path


if __name__ == "__main__":
    main()
