"""
Synthetic measurement data in the published NDJSON schema.

Two uses:

1. Tests. Records are built from explicit 2x2 contingency tables, so the ΔC / KL a test
   expects can be computed by hand from those tables and compared against what the analysis
   extracts from the files.

2. Smoke-running the pipeline with no GPU:

       python -m revisions.fixtures --out /tmp/fix
       python -m revisions.joint_measurement --steering /tmp/fix/steering.ndjson \
           --joint /tmp/fix/joint.ndjson --models synthetic --max-index 4 --out /tmp/out

These files are synthetic. They are never written to a results directory and no number in
them means anything about a real model.

- EMNLP revisions, 2026
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from winogender_contextuality.utils import Context, Measurement

DEFAULT_PRONOUNS = ["he", "she"]


def logits_for(p_female: float | None) -> list[float] | None:
    """
    A 2-vector over the canonical (male, female) options whose softmax gives `p_female`.

    Collection stores logits in canonical option order, so position 1 is female; this lets a
    fixture pin an exact internal probability and lets tests assert on it.
    """
    if p_female is None:
        return None
    p = float(np.clip(p_female, 1e-6, 1 - 1e-6))
    return [0.0, float(np.log(p / (1 - p)))]


def _records_from_table(
    idx: int,
    table: np.ndarray,
    sent_order: list,
    pnoun_order: tuple,
    keys: tuple[str, str] | tuple[str],
    pronouns_1,
    pronouns_2,
    sentence_1: str | None = "PRIME.",
    sentence_2: str = "Free sentence with BLANK.",
    case_1: str = "$NOM_PRONOUN",
    case_2: str = "$NOM_PRONOUN",
    prime_from_table: bool = False,
    internal_p: float | None = None,
) -> list[dict]:
    """
    Expand a 2x2 count table into individual Measurement records.

    table[a, b] = number of trials where the first variable took outcome `a` and the second
    took outcome `b` (0 = male, 1 = female).

    For joint data (keys = ('BLANK1','BLANK2')) both outcomes become generations.
    For steering data (keys = ('BLANK',)) the FIRST outcome is the prime -- it is written into
    context.pnoun_order[0], not into the measurement -- and the second is the generation.
    """
    out = []
    opts1 = list(pronouns_1)
    opts2 = list(pronouns_2)

    for a in (0, 1):
        for b in (0, 1):
            for _ in range(int(table[a, b])):
                if prime_from_table:
                    prime = opts1[a]  # opts1 = [male, female] canonical
                    measurement = {"BLANK": opts2[b]}
                    pn_order = (prime, pnoun_order[1])
                else:
                    measurement = {keys[0]: opts1[a], keys[1]: opts2[b]}
                    pn_order = tuple(pnoun_order)

                ctx = Context(
                    sent_order=list(sent_order),
                    pnoun_order=pn_order,
                    sentence_1=sentence_1,
                    sentence_2=sentence_2,
                    pronouns_1=opts1,
                    pronouns_2=opts2,
                    case_1=case_1,
                    case_2=case_2,
                )
                m = Measurement(
                    index=int(idx),
                    context=ctx,
                    measurement=measurement,
                    probabilities=None,
                    logits=logits_for(internal_p),
                )
                out.append(asdict(m))

    return out


def joint_records(idx: int, table: np.ndarray, sent_order: list, i: int, j: int, **kw) -> list[dict]:
    """Joint (two-pronoun) records; table indexed [BLANK1_female, BLANK2_female]."""
    return _records_from_table(
        idx,
        table,
        sent_order=sent_order,
        pnoun_order=(i, j),
        keys=("BLANK1", "BLANK2"),
        pronouns_1=kw.pop("pronouns_1", DEFAULT_PRONOUNS),
        pronouns_2=kw.pop("pronouns_2", DEFAULT_PRONOUNS),
        **kw,
    )


def steering_records(idx: int, table: np.ndarray, sent_order: list, j: int, **kw) -> list[dict]:
    """Steering (one-pronoun) records; table indexed [prime_female, generated_female]."""
    return _records_from_table(
        idx,
        table,
        sent_order=sent_order,
        pnoun_order=(None, j),
        keys=("BLANK",),
        pronouns_1=kw.pop("pronouns_1", DEFAULT_PRONOUNS),
        pronouns_2=kw.pop("pronouns_2", DEFAULT_PRONOUNS),
        prime_from_table=True,
        **kw,
    )


def unprimed_records(
    idx: int,
    n_female: int,
    n_male: int,
    sent_order: list,
    j: int = 0,
    pronouns=DEFAULT_PRONOUNS,
    internal_p: float | None = None,
) -> list[dict]:
    """
    Baseline records with no prime: context.sentence_1 is None and pnoun_order[0] is None,
    which is how get_single_sentences() and get_filled_pnoun() tell them apart from primed ones.
    """
    out = []
    for gen, count in ((pronouns[1], n_female), (pronouns[0], n_male)):
        for _ in range(int(count)):
            ctx = Context(
                sent_order=list(sent_order),
                pnoun_order=(None, j),
                sentence_1=None,
                sentence_2="Free sentence with BLANK.",
                pronouns_1=list(pronouns),
                pronouns_2=list(pronouns),
                case_1="$NOM_PRONOUN",
                case_2="$NOM_PRONOUN",
            )
            out.append(
                asdict(
                    Measurement(
                        index=int(idx),
                        context=ctx,
                        measurement={"BLANK": gen},
                        probabilities=None,
                        logits=logits_for(internal_p),
                    )
                )
            )
    return out


def null_records(
    idx: int,
    n_female: int,
    n_male: int,
    prime_tag: str = "null_0",
    n_order: int = 0,
    pronouns=DEFAULT_PRONOUNS,
    prime: str = "The sky is blue.",
) -> list[dict]:
    """
    Generic-null-prime records, in the schema generate_one_null_context writes.

    `pronouns` is the PRESENTED option order, which the null runs reverse half the time. The
    female/male generations are assigned by string, not by position, so passing ['she','he']
    still produces `n_female` generations of "she" -- otherwise the fixture would bake in the
    very positional assumption the analysis is supposed to avoid.
    """
    from revisions.common import female_option, male_option

    out = []
    for gen, count in ((female_option(pronouns), n_female), (male_option(pronouns), n_male)):
        for _ in range(count):
            ctx = Context(
                sent_order=[prime_tag, 0],
                pnoun_order=("null", n_order),
                sentence_1=prime,
                sentence_2="Free sentence with BLANK.",
                pronouns_1="null",
                pronouns_2=list(pronouns),
                case_1="null",
                case_2="$NOM_PRONOUN",
            )
            out.append(
                asdict(
                    Measurement(
                        index=int(idx),
                        context=ctx,
                        measurement={"BLANK": gen},
                        probabilities=None,
                        logits=None,
                    )
                )
            )
    return out


def refnull_records(
    idx: int,
    n_female: int,
    n_male: int,
    n_order: int = 0,
    pronouns=DEFAULT_PRONOUNS,
    prime: str = "The technician told the customer that the technician had completed the repair.",
) -> list[dict]:
    """Referent-only null-prime records, in the schema revisions.collect_referent_null writes."""
    recs = null_records(idx, n_female, n_male, prime_tag="refnull", n_order=n_order, pronouns=pronouns, prime=prime)
    for r in recs:
        r["context"]["sent_order"] = ["refnull", 0]
        r["context"]["pnoun_order"] = ["refnull", n_order]
        r["context"]["pronouns_1"] = "refnull"
        r["context"]["case_1"] = "refnull"
    return recs


def write_ndjson(path: Path, records: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def build_dataset(n_items: int = 4, n: int = 100, seed: int = 0) -> dict[str, list[dict]]:
    """
    A small synthetic dataset covering every condition the analyses read.

    Item 0 is built to be strongly order-dependent (the joint distribution flips with sentence
    order), items 1+ are progressively milder. Steering tables are set equal to the joint
    tables, so the two formulations must return the same ΔC -- the invariant the tests check.
    """
    rng = np.random.default_rng(seed)
    steering, joint, null, refnull = [], [], [], []

    for idx in range(n_items):
        # Strength of the order effect, decaying across items.
        skew = 0.4 / (idx + 1)

        for sent_order, sign in (([0, 1], +1), ([1, 0], -1)):
            p_ff = 0.25 + sign * skew  # P(both female)
            p_mm = 0.25 + sign * skew
            p_fm = 0.5 - (p_ff + p_mm) / 2 * 1.0
            p_mf = 1.0 - p_ff - p_mm - p_fm

            probs = np.array([[p_mm, p_mf], [p_fm, p_ff]], dtype=float)
            probs = np.clip(probs, 1e-6, None)
            probs /= probs.sum()

            table = rng.multinomial(n, probs.ravel()).reshape(2, 2).astype(float)

            # Deterministic internal probabilities, pinned per (item, order) so the
            # quantization sanity check has something exact to compare.
            p_int = 0.5 + sign * skew / 2

            for k in (0, 1):  # both option-order conditions
                joint += joint_records(idx, table, sent_order=sent_order, i=k, j=k)
                steering += steering_records(
                    idx, table, sent_order=sent_order, j=k, internal_p=p_int
                )

            # Off-diagonal option orders, only needed by the rank-4 system.
            for i, j in ((0, 1), (1, 0)):
                joint += joint_records(idx, table, sent_order=sent_order, i=i, j=j)

            # Unprimed baselines for the KL analyses.
            n_fem = int(table[:, 1].sum())
            steering += unprimed_records(
                idx,
                n_female=n_fem,
                n_male=n - n_fem,
                sent_order=sent_order,
                internal_p=p_int,
            )

        # Null-prime conditions are indexed per sentence slot: rows 2k and 2k+1.
        for slot, row in enumerate((2 * idx, 2 * idx + 1)):
            base = 40 + 5 * slot
            for tag in ("null_0", "null_1", "null_2"):
                for n_order in (0, 1):
                    null += null_records(row, n_female=base, n_male=n - base, prime_tag=tag, n_order=n_order)
            for n_order in (0, 1):
                refnull += refnull_records(row, n_female=base + 3, n_male=n - base - 3, n_order=n_order)

    return {"steering": steering, "joint": joint, "null": null, "refnull": refnull}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        prog="python -m revisions.fixtures",
        description="Write synthetic measurement NDJSONs for smoke-testing the analyses.",
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--n-items", type=int, default=4)
    parser.add_argument("--n", type=int, default=100, help="Trials per context.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    data = build_dataset(n_items=args.n_items, n=args.n, seed=args.seed)
    for name, records in data.items():
        path = write_ndjson(out / f"{name}.ndjson", records)
        print(f"wrote {len(records):6d} records -> {path}")
    return out


if __name__ == "__main__":
    main()
