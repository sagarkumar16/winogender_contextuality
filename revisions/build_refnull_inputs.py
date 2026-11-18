"""
Builds the input file for the referent-only null primes.

Every sentence of all_sentences_wp.csv gets a prime that names both referents of its pair and
contains no pronoun, e.g.

    The technician and the customer were both present.

with the referents in the order the sentence mentions them. The output has the columns
generate_one_null_context reads (template, differences, referent, case) plus `prime`, in the same
row order as the notebook's all_sentences_wp (template_1, then template_2, of each pair), so the
referent-only run is the published null run with a per-row prime:

    python -m revisions.build_refnull_inputs \
        --pairs <interim>/winopron_pairs.tsv --output <interim>/all_sentences_wp_refnull.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

REFNULL_FRAME = "The {first} and the {second} were both present."

GENDERED_PRONOUNS = re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", flags=re.IGNORECASE)


def mention_order(sentence: str, referents: list[str]) -> list[str]:
    """
    Referents sorted by where the sentence first mentions them.

    :param sentence: template with BLANK
    :param referents: the pair's two referents
    :return: referents, earliest mention first
    """
    positions = {}
    for referent in referents:
        match = re.search(r"\b" + re.escape(referent.lower()) + r"\b", sentence.lower())
        if match is None:
            raise ValueError(f"{referent!r} is not mentioned in {sentence!r}")
        positions[referent] = match.start()
    return sorted(referents, key=positions.get)


def build_refnull_inputs(pairs: pd.DataFrame, frame: str = REFNULL_FRAME) -> pd.DataFrame:
    """
    One row per sentence of pairs, with its referent-only prime.

    :param pairs: winopron_pairs.tsv
    :param frame: prime template with {first} and {second} placeholders
    :return: DataFrame with columns template, differences, referent, case, prime
    """
    rows = []
    for idx in pairs.index:
        referents = [pairs["referent_1"][idx], pairs["referent_2"][idx]]
        for n in [1, 2]:
            template = pairs[f"template_{n}"][idx]
            first, second = mention_order(template, referents)
            prime = frame.format(first=first, second=second)
            if GENDERED_PRONOUNS.search(prime):
                raise ValueError(f"Referent-only prime contains a gendered pronoun: {prime!r}")
            rows.append({
                "template": template,
                "differences": pairs[f"differences_{n}"][idx],
                "referent": pairs[f"referent_{n}"][idx],
                "case": pairs[f"case_{n}"][idx],
                "prime": prime,
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", required=True, type=Path, help="winopron_pairs.tsv")
    parser.add_argument("--output", required=True, type=Path, help="CSV to write")
    parser.add_argument("--frame", default=REFNULL_FRAME, help="Prime template with {first} and {second}")
    args = parser.parse_args(argv)

    out = build_refnull_inputs(pd.read_csv(args.pairs, sep="\t"), frame=args.frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} referent-only primes to {args.output}, e.g. {out.prime[0]!r}")
    return args.output


if __name__ == "__main__":
    main()
